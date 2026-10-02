# 1018_mha_fused_bwd — FlashAttention 反向 fused 形态（单次调用同时算
# dQ/dK/dV）的 KernelBench 兼容题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=1018, backend=triton。

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """FlashAttention 反向（fused 形态）：给定前向输入 Q/K/V 与上游梯度 dO，
    一次调用同时计算 scaled dot-product attention 前向
    O = softmax(scale·QK^T)·V 对 Q、K、V 三者的梯度 dQ/dK/dV。支持
    MHA/GQA，稠密定长 [batch, seqlen, heads, head_dim]（bshd）布局、非因果
    （seqlen_q 与 seqlen_k 可不同）。

    数学定义（g = num_q_heads // num_kv_heads，scale = 1/sqrt(head_dim)，
    S_q/S_k 为 q/k 的序列长度，query 头 h 对应的 kv 头编号 h_kv = h // g）：
      S[b,h,i,j]     = scale * <q[b,i,h,:], k[b,j,h_kv,:]>
      P[b,h,i,j]     = softmax_j(S[b,h,i,:])     （非因果，无掩码）
      O[b,h,i,:]     = sum_j P[b,h,i,j] * v[b,j,h_kv,:]
      delta[b,h,i]   = <dO[b,i,h,:], O[b,h,i,:]> （= rowsum(dP∘P)）
      dP[b,h,i,j]    = <dO[b,i,h,:], v[b,j,h_kv,:]>
      dS[b,h,i,j]    = P[b,h,i,j] * (dP[b,h,i,j] - delta[b,h,i])
      dQ[b,i,h,:]    = scale * sum_j dS[b,h,i,j] * k[b,j,h_kv,:]
      dK[b,j,h_kv,:] = scale * sum_{i, h: h//g = h_kv} dS[b,h,i,j] * q[b,i,h,:]
      dV[b,j,h_kv,:] = sum_{i, h: h//g = h_kv} P[b,h,i,j] * dO[b,i,h,:]
    即 dQ = scale·(dS·K)，dK = scale·(dS^T·Q)（按 g 组内 query 头求和），
    dV = P^T·dO（按 g 组内 query 头求和）。中间计算在 float32 中进行
    （softmax 数值稳定实现），三个梯度分别 cast 回输入 dtype。
    边界行为：softmax 沿 k 维归一化，分母恒为正；GQA 时 dK/dV 为同一
    kv 头对应的全部 query 头的梯度之和，形状与 K/V 一致。

    输入输出规格：
      q, do [batch, seqlen_q, num_q_heads, head_dim]  float16/bfloat16
      k, v  [batch, seqlen_k, num_kv_heads, head_dim] 与 q 同 dtype
      输出  1-D 张量，dtype 与 q 一致：dQ、dK、dV 依序按各自 bshd 行主序
            平铺后拼接，长度 = batch*seqlen_q*num_q_heads*head_dim
            + 2*batch*seqlen_k*num_kv_heads*head_dim
    全域约束：num_q_heads 是 num_kv_heads 的整数倍（相等即 MHA，否则
    GQA）；1 <= head_dim <= 256；seqlen_q >= 1、seqlen_k >= 1。

    实现约束（违规判负）：
      - 核心计算（QK^T / P 与 delta 重算 / 梯度链式乘）必须在提交文件内
        完成，禁止调用 scaled_dot_product_attention / sdpa / flash_attn /
        torch.matmul / torch.bmm / torch.einsum / torch.softmax，也禁止
        借助 torch.autograd 或 .backward() 反传求梯度。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：非因果、稠密定长 batch 布局、无 dropout、无 bias/ALiBi、
    非量化输入（fp16/bf16）。
    """

    def __init__(self, head_size: int, scale=None):
        super().__init__()
        self.sm_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_size)

    def forward(self, q, k, v, do):
        # 全部输入为浮点张量（bshd 布局），无需整型恢复
        B, seqlen_q, num_q_heads, head_dim = q.shape
        _, seqlen_k, num_kv_heads, _ = k.shape
        group = num_q_heads // num_kv_heads
        assert num_q_heads == num_kv_heads * group, "num_q_heads 必须是 num_kv_heads 的整数倍"

        # bshd -> bhsd，中间计算全部 float32
        qf = q.to(torch.float32).permute(0, 2, 1, 3)     # [B, Hq, Sq, D]
        kf = k.to(torch.float32).permute(0, 2, 1, 3)     # [B, Hk, Sk, D]
        vf = v.to(torch.float32).permute(0, 2, 1, 3)
        dof = do.to(torch.float32).permute(0, 2, 1, 3)

        # GQA：同一 kv 头的 K/V 复制给 group 内的每个 query 头（MHA 时恒等）
        kf = kf.repeat_interleave(group, dim=1) if group > 1 else kf     # [B, Hq, Sk, D]
        vf = vf.repeat_interleave(group, dim=1) if group > 1 else vf

        # 前向统计量（反向需要重算的 P 与 delta）
        scores = torch.matmul(qf, kf.transpose(-2, -1)) * self.sm_scale   # [B, Hq, Sq, Sk]
        p = torch.softmax(scores, dim=-1)
        o = torch.matmul(p, vf)                                          # [B, Hq, Sq, D]
        delta = (dof * o).sum(dim=-1, keepdim=True)                      # [B, Hq, Sq, 1]

        # 梯度链：dP/dS/dQ/dK/dV
        dp = torch.matmul(dof, vf.transpose(-2, -1))                     # [B, Hq, Sq, Sk]
        ds = p * (dp - delta)
        dq = torch.matmul(ds, kf) * self.sm_scale                        # [B, Hq, Sq, D]
        dk_full = torch.matmul(ds.transpose(-2, -1), qf) * self.sm_scale # [B, Hq, Sk, D]
        dv_full = torch.matmul(p.transpose(-2, -1), dof)                 # [B, Hq, Sk, D]

        # GQA：dK/dV 在 group 内对 query 头求和（MHA 时恒等）
        dk = dk_full.reshape(B, num_kv_heads, group, seqlen_k, head_dim).sum(dim=2) if group > 1 else dk_full
        dv = dv_full.reshape(B, num_kv_heads, group, seqlen_k, head_dim).sum(dim=2) if group > 1 else dv_full

        # bhsd -> bshd，cast 回输入 dtype，按 dQ|dK|dV 顺序平铺拼接
        dq_flat = dq.transpose(1, 2).to(q.dtype).reshape(-1)
        dk_flat = dk.transpose(1, 2).to(k.dtype).reshape(-1)
        dv_flat = dv.transpose(1, 2).to(k.dtype).reshape(-1)
        return torch.cat([dq_flat, dk_flat, dv_flat])


def get_init_inputs():
    return [64]  # head_size；sm_scale 缺省 1/sqrt(head_size)


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    batch = 2
    seqlen_q = 128
    seqlen_k = 256
    num_kv_heads = 4
    query_group_size = 4      # num_q_heads = 16
    head_size = 64

    num_q_heads = num_kv_heads * query_group_size
    q = torch.randn(batch, seqlen_q, num_q_heads, head_size).to(torch.float16)
    k = torch.randn(batch, seqlen_k, num_kv_heads, head_size).to(torch.float16)
    v = torch.randn(batch, seqlen_k, num_kv_heads, head_size).to(torch.float16)
    do = torch.randn(batch, seqlen_q, num_q_heads, head_size).to(torch.float16)
    return [q, k, v, do]


def make_inputs(batch: int, seqlen_q: int, seqlen_k: int, num_q_heads: int,
                num_kv_heads: int, head_size: int, dtype: str = "float16",
                seed: int = 0, seq_lens=None):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    稠密定长布局：每个 batch 元素的 q/k 序列长度分别固定为 seqlen_q/seqlen_k，
    两者可独立取值；seq_lens 仅为与离线审计器的调用约定兼容而保留（恒为
    None，稠密布局无变长序列）。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert seq_lens is None, "稠密定长布局无变长序列，不接受 seq_lens"
    assert num_q_heads % num_kv_heads == 0, "num_q_heads 必须是 num_kv_heads 的整数倍"

    q = torch.randn(batch, seqlen_q, num_q_heads, head_size, generator=gen).to(dt)
    k = torch.randn(batch, seqlen_k, num_kv_heads, head_size, generator=gen).to(dt)
    v = torch.randn(batch, seqlen_k, num_kv_heads, head_size, generator=gen).to(dt)
    do = torch.randn(batch, seqlen_q, num_q_heads, head_size, generator=gen).to(dt)
    return q, k, v, do
