# 1019_mha_onekernel_bwd — FlashAttention 反向 one-kernel（KernelBench 兼容题目文件）。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=1019, backend=triton。

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """FlashAttention 反向（one-kernel 形态）：给定前向输出 o 与逐行
    log-sum-exp 统计量 lse，由 (q, k, v, o, lse, do) 计算梯度 (dq, dk, dv)，
    dK/dV 与 dQ 在同一 kernel 内完成（dK/dV 先、dQ 后）。

    数学定义（记 g = num_q_heads // num_kv_heads，缩放后得分
    s[b,h,i,j] = sm_scale * <q[b,i,h,:], k[b,j,h//g,:]>，
    sm_scale 缺省 1/sqrt(head_dim)）：
      keep[b,i,j] = (j <= i + seqlen_k - seqlen_q) 当 causal=1（右下对齐），
                    恒真当 causal=0
      p[b,h,i,j]   = exp(s[b,h,i,j] - lse[b,h,i])，掩码位置（keep=0）恒为 0
      delta[b,h,i] = sum_d o[b,i,h,d] * do[b,i,h,d]   （由给定 o 计算）
      dp[b,h,i,j]  = <do[b,i,h,:], v[b,j,h//g,:]>
      ds[b,h,i,j]  = p[b,h,i,j] * (dp[b,h,i,j] - delta[b,h,i])
      dv[b,j,h_k,d] = sum_{h = h_k*g .. h_k*g+g-1} sum_i p[b,h,i,j] * do[b,i,h,d]
      dk[b,j,h_k,d] = sm_scale * sum_{h 同组} sum_i ds[b,h,i,j] * q[b,i,h,d]
      dq[b,i,h,d]   = sm_scale * sum_j ds[b,h,i,j] * k[b,j,h//g,d]
    注意 dK/dQ 末端乘 sm_scale 而 dV 不乘；delta 用输入 o（而非理想前向输出）。
    边界行为：causal=1 且 i < seqlen_q - seqlen_k 的行为全掩码行——p/ds 恒 0，
    dq 对应行全 0，对 dk/dv 无贡献（题面输入约定这些行 o=0、lse=0）。

    输入输出规格：
      q, o, do  [batch, seqlen_q, num_q_heads, head_dim] float16/bfloat16
      k, v      [batch, seqlen_k, num_kv_heads, head_dim] 与 q 同 dtype
      lse       [batch, num_q_heads, seqlen_q] float32——缩放后得分的自然对数
                log-sum-exp（全掩码行为 0）
      输出 (dq, dk, dv)：dq 与 q 同形、dk/dv 与 k 同形，dtype 同输入
    全域约束：num_q_heads 是 num_kv_heads 的整数倍（GQA，相等即 MHA）；
      1 <= head_dim <= 256（可非 2 次幂）；1 <= seqlen_q, seqlen_k；
      o 与 lse 是 (q, k, v) 的自洽前向统计量（题面生成器保证）。

    实现约束（违规判负）：
      - 核心计算（delta / p / ds / dV / dK / dQ 的全部矩阵乘与逐元素运算）
        必须在提交文件内完成，禁止调用 scaled_dot_product_attention / sdpa /
        flash_attn / torch.matmul / torch.bmm / torch.einsum /
        torch.softmax，也禁止 torch.autograd / autograd.grad / .backward(
        一类的自动微分捷径。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：稠密定长（非 varlen）、非量化输入、无 dropout / ALiBi / bias /
    fp8；causal 与 non-causal 经 __init__ 的 causal 开关选择。
    """

    def __init__(self, head_dim: int, causal: int, scale=None):
        super().__init__()
        self.head_dim = int(head_dim)
        self.causal = bool(int(causal))
        self.sm_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)

    def forward(self, q, k, v, o, lse, do):
        # 本题 forward 无整型输入（无索引/表/长度张量），无需无损恢复
        B, M, H_Q, D = q.shape
        _, N, H_K, _ = k.shape
        group = H_Q // H_K
        assert H_Q == H_K * group, "num_q_heads 必须是 num_kv_heads 的整数倍"
        causal = bool(self.causal)

        dq = torch.empty_like(q)
        dk = torch.zeros(B, N, H_K, D, dtype=torch.float32, device=q.device)
        dv = torch.zeros_like(dk)

        for b in range(B):
            for h in range(H_Q):
                hk = h // group    # GQA：组内全部 query head 共享一个 KV head
                qh = q[b, :, h, :].to(torch.float32)    # [M, D]
                kh = k[b, :, hk, :].to(torch.float32)   # [N, D]
                vh = v[b, :, hk, :].to(torch.float32)
                doh = do[b, :, h, :].to(torch.float32)
                oh = o[b, :, h, :].to(torch.float32)

                scores = torch.matmul(qh, kh.transpose(-2, -1)) * self.sm_scale
                if causal:
                    blocked = torch.triu(
                        torch.ones(M, N, dtype=torch.bool, device=q.device),
                        diagonal=N - M + 1,
                    )
                    scores = scores.masked_fill(blocked, float("-inf"))

                p = torch.exp(scores - lse[b, h].to(torch.float32).unsqueeze(-1))
                delta = torch.sum(oh * doh, dim=-1)                      # [M]
                dp = torch.matmul(doh, vh.transpose(-2, -1))             # [M, N]
                ds = p * (dp - delta.unsqueeze(-1))

                dv[b, :, hk, :] += torch.matmul(p.transpose(-2, -1), doh)
                dk[b, :, hk, :] += (
                    torch.matmul(ds.transpose(-2, -1), qh) * self.sm_scale
                )
                dq[b, :, h, :] = torch.matmul(ds, kh) * self.sm_scale

        return dq, dk.to(k.dtype), dv.to(v.dtype)


def _forward_stats(q, k, v, sm_scale, causal):
    """由 (q, k, v) 计算前向统计量：输出 o（输入 dtype）与 lse（float32，
    自然对数 log-sum-exp；causal 全掩码行 o=0、lse=0）。仅供输入生成器使用。
    """
    B, M, H_Q, D = q.shape
    _, N, H_K, _ = k.shape
    group = H_Q // H_K

    o = torch.empty_like(q)
    lse = torch.zeros(B, H_Q, M, dtype=torch.float32)
    for b in range(B):
        for h in range(H_Q):
            qh = q[b, :, h, :].to(torch.float32)
            kh = k[b, :, h // group, :].to(torch.float32)
            vh = v[b, :, h // group, :].to(torch.float32)

            scores = torch.matmul(qh, kh.transpose(-2, -1)) * sm_scale
            if causal:
                blocked = torch.triu(
                    torch.ones(M, N, dtype=torch.bool), diagonal=N - M + 1
                )
                scores = scores.masked_fill(blocked, float("-inf"))

            lse_h = torch.logsumexp(scores, dim=-1)
            if causal and M > N:
                # 全掩码行（i < M - N）：lse 置 0（与 FlashAttention 前向约定
                # 一致），随后 p=exp(-inf-0)=0、o 行为 0
                lse_h = torch.where(
                    torch.arange(M) < M - N, torch.zeros_like(lse_h), lse_h
                )
            p = torch.exp(scores - lse_h.unsqueeze(-1))
            o[b, :, h, :] = torch.matmul(p, vh).to(q.dtype)
            lse[b, h] = lse_h
    return o, lse


def get_init_inputs():
    return [64, 1]  # head_dim, causal；sm_scale 缺省 1/sqrt(head_dim)


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性（causal 与 get_init_inputs 一致取 1）
    batch = 2
    num_kv_heads = 2
    query_group_size = 4      # num_q_heads = 8
    head_dim = 64
    seqlen_q = 128
    seqlen_k = 128

    num_q_heads = num_kv_heads * query_group_size
    q = torch.randn(batch, seqlen_q, num_q_heads, head_dim).to(torch.float16)
    k = torch.randn(batch, seqlen_k, num_kv_heads, head_dim).to(torch.float16)
    v = torch.randn(batch, seqlen_k, num_kv_heads, head_dim).to(torch.float16)
    do = torch.randn(batch, seqlen_q, num_q_heads, head_dim).to(torch.float16)
    o, lse = _forward_stats(q, k, v, head_dim ** -0.5, causal=True)
    return [q, k, v, o, lse, do]


def make_inputs(batch: int, num_q_heads: int, num_kv_heads: int, head_dim: int,
                seqlen_q: int, seqlen_k: int, dtype: str = "float16",
                causal: int = 1, seed: int = 0):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到
    设备）。o 与 lse 由 (q, k, v) 以 float32 精确前向计算得到（全掩码行 o=0、
    lse=0），保证与题面语义自洽。"""
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert num_q_heads % num_kv_heads == 0, "num_q_heads 必须是 num_kv_heads 的整数倍"

    q = torch.randn(batch, seqlen_q, num_q_heads, head_dim, generator=gen).to(dt)
    k = torch.randn(batch, seqlen_k, num_kv_heads, head_dim, generator=gen).to(dt)
    v = torch.randn(batch, seqlen_k, num_kv_heads, head_dim, generator=gen).to(dt)
    do = torch.randn(batch, seqlen_q, num_q_heads, head_dim, generator=gen).to(dt)
    o, lse = _forward_stats(q, k, v, head_dim ** -0.5, causal=bool(causal))
    return q, k, v, o, lse, do
