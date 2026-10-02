# 1017_mha — 稠密（bshd 4D batch）FlashAttention 前向的自包含 KernelBench 题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=1017, backend=triton。

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """稠密（padded 4D batch）FlashAttention 前向： scaled dot-product attention
    带在线 softmax、右下对齐因果掩码、ALiBi 位置偏置、MHA/GQA，并同时给出
    每行 log-sum-exp（LSE）。

    数学定义（对 batch 元素 b、query head h、query 位置 i、key 位置 j，
    g = num_q_heads // num_kv_heads，query head h 对应 KV head h // g）：
      score[b,h,i,j] = sm_scale * <q[b,i,h,:], k[b,j,h//g,:]>
                       - alibi_slopes[b,h] * |i + seqlen_k - seqlen_q - j|
      causal=1 时施加右下对齐因果掩码：j > i + seqlen_k - seqlen_q 的位置
      记 -inf（即 query i 只 attend key j <= i + seqlen_k - seqlen_q）；
      causal=0 时不施加任何掩码。ALiBi 偏置在因果/非因果两种模式下都施加，
      全 0 斜率等价于无 ALiBi。
      p[b,h,i,:] = softmax_j(score[b,h,i,:])   （数值稳定的在线实现）
      out[b,i,h,:] = sum_j p[b,h,i,j] * v[b,j,h//g,:]
      lse[b,h,i] = ln( sum_j exp(score[b,h,i,j]) )   （自然对数）
    softmax/累加在 float32 中完成，out cast 回输入 dtype，lse 为 float32。
    边界行为：causal=1 且 seqlen_q > seqlen_k 时，行 i < seqlen_q - seqlen_k
    整行被掩码——这些行的 out 为 0、lse 为 0（不是 -inf）。

    输入输出规格：
      q            [batch, seqlen_q, num_q_heads, head_dim]   float16/bfloat16
      k, v         [batch, seqlen_k, num_kv_heads, head_dim]  与 q 同 dtype
      causal       0-dim int32 张量，1=因果掩码，0=无掩码
      alibi_slopes [batch, num_q_heads] float32，元素 >= 0（全 0 即无 ALiBi）
      返回 (out, lse)：
      out          [batch, seqlen_q, num_q_heads, head_dim]，与 q 同 dtype
      lse          [batch, num_q_heads, seqlen_q]，float32
    全域约束：num_q_heads 是 num_kv_heads 的整数倍（相等即 MHA，否则 GQA）；
    1 <= seqlen_q、1 <= seqlen_k（任意组合，含 seqlen_q > seqlen_k）；
    1 <= head_dim <= 256。

    实现约束（违规判负）：
      - 核心计算（QK^T / ALiBi 偏置 / 因果掩码 / softmax / PV / LSE）必须在
        提交文件内完成，禁止调用现成的注意力 / 矩阵乘 / softmax 例程：
        scaled_dot_product_attention / sdpa / flash_attn / torch.matmul /
        torch.bmm / torch.einsum / torch.softmax / torch.logsumexp，以及任何
        第三方算子库的同义封装。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；out cast 回输入 dtype，lse 为 float32。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：稠密 4D 前向、非量化输入、无 bias、无 dropout、无滑窗。
    """

    def __init__(self, head_dim: int, sm_scale=None):
        super().__init__()
        self.sm_scale = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(head_dim)

    def forward(self, q, k, v, causal, alibi_slopes):
        # KernelBench 评测器会把全部输入张量 cast 成评测 precision（triton 后端
        # 为 fp32）；0/1 开关是小整数，fp32 可精确表示，此处无损恢复
        if not torch.is_tensor(causal):
            causal = torch.tensor(causal)
        is_causal = bool(int(causal.to(torch.int32)))
        alibi_slopes = alibi_slopes.to(torch.float32)

        batch, seqlen_q, nheads_q, head_dim = q.shape
        _, seqlen_k, nheads_k, _ = k.shape
        group = nheads_q // nheads_k
        assert nheads_q == nheads_k * group, "num_q_heads 必须是 num_kv_heads 的整数倍"

        qf = q.to(torch.float32).permute(0, 2, 1, 3)         # [B, Hq, Sq, D]
        kf = k.to(torch.float32).permute(0, 2, 1, 3)         # [B, Hk, Sq, D]
        vf = v.to(torch.float32).permute(0, 2, 1, 3)         # [B, Hk, Sk, D]

        # QK^T（先缩放 query），GQA：同一 KV head 的 K/V 复制给 group 内每个 head
        scores = torch.matmul(qf, kf.repeat_interleave(group, dim=1).transpose(-2, -1))
        scores = scores * self.sm_scale                       # [B, Hq, Sq, Sk]

        # ALiBi 偏置：-slope * |i + Sk - Sq - j|，加到缩放后的 score 上
        pos_q = torch.arange(seqlen_q, device=q.device, dtype=torch.float32)
        pos_k = torch.arange(seqlen_k, device=q.device, dtype=torch.float32)
        relative = (pos_q[:, None] + (seqlen_k - seqlen_q)) - pos_k[None, :]
        scores = scores - alibi_slopes[:, :, None, None] * relative.abs()

        if is_causal:
            # 右下对齐因果掩码：j > i + (Sk - Sq) 记 -inf
            causal_mask = torch.triu(
                torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device),
                diagonal=seqlen_k - seqlen_q + 1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # LSE：自然对数 log-sum-exp；整行掩码（causal 且 i < Sq - Sk）置 0
        lse = torch.logsumexp(scores, dim=-1)                 # [B, Hq, Sq]
        fully_masked = torch.isinf(lse) & (lse < 0)
        lse = torch.where(fully_masked, torch.zeros_like(lse), lse)

        # softmax + PV；整行掩码的行输出 0
        probs = torch.softmax(scores, dim=-1)                 # NaN 行 = 全掩码行
        acc = torch.matmul(probs, vf.repeat_interleave(group, dim=1))
        acc = torch.where(fully_masked[:, :, :, None], torch.zeros_like(acc), acc)

        out = acc.permute(0, 2, 1, 3).to(q.dtype)             # [B, Sq, Hq, D]
        return out, lse


def get_init_inputs():
    return [64]  # head_dim；sm_scale 缺省 1/sqrt(head_dim)


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    batch = 2
    seqlen_q = 256
    seqlen_k = 192            # Sq > Sk：causal 下前 64 行整行掩码（out=0, lse=0）
    num_kv_heads = 4
    query_group_size = 4      # num_q_heads = 16
    head_dim = 64

    num_q_heads = num_kv_heads * query_group_size
    q = (torch.randn(batch, seqlen_q, num_q_heads, head_dim) * 0.5).to(torch.float16)
    k = (torch.randn(batch, seqlen_k, num_kv_heads, head_dim) * 0.5).to(torch.float16)
    v = (torch.randn(batch, seqlen_k, num_kv_heads, head_dim) * 0.5).to(torch.float16)
    causal = torch.tensor(1, dtype=torch.int32)
    alibi_slopes = torch.rand(batch, num_q_heads)
    return [q, k, v, causal, alibi_slopes]


def make_inputs(batch: int, seqlen_q: int, seqlen_k: int, num_q_heads: int,
                num_kv_heads: int, head_dim: int, dtype: str = "float16",
                causal: int = 1, alibi: int = 1, seed: int = 0):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    alibi=1 生成 U[0,1) 随机斜率，alibi=0 生成全 0 斜率（等价于无 ALiBi 偏置）。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    q = (torch.randn(batch, seqlen_q, num_q_heads, head_dim, generator=gen) * 0.5).to(dt)
    k = (torch.randn(batch, seqlen_k, num_kv_heads, head_dim, generator=gen) * 0.5).to(dt)
    v = (torch.randn(batch, seqlen_k, num_kv_heads, head_dim, generator=gen) * 0.5).to(dt)
    if alibi:
        alibi_slopes = torch.rand(batch, num_q_heads, generator=gen)
    else:
        alibi_slopes = torch.zeros(batch, num_q_heads)
    causal_flag = torch.tensor(int(causal), dtype=torch.int32)
    return q, k, v, causal_flag, alibi_slopes
