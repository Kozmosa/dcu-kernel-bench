# 1017_mha -- self-contained KernelBench task file for dense (bshd 4D batch)
# FlashAttention forward.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=1017, backend=triton.

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """Dense (padded 4D batch) FlashAttention forward: scaled dot-product attention
    with softmax, bottom-right-aligned causal mask, ALiBi positional bias,
    MHA/GQA, returning both the output and the per-row log-sum-exp (LSE).

    Mathematical definition (for batch element b, query head h, query position i,
    key position j, g = num_q_heads // num_kv_heads, query head h maps to KV head
    h // g):
      score[b,h,i,j] = sm_scale * <q[b,i,h,:], k[b,j,h//g,:]>
                       - alibi_slopes[b,h] * |i + seqlen_k - seqlen_q - j|
      With causal=1 the bottom-right-aligned causal mask is applied: positions
      j > i + seqlen_k - seqlen_q are set to -inf (i.e. query i only attends to
      keys j <= i + seqlen_k - seqlen_q); with causal=0 no mask is applied at all.
      The ALiBi bias is applied in both causal and non-causal mode; all-zero
      slopes are equivalent to no ALiBi.
      p[b,h,i,:] = softmax_j(score[b,h,i,:])   (numerically stable online form)
      out[b,i,h,:] = sum_j p[b,h,i,j] * v[b,j,h//g,:]
      lse[b,h,i] = ln( sum_j exp(score[b,h,i,j]) )   (natural logarithm)
    softmax/accumulation are done in float32, out is cast back to the input dtype,
    lse is float32.
    Boundary behaviour: with causal=1 and seqlen_q > seqlen_k, rows
    i < seqlen_q - seqlen_k are masked out entirely -- those rows have out = 0 and
    lse = 0 (not -inf).

    Input/output specification:
      q            [batch, seqlen_q, num_q_heads, head_dim]   float16/bfloat16
      k, v         [batch, seqlen_k, num_kv_heads, head_dim]  same dtype as q
      causal       0-dim int32 tensor, 1=causal mask, 0=no mask
      alibi_slopes [batch, num_q_heads] float32, elements >= 0 (all zeros = no ALiBi)
      Returns a single 1-D tensor packed (multi-output packing protocol):
      packed = cat([out.reshape(-1).to(float32), lse.reshape(-1)])
      where out [batch, seqlen_q, num_q_heads, head_dim] (computed in the input
      dtype, then cast to float32 for the concatenation) and lse
      [batch, num_q_heads, seqlen_q] float32; total length of packed =
      batch * seqlen_q * num_q_heads * (head_dim + 1).
    Global-domain constraints: num_q_heads is an integer multiple of num_kv_heads
    (equal means MHA, otherwise GQA); 1 <= seqlen_q, 1 <= seqlen_k (any
    combination, including seqlen_q > seqlen_k); 1 <= head_dim <= 256.

    Implementation constraints (violations are scored as failures):
      - The core computation (QK^T / ALiBi bias / causal mask / softmax / PV / LSE)
        must be completed inside the submitted file; calling ready-made attention /
        matrix-multiplication / softmax routines is forbidden:
        scaled_dot_product_attention / sdpa / flash_attn / torch.matmul /
        torch.bmm / torch.einsum / torch.softmax / torch.logsumexp, as is any
        synonymous wrapper from a third-party operator library.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation; cast out back to the input
        dtype, keep lse in float32.
      - forward must return a single 1-D float32 tensor following the packing
        protocol above (compute out and lse internally first, then flatten and
        concatenate them).

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: dense 4D forward, non-quantized inputs, no bias, no dropout, no
    sliding window.
    """

    def __init__(self, head_dim: int, sm_scale=None):
        super().__init__()
        self.sm_scale = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(head_dim)

    def forward(self, q, k, v, causal, alibi_slopes):
        # The KernelBench evaluator casts every input tensor to the evaluation
        # precision (fp32 for the triton backend); the 0/1 switch is a small
        # integer exactly representable in fp32, so recovering it here is lossless
        if not torch.is_tensor(causal):
            causal = torch.tensor(causal)
        is_causal = bool(int(causal.to(torch.int32)))
        alibi_slopes = alibi_slopes.to(torch.float32)

        batch, seqlen_q, nheads_q, head_dim = q.shape
        _, seqlen_k, nheads_k, _ = k.shape
        group = nheads_q // nheads_k
        assert nheads_q == nheads_k * group, "num_q_heads must be an integer multiple of num_kv_heads"

        qf = q.to(torch.float32).permute(0, 2, 1, 3)         # [B, Hq, Sq, D]
        kf = k.to(torch.float32).permute(0, 2, 1, 3)         # [B, Hk, Sk, D]
        vf = v.to(torch.float32).permute(0, 2, 1, 3)         # [B, Hk, Sk, D]

        # QK^T (scale the query first), GQA: the K/V of one KV head is replicated to
        # every head in the group
        scores = torch.matmul(qf, kf.repeat_interleave(group, dim=1).transpose(-2, -1))
        scores = scores * self.sm_scale                       # [B, Hq, Sq, Sk]

        # ALiBi bias: -slope * |i + Sk - Sq - j|, added to the scaled scores
        pos_q = torch.arange(seqlen_q, device=q.device, dtype=torch.float32)
        pos_k = torch.arange(seqlen_k, device=q.device, dtype=torch.float32)
        relative = (pos_q[:, None] + (seqlen_k - seqlen_q)) - pos_k[None, :]
        scores = scores - alibi_slopes[:, :, None, None] * relative.abs()

        if is_causal:
            # Bottom-right-aligned causal mask: j > i + (Sk - Sq) becomes -inf
            causal_mask = torch.triu(
                torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device),
                diagonal=seqlen_k - seqlen_q + 1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # LSE: natural-log log-sum-exp; fully masked rows (causal with i < Sq - Sk)
        # are set to 0
        lse = torch.logsumexp(scores, dim=-1)                 # [B, Hq, Sq]
        fully_masked = torch.isinf(lse) & (lse < 0)
        lse = torch.where(fully_masked, torch.zeros_like(lse), lse)

        # softmax + PV; fully masked rows output 0
        probs = torch.softmax(scores, dim=-1)                 # NaN rows = fully masked rows
        acc = torch.matmul(probs, vf.repeat_interleave(group, dim=1))
        acc = torch.where(fully_masked[:, :, :, None], torch.zeros_like(acc), acc)

        out = acc.permute(0, 2, 1, 3).to(q.dtype)             # [B, Sq, Hq, D]
        # Multi-output packing protocol: single-tensor 1-D concatenation (the
        # evaluator only supports a single tensor output)
        return torch.cat([out.reshape(-1).to(torch.float32), lse.reshape(-1)])


def get_init_inputs():
    return [64]  # head_dim; sm_scale defaults to 1/sqrt(head_dim)


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the evaluator
    # calls this function after set_seed, so multiple correctness trials therefore
    # see diverse inputs
    batch = 2
    seqlen_q = 256
    seqlen_k = 192            # Sq > Sk: under causal the first 64 rows are fully masked (out=0, lse=0)
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
    """Generate inputs from a public/hidden/perf case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the
    device).

    alibi=1 generates U[0,1) random slopes, alibi=0 generates all-zero slopes
    (equivalent to no ALiBi bias).
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
