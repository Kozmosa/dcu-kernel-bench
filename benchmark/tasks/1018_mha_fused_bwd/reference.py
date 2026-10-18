# 1018_mha_fused_bwd -- KernelBench-compatible task file for the FlashAttention
# backward fused form (a single call computes dQ/dK/dV together).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=1018, backend=triton.

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """FlashAttention backward (fused form): given the forward inputs Q/K/V and the
    upstream gradient dO, a single call computes the gradients dQ/dK/dV of the
    scaled dot-product attention forward O = softmax(scale·QK^T)·V with respect to
    Q, K and V. Supports MHA/GQA, dense fixed-length
    [batch, seqlen, heads, head_dim] (bshd) layout, non-causal (seqlen_q and
    seqlen_k may differ).

    Mathematical definition (g = num_q_heads // num_kv_heads,
    scale = 1/sqrt(head_dim), S_q/S_k are the sequence lengths of q/k, the KV head
    index for query head h is h_kv = h // g):
      S[b,h,i,j]     = scale * <q[b,i,h,:], k[b,j,h_kv,:]>
      P[b,h,i,j]     = softmax_j(S[b,h,i,:])     (non-causal, no mask)
      O[b,h,i,:]     = sum_j P[b,h,i,j] * v[b,j,h_kv,:]
      delta[b,h,i]   = <dO[b,i,h,:], O[b,h,i,:]> (= rowsum(dP∘P))
      dP[b,h,i,j]    = <dO[b,i,h,:], v[b,j,h_kv,:]>
      dS[b,h,i,j]    = P[b,h,i,j] * (dP[b,h,i,j] - delta[b,h,i])
      dQ[b,i,h,:]    = scale * sum_j dS[b,h,i,j] * k[b,j,h_kv,:]
      dK[b,j,h_kv,:] = scale * sum_{i, h: h//g = h_kv} dS[b,h,i,j] * q[b,i,h,:]
      dV[b,j,h_kv,:] = sum_{i, h: h//g = h_kv} P[b,h,i,j] * dO[b,i,h,:]
    i.e. dQ = scale·(dS·K), dK = scale·(dS^T·Q) (summed over the query heads inside
    each group of g), dV = P^T·dO (summed over the query heads inside each group of
    g). Intermediates are computed in float32 (numerically stable softmax), and
    each of the three gradients is cast back to the input dtype.
    Boundary behaviour: softmax normalizes along the k dimension, so the
    denominator is always positive; under GQA, dK/dV are the sum of the gradients
    over all query heads belonging to the same KV head, with a shape matching K/V.

    Input/output specification:
      q, do [batch, seqlen_q, num_q_heads, head_dim]  float16/bfloat16
      k, v  [batch, seqlen_k, num_kv_heads, head_dim] same dtype as q
      output: a 1-D tensor with the same dtype as q: dQ, dK, dV flattened in that
              order in their own bshd row-major layouts and then concatenated,
              length = batch*seqlen_q*num_q_heads*head_dim
                       + 2*batch*seqlen_k*num_kv_heads*head_dim
    Global-domain constraints: num_q_heads is an integer multiple of num_kv_heads
    (equal means MHA, otherwise GQA); 1 <= head_dim <= 256; seqlen_q >= 1,
    seqlen_k >= 1.

    Implementation constraints (violations are scored as failures):
      - The core computation (QK^T / recomputation of P and delta / chained
        gradient products) must be completed inside the submitted file; calling
        scaled_dot_product_attention / sdpa / flash_attn / torch.matmul /
        torch.bmm / torch.einsum / torch.softmax is forbidden, as is obtaining the
        gradients through torch.autograd or .backward().
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation; cast the output back to the
        input dtype.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: non-causal, dense fixed-length batch layout, no dropout, no
    bias/ALiBi, non-quantized inputs (fp16/bf16).
    """

    def __init__(self, head_size: int, scale=None):
        super().__init__()
        self.sm_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_size)

    def forward(self, q, k, v, do):
        # All inputs are float tensors (bshd layout); no integer recovery needed
        B, seqlen_q, num_q_heads, head_dim = q.shape
        _, seqlen_k, num_kv_heads, _ = k.shape
        group = num_q_heads // num_kv_heads
        assert num_q_heads == num_kv_heads * group, "num_q_heads must be an integer multiple of num_kv_heads"

        # bshd -> bhsd, all intermediate computation in float32
        qf = q.to(torch.float32).permute(0, 2, 1, 3)     # [B, Hq, Sq, D]
        kf = k.to(torch.float32).permute(0, 2, 1, 3)     # [B, Hk, Sk, D]
        vf = v.to(torch.float32).permute(0, 2, 1, 3)
        dof = do.to(torch.float32).permute(0, 2, 1, 3)

        # GQA: the K/V of one KV head is replicated to every query head in the group
        # (identity for MHA)
        kf = kf.repeat_interleave(group, dim=1) if group > 1 else kf     # [B, Hq, Sk, D]
        vf = vf.repeat_interleave(group, dim=1) if group > 1 else vf

        # Forward statistics (P and delta must be recomputed for the backward pass)
        scores = torch.matmul(qf, kf.transpose(-2, -1)) * self.sm_scale   # [B, Hq, Sq, Sk]
        p = torch.softmax(scores, dim=-1)
        o = torch.matmul(p, vf)                                          # [B, Hq, Sq, D]
        delta = (dof * o).sum(dim=-1, keepdim=True)                      # [B, Hq, Sq, 1]

        # Gradient chain: dP/dS/dQ/dK/dV
        dp = torch.matmul(dof, vf.transpose(-2, -1))                     # [B, Hq, Sq, Sk]
        ds = p * (dp - delta)
        dq = torch.matmul(ds, kf) * self.sm_scale                        # [B, Hq, Sq, D]
        dk_full = torch.matmul(ds.transpose(-2, -1), qf) * self.sm_scale # [B, Hq, Sk, D]
        dv_full = torch.matmul(p.transpose(-2, -1), dof)                 # [B, Hq, Sk, D]

        # GQA: dK/dV are summed over the query heads inside each group (identity for
        # MHA)
        dk = dk_full.reshape(B, num_kv_heads, group, seqlen_k, head_dim).sum(dim=2) if group > 1 else dk_full
        dv = dv_full.reshape(B, num_kv_heads, group, seqlen_k, head_dim).sum(dim=2) if group > 1 else dv_full

        # bhsd -> bshd, cast back to the input dtype, flatten and concatenate in
        # dQ|dK|dV order
        dq_flat = dq.transpose(1, 2).to(q.dtype).reshape(-1)
        dk_flat = dk.transpose(1, 2).to(k.dtype).reshape(-1)
        dv_flat = dv.transpose(1, 2).to(k.dtype).reshape(-1)
        return torch.cat([dq_flat, dk_flat, dv_flat])


def get_init_inputs():
    return [64]  # head_size; sm_scale defaults to 1/sqrt(head_size)


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the evaluator
    # calls this function after set_seed, so multiple correctness trials therefore
    # see diverse inputs
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
    """Generate inputs from a public/hidden/perf case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the
    device).

    Dense fixed-length layout: for every batch element the q/k sequence lengths
    are fixed at seqlen_q/seqlen_k respectively, and the two may be chosen
    independently; seq_lens is kept only for call-convention compatibility with the
    offline auditor (always None -- the dense layout has no variable-length
    sequences).
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert seq_lens is None, "the dense fixed-length layout has no variable-length sequences; seq_lens is not accepted"
    assert num_q_heads % num_kv_heads == 0, "num_q_heads must be an integer multiple of num_kv_heads"

    q = torch.randn(batch, seqlen_q, num_q_heads, head_size, generator=gen).to(dt)
    k = torch.randn(batch, seqlen_k, num_kv_heads, head_size, generator=gen).to(dt)
    v = torch.randn(batch, seqlen_k, num_kv_heads, head_size, generator=gen).to(dt)
    do = torch.randn(batch, seqlen_q, num_q_heads, head_size, generator=gen).to(dt)
    return q, k, v, do
