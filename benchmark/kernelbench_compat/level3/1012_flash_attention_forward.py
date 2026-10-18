# 1012_flash_attention_forward -- FlashAttention v2 forward pass over
# variable-length sequences (KernelBench compatible task file).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=1012, backend=triton.

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """Variable-length (varlen) FlashAttention v2 forward pass: dense Q/K/V is
    segmented by cu_seqlens and each segment computes scaled dot-product attention
    independently (online softmax semantics), supporting MHA/GQA and a causal
    mask.

    Mathematical definition: for segment b of the batch (let
    L_q = cu_seqlens_q[b+1]-cu_seqlens_q[b],
    L_k = cu_seqlens_k[b+1]-cu_seqlens_k[b]; the task constrains 0 < L_q <= L_k, and
    g = num_q_heads // num_kv_heads):
      scores[h,i,j] = sm_scale * <q[start_q+i,h,:], k[start_k+j,h//g,:]>
      with causal=1 a bottom-right-aligned causal mask is applied: positions with
      j > i + (L_k - L_q) are set to -inf (i.e. query position i only attends to key
      positions j <= i + L_k - L_q); with causal=0 no mask is applied.
      p[h,i,:] = softmax_j(scores[h,i,:])
      out[start_q+i,h,:] = sum_j p[h,i,j] * v[start_k+j,h//g,:]
    The softmax uses a numerically stable formulation; intermediates are accumulated
    in float32 and the output is cast back to the input dtype.
    Boundary behaviour: the task guarantees L_q >= 1 for every segment, so the
    softmax denominator is always positive and no row is fully masked; segments
    other than the last do not affect each other.

    Input/output specification:
      q            [total_q, num_q_heads, head_dim]  float16/bfloat16
      k, v         [total_k, num_kv_heads, head_dim] same dtype as q
      cu_seqlens_q [num_seqs+1] int32, monotonically non-decreasing, first element 0,
                   last element total_q
      cu_seqlens_k [num_seqs+1] int32, monotonically non-decreasing, first element 0,
                   last element total_k
      causal       0-dim int32 tensor, 1 = causal mask, 0 = no mask
      out          [total_q, num_q_heads, head_dim], same dtype as q
    Global constraints: num_q_heads is an integer multiple of num_kv_heads (GQA;
    equality means MHA); 1 <= head_dim <= 256; for every sequence
    0 < L_q <= L_k.

    Implementation constraints (violations are scored as failures):
      - The core computation (QK^T / softmax / PV) must be completed inside the
        submitted file; calling scaled_dot_product_attention / sdpa / flash_attn /
        aiter / torch.matmul / torch.bmm / torch.einsum / torch.softmax is
        forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Accumulate intermediates in float32; cast the output back to the input
        dtype.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: varlen forward, non-quantized inputs, no bias/ALiBi, no dropout, no
    LSE output.
    """

    def __init__(self, head_dim: int, scale=None):
        super().__init__()
        self.sm_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, causal):
        # The KernelBench evaluator casts every input tensor to the evaluation
        # precision (fp32 for the triton backend); the cumulative lengths and the
        # 0/1 flag are small integers exactly representable in fp32, restored
        # losslessly here.
        cu_seqlens_q = cu_seqlens_q.to(torch.int32)
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)
        if not torch.is_tensor(causal):
            causal = torch.tensor(causal)
        is_causal = bool(int(causal.to(torch.int32)))

        nheads_q = q.shape[1]
        nheads_k = k.shape[1]
        group = nheads_q // nheads_k
        assert nheads_q == nheads_k * group, "num_q_heads must be an integer multiple of num_kv_heads"
        num_seqs = int(cu_seqlens_q.numel()) - 1

        out = torch.empty_like(q)
        for b in range(num_seqs):
            start_q, end_q = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
            start_k, end_k = int(cu_seqlens_k[b]), int(cu_seqlens_k[b + 1])
            seq_len_q = end_q - start_q
            seq_len_k = end_k - start_k
            assert 0 < seq_len_q <= seq_len_k, "task constraint: every sequence satisfies 0 < L_q <= L_k"

            q_b = q[start_q:end_q].transpose(0, 1).to(torch.float32)   # [H_Q, Lq, D]
            k_b = k[start_k:end_k].transpose(0, 1).to(torch.float32)   # [H_K, Lk, D]
            v_b = v[start_k:end_k].transpose(0, 1).to(torch.float32)
            # GQA: the K/V of one KV head is replicated to every query head in the group
            if group > 1:
                k_b = k_b.repeat_interleave(group, dim=0)              # [H_Q, Lk, D]
                v_b = v_b.repeat_interleave(group, dim=0)

            scores = torch.matmul(q_b, k_b.transpose(-2, -1)) * self.sm_scale
            if is_causal:
                # Bottom-right-aligned causal mask (consistent with FlashAttention
                # semantics when L_q != L_k)
                mask = torch.triu(
                    torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=q.device),
                    diagonal=seq_len_k - seq_len_q + 1,
                )
                scores = scores.masked_fill(mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            o_b = torch.matmul(probs, v_b)                             # [H_Q, Lq, D]
            out[start_q:end_q] = o_b.transpose(0, 1).to(q.dtype)
        return out


def get_init_inputs():
    return [64]  # head_dim; sm_scale defaults to 1/sqrt(head_dim)


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the evaluator
    # calls this function after set_seed, so multiple correctness trials therefore
    # see diverse inputs.
    num_seqs = 4
    num_kv_heads = 4
    query_group_size = 4      # num_q_heads = 16
    head_dim = 64
    max_seq_len_q = 256
    max_seq_len_k = 512

    num_q_heads = num_kv_heads * query_group_size
    seq_lens_q = torch.randint(1, max_seq_len_q + 1, (num_seqs,), dtype=torch.int64)
    seq_lens_k = torch.randint(1, max_seq_len_k + 1, (num_seqs,), dtype=torch.int64)
    seq_lens_k = torch.maximum(seq_lens_k, seq_lens_q)   # every sequence satisfies L_q <= L_k

    cu_seqlens_q = torch.zeros(num_seqs + 1, dtype=torch.int32)
    cu_seqlens_k = torch.zeros(num_seqs + 1, dtype=torch.int32)
    for i in range(num_seqs):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + int(seq_lens_q[i])
        cu_seqlens_k[i + 1] = cu_seqlens_k[i] + int(seq_lens_k[i])

    total_q = int(seq_lens_q.sum())
    total_k = int(seq_lens_k.sum())
    q = (torch.randn(total_q, num_q_heads, head_dim) * 0.5).to(torch.float16)
    k = (torch.randn(total_k, num_kv_heads, head_dim) * 0.5).to(torch.float16)
    v = (torch.randn(total_k, num_kv_heads, head_dim) * 0.5).to(torch.float16)
    causal = torch.tensor(1, dtype=torch.int32)
    return [q, k, v, cu_seqlens_q, cu_seqlens_k, causal]


def make_inputs(num_seqs: int, num_q_heads: int, num_kv_heads: int,
                head_dim: int, max_seq_len_q: int, max_seq_len_k: int,
                dtype: str = "float16", causal: int = 1, seed: int = 0,
                seq_lens_q=None, seq_lens_k=None):
    """Generate inputs from a public/hidden/performance case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    seq_lens_q / seq_lens_k may be given explicitly (to construct boundary cases
    exactly; a single value broadcasts over the whole batch); when omitted,
    L_q ~ U[1, max_seq_len_q] and L_k ~ U[1, max_seq_len_k] followed by
    max(L_k, L_q) are sampled, always satisfying the task constraint
    0 < L_q <= L_k.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    if seq_lens_q is None:
        seq_lens_q = torch.randint(1, max_seq_len_q + 1, (num_seqs,), generator=gen, dtype=torch.int64)
    else:
        if len(seq_lens_q) == 1 and num_seqs > 1:   # a single value broadcasts over the whole batch
            seq_lens_q = seq_lens_q * num_seqs
        seq_lens_q = torch.tensor(seq_lens_q, dtype=torch.int64)
    if seq_lens_k is None:
        seq_lens_k = torch.randint(1, max_seq_len_k + 1, (num_seqs,), generator=gen, dtype=torch.int64)
        seq_lens_k = torch.maximum(seq_lens_k, seq_lens_q)
    else:
        if len(seq_lens_k) == 1 and num_seqs > 1:
            seq_lens_k = seq_lens_k * num_seqs
        seq_lens_k = torch.tensor(seq_lens_k, dtype=torch.int64)

    assert len(seq_lens_q) == num_seqs and len(seq_lens_k) == num_seqs
    assert int(seq_lens_q.max()) <= max_seq_len_q and int(seq_lens_k.max()) <= max_seq_len_k
    assert bool((seq_lens_q >= 1).all()), "every sequence satisfies L_q >= 1"
    assert bool((seq_lens_q <= seq_lens_k).all()), "task constraint: every sequence satisfies 0 < L_q <= L_k"

    cu_seqlens_q = torch.zeros(num_seqs + 1, dtype=torch.int32)
    cu_seqlens_k = torch.zeros(num_seqs + 1, dtype=torch.int32)
    for i in range(num_seqs):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + int(seq_lens_q[i])
        cu_seqlens_k[i + 1] = cu_seqlens_k[i] + int(seq_lens_k[i])

    total_q = int(seq_lens_q.sum())
    total_k = int(seq_lens_k.sum())
    q = (torch.randn(total_q, num_q_heads, head_dim, generator=gen) * 0.5).to(dt)
    k = (torch.randn(total_k, num_kv_heads, head_dim, generator=gen) * 0.5).to(dt)
    v = (torch.randn(total_k, num_kv_heads, head_dim, generator=gen) * 0.5).to(dt)
    causal_flag = torch.tensor(int(causal), dtype=torch.int32)
    return q, k, v, cu_seqlens_q, cu_seqlens_k, causal_flag
