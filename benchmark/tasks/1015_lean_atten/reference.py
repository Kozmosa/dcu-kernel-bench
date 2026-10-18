# 1015_lean_atten -- Lean Attention forward pass over ragged batches with
# variable-length KV (contiguous, non-paged KV; KernelBench compatible task file).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=1015, backend=triton.

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """Attention forward pass over ragged batches with variable-length KV (Lean
    Attention semantics: contiguous, non-paged KV, online softmax inside a single
    kernel), covering both decode (independent KV lengths per batch entry) and
    prefill (square causal self-attention) forms, with MHA (Q/K/V head counts
    equal).

    Mathematical definition: the batch holds num_seqs sequences, sequence b has an
    actually participating KV length kv_lens[b] (>= 1), and every query segment has
    the same length q_len (total_q = num_seqs * q_len, with q stored in
    sequence-major order). Let start_k = kv_lens[0] + ... + kv_lens[b-1] (k/v are
    likewise stored contiguously in sequence-major order). For every sequence b,
    head h, query position i (0 <= i < q_len) and KV position j
    (0 <= j < kv_lens[b]):
      scores[i, j] = sm_scale * <q[b*q_len+i, h, :], k[start_k+j, h, :]>
      with causal=1 (the task constrains kv_lens[b] == q_len for all b in this form)
      a square causal mask is applied: positions with j > i are set to -inf; with
      causal=0 no mask is applied;
      p[i, :] = softmax_j(scores[i, :])
      out[b*q_len+i, h, :] = sum_j p[i, j] * v[start_k+j, h, :]
    The softmax uses a numerically stable (online) formulation; intermediates are
    accumulated in float32 and the output is cast back to the input dtype.
    Boundary behaviour: kv_lens[b] >= 1 always holds, so the softmax denominator is
    always positive and no row is fully masked; sequences do not affect each other;
    slots in k/v beyond each sequence's cumulative length boundary do not
    participate in the computation.

    Input/output specification:
      q        [num_seqs*q_len, num_heads, head_dim]  float16/bfloat16, sequence-major
      k, v     [total_kv, num_heads, head_dim]        same dtype as q, total_kv = sum(kv_lens)
      kv_lens  [num_seqs] int32 -- the KV length actually participating per sequence
      causal   0-dim int32 tensor, 1 = square causal mask, 0 = no mask
      out      [num_seqs*q_len, num_heads, head_dim]  same dtype as q
    Global constraints: Q/K/V head counts are equal (MHA, no GQA); head_dim in
    {16, 32, 64, 128, 256} and identical across Q/K/V; q_len >= 1; with causal=1,
    kv_lens[b] == q_len for all b.

    Implementation constraints (violations are scored as failures):
      - The core computation (QK^T / softmax / PV) must be completed inside the
        submitted file; calling scaled_dot_product_attention / sdpa / flash_attn /
        aiter / torch.matmul / torch.bmm / torch.einsum / torch.softmax is
        forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Accumulate intermediates in float32; cast the output back to the input
        dtype.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: contiguous non-paged KV, variable-length ragged batch, MHA, both
    causal states, fp16/bf16 non-quantized inputs; no GQA, no dropout, no
    bias/ALiBi, no LSE output.
    """

    def __init__(self, head_dim: int, scale=None):
        super().__init__()
        self.sm_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)

    def forward(self, q, k, v, kv_lens, causal):
        # The KernelBench evaluator casts every input tensor to the evaluation
        # precision (fp32 for the triton backend); the KV lengths and the 0/1 flag
        # are small integers exactly representable in fp32, restored losslessly
        # here.
        kv_lens = kv_lens.to(torch.int32)
        if not torch.is_tensor(causal):
            causal = torch.tensor(causal)
        is_causal = bool(int(causal.to(torch.int32)))

        num_heads = q.shape[1]
        assert num_heads == k.shape[1] == v.shape[1], "MHA: Q/K/V must have the same number of heads"
        num_seqs = int(kv_lens.numel())
        q_len = q.shape[0] // num_seqs
        assert q_len * num_seqs == q.shape[0], "total_q must be an integer multiple of num_seqs"
        if is_causal:
            assert bool((kv_lens == q_len).all()), "with causal=1 every batch entry's kv_len must equal q_len"

        out = torch.empty_like(q)
        start_k = 0
        for b in range(num_seqs):
            kv_len = int(kv_lens[b])
            q_b = q[b * q_len : (b + 1) * q_len].transpose(0, 1).to(torch.float32)  # [H, q_len, D]
            k_b = k[start_k : start_k + kv_len].transpose(0, 1).to(torch.float32)   # [H, kv_len, D]
            v_b = v[start_k : start_k + kv_len].transpose(0, 1).to(torch.float32)

            scores = torch.matmul(q_b, k_b.transpose(-2, -1)) * self.sm_scale
            if is_causal:
                mask = torch.triu(
                    torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device),
                    diagonal=1,
                )
                scores = scores.masked_fill(mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            o_b = torch.matmul(probs, v_b)                                          # [H, q_len, D]
            out[b * q_len : (b + 1) * q_len] = o_b.transpose(0, 1).to(q.dtype)
            start_k += kv_len
        return out


def get_init_inputs():
    return [128]  # head_dim; sm_scale defaults to 1/sqrt(head_dim)


def get_inputs():
    # Fixed shape family (the decode form: causal=0, independent KV lengths per
    # batch entry); the random part consumes the global RNG -- the evaluator calls
    # this function after set_seed, so multiple correctness trials therefore see
    # diverse inputs.
    num_seqs = 4
    num_heads = 8
    head_dim = 128
    q_len = 16
    max_kv_len = 2048

    kv_lens = torch.randint(1, max_kv_len + 1, (num_seqs,), dtype=torch.int32)
    total_kv = int(kv_lens.sum())

    q = (torch.randn(num_seqs * q_len, num_heads, head_dim) * 0.5).to(torch.float16)
    k = (torch.randn(total_kv, num_heads, head_dim) * 0.5).to(torch.float16)
    v = (torch.randn(total_kv, num_heads, head_dim) * 0.5).to(torch.float16)
    causal = torch.tensor(0, dtype=torch.int32)
    return [q, k, v, kv_lens, causal]


def make_inputs(num_seqs: int, num_heads: int, head_dim: int, q_len: int,
                max_kv_len: int, dtype: str = "float16", causal: int = 0,
                seed: int = 0, kv_lens=None):
    """Generate inputs from a public/hidden/performance case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    causal=1 (square causal self-attention): every batch entry has
    kv_len == q_len; when kv_lens is omitted it is filled with q_len, and when
    given explicitly every entry must equal q_len (a single value [q_len] may
    broadcast).
    causal=0 (decode / non-causal): when kv_lens is omitted it is sampled uniformly
    from [1, max_kv_len]; when given explicitly (a single value broadcasts over the
    whole batch) it is used to construct boundary cases exactly.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert head_dim in (16, 32, 64, 128, 256), "head_dim must be in {16,32,64,128,256}"

    if causal:
        if kv_lens is None:
            kv_lens = [q_len] * num_seqs
        else:
            if len(kv_lens) == 1 and num_seqs > 1:   # a single value broadcasts over the whole batch
                kv_lens = kv_lens * num_seqs
            assert all(int(n) == q_len for n in kv_lens), "with causal=1 every batch entry's kv_len must equal q_len"
    else:
        if kv_lens is None:
            kv_lens = torch.randint(1, max_kv_len + 1, (num_seqs,), generator=gen, dtype=torch.int64).tolist()
        elif len(kv_lens) == 1 and num_seqs > 1:
            kv_lens = kv_lens * num_seqs
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32)
    assert int(kv_lens.numel()) == num_seqs
    assert 1 <= int(kv_lens.max()) <= max_kv_len, "kv_lens is outside [1, max_kv_len]"

    total_kv = int(kv_lens.sum())
    q = (torch.randn(num_seqs * q_len, num_heads, head_dim, generator=gen) * 0.5).to(dt)
    k = (torch.randn(total_kv, num_heads, head_dim, generator=gen) * 0.5).to(dt)
    v = (torch.randn(total_kv, num_heads, head_dim, generator=gen) * 0.5).to(dt)
    causal_flag = torch.tensor(int(causal), dtype=torch.int32)
    return q, k, v, kv_lens, causal_flag
