# 1024_prefill_attention -- prefill context attention for a non-paged (contiguous
# KV) ragged variable-length batch: the multi-token query of each sequence runs
# (optionally causal) online-softmax self-attention over that sequence's
# contiguously stored KV, with GQA support -- a model_class (KernelBench
# compatible) task file.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=1024, backend=triton.

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """Non-paged prefill self-attention over a ragged variable-length batch (contiguous KV, GQA, causal optional).

    Operator semantics: the tokens of num_seqs sequences in the batch are stored
    flattened in dimension 0 of q/k/v; sequence i occupies the row range
    [b_start_loc[i], b_start_loc[i] + b_seq_len[i]), and b_start_loc is the
    exclusive prefix sum of the sequence lengths (b_start_loc[0] = 0,
    b_start_loc[i+1] = b_start_loc[i] + b_seq_len[i]), with no padding; the
    q/k/v of each sequence are the query/key/value of the same token batch (self
    attention). For the h-th query head (its KV head is
    g = h // query_group_size, query_group_size = num_q_heads // num_kv_heads)
    and in-sequence relative position t of sequence i
    (0 <= t < L_i, L_i = b_seq_len[i], row index s = b_start_loc[i] + t):
      logits[j] = sm_scale * <q[s, h, :], k[b_start_loc[i]+j, g, :]>
                  (sm_scale defaults to 1 / sqrt(head_dim))
      with causal=True, positions j > t are set to -inf; with causal=False all j are visible
      p[j] = softmax_j(logits[j])
      out[s, h, :] = sum_{j=0}^{L_i-1} p[j] * v[b_start_loc[i]+j, g, :]
    The softmax uses a numerically stable online implementation, intermediate
    accumulation is in float32, and the output is cast back to the input dtype.
    Boundary behaviour: b_seq_len[i] >= 1 always holds, the softmax denominator
    is always positive and there is no fully masked row (under causal masking row
    t sees at least itself); sequences do not affect each other.

    Input/output specification (evaluation dtype is float16 / bfloat16, all
    tensors contiguous):
      q            eval dtype (total_tokens, num_q_heads, head_dim)
      k            eval dtype (total_tokens, num_kv_heads, head_dim)
      v            eval dtype (total_tokens, num_kv_heads, head_dim)
      b_start_loc  int32 (num_seqs) -- exclusive prefix sum of the sequence lengths
      b_seq_len    int32 (num_seqs) -- length of each sequence (>= 1, may all differ)
      returns out  eval dtype (total_tokens, num_q_heads, head_dim)
    Global-domain constraints: num_q_heads is an integer multiple of num_kv_heads
    (GQA; equality means MHA); Q/K/V share the same head dim and it is >= 16,
    non-power-of-2 is allowed (out-of-range channels must be masked to zero by
    head_dim); total_tokens = sum(b_seq_len).

    Implementation constraints (violations are scored as failures):
      - The core computation (QK^T / softmax / PV) must be completed inside the
        submitted file; calling scaled_dot_product_attention / sdpa / flash_attn /
        aiter / torch.matmul / torch.bmm / torch.einsum / torch.softmax is
        forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation; the output is cast back to
        the input dtype.
      - The evaluator casts every input tensor to fp32 before calling forward;
        the prefix sum and the sequence lengths are small integers that fp32
        represents exactly, so .to(torch.int32) at the entry recovers them
        losslessly.
      - The maximum sequence length is not an input: an upper bound for the grid
        along the in-sequence position dimension can be derived inside forward
        from int(b_seq_len.max()), and the kernel masks out-of-range rows/columns
        per sequence using L_i.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: contiguous non-paged KV, ragged variable-length batch, GQA,
    causal/non-causal, fp16/bf16 non-quantized inputs; no paged KV cache, no
    prefix-pool indirect addressing, no custom mask, no sliding window, no
    attention sink, no dropout, no LSE output, no MLA absorb form -- those
    variants are left as separate follow-up tasks.
    """

    def __init__(self, causal: bool = True, sm_scale=None):
        super().__init__()
        self.causal = bool(causal)
        self.sm_scale = None if sm_scale is None else float(sm_scale)

    def forward(self, q, k, v, b_start_loc, b_seq_len):
        # The KernelBench evaluator casts every input tensor to the evaluation
        # precision (fp32 for the triton backend); the prefix sum and the
        # sequence lengths are small integers that fp32 represents exactly, so
        # the recovery here is lossless
        b_start_loc = b_start_loc.to(torch.int32)
        b_seq_len = b_seq_len.to(torch.int32)

        total, H_Q, D = q.shape
        H_KV = k.shape[1]
        group = H_Q // H_KV
        assert H_Q == H_KV * group, "num_q_heads must be an integer multiple of num_kv_heads"
        assert k.shape[-1] == D and v.shape[-1] == D, "the Q/K/V head dims must be identical"
        scale = self.sm_scale if self.sm_scale is not None else 1.0 / math.sqrt(D)
        B = b_seq_len.shape[0]

        out = torch.empty_like(q)
        for i in range(B):
            start = int(b_start_loc[i])
            seq_len = int(b_seq_len[i])

            q_i = q[start:start + seq_len].to(torch.float32)   # [L, H_Q, D]
            k_i = k[start:start + seq_len].to(torch.float32)   # [L, H_KV, D]
            v_i = v[start:start + seq_len].to(torch.float32)

            # GQA: the K/V of one KV head are replicated to every query head in the group
            if group > 1:
                k_i = k_i.repeat_interleave(group, dim=1)      # [L, H_Q, D]
                v_i = v_i.repeat_interleave(group, dim=1)

            scores = torch.einsum("qhd,khd->hqk", q_i, k_i) * scale   # [H_Q, L, L]
            if self.causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1
                )
                scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
            p = torch.softmax(scores, dim=-1)
            out[start:start + seq_len] = torch.einsum("hqk,khd->qhd", p, v_i).to(q.dtype)
        return out


def get_init_inputs():
    return [True]   # causal; sm_scale defaults to 1/sqrt(head_dim)


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs
    num_seqs = 4
    num_kv_heads = 4
    query_group_size = 4      # num_q_heads = 16
    head_dim = 64
    max_seq_len = 256

    num_q_heads = num_kv_heads * query_group_size
    seq_lens = torch.randint(1, max_seq_len + 1, (num_seqs,), dtype=torch.int32)
    b_start_loc = torch.cat(
        [torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0, dtype=torch.int32)[:-1]]
    )
    total_tokens = int(b_start_loc[-1] + seq_lens[-1])

    q = torch.randn(total_tokens, num_q_heads, head_dim).to(torch.float16)
    k = torch.randn(total_tokens, num_kv_heads, head_dim).to(torch.float16)
    v = torch.randn(total_tokens, num_kv_heads, head_dim).to(torch.float16)
    return [q, k, v, b_start_loc, seq_lens]


def make_inputs(num_seqs: int, num_q_heads: int, num_kv_heads: int,
                head_dim: int, max_seq_len: int, dtype: str = "float16",
                seed: int = 0, seq_lens=None, causal: bool = True):
    """Generate inputs from a public/hidden/perf case description. Runs on the evaluation side only (generated on CPU; the evaluator moves them to the device).

    seq_lens may be given explicitly (to construct boundary cases exactly; a
    single value is broadcast to the whole batch); by default it is sampled
    uniformly from [1, max_seq_len]. causal does not affect the values of the
    input tensors, it only records the Model init hyper-parameter this case
    corresponds to. The consumption order is fixed: seq_lens -> q -> k -> v, all
    from the same torch.Generator(seed), so the same seed reproduces the result
    bit for bit.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    if seq_lens is None:
        seq_lens = torch.randint(1, max_seq_len + 1, (num_seqs,),
                                 generator=gen, dtype=torch.int32)
    else:
        if len(seq_lens) == 1 and num_seqs > 1:   # a single value is broadcast to the whole batch
            seq_lens = list(seq_lens) * num_seqs
        seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        assert len(seq_lens) == num_seqs and int(seq_lens.min()) >= 1
        assert int(seq_lens.max()) <= max_seq_len

    b_start_loc = torch.cat(
        [torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0, dtype=torch.int32)[:-1]]
    )
    total_tokens = int(seq_lens.sum())

    q = torch.randn(total_tokens, num_q_heads, head_dim, generator=gen).to(dt)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, generator=gen).to(dt)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, generator=gen).to(dt)
    return q, k, v, b_start_loc, seq_lens
