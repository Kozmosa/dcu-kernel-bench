# 1021 -- long-context (split two-pass) variant of paged KV decode attention:
# task prompt and evaluation ground truth.
#
# This file must be self-contained: the evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here. The semantics follow
# the official test's PyTorch reference implementation (single-token decode paged
# attention; for long sequences with max_seq_len > 8192 the production schedule
# switches to a two-pass form of "1024-token splits in parallel + cross-split
# merging"), intermediate computation is in float32, and the output is cast back
# to the input dtype. Boundary against the already-onboarded 1002: 1002 takes the
# short-context single-pass regime (max_seq_len <= 8192), while this task takes
# the long-context split two-pass regime -- the variant subsets do not overlap.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=1021, backend=triton.

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """Paged attention for the long-context decode phase (single-token query), for both MHA and GQA.

    Operator semantics: the KV cache is paged with a fixed block_size in the
    physical block pool key_cache / value_cache, and each sequence is addressed
    indirectly through the physical block ids in block_tables; the KV length that
    actually takes part in attention for each sequence is given by seq_lens[s]
    (>=1); slots of the physical block pool that nothing references, and slots of
    the tail block beyond seq_lens[s], are garbage data and must not take part in
    the computation. For query head h of sequence s (its KV head is
    h // query_group_size; let bs = block_size, bt = block_tables):
      logits[l] = scale * <query[s,h,:], key_cache[bt[s,l//bs], h_kv, l%bs, :]>
      p = softmax(logits[0:seq_lens[s]])
      out[s,h,:] = sum_l p[l] * value_cache[bt[s,l//bs], h_kv, l%bs, :]
    softmax is a numerically stable reduction over the whole
    [0, seq_lens[s]) interval.

    This task is in the long-context regime: the block-table capacity max_seq_len
    (16384 in the fixed family) far exceeds 8192, so a single sequence's KV can be
    long. How the work is parallelized is **not** a scoring criterion -- any
    implementation that is numerically equivalent to the full softmax within the
    tolerance is acceptable, and the implementation path is free. Core boundary
    cases: partially filled tail blocks (seq_len not a multiple of the block
    size), seq_len exactly a multiple of the block size, wildly different
    sequence lengths within one batch (as short as 1), and tail-block slot
    masking -- positions beyond seq_len must not contribute any probability mass.

    Input/output specification:
      query       [num_seqs, num_q_heads, head_size], fp16/bf16
      key_cache   [num_blocks, num_kv_heads, block_size, head_size], fp16/bf16
      value_cache same as key_cache
      block_tables[num_seqs, max_blocks_per_seq] int32, logical block -> physical block id
      seq_lens    [num_seqs] int32, each sequence's valid KV length, 1 <= seq_lens[s] <= max_seq_len
      output out  [num_seqs, num_q_heads, head_size], dtype matches query

    Implementation constraints (violations are scored as failures):
      - The core computation (QK^T / softmax / PV) must be done inside the
        submitted file; calling external attention operator libraries such as
        scaled_dot_product_attention / sdpa / flash_attn, and torch.matmul /
        torch.bmm / torch.einsum / torch.softmax, is forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation; cast the output back to the
        input dtype.
      - Unreferenced physical blocks and slots beyond seq_len must not affect the
        output.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: single-token decode, non-quantized KV, no ALiBi, no sliding
    window; the long-context regime (max(seq_lens) > 8192).
    """

    def __init__(self, head_size: int, scale=None):
        super().__init__()
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_size)

    def forward(self, query, key_cache, value_cache, block_tables, seq_lens):
        # The online evaluator casts every input tensor to the evaluation precision
        # (fp32 for the triton backend); block ids and sequence lengths are small
        # integers that fp32 represents exactly, so they are recovered losslessly
        block_tables = block_tables.to(torch.long)
        seq_lens = seq_lens.to(torch.int32)

        B, H_Q, D = query.shape
        H_KV = key_cache.shape[1]
        bs = key_cache.shape[2]
        group = H_Q // H_KV
        assert H_Q == H_KV * group, "num_q_heads must be an integer multiple of num_kv_heads"

        out = torch.empty_like(query)
        for s in range(B):
            seq_len = int(seq_lens[s])
            num_blocks = (seq_len + bs - 1) // bs
            physical = block_tables[s, :num_blocks]

            # Gather this sequence's K/V through the block table: [num_blocks, H_KV, bs, D] -> [H_KV, seq_len, D]
            k = key_cache[physical].permute(1, 0, 2, 3).reshape(H_KV, num_blocks * bs, D)[:, :seq_len].to(torch.float32)
            v = value_cache[physical].permute(1, 0, 2, 3).reshape(H_KV, num_blocks * bs, D)[:, :seq_len].to(torch.float32)

            # GQA: the K/V of one KV head is replicated to every query head in the group
            if group > 1:
                k = k.repeat_interleave(group, dim=0)   # [H_Q, seq_len, D]
                v = v.repeat_interleave(group, dim=0)

            q = query[s].to(torch.float32) * self.scale      # [H_Q, D]
            logits = torch.einsum("hd,hld->hl", q, k)   # [H_Q, seq_len]
            probs = torch.softmax(logits, dim=-1)
            out[s] = torch.einsum("hl,hld->hd", probs, v).to(query.dtype)
        return out


def get_init_inputs():
    return [128]  # head_size; scale defaults to 1/sqrt(head_size)


def get_inputs():
    # Fixed shape family (long-context regime: max_seq_len=16384 > 8192, the split
    # two-pass schedule triggers when max(seq_lens) > 8192); the random part
    # consumes the global RNG -- the evaluator calls this function after set_seed,
    # so multiple correctness trials therefore see diverse inputs
    num_seqs = 4
    num_kv_heads = 8
    query_group_size = 4      # num_q_heads = 32
    head_size = 128
    block_size = 16
    max_seq_len = 16384

    num_q_heads = num_kv_heads * query_group_size
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    # The physical block pool exceeds the actual usage: unreferenced slots are garbage data, used to check mask correctness
    num_blocks = num_seqs * max_blocks_per_seq + 8

    seq_lens = torch.randint(1, max_seq_len + 1, (num_seqs,), dtype=torch.int32)
    perm = torch.randperm(num_blocks)
    block_tables = perm[: num_seqs * max_blocks_per_seq].to(torch.int32).reshape(num_seqs, max_blocks_per_seq)

    query = torch.randn(num_seqs, num_q_heads, head_size).to(torch.float16)
    key_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_size).to(torch.float16)
    value_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_size).to(torch.float16)
    return [query, key_cache, value_cache, block_tables, seq_lens]


def make_inputs(num_seqs: int, num_q_heads: int, num_kv_heads: int,
                head_size: int, block_size: int, max_seq_len: int,
                dtype: str = "float16", seed: int = 0, seq_lens=None):
    """Deterministic case generator (shared by hidden/perf cases).

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device); all randomness comes from
    torch.Generator().manual_seed(seed). The parameter fields correspond one to
    one with the hidden/perf cases; seq_lens may be given explicitly (to construct
    split/tail-block boundaries exactly) and by default is sampled uniformly in
    [1, max_seq_len].
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    if seq_lens is None:
        seq_lens = torch.randint(1, max_seq_len + 1, (num_seqs,), generator=gen, dtype=torch.int32)
    else:
        if len(seq_lens) == 1 and num_seqs > 1:   # a single value broadcasts to the whole batch
            seq_lens = seq_lens * num_seqs
        seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        assert len(seq_lens) == num_seqs and int(seq_lens.max()) <= max_seq_len
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size

    # The physical block pool exceeds the actual usage: unreferenced slots are "garbage data", used to check mask correctness
    num_blocks = num_seqs * max_blocks_per_seq + 8
    perm = torch.randperm(num_blocks, generator=gen)
    block_tables = perm[: num_seqs * max_blocks_per_seq].reshape(num_seqs, max_blocks_per_seq).to(torch.int32)

    query = torch.randn(num_seqs, num_q_heads, head_size, generator=gen).to(dt)
    key_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_size, generator=gen).to(dt)
    value_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_size, generator=gen).to(dt)
    return query, key_cache, value_cache, block_tables, seq_lens
