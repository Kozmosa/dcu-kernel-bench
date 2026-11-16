# 1029_triton_decode_attention -- two-stage GQA flash decoding (single-token
# decode, paged KV cache, split-KV two-stage reduction), a self-contained
# KernelBench task file.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=1029, backend=triton.

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """Decode-stage attention (single-token query) with paged KV.

    The KV cache is paged at page_size granularity: slot s of physical page p in
    key_cache / value_cache corresponds to flat slot number p * page_size + s.
    Sequence b is addressed indirectly through page_tables, and its l-th KV token
    (0 <= l < seq_lens[b]) lives in slot l % page_size of physical page
    page_tables[b, l // page_size]; slots in the page pool that are not
    referenced are garbage data and must not affect the output. Query head h
    corresponds to KV head h // query_group_size
    (query_group_size = num_q_heads // num_kv_heads, equal to 1 means MHA, equal
    to num_q_heads means MQA). For every sequence b and every query head h:
      logits[l] = scale * <q[b,h,:head_dim_qk],
                            key_flat[slot(b,l), h//group, :head_dim_qk]>
      p[l]      = exp(logits[l] - max(logits)) / sum(exp(logits - max(logits)))
      out[b,h,:head_dim_v] = sum_l p[l] * value_flat[slot(b,l), h//group, :head_dim_v]
    where key_flat / value_flat is the [num_pages*page_size, H_KV, D] flat view of
    the page pool, scale defaults to 1/sqrt(head_dim_qk), and the softmax must be
    numerically stable.

    Any implementation that is numerically equivalent within the tolerance is
    acceptable; how the work is parallelized is **not** a scoring criterion.

    Boundary behaviour:
      - Only the first seq_lens[b] KV tokens participate in the computation;
        seq_lens[b] >= 1 (an empty sequence is undefined).
      - page_size >= 1 is an arbitrary value (with page_size == 1 the page_tables
        are simply a per-token slot table).
      - head_dim_qk and head_dim_v may differ, and non-power-of-2 is allowed (for
        example 192/128, 576/512); the channels short of a power of 2 are handled
        by masking, so no out-of-bounds access occurs.
      - The page pool may be larger than what is actually used; page_tables
        entries beyond ceil(seq_lens[b]/page_size) point at valid but garbage
        pages that do not belong to this sequence.

    Inputs:
      q           [num_seqs, num_q_heads, head_dim_qk]  fp16/bf16 -- the query of
                  one new token per batch
      key_cache   [num_pages, page_size, num_kv_heads, head_dim_qk]
      value_cache [num_pages, page_size, num_kv_heads, head_dim_v]
      page_tables [num_seqs, max_pages_per_seq] int32 -- logical page -> physical page id
      seq_lens    [num_seqs] int32 -- the actually valid KV length of each sequence
    Output:
      out [num_seqs, num_q_heads, head_dim_v], dtype same as q.

    Implementation constraints (violations are scored as failures):
      - The core computation (QK^T / softmax / PV / cross-slice reduction) must
        be completed inside the submitted file; calling external attention
        operator libraries such as scaled_dot_product_attention / sdpa /
        flash_attn, as well as torch.matmul / torch.bmm / torch.einsum /
        torch.softmax, is forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation; the output is cast back to
        the input dtype.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: single-token decode, non-quantized KV, no logit_cap clamping, no
    ALiBi, no sliding window.
    """

    def __init__(self, page_size: int, head_dim_qk: int, head_dim_v: int, num_kv_splits: int = 8, scale=None):
        super().__init__()
        self.page_size = int(page_size)
        self.head_dim_qk = int(head_dim_qk)
        self.head_dim_v = int(head_dim_v)
        self.num_kv_splits = int(num_kv_splits)
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim_qk)

    def forward(self, q, key_cache, value_cache, page_tables, seq_lens):
        # The KernelBench evaluator casts every input tensor to the evaluation
        # precision (fp32 for the triton backend); the page table and the
        # sequence lengths are small integers that fp32 represents exactly, so
        # the recovery here is lossless
        page_tables = page_tables.to(torch.long)
        seq_lens = seq_lens.to(torch.int32)

        B, H_Q, D_QK = q.shape
        num_pages, page_size, H_KV, _ = key_cache.shape
        D_V = value_cache.shape[-1]
        group = H_Q // H_KV
        assert H_Q == H_KV * group, "num_q_heads must be an integer multiple of num_kv_heads"
        assert page_size == self.page_size, "the key_cache page size does not match the initialized page_size"

        # Flat view of the page pool: physical page p slot s -> flat slot number p*page_size+s
        key_flat = key_cache.view(num_pages * page_size, H_KV, D_QK)
        value_flat = value_cache.view(num_pages * page_size, H_KV, D_V)

        out = torch.empty(B, H_Q, D_V, dtype=q.dtype, device=q.device)
        for b in range(B):
            seq_len = int(seq_lens[b])
            offs = torch.arange(seq_len, device=q.device)
            slots = page_tables[b, offs // page_size] * page_size + offs % page_size

            # Gather this sequence's K/V through the page table: [seq_len, H_KV, D],
            # cast up to float32 after the gather
            k = key_flat[slots].to(torch.float32)
            v = value_flat[slots].to(torch.float32)

            # GQA: the K/V of one KV head are replicated to the contiguous group of query heads
            if group > 1:
                k = k.repeat_interleave(group, dim=1)   # [seq_len, H_Q, D_QK]
                v = v.repeat_interleave(group, dim=1)   # [seq_len, H_Q, D_V]

            qf = q[b].to(torch.float32) * self.scale     # [H_Q, D_QK]
            logits = torch.einsum("hd,lhd->hl", qf, k)   # [H_Q, seq_len]
            probs = torch.softmax(logits, dim=-1)
            out[b] = torch.einsum("hl,lhd->hd", probs, v).to(q.dtype)
        return out


def get_init_inputs():
    # page_size / head_dim_qk / head_dim_v / num_kv_splits;
    # scale defaults to 1/sqrt(head_dim_qk)
    return [16, 192, 128, 8]


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs
    num_seqs = 4
    num_kv_heads = 8
    query_group_size = 4      # num_q_heads = 32 -> GQA (grouped path)
    head_dim_qk = 192
    head_dim_v = 128
    page_size = 16
    max_seq_len = 1024

    num_q_heads = num_kv_heads * query_group_size
    max_pages_per_seq = (max_seq_len + page_size - 1) // page_size
    # The physical page pool is larger than what is used: unreferenced slots are
    # garbage data, used to check mask correctness
    num_pages = num_seqs * max_pages_per_seq + 8

    seq_lens = torch.randint(1, max_seq_len + 1, (num_seqs,), dtype=torch.int32)
    perm = torch.randperm(num_pages)
    page_tables = perm[: num_seqs * max_pages_per_seq].to(torch.int32).reshape(num_seqs, max_pages_per_seq)

    q = torch.randn(num_seqs, num_q_heads, head_dim_qk).to(torch.bfloat16)
    key_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim_qk).to(torch.bfloat16)
    value_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim_v).to(torch.bfloat16)
    return [q, key_cache, value_cache, page_tables, seq_lens]


def make_inputs(num_seqs: int, num_q_heads: int, num_kv_heads: int,
                head_dim_qk: int, head_dim_v: int, page_size: int,
                max_seq_len: int, dtype: str = "bfloat16", seed: int = 0,
                seq_lens=None):
    """Generate inputs from a public/hidden case description. Runs on the evaluation side only (generated on CPU; the evaluator moves them to the device).

    seq_lens may be given explicitly (to construct page-boundary cases exactly);
    by default it is sampled uniformly from [1, max_seq_len]. Returns
    [q, key_cache, value_cache, page_tables, seq_lens], where the physical page
    ids within a row of page_tables are all distinct (drawn by randperm).
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    if seq_lens is None:
        seq_lens = torch.randint(1, max_seq_len + 1, (num_seqs,), generator=gen, dtype=torch.int32)
    else:
        if len(seq_lens) == 1 and num_seqs > 1:   # a single value is broadcast to the whole batch
            seq_lens = seq_lens * num_seqs
        seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        assert len(seq_lens) == num_seqs and int(seq_lens.max()) <= max_seq_len
    max_pages_per_seq = (max_seq_len + page_size - 1) // page_size

    # The physical page pool is larger than what is used: unreferenced slots are
    # "garbage data", used to check mask correctness
    num_pages = num_seqs * max_pages_per_seq + 8
    perm = torch.randperm(num_pages, generator=gen)
    page_tables = perm[: num_seqs * max_pages_per_seq].reshape(num_seqs, max_pages_per_seq).to(torch.int32)

    q = torch.randn(num_seqs, num_q_heads, head_dim_qk, generator=gen).to(dt)
    key_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim_qk, generator=gen).to(dt)
    value_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim_v, generator=gen).to(dt)
    return [q, key_cache, value_cache, page_tables, seq_lens]
