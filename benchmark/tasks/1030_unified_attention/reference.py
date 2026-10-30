# 1030_unified_attention -- unified variable-length attention: one entry point
# handles prefill (q_s > 1 per sequence) and decode (q_s == 1) sequences mixed
# together, with the KV stored by fixed block_size pages in a physical block
# pool, supporting GQA, sliding window, logit softcap and attention sinks.
# KernelBench compatible form (Model / get_inputs / get_init_inputs).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=1030, backend=triton.

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """Unified variable-length attention: paged attention with prefill and decode sequences mixed.

    The num_seqs sequences in the batch share one packed query tensor: the
    queries of sequence s are stored contiguously in the row range
    [cu_seqlens_q[s], cu_seqlens_q[s+1]), q_s = cu_seqlens_q[s+1] -
    cu_seqlens_q[s] rows in total (q_s == 1 is a decode sequence, q_s > 1 is a
    prefill sequence, and the two may be mixed arbitrarily within the batch); the
    KV length of that sequence is kv_s = seq_lens[s], and kv_s >= q_s is
    guaranteed. The KV is not stored contiguously but addressed page-wise: the
    physical address of logical token position j (0 <= j < kv_s) is
      key_cache[block_tables[s, j // block_size], j % block_size, g, :]
      value_cache[block_tables[s, j // block_size], j % block_size, g, :]
    where g is the KV head index. Slots in the physical block pool that are not
    referenced are garbage data and must not participate in the computation; only
    the first ceil(kv_s / block_size) entries of each row of block_tables are
    valid.

    For the i-th query of sequence s (0 <= i < q_s, with in-KV-sequence absolute
    position T = kv_s - q_s + i and row index r = cu_seqlens_q[s] + i) and query
    head h (the corresponding KV head is g = h // query_group_size,
    query_group_size = num_q_heads // num_kv_heads):
      logits[j] = scale * <query[r, h, :], K_phys(j, g, :)>
                  scale defaults to 1 / sqrt(head_size)
      visibility (causal, query right-aligned against the KV): j <= T; if
      sliding_window > 0, additionally T - j < sliding_window (the left window edge)
      if use_sinks=True: one extra virtual sink column is introduced whose logit
      is always sinks[h] and whose value is always the zero vector, and which is
      never masked
      if softcap > 0: all logits (including the sink column) first go through
      c * tanh(logit / c) (c = softcap), and only then the mask and the softmax
      p = softmax(visible logits)
      out[r, h, :] = sum_j p[j] * V_phys(j, g, :)   (the sink column contributes the zero vector)
    The causal right alignment guarantees that every query sees at least its own
    position, so the softmax denominator is always > 0.

    Input/output specification (evaluation dtype is float16 / bfloat16, all
    tensors contiguous):
      query         eval dtype (total_q, num_q_heads, head_size)
                    total_q = cu_seqlens_q[-1]
      key_cache     eval dtype (num_blocks, block_size, num_kv_heads, head_size)
      value_cache   eval dtype, same shape as key_cache
      block_tables  int32 (num_seqs, max_blocks_per_seq) logical block -> physical block id
      seq_lens      int32 (num_seqs) valid KV length kv_s of each sequence (>= q_s)
      cu_seqlens_q  int32 (num_seqs + 1) prefix sum of the query lengths, first element 0
      sinks         eval dtype (num_q_heads) per-head sink logit
                    (ignored when use_sinks=False)
      returns out   eval dtype (total_q, num_q_heads, head_size), aligned with the query rows

    Implementation constraints (violations are scored as failures):
      - The core computation (QK^T / softmax / PV) must be completed inside the
        submitted file; calling scaled_dot_product_attention / sdpa / flash_attn /
        flex_attention / aiter / torch.matmul / torch.bmm / torch.einsum /
        torch.softmax is forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation; the output is cast back to the
        input dtype.
      - The evaluator casts every input tensor to fp32 before calling forward;
        block table ids, sequence lengths and the position prefix sums are small
        integers that fp32 represents exactly, so .to(torch.int32) /
        .to(torch.long) at the entry recovers them losslessly.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: paged KV + prefill/decode mixed ragged batch + GQA + causal right
    alignment + sliding window + logit softcap + attention sinks; no ALiBi, no
    query-query bias, no multimodal bidirectional prefix, no fp8 quantized KV with
    dequantization scales -- those variants are left as separate follow-up tasks.
    """

    def __init__(self, head_size: int, sliding_window: int = 0, softcap: float = 0.0, use_sinks: bool = True):
        super().__init__()
        self.scale = 1.0 / math.sqrt(head_size)
        self.sliding_window = int(sliding_window)
        self.softcap = float(softcap)
        self.use_sinks = bool(use_sinks)

    def forward(self, query, key_cache, value_cache, block_tables, seq_lens, cu_seqlens_q, sinks):
        # The KernelBench evaluator casts every input tensor to the evaluation
        # precision (fp32 for the triton backend); block table ids and the small
        # integer length inputs are exactly representable in fp32, so the
        # recovery here is lossless
        block_tables = block_tables.to(torch.long)
        seq_lens = seq_lens.to(torch.int32)
        cu_seqlens_q = cu_seqlens_q.to(torch.int32)

        num_seqs = seq_lens.shape[0]
        block_size = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        head_size = key_cache.shape[3]
        num_q_heads = query.shape[1]
        group = num_q_heads // num_kv_heads
        assert num_q_heads == num_kv_heads * group, \
            "num_q_heads must be an integer multiple of num_kv_heads"

        outputs = []
        for s in range(num_seqs):
            start = int(cu_seqlens_q[s])
            q_len = int(cu_seqlens_q[s + 1]) - start
            kv_len = int(seq_lens[s])

            q = query[start:start + q_len].to(torch.float32) * self.scale

            # Collect this sequence's K/V through the block table:
            # [nblk, bs, H_KV, D] -> [kv_len, H_KV, D]
            num_kv_blocks = (kv_len + block_size - 1) // block_size
            physical = block_tables[s, :num_kv_blocks]
            k = key_cache[physical].reshape(
                num_kv_blocks * block_size, num_kv_heads, head_size
            )[:kv_len].to(torch.float32)
            v = value_cache[physical].reshape(
                num_kv_blocks * block_size, num_kv_heads, head_size
            )[:kv_len].to(torch.float32)

            # GQA: the K/V of one KV head are replicated to every query head in the group
            if group > 1:
                k = k.repeat_interleave(group, dim=1)   # [kv_len, H_Q, D]
                v = v.repeat_interleave(group, dim=1)

            attn = torch.einsum("qhd,khd->hqk", q, k)   # [H_Q, q_len, kv_len]

            if self.use_sinks:
                # sink column: the logit is always sinks[h], the value is always the zero vector
                sink_logits = sinks[:num_q_heads].to(torch.float32)[:, None, None]
                sink_logits = sink_logits.expand(-1, q_len, 1)
                attn = torch.cat([sink_logits, attn], dim=-1)
                v = torch.cat([torch.zeros(1, num_q_heads, head_size,
                                           dtype=v.dtype, device=v.device),
                               v], dim=0)

            # causal mask (query right-aligned against the KV): j - i >
            # kv_len - q_len is invisible
            mask = torch.triu(torch.ones(q_len, kv_len, dtype=torch.bool,
                                         device=attn.device),
                              diagonal=kv_len - q_len + 1)
            if self.sliding_window > 0:
                sw_mask = ~torch.triu(
                    torch.ones(q_len, kv_len, dtype=torch.bool,
                               device=attn.device),
                    diagonal=kv_len - (q_len + self.sliding_window) + 1)
                mask = mask | sw_mask
            if self.softcap > 0:
                attn = self.softcap * torch.tanh(attn / self.softcap)
            if self.use_sinks:
                mask = torch.cat([torch.zeros(q_len, 1, dtype=torch.bool,
                                              device=mask.device), mask], dim=-1)

            attn = attn.masked_fill(mask, float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            outputs.append(torch.einsum("hqk,khd->qhd", attn, v))

        return torch.cat(outputs, dim=0).to(query.dtype)


def get_init_inputs():
    # head_size / sliding_window / softcap / use_sinks, matching the shape family
    # of get_inputs one to one (sliding window 64, softcap 50.0, sinks enabled)
    return [128, 64, 50.0, True]


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs
    num_seqs = 4
    num_kv_heads = 2
    query_group_size = 4      # num_q_heads = 8
    head_size = 128
    block_size = 16
    max_seq_len = 512

    num_q_heads = num_kv_heads * query_group_size
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    # The physical block pool is larger than what is used: unreferenced slots are
    # garbage data, used to check mask correctness
    num_blocks = num_seqs * max_blocks_per_seq + 8

    # Fixed mixing pattern: 2 decode (q=1) + 2 prefill (q=24 / q=9)
    query_lens = torch.tensor([1, 24, 1, 9], dtype=torch.int32)
    ctx = torch.randint(0, max_seq_len - 24 + 1, (num_seqs,), dtype=torch.int32)
    seq_lens = (query_lens + ctx).to(torch.int32)

    perm = torch.randperm(num_blocks)
    block_tables = perm[: num_seqs * max_blocks_per_seq].to(torch.int32) \
        .reshape(num_seqs, max_blocks_per_seq)
    cu_seqlens_q = torch.tensor([0] + query_lens.tolist(),
                                dtype=torch.int32).cumsum(0, dtype=torch.int32)

    total_q = int(cu_seqlens_q[-1])
    query = torch.randn(total_q, num_q_heads, head_size).to(torch.bfloat16)
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads,
                            head_size).to(torch.bfloat16)
    value_cache = torch.randn_like(key_cache)
    sinks = torch.randn(num_q_heads).to(torch.bfloat16)
    return [query, key_cache, value_cache, block_tables, seq_lens,
            cu_seqlens_q, sinks]


def make_inputs(num_seqs: int, num_q_heads: int, num_kv_heads: int,
                head_size: int, block_size: int, max_seq_len: int,
                sliding_window: int = 0, softcap: float = 0.0,
                use_sinks: bool = True, dtype: str = "bfloat16",
                seed: int = 0, query_lens=None, kv_lens=None, name=None):
    """Deterministic case generator (torch.Generator().manual_seed(seed)), generated on CPU.

    The fields map one to one onto the hidden/perf cases; sliding_window /
    softcap / use_sinks / name are init-side configuration (passed to
    Model.__init__ when the task is constructed) and do not participate in tensor
    generation; the sinks tensor is always generated (the forward signature has a
    fixed number of parameters) and is ignored when use_sinks=False. query_lens
    defaults to the mixed pattern (decode q=1 at even indices, prefill q=17 at odd
    indices); kv_lens defaults to sampling in [q_s, max_seq_len]; either of them
    can be broadcast to the whole batch from a single value. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the
    device).
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert num_q_heads % num_kv_heads == 0, \
        "num_q_heads must be an integer multiple of num_kv_heads"

    if query_lens is None:
        query_lens = [1 if i % 2 == 0 else 17 for i in range(num_seqs)]
    if len(query_lens) == 1 and num_seqs > 1:
        query_lens = list(query_lens) * num_seqs
    assert len(query_lens) == num_seqs
    query_lens_t = torch.tensor(query_lens, dtype=torch.int32)

    if kv_lens is None:
        max_q = int(query_lens_t.max())
        ctx = torch.randint(0, max_seq_len - max_q + 1, (num_seqs,),
                            generator=gen, dtype=torch.int32)
        kv_lens_t = query_lens_t + ctx
    else:
        if len(kv_lens) == 1 and num_seqs > 1:
            kv_lens = list(kv_lens) * num_seqs
        kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
    assert len(kv_lens_t) == num_seqs
    assert bool((kv_lens_t >= query_lens_t).all()), "kv_len must be >= query_len"
    assert int(kv_lens_t.max()) <= max_seq_len

    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    # The physical block pool is larger than what is used: unreferenced slots are
    # garbage data, used to check mask correctness
    num_blocks = num_seqs * max_blocks_per_seq + 8
    perm = torch.randperm(num_blocks, generator=gen)
    block_tables = perm[: num_seqs * max_blocks_per_seq] \
        .reshape(num_seqs, max_blocks_per_seq).to(torch.int32)

    cu_seqlens_q = torch.cat([
        torch.zeros(1, dtype=torch.int32),
        query_lens_t.cumsum(0, dtype=torch.int32)])
    total_q = int(cu_seqlens_q[-1])

    query = torch.randn(total_q, num_q_heads, head_size, generator=gen).to(dt)
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size,
                            generator=gen).to(dt)
    value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size,
                              generator=gen).to(dt)
    sinks = torch.randn(num_q_heads, generator=gen).to(dt)
    return query, key_cache, value_cache, block_tables, kv_lens_t, \
        cu_seqlens_q, sinks
