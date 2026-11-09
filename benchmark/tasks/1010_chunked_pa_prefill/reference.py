# 1010_chunked_pa_prefill -- paged attention over a chunked prefill/decode mixed
# batch (model_class form).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here; make_inputs allows
# the offline public/hidden/perf cases to be regenerated deterministically
# (independent torch.Generator seed, generated on CPU).
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=1010, backend=triton.

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """Chunked prefill / decode mixed-batch attention over a paged KV cache (MHA and GQA, causal mask).

    The batch holds num_seqs sequences running in parallel; sequence b computes
    query_len[b] = query_start_loc[b+1] - query_start_loc[b] new query tokens
    this round, and the per-sequence slices are packed back to back along the
    token dimension according to query_start_loc (no padding). Sequence b
    currently has total length seq_lens[b], of which the first
    ctx_len[b] = seq_lens[b] - query_len[b] tokens are already-admitted prefix
    K/V that live only in the paged KV cache; the K/V of this round's new tokens
    are given separately and explicitly as key / value (consistent with the
    corresponding cache slots). A sequence with query_len[b] == 1 is decode,
    > 1 is chunked prefill, and the two kinds may be mixed in the same batch.

    Paged layout: the physical block pool has num_blocks blocks, each holding
    block_size token slots; block_tables[b] is that sequence's "logical block ->
    physical block number" table, covering the whole seq_lens[b] (including this
    round's new tokens). K uses a channel-packed layout: K[tok, h, d] lives at
    key_cache[pblk, h, d // x, tok % block_size, d % x] (x = 8, head_size is a
    multiple of 8); V is value_cache[pblk, h, d, tok % block_size]. Blocks of the
    pool that no block table references, and slots past a sequence's valid
    length, hold garbage data (magnitudes up to +/-10) and must not take part in
    the computation.

    For query head h of sequence b (its KV head is h // query_group_size, where
    query_group_size = num_heads / num_kv_heads) and its j-th new token (absolute
    position within the sequence pos = ctx_len[b] + j):

      logits[l] = scale * <query[pos, h, :], K[l]>,  l = 0 .. seq_lens[b]-1
      causal mask: logits at positions l > pos are set to -inf
      p[l] = exp(logits[l] - max_l logits) / sum_l' exp(logits[l'] - max_l logits)
      out[pos, h, :] = sum_l p[l] * V[l]

    The first ctx_len[b] K / V tokens come from the paged cache (indirectly
    addressed through the block table), the remaining query_len[b] come from
    key / value. softmax uses an online (numerically stable) implementation,
    intermediate accumulation is in float32, and the output is cast back to the
    input dtype.

    Input/output specification (forward parameter order):
      query           [num_tokens, num_heads, head_size]          fp16/bf16
      key             [num_tokens, num_kv_heads, head_size]       same as query
      value           [num_tokens, num_kv_heads, head_size]       same as query
      key_cache       [num_blocks, num_kv_heads, head_size//8, block_size, 8]  same as query
      value_cache     [num_blocks, num_kv_heads, head_size, block_size]        same as query
      block_tables    [num_seqs, max_blocks_per_seq]              int32
      query_start_loc [num_seqs+1] int32 (prefix sum of query_len, first element 0)
      seq_lens        [num_seqs] int32
      output out      [num_tokens, num_heads, head_size], dtype matches query.

    Implementation constraints (violations are scored as failures):
      - The core computation (QK^T / softmax / PV) must be done inside the
        submitted file; calling scaled_dot_product_attention / sdpa / flash_attn
        / torch.matmul / torch.bmm / torch.einsum / torch.softmax is forbidden
        (the complete forbidden list is defined by the task configuration's
        forbidden field).
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation; cast the output back to the
        input dtype.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: chunked prefill/decode mixed batch, non-quantized KV, no ALiBi,
    no sliding window.
    """

    def __init__(self, head_size: int, scale=None):
        super().__init__()
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_size)

    def forward(self, query, key, value, key_cache, value_cache, block_tables, query_start_loc, seq_lens):
        # The KernelBench evaluator casts every input tensor to the evaluation
        # precision (fp32 for the triton backend); block ids, lengths and the
        # prefix sum are small integers that fp32 represents exactly, so they
        # are recovered losslessly here
        block_tables = block_tables.to(torch.long)
        query_start_loc = query_start_loc.to(torch.int32)
        seq_lens = seq_lens.to(torch.int32)

        num_tokens, num_heads, head_size = query.shape
        num_kv_heads = key_cache.shape[1]
        block_size = value_cache.shape[3]
        x = key_cache.shape[4]
        group = num_heads // num_kv_heads
        assert num_heads == num_kv_heads * group, "num_heads must be an integer multiple of num_kv_heads"

        out = torch.empty_like(query)
        for b in range(int(seq_lens.shape[0])):
            q_start = int(query_start_loc[b])
            q_len = int(query_start_loc[b + 1]) - q_start
            seq_len = int(seq_lens[b])
            ctx_len = seq_len - q_len
            num_blocks = (seq_len + block_size - 1) // block_size
            physical = block_tables[b, :num_blocks]

            # Gather prefix K/V through the block table (channel-packed -> logical layout):
            #   key_cache  [nblk, Hkv, D//x, bs, x] -> permute(1,0,3,2,4) -> [Hkv, nblk*bs, D]
            #   value_cache[nblk, Hkv, D, bs]      -> permute(1,0,3,2)   -> [Hkv, nblk*bs, D]
            k_ctx = key_cache[physical].permute(1, 0, 3, 2, 4).reshape(
                num_kv_heads, num_blocks * block_size, head_size)[:, :ctx_len].to(torch.float32)
            v_ctx = value_cache[physical].permute(1, 0, 3, 2).reshape(
                num_kv_heads, num_blocks * block_size, head_size)[:, :ctx_len].to(torch.float32)

            # K/V of this round's new tokens (token dimension aligned with query)
            k_new = key[q_start:q_start + q_len].to(torch.float32).permute(1, 0, 2)   # [Hkv, q_len, D]
            v_new = value[q_start:q_start + q_len].to(torch.float32).permute(1, 0, 2)

            k_full = torch.cat([k_ctx, k_new], dim=1)   # [Hkv, seq_len, D]
            v_full = torch.cat([v_ctx, v_new], dim=1)

            # GQA: the K/V of one KV head is replicated to every query head in the group
            if group > 1:
                k_full = k_full.repeat_interleave(group, dim=0)   # [H, seq_len, D]
                v_full = v_full.repeat_interleave(group, dim=0)

            q = query[q_start:q_start + q_len].to(torch.float32) * self.scale      # [q_len, H, D]

            logits = torch.einsum("qhd,hld->hql", q, k_full)                       # [H, q_len, seq_len]
            pos = ctx_len + torch.arange(q_len, device=query.device)
            causal = torch.arange(seq_len, device=query.device)[None, :] > pos[:, None]   # [1, q_len, seq_len]
            logits = logits.masked_fill(causal, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            out[q_start:q_start + q_len] = torch.einsum(
                "hql,hld->qhd", probs, v_full).to(query.dtype)
        return out


def get_init_inputs():
    return [128]  # head_size; scale defaults to 1/sqrt(head_size)


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the evaluator
    # calls this function after set_seed, so multiple correctness trials therefore
    # see diverse inputs. The length sampling mixes in decode
    # (query_lens[1] = query_lens[4] = query_lens[-1] = 1), exercising mixed-batch
    # semantics.
    return list(_build_case(
        num_seqs=8, num_heads=32, num_queries_per_kv=4, head_size=128,
        block_size=32, max_query_len=32, max_ctx_len=64,
        dtype="float16",
    ))


def make_inputs(num_seqs: int, num_heads: int, num_queries_per_kv: int,
                head_size: int, block_size: int, max_query_len: int,
                max_ctx_len: int, dtype: str = "float16", seed: int = 0,
                query_lens=None, ctx_lens=None, name: str = ""):
    """Generate inputs deterministically from a public/hidden/perf case description.

    Runs on the evaluation side only. query_lens / ctx_lens may be given
    explicitly (to construct boundary cases exactly); by default they are sampled
    following the official test convention: query_len uniform in
    [16, max_query_len], ctx_len uniform in [16, max_ctx_len], and when
    num_seqs > 5 decode sequences are always mixed in
    (query_lens[1] = query_lens[4] = query_lens[-1] = 1). A single-element list
    is broadcast to the whole batch. name is case-identifying metadata and does
    not take part in generation (it makes per-case direct calls
    make_inputs(**case) possible).
    """
    return _build_case(
        num_seqs=num_seqs, num_heads=num_heads,
        num_queries_per_kv=num_queries_per_kv, head_size=head_size,
        block_size=block_size, max_query_len=max_query_len,
        max_ctx_len=max_ctx_len, dtype=dtype,
        query_lens=query_lens, ctx_lens=ctx_lens,
        gen=torch.Generator().manual_seed(seed),
    )


def _build_case(num_seqs, num_heads, num_queries_per_kv, head_size,
                block_size, max_query_len, max_ctx_len, dtype="float16",
                query_lens=None, ctx_lens=None, gen=None):
    """Build one complete case: gen=None consumes the global RNG (get_inputs path)."""
    dt = getattr(torch, dtype)
    num_kv_heads = num_heads // num_queries_per_kv
    assert num_heads == num_kv_heads * num_queries_per_kv, \
        "num_heads must be an integer multiple of num_queries_per_kv"
    assert head_size % 8 == 0, "head_size must be a multiple of 8 (K channel packing x=8)"
    assert block_size >= 1

    # --- length sampling (endpoints inclusive, matching the official test) ---
    if query_lens is None:
        assert max_query_len >= 16, "default sampling needs max_query_len >= 16 (otherwise pass query_lens explicitly)"
        query_lens = torch.randint(16, max_query_len + 1, (num_seqs,), generator=gen).tolist()
        if num_seqs > 5:   # official test convention: always mix decode sequences into the batch
            query_lens[1] = 1
            query_lens[4] = 1
        query_lens[-1] = 1
    else:
        if len(query_lens) == 1 and num_seqs > 1:
            query_lens = list(query_lens) * num_seqs
        assert len(query_lens) == num_seqs, "query_lens length must equal num_seqs"
        assert all(1 <= l_ <= max_query_len for l_ in query_lens), "query_lens out of range"
    if ctx_lens is None:
        assert max_ctx_len >= 16, "default sampling needs max_ctx_len >= 16 (otherwise pass ctx_lens explicitly)"
        ctx_lens = torch.randint(16, max_ctx_len + 1, (num_seqs,), generator=gen).tolist()
    else:
        if len(ctx_lens) == 1 and num_seqs > 1:
            ctx_lens = list(ctx_lens) * num_seqs
        assert len(ctx_lens) == num_seqs, "ctx_lens length must equal num_seqs"
        assert all(0 <= c <= max_ctx_len for c in ctx_lens), "ctx_lens out of range"
    seq_lens = [q + c for q, c in zip(query_lens, ctx_lens)]
    assert all(s >= 1 for s in seq_lens), "seq_lens must be >= 1"

    num_tokens = sum(query_lens)
    sum_seq_lens = sum(seq_lens)

    # Block-table width is taken from the upper bound (max_query_len + max_ctx_len)
    # so coverage is guaranteed; the pool keeps 8 extra unreferenced garbage
    # blocks at the tail to check mask correctness
    max_blocks_per_seq = (max_query_len + max_ctx_len + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks_per_seq + 8

    perm = torch.randperm(num_blocks, generator=gen)
    block_tables = perm[: num_seqs * max_blocks_per_seq].reshape(
        num_seqs, max_blocks_per_seq).to(torch.int32)

    query = (torch.rand(num_tokens, num_heads, head_size, generator=gen) * 0.2 - 0.1).to(dt)

    # Whole-sequence K/V master copy (fp32, sequence-major); the new-token part is
    # also copied into the token-major k/v
    kv = torch.rand(sum_seq_lens, 2, num_kv_heads, head_size, generator=gen) * 0.2 - 0.1
    key_full, value_full = kv.unbind(dim=1)

    k = torch.zeros(num_tokens, num_kv_heads, head_size, dtype=dt)
    v = torch.zeros(num_tokens, num_kv_heads, head_size, dtype=dt)

    # Fill the block pool with large garbage (+/-10) first, then overwrite the
    # valid slots with whole sequences
    k_cache = (torch.rand(num_blocks, block_size, num_kv_heads, head_size,
                          generator=gen) * 20.0 - 10.0).to(dt)
    v_cache = (torch.rand(num_blocks, block_size, num_kv_heads, head_size,
                          generator=gen) * 20.0 - 10.0).to(dt)
    k_cache_view = k_cache.view(-1, num_kv_heads, head_size)
    v_cache_view = v_cache.view(-1, num_kv_heads, head_size)

    b_start_loc = [0]
    seq_start = 0
    token_cursor = 0
    for i in range(num_seqs):
        q_len_i, seq_len_i = query_lens[i], seq_lens[i]
        ctx_len_i = seq_len_i - q_len_i
        b_start_loc.append(b_start_loc[-1] + q_len_i)
        # The new tokens' K/V come from the sequence tail [seq_start+ctx, seq_start+seq_len)
        k[token_cursor: token_cursor + q_len_i] = key_full[seq_start + ctx_len_i: seq_start + seq_len_i].to(dt)
        v[token_cursor: token_cursor + q_len_i] = value_full[seq_start + ctx_len_i: seq_start + seq_len_i].to(dt)
        # The whole sequence (prefix + this round's new tokens) is written into the
        # physical blocks named by the block table (cur is the sequence cursor
        # starting at in-block offset 0; a physical block receives [slot, slot+span))
        cur = 0
        block_id = 0
        while cur < seq_len_i:
            end = min(cur + block_size, seq_len_i)
            span = end - cur
            slot = int(block_tables[i, block_id]) * block_size
            k_cache_view[slot: slot + span] = key_full[seq_start + cur: seq_start + end].to(dt)
            v_cache_view[slot: slot + span] = value_full[seq_start + cur: seq_start + end].to(dt)
            cur = end
            block_id += 1
        seq_start += seq_len_i
        token_cursor += q_len_i

    # Convert to the paged layout the kernel expects:
    #   K: [num_blocks, bs, Hkv, D] -> channel-packed [num_blocks, Hkv, D//8, bs, 8]
    #   V: [num_blocks, bs, Hkv, D] ->                [num_blocks, Hkv, D, bs]
    key_cache = k_cache.view(num_blocks, block_size, num_kv_heads,
                             head_size // 8, 8).permute(0, 2, 3, 1, 4).contiguous()
    value_cache = v_cache.view(num_blocks, block_size, num_kv_heads,
                               head_size).permute(0, 2, 3, 1).contiguous()

    query_start_loc = torch.tensor(b_start_loc, dtype=torch.int32)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    return query, k, v, key_cache, value_cache, block_tables, query_start_loc, seq_lens_t
