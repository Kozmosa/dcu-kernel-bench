# 1022_pa_prefill -- prefill multi-token-query variant of paged KV prefix
# (context) attention (model_class self-contained task file).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here; make_inputs allows
# deterministic regeneration of the offline public/hidden/perf cases
# (independent torch.Generator seed, generated on CPU).
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=1022, backend=triton.

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """Paged KV prefix (context) attention: attention for prefill multi-token queries.

    (MHA/GQA, causal + optional ALiBi.)

    Within a batch, num_seqs sequences run in parallel; sequence b has
    query_len[b] = query_start_loc[b+1] - query_start_loc[b] newly computed query
    tokens in this round, and the slices of all sequences are flattened and
    concatenated along the token dimension in query_start_loc order (no
    padding). The current total length of the sequence is seq_lens[b], of which
    the first ctx_len[b] = seq_lens[b] - query_len[b] tokens are the already
    admitted prefix K/V, stored only in the paged KV cache; the K/V of the new
    tokens of this round are given explicitly only as key / value (the cache
    does not contain the new tokens). query_len[b] > 1 means prefill, and == 1
    is also legal (a single-token query against the pure prefix).

    Paged layout: the physical block pool has num_blocks blocks, each holding
    block_size token slots; block_tables[b] is the "prefix logical block ->
    physical block id" table of that sequence, where only the first
    ceil(ctx_len[b]/block_size) columns are valid (the remaining columns are
    garbage). K uses the channel-packed layout: the K channel d of the l-th
    prefix token is stored at
    key_cache[bt[b, l//block_size], h_kv, d//x, l%block_size, d%x]
    (x = 8, head_size is a multiple of 8); V is stored at
    value_cache[bt[b, l//block_size], h_kv, d, l%block_size]. Blocks in the pool
    that are not referenced, as well as slots beyond the prefix length inside
    referenced blocks, are garbage data (magnitudes up to +/-10) and must not
    participate in the computation.

    For the j-th new token of the h-th query head of sequence b (the
    corresponding KV head is h // query_group_size, query_group_size =
    num_heads / num_kv_heads), at in-sequence absolute position pos =
    ctx_len[b] + j:

      logits[l] = scale * <query[pos, h, :], K[l]> + alibi_slopes[h] * (l - pos)
      causal mask: logits at positions l > pos are set to -inf
      p[l] = exp(logits[l] - max_l' logits[l']) / sum_l' exp(logits[l'] - max_l' logits[l'])
      out[pos, h, :] = sum_{l = 0..pos} p[l] * V[l]

    where K[l] / V[l]: for l < ctx_len[b] they come from the paged cache
    (indirect addressing through the block table), and for l >= ctx_len[b] they
    come from key / value. When alibi_slopes is None no bias term is added (it
    degenerates to the unbiased baseline); when a float32 [num_heads] tensor is
    given, the ALiBi bias is added as in the formula above (the bias is 0 at
    l == pos). The softmax uses an online (numerically stable) implementation,
    intermediate accumulation is in float32, and the output is cast back to the
    input dtype.

    Input/output specification (forward parameter order):
      query           [num_tokens, num_heads, head_size]                    fp16/bf16
      key             [num_tokens, num_kv_heads, head_size]                 same as query
      value           [num_tokens, num_kv_heads, head_size]                 same as query
      key_cache       [num_blocks, num_kv_heads, head_size//8, block_size, 8]   same as query
      value_cache     [num_blocks, num_kv_heads, head_size, block_size]         same as query
      block_tables    [num_seqs, max_blocks_per_seq]                         int32
      query_start_loc [num_seqs+1] int32 (prefix sum of query_len, first element 0)
      seq_lens        [num_seqs] int32
      alibi_slopes    [num_heads] float32, or None (no ALiBi bias)
      returns out     [num_tokens, num_heads, head_size], dtype same as query.
      scale defaults to 1/sqrt(head_size) (given by head_size in __init__).

    Implementation constraints (violations are scored as failures):
      - The core computation (QK^T / softmax / PV) must be completed inside the
        submitted file; calling scaled_dot_product_attention / sdpa / flash_attn /
        torch.matmul / torch.bmm / torch.einsum / torch.softmax is forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation; the output is cast back to
        the input dtype.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: prefill multi-token queries, non-quantized KV; includes the ALiBi
    bias variant and the unbiased baseline (alibi_slopes passed as None); does
    not include sliding window, does not include a quantized KV cache, and the
    K/V of the new tokens of this round are not written into the cache.
    """

    def __init__(self, head_size: int, scale=None):
        super().__init__()
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_size)

    def forward(self, query, key, value, key_cache, value_cache, block_tables, query_start_loc, seq_lens, alibi_slopes=None):
        # The KernelBench evaluator casts every input tensor to the evaluation
        # precision (fp32 for the triton backend); block ids, lengths and prefix
        # sums are small integers that fp32 represents exactly, so the recovery
        # here is lossless
        block_tables = block_tables.to(torch.long)
        query_start_loc = query_start_loc.to(torch.int32)
        seq_lens = seq_lens.to(torch.int32)
        slopes = None if alibi_slopes is None else alibi_slopes.to(torch.float32)

        num_tokens, num_heads, head_size = query.shape
        num_kv_heads = key_cache.shape[1]
        block_size = value_cache.shape[3]
        x = key_cache.shape[4]
        group = num_heads // num_kv_heads
        assert num_heads == num_kv_heads * group, "num_heads must be an integer multiple of num_kv_heads"
        assert head_size == (head_size // x) * x, "head_size must be an integer multiple of x"

        out = torch.empty_like(query)
        for b in range(int(seq_lens.shape[0])):
            q_start = int(query_start_loc[b])
            q_len = int(query_start_loc[b + 1]) - q_start
            seq_len = int(seq_lens[b])
            ctx_len = seq_len - q_len
            num_ctx_blocks = (ctx_len + block_size - 1) // block_size
            physical = block_tables[b, :num_ctx_blocks]

            # Collect the prefix K/V through the block table (channel packing ->
            # logical layout):
            #   key_cache  [nblk, Hkv, D//x, bs, x] -> permute(1,0,3,2,4) -> [Hkv, nblk*bs, D]
            #   value_cache[nblk, Hkv, D, bs]      -> permute(1,0,3,2)   -> [Hkv, nblk*bs, D]
            k_ctx = key_cache[physical].permute(1, 0, 3, 2, 4).reshape(
                num_kv_heads, num_ctx_blocks * block_size, head_size)[:, :ctx_len].to(torch.float32)
            v_ctx = value_cache[physical].permute(1, 0, 3, 2).reshape(
                num_kv_heads, num_ctx_blocks * block_size, head_size)[:, :ctx_len].to(torch.float32)

            # K/V of the new tokens of this round (token dimension aligned with the
            # query, present only in key/value)
            k_new = key[q_start:q_start + q_len].to(torch.float32).permute(1, 0, 2)   # [Hkv, q_len, D]
            v_new = value[q_start:q_start + q_len].to(torch.float32).permute(1, 0, 2)

            k_full = torch.cat([k_ctx, k_new], dim=1)   # [Hkv, seq_len, D]
            v_full = torch.cat([v_ctx, v_new], dim=1)

            # GQA: the K/V of one KV head are replicated to every query head in the group
            if group > 1:
                k_full = k_full.repeat_interleave(group, dim=0)   # [H, seq_len, D]
                v_full = v_full.repeat_interleave(group, dim=0)

            q = query[q_start:q_start + q_len].to(torch.float32) * self.scale      # [q_len, H, D]

            logits = torch.einsum("qhd,hld->hql", q, k_full)                       # [H, q_len, seq_len]
            pos = ctx_len + torch.arange(q_len, device=query.device)
            causal = torch.arange(seq_len, device=query.device)[None, :] > pos[:, None]   # [q_len, seq_len]
            logits = logits.masked_fill(causal[None], float("-inf"))
            if slopes is not None:
                rel = torch.arange(seq_len, device=query.device)[None, :] - pos[:, None]
                bias = rel[None].to(torch.float32) * slopes.view(num_heads, 1, 1)
                logits = logits + bias.masked_fill(causal[None], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            out[q_start:q_start + q_len] = torch.einsum(
                "hql,hld->qhd", probs, v_full).to(query.dtype)
        return out


def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
    """ALiBi slopes (standard construction: geometric sequences in power-of-2 groups), float32 [total_num_heads]."""
    closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), dtype=torch.float32)
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)
    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), dtype=torch.float32)
        num_remaining_heads = min(closest_power_of_2, total_num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)])
    return slopes


def get_init_inputs():
    return [128]  # head_size; scale defaults to 1/sqrt(head_size)


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs. The online setting is the ALiBi bias
    # variant (alibi_slopes given as a tensor); the unbiased baseline is covered
    # by hidden cases with alibi_slopes=None.
    return list(_build_case(
        num_seqs=8, num_heads=32, num_queries_per_kv=4, head_size=128,
        block_size=32, max_query_len=32, max_ctx_len=64,
        dtype="float16", use_alibi=True,
    ))


def make_inputs(num_seqs: int, num_heads: int, num_queries_per_kv: int,
                head_size: int, block_size: int, max_query_len: int,
                max_ctx_len: int, dtype: str = "float16", seed: int = 0,
                query_lens=None, ctx_lens=None, use_alibi: bool = False,
                name: str = ""):
    """Deterministically generate inputs from a public/hidden/perf case description.

    Runs on the evaluation side only (independent seed, generated on CPU).
    query_lens / ctx_lens may be given explicitly (to construct boundary cases
    exactly); by default they are sampled following the official test
    convention: query_len uniformly in [16, max_query_len] and ctx_len uniformly
    in [16, max_ctx_len] (prefill semantics: every sequence has multiple tokens;
    the single-token and zero-prefix boundaries are constructed with explicit
    lens). A single-element list is broadcast to the whole batch. When use_alibi
    is True an ALiBi slope tensor is returned, otherwise that position is None.
    name is case-identifying metadata and does not participate in generation (it
    makes it easy to call make_inputs(**case) directly per case).
    """
    return _build_case(
        num_seqs=num_seqs, num_heads=num_heads,
        num_queries_per_kv=num_queries_per_kv, head_size=head_size,
        block_size=block_size, max_query_len=max_query_len,
        max_ctx_len=max_ctx_len, dtype=dtype,
        query_lens=query_lens, ctx_lens=ctx_lens, use_alibi=use_alibi,
        gen=torch.Generator().manual_seed(seed),
    )


def _build_case(num_seqs, num_heads, num_queries_per_kv, head_size,
                block_size, max_query_len, max_ctx_len, dtype="float16",
                query_lens=None, ctx_lens=None, use_alibi=False, gen=None):
    """Generate one complete case: gen=None consumes the global RNG (the get_inputs path)."""
    dt = getattr(torch, dtype)
    num_kv_heads = num_heads // num_queries_per_kv
    assert num_heads == num_kv_heads * num_queries_per_kv, \
        "num_heads must be an integer multiple of num_queries_per_kv"
    assert head_size % 8 == 0, "head_size must be a multiple of 8 (K channel packing x=8)"
    assert block_size >= 1

    # --- length sampling (endpoints closed, matching the official tests) ---
    if query_lens is None:
        assert max_query_len >= 16, "the default sampling requires max_query_len >= 16 (otherwise pass query_lens explicitly)"
        query_lens = torch.randint(16, max_query_len + 1, (num_seqs,), generator=gen).tolist()
    else:
        if len(query_lens) == 1 and num_seqs > 1:
            query_lens = list(query_lens) * num_seqs
        assert len(query_lens) == num_seqs, "the length of query_lens must equal num_seqs"
        assert all(1 <= l_ <= max_query_len for l_ in query_lens), "query_lens out of range"
    if ctx_lens is None:
        assert max_ctx_len >= 16, "the default sampling requires max_ctx_len >= 16 (otherwise pass ctx_lens explicitly)"
        ctx_lens = torch.randint(16, max_ctx_len + 1, (num_seqs,), generator=gen).tolist()
    else:
        if len(ctx_lens) == 1 and num_seqs > 1:
            ctx_lens = list(ctx_lens) * num_seqs
        assert len(ctx_lens) == num_seqs, "the length of ctx_lens must equal num_seqs"
        assert all(0 <= c <= max_ctx_len for c in ctx_lens), "ctx_lens out of range"
    seq_lens = [q + c for q, c in zip(query_lens, ctx_lens)]
    assert all(s >= 1 for s in seq_lens), "seq_lens must be >= 1"

    num_tokens = sum(query_lens)
    sum_seq_lens = sum(seq_lens)

    # The block table width covers the longest prefix plus 2 extra garbage
    # columns (checking "extra table columns"); the tail of the block pool holds
    # 8 additional unreferenced garbage blocks (checking "extra pool blocks +
    # garbage magnitudes")
    blocks_needed = (max_ctx_len + block_size - 1) // block_size
    max_blocks_per_seq = blocks_needed + 2
    num_blocks = num_seqs * max_blocks_per_seq + 8

    perm = torch.randperm(num_blocks, generator=gen)
    block_tables = perm[: num_seqs * max_blocks_per_seq].reshape(
        num_seqs, max_blocks_per_seq).to(torch.int32)

    query = (torch.rand(num_tokens, num_heads, head_size, generator=gen) * 0.2 - 0.1).to(dt)

    # Full-sequence K/V backing store (fp32, sequence-major); only the prefix
    # segment is written into the block pool, the new-token segment of this round
    # is copied into the token-major k/v (the cache does not contain the new
    # tokens)
    kv = torch.rand(sum_seq_lens, 2, num_kv_heads, head_size, generator=gen) * 0.2 - 0.1
    key_full, value_full = kv.unbind(dim=1)

    k = torch.zeros(num_tokens, num_kv_heads, head_size, dtype=dt)
    v = torch.zeros(num_tokens, num_kv_heads, head_size, dtype=dt)

    # Fill the pool with large garbage first (+/-10); the prefix slots are
    # overwritten afterwards, and the slots after the prefix keep the garbage
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
        q_len_i, ctx_len_i, seq_len_i = query_lens[i], ctx_lens[i], seq_lens[i]
        b_start_loc.append(b_start_loc[-1] + q_len_i)
        # The K/V of the new tokens come from the sequence tail
        # [seq_start+ctx, seq_start+seq_len)
        k[token_cursor: token_cursor + q_len_i] = key_full[seq_start + ctx_len_i: seq_start + seq_len_i].to(dt)
        v[token_cursor: token_cursor + q_len_i] = value_full[seq_start + ctx_len_i: seq_start + seq_len_i].to(dt)
        # Only the prefix [seq_start, seq_start+ctx) is written into the physical
        # blocks named by the block table (cur is the in-sequence prefix cursor,
        # and [slot, slot+span) is written inside the physical block)
        cur = 0
        block_id = 0
        while cur < ctx_len_i:
            end = min(cur + block_size, ctx_len_i)
            span = end - cur
            slot = int(block_tables[i, block_id]) * block_size
            k_cache_view[slot: slot + span] = key_full[seq_start + cur: seq_start + end].to(dt)
            v_cache_view[slot: slot + span] = value_full[seq_start + cur: seq_start + end].to(dt)
            cur = end
            block_id += 1
        seq_start += seq_len_i
        token_cursor += q_len_i

    # Convert to the paged layout expected by the kernel:
    #   K: [num_blocks, bs, Hkv, D] -> channel packed [num_blocks, Hkv, D//8, bs, 8]
    #   V: [num_blocks, bs, Hkv, D] ->               [num_blocks, Hkv, D, bs]
    key_cache = k_cache.view(num_blocks, block_size, num_kv_heads,
                             head_size // 8, 8).permute(0, 2, 3, 1, 4).contiguous()
    value_cache = v_cache.view(num_blocks, block_size, num_kv_heads,
                               head_size).permute(0, 2, 3, 1).contiguous()

    query_start_loc = torch.tensor(b_start_loc, dtype=torch.int32)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    alibi_slopes = _get_alibi_slopes(num_heads) if use_alibi else None
    return query, k, v, key_cache, value_cache, block_tables, query_start_loc, seq_lens_t, alibi_slopes
