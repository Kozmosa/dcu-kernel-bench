"""1001_paged_attention -- standalone PyTorch reference implementation (the sole
semantic authority).

The evaluator uses the output produced by this file as `expected`. The
computation is done in float32 and finally cast back to the input dtype; the
kernel written by the generating Agent only needs to match within tolerance.
"""


def reference(query, key_cache, value_cache, block_tables, seq_lens, scale=None):
    """Paged attention for single-token decode (works for both MHA and GQA).

    query       [B, H_Q, D]
    key_cache   [num_blocks, H_KV, block_size, D]
    value_cache [num_blocks, H_KV, block_size, D]
    block_tables[B, max_blocks_per_seq] int32 -- logical block -> physical block id
    seq_lens    [B] int32 -- the actually valid KV length of each sequence
    scale       float, defaults to 1/sqrt(D)

    Returns out [B, H_Q, D] with the same dtype as query.
    """
    import torch

    B, H_Q, D = query.shape
    H_KV = key_cache.shape[1]
    bs = key_cache.shape[2]
    group = H_Q // H_KV
    assert H_Q == H_KV * group, "num_q_heads must be an integer multiple of num_kv_heads"

    if scale is None:
        import math
        scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(query)
    for s in range(B):
        seq_len = int(seq_lens[s])
        num_blocks = (seq_len + bs - 1) // bs
        physical = block_tables[s, :num_blocks].to(torch.long)

        # Gather this sequence's K/V according to the block table:
        # [num_blocks, H_KV, bs, D] -> [H_KV, seq_len, D]
        k = key_cache[physical].permute(1, 0, 2, 3).reshape(H_KV, num_blocks * bs, D)[:, :seq_len].to(torch.float32)
        v = value_cache[physical].permute(1, 0, 2, 3).reshape(H_KV, num_blocks * bs, D)[:, :seq_len].to(torch.float32)

        # GQA: the K/V of one KV head is replicated to every query head in the group
        if group > 1:
            k = k.repeat_interleave(group, dim=0)   # [H_Q, seq_len, D]
            v = v.repeat_interleave(group, dim=0)

        q = query[s].to(torch.float32) * scale      # [H_Q, D]
        logits = torch.einsum("hd,hld->hl", q, k)   # [H_Q, seq_len]
        probs = torch.softmax(logits, dim=-1)
        out[s] = torch.einsum("hl,hld->hd", probs, v).to(query.dtype)
    return out


def make_inputs(num_seqs: int, num_q_heads: int, num_kv_heads: int,
                head_size: int, block_size: int, max_seq_len: int,
                dtype: str = "float16", seed: int = 0, seq_lens=None):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    `seq_lens` may be given explicitly (to construct boundary cases exactly); when
    omitted it is sampled uniformly from [1, max_seq_len].
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    if seq_lens is None:
        seq_lens = torch.randint(1, max_seq_len + 1, (num_seqs,), generator=gen, dtype=torch.int32)
    else:
        if len(seq_lens) == 1 and num_seqs > 1:   # a single value broadcasts over the whole batch
            seq_lens = seq_lens * num_seqs
        seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        assert len(seq_lens) == num_seqs and int(seq_lens.max()) <= max_seq_len
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size

    # The physical block pool is larger than what is actually used: unreferenced
    # slots hold "garbage data", which exercises the masking.
    num_blocks = num_seqs * max_blocks_per_seq + 8
    perm = torch.randperm(num_blocks, generator=gen)
    block_tables = perm[: num_seqs * max_blocks_per_seq].reshape(num_seqs, max_blocks_per_seq).to(torch.int32)

    query = torch.randn(num_seqs, num_q_heads, head_size, generator=gen).to(dt)
    key_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_size, generator=gen).to(dt)
    value_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_size, generator=gen).to(dt)
    return query, key_cache, value_cache, block_tables, seq_lens
