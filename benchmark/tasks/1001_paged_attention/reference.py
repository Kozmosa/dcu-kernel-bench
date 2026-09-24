"""1001_paged_attention — 独立 PyTorch 参考实现（语义唯一依据）。

评测器以本文件生成的输出为 expected。计算在 float32 中完成，
最后 cast 回输入 dtype；生成 Agent 的 kernel 只需在容差内匹配。
"""


def reference(query, key_cache, value_cache, block_tables, seq_lens, scale=None):
    """单 token decode 的 paged attention（MHA / GQA 通用）。

    query       [B, H_Q, D]
    key_cache   [num_blocks, H_KV, block_size, D]
    value_cache [num_blocks, H_KV, block_size, D]
    block_tables[B, max_blocks_per_seq] int32 —— 逻辑块 -> 物理块编号
    seq_lens    [B] int32 —— 每个序列实际有效的 KV 长度
    scale       float，默认 1/sqrt(D)

    返回 out [B, H_Q, D]，dtype 与 query 一致。
    """
    import torch

    B, H_Q, D = query.shape
    H_KV = key_cache.shape[1]
    bs = key_cache.shape[2]
    group = H_Q // H_KV
    assert H_Q == H_KV * group, "num_q_heads 必须是 num_kv_heads 的整数倍"

    if scale is None:
        import math
        scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(query)
    for s in range(B):
        seq_len = int(seq_lens[s])
        num_blocks = (seq_len + bs - 1) // bs
        physical = block_tables[s, :num_blocks].to(torch.long)

        # 按块表收集本序列的 K/V：[num_blocks, H_KV, bs, D] -> [H_KV, seq_len, D]
        k = key_cache[physical].permute(1, 0, 2, 3).reshape(H_KV, num_blocks * bs, D)[:, :seq_len].to(torch.float32)
        v = value_cache[physical].permute(1, 0, 2, 3).reshape(H_KV, num_blocks * bs, D)[:, :seq_len].to(torch.float32)

        # GQA：同一 KV head 的 K/V 复制给 group 内的每个 query head
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
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    seq_lens 可显式指定（用于精确构造边界案例）；缺省时在 [1, max_seq_len] 均匀采样。
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    if seq_lens is None:
        seq_lens = torch.randint(1, max_seq_len + 1, (num_seqs,), generator=gen, dtype=torch.int32)
    else:
        if len(seq_lens) == 1 and num_seqs > 1:   # 单值广播到整个 batch
            seq_lens = seq_lens * num_seqs
        seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        assert len(seq_lens) == num_seqs and int(seq_lens.max()) <= max_seq_len
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size

    # 物理块池大于实际用量：未引用槽位是"垃圾数据"，用于检验掩码正确性
    num_blocks = num_seqs * max_blocks_per_seq + 8
    perm = torch.randperm(num_blocks, generator=gen)
    block_tables = perm[: num_seqs * max_blocks_per_seq].reshape(num_seqs, max_blocks_per_seq).to(torch.int32)

    query = torch.randn(num_seqs, num_q_heads, head_size, generator=gen).to(dt)
    key_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_size, generator=gen).to(dt)
    value_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_size, generator=gen).to(dt)
    return query, key_cache, value_cache, block_tables, seq_lens
