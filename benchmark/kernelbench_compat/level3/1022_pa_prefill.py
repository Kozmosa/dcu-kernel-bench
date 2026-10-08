# 1022_pa_prefill —— 分页 KV 前缀（context）attention 的 prefill 多 token query
# 变体（model_class 自包含题目文件）。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联；make_inputs 供离线公开/隐藏/性能
# case 确定性复生成（独立 torch.Generator 种子，CPU 生成）。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=1022, backend=triton。

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """分页 KV 前缀（context）attention：prefill 多 token query 的注意力（MHA/GQA，因果 + 可选 ALiBi）。

    批次内 num_seqs 个序列并行；第 b 个序列本轮新算的 query token 共
    query_len[b] = query_start_loc[b+1] - query_start_loc[b] 个，各序列的切片
    按 query_start_loc 展平拼接在 token 维上（无 padding）。序列当前总长
    seq_lens[b]，其中前 ctx_len[b] = seq_lens[b] - query_len[b] 个 token 是
    已入场的前缀 K/V，仅存于分页 KV Cache；本轮新 token 的 K/V 只以 key /
    value 显式给出（缓存中不含新 token）。query_len[b] > 1 即 prefill，
    == 1 亦合法（对纯前缀的单 token 查询）。

    分页布局：物理块池共 num_blocks 个块，每块 block_size 个 token 槽位；
    block_tables[b] 为该序列「前缀逻辑块 -> 物理块编号」表，仅前
    ceil(ctx_len[b]/block_size) 列有效（其余列为垃圾）。K 为通道打包布局：
    前缀第 l 个 token 的 K 通道 d 存于
    key_cache[bt[b, l//block_size], h_kv, d//x, l%block_size, d%x]
    （x = 8，head_size 为 8 的倍数）；V 存于
    value_cache[bt[b, l//block_size], h_kv, d, l%block_size]。块池中未被引用
    的块、以及被引用块中前缀长度之后的槽位均为垃圾数据（幅值可达 ±10），
    不得参与计算。

    对序列 b 的第 h 个 query head（对应 KV head 为 h // query_group_size，
    query_group_size = num_heads / num_kv_heads）的第 j 个新 token（序列内
    绝对位置 pos = ctx_len[b] + j）：

      logits[l] = scale * <query[pos, h, :], K[l]> + alibi_slopes[h] * (l - pos)
      因果掩码：l > pos 的位置 logits 置 -inf
      p[l] = exp(logits[l] - max_l' logits[l']) / sum_l' exp(logits[l'] - max_l' logits[l'])
      out[pos, h, :] = sum_{l = 0..pos} p[l] * V[l]

    其中 K[l] / V[l]：l < ctx_len[b] 取自分页缓存（块表间接寻址），
    l >= ctx_len[b] 取自 key / value。alibi_slopes 为 None 时不加偏置项
    （退化为无偏置基线）；给定 float32 [num_heads] 张量时按上式加 ALiBi
    偏置（l == pos 处偏置为 0）。softmax 采用在线（数值稳定）实现，中间
    累加为 float32，输出 cast 回输入 dtype。

    输入输出规格（forward 形参顺序）：
      query           [num_tokens, num_heads, head_size]                    fp16/bf16
      key             [num_tokens, num_kv_heads, head_size]                 同 query
      value           [num_tokens, num_kv_heads, head_size]                 同 query
      key_cache       [num_blocks, num_kv_heads, head_size//8, block_size, 8]   同 query
      value_cache     [num_blocks, num_kv_heads, head_size, block_size]        同 query
      block_tables    [num_seqs, max_blocks_per_seq]                         int32
      query_start_loc [num_seqs+1] int32（query_len 前缀和，首元素 0）
      seq_lens        [num_seqs] int32
      alibi_slopes    [num_heads] float32，或 None（无 ALiBi 偏置）
      返回 out        [num_tokens, num_heads, head_size]，dtype 与 query 一致。
      scale 缺省 1/sqrt(head_size)（由 __init__ 的 head_size 给出）。

    实现约束（违规判负）：
      - 核心计算（QK^T / softmax / PV）必须在提交文件内完成，禁止调用
        scaled_dot_product_attention / sdpa / flash_attn /
        torch.matmul / torch.bmm / torch.einsum / torch.softmax。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：prefill 多 token query、非量化 KV；含 ALiBi 偏置变体与无偏置
    基线（alibi_slopes 传 None）；不含滑窗、不含量化 KV cache、本轮新 token
    的 K/V 不写入缓存。
    """

    def __init__(self, head_size: int, scale=None):
        super().__init__()
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_size)

    def forward(self, query, key, value, key_cache, value_cache, block_tables, query_start_loc, seq_lens, alibi_slopes=None):
        # KernelBench 评测器会把全部输入张量 cast 成评测 precision（triton 后端
        # 为 fp32）；块编号、长度、前缀和都是小整数，fp32 可精确表示，此处无损恢复
        block_tables = block_tables.to(torch.long)
        query_start_loc = query_start_loc.to(torch.int32)
        seq_lens = seq_lens.to(torch.int32)
        slopes = None if alibi_slopes is None else alibi_slopes.to(torch.float32)

        num_tokens, num_heads, head_size = query.shape
        num_kv_heads = key_cache.shape[1]
        block_size = value_cache.shape[3]
        x = key_cache.shape[4]
        group = num_heads // num_kv_heads
        assert num_heads == num_kv_heads * group, "num_heads 必须是 num_kv_heads 的整数倍"
        assert head_size == (head_size // x) * x, "head_size 必须是 x 的整数倍"

        out = torch.empty_like(query)
        for b in range(int(seq_lens.shape[0])):
            q_start = int(query_start_loc[b])
            q_len = int(query_start_loc[b + 1]) - q_start
            seq_len = int(seq_lens[b])
            ctx_len = seq_len - q_len
            num_ctx_blocks = (ctx_len + block_size - 1) // block_size
            physical = block_tables[b, :num_ctx_blocks]

            # 前缀 K/V 按块表收集（通道打包 -> 逻辑布局）：
            #   key_cache  [nblk, Hkv, D//x, bs, x] -> permute(1,0,3,2,4) -> [Hkv, nblk*bs, D]
            #   value_cache[nblk, Hkv, D, bs]      -> permute(1,0,3,2)   -> [Hkv, nblk*bs, D]
            k_ctx = key_cache[physical].permute(1, 0, 3, 2, 4).reshape(
                num_kv_heads, num_ctx_blocks * block_size, head_size)[:, :ctx_len].to(torch.float32)
            v_ctx = value_cache[physical].permute(1, 0, 3, 2).reshape(
                num_kv_heads, num_ctx_blocks * block_size, head_size)[:, :ctx_len].to(torch.float32)

            # 本轮新 token 的 K/V（token 维与 query 对齐，仅存在于 key/value）
            k_new = key[q_start:q_start + q_len].to(torch.float32).permute(1, 0, 2)   # [Hkv, q_len, D]
            v_new = value[q_start:q_start + q_len].to(torch.float32).permute(1, 0, 2)

            k_full = torch.cat([k_ctx, k_new], dim=1)   # [Hkv, seq_len, D]
            v_full = torch.cat([v_ctx, v_new], dim=1)

            # GQA：同一 KV head 的 K/V 复制给 group 内每个 query head
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
    """ALiBi 斜率（标准构造：2 的幂次分组几何序列），float32 [total_num_heads]。"""
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
    return [128]  # head_size；scale 缺省 1/sqrt(head_size)


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性。在线口径为 ALiBi 偏置变体
    # （alibi_slopes 给定张量）；无偏置基线由隐藏 case 以 alibi_slopes=None
    # 覆盖。
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
    """按公开/隐藏/性能案例描述确定性生成输入（独立种子，CPU 生成）。

    仅在评测端运行。query_lens / ctx_lens 可显式指定（精确构造边界 case）；
    缺省按官方测试惯例采样：query_len 在 [16, max_query_len]、ctx_len 在
    [16, max_ctx_len] 内均匀采样（prefill 语义：全部序列多 token；单 token
    与零前缀等边界由显式 lens 构造）。单元素列表广播到整个 batch。
    use_alibi 为 True 时返回 ALiBi 斜率张量，否则该位置为 None。
    name 为 case 标识元数据，不参与生成（便于逐 case 直调
    make_inputs(**case)）。
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
    """生成一个完整 case：gen=None 时消费全局 RNG（get_inputs 路径）。"""
    dt = getattr(torch, dtype)
    num_kv_heads = num_heads // num_queries_per_kv
    assert num_heads == num_kv_heads * num_queries_per_kv, \
        "num_heads 必须是 num_queries_per_kv 的整数倍"
    assert head_size % 8 == 0, "head_size 必须是 8 的倍数（K 通道打包 x=8）"
    assert block_size >= 1

    # --- 长度采样（端点含闭，与官方测试一致） ---
    if query_lens is None:
        assert max_query_len >= 16, "缺省采样需要 max_query_len >= 16（否则请显式给出 query_lens）"
        query_lens = torch.randint(16, max_query_len + 1, (num_seqs,), generator=gen).tolist()
    else:
        if len(query_lens) == 1 and num_seqs > 1:
            query_lens = list(query_lens) * num_seqs
        assert len(query_lens) == num_seqs, "query_lens 长度必须等于 num_seqs"
        assert all(1 <= l_ <= max_query_len for l_ in query_lens), "query_lens 越界"
    if ctx_lens is None:
        assert max_ctx_len >= 16, "缺省采样需要 max_ctx_len >= 16（否则请显式给出 ctx_lens）"
        ctx_lens = torch.randint(16, max_ctx_len + 1, (num_seqs,), generator=gen).tolist()
    else:
        if len(ctx_lens) == 1 and num_seqs > 1:
            ctx_lens = list(ctx_lens) * num_seqs
        assert len(ctx_lens) == num_seqs, "ctx_lens 长度必须等于 num_seqs"
        assert all(0 <= c <= max_ctx_len for c in ctx_lens), "ctx_lens 越界"
    seq_lens = [q + c for q, c in zip(query_lens, ctx_lens)]
    assert all(s >= 1 for s in seq_lens), "seq_lens 必须 >= 1"

    num_tokens = sum(query_lens)
    sum_seq_lens = sum(seq_lens)

    # 块表宽度覆盖最长前缀并另加 2 列垃圾列（检验「表列多余」）；块池尾部
    # 多出 8 个未引用垃圾块，检验「块池多余 + 垃圾幅值」
    blocks_needed = (max_ctx_len + block_size - 1) // block_size
    max_blocks_per_seq = blocks_needed + 2
    num_blocks = num_seqs * max_blocks_per_seq + 8

    perm = torch.randperm(num_blocks, generator=gen)
    block_tables = perm[: num_seqs * max_blocks_per_seq].reshape(
        num_seqs, max_blocks_per_seq).to(torch.int32)

    query = (torch.rand(num_tokens, num_heads, head_size, generator=gen) * 0.2 - 0.1).to(dt)

    # 全序列 K/V 主存（fp32，序列主序）；仅前缀段写入块池，本轮新 token 段
    # 拷入 token 主序 k/v（缓存不含新 token）
    kv = torch.rand(sum_seq_lens, 2, num_kv_heads, head_size, generator=gen) * 0.2 - 0.1
    key_full, value_full = kv.unbind(dim=1)

    k = torch.zeros(num_tokens, num_kv_heads, head_size, dtype=dt)
    v = torch.zeros(num_tokens, num_kv_heads, head_size, dtype=dt)

    # 块池先填大幅垃圾（±10），前缀槽位随后覆写；前缀之后的槽位保持垃圾
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
        # 新 token 的 K/V 取自序列尾部 [seq_start+ctx, seq_start+seq_len)
        k[token_cursor: token_cursor + q_len_i] = key_full[seq_start + ctx_len_i: seq_start + seq_len_i].to(dt)
        v[token_cursor: token_cursor + q_len_i] = value_full[seq_start + ctx_len_i: seq_start + seq_len_i].to(dt)
        # 仅前缀 [seq_start, seq_start+ctx) 写入块表指定的物理块（cur 为序列内
        # 前缀游标，物理块内写 [slot, slot+span)）
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

    # 转成 kernel 期望的分页布局：
    #   K: [num_blocks, bs, Hkv, D] -> 通道打包 [num_blocks, Hkv, D//8, bs, 8]
    #   V: [num_blocks, bs, Hkv, D] ->            [num_blocks, Hkv, D, bs]
    key_cache = k_cache.view(num_blocks, block_size, num_kv_heads,
                             head_size // 8, 8).permute(0, 2, 3, 1, 4).contiguous()
    value_cache = v_cache.view(num_blocks, block_size, num_kv_heads,
                               head_size).permute(0, 2, 3, 1).contiguous()

    query_start_loc = torch.tensor(b_start_loc, dtype=torch.int32)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    alibi_slopes = _get_alibi_slopes(num_heads) if use_alibi else None
    return query, k, v, key_cache, value_cache, block_tables, query_start_loc, seq_lens_t, alibi_slopes
