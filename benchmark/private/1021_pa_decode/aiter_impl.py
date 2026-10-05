# aiter_impl.py — 1021_pa_decode 的 aiter 官方实现适配器
#
# 来源（aiter 本地 pinned 检出 .dcu_runs/aiter_pinned，commit c39fff8c）：
#   aiter/ops/triton/pa_decode.py
#   sha256 02bc511ba88a6faef06aa4cd6f348adf0ae8c23a81e18122dada5b399be34bf7
#   （与 benchmark/sources/1021_pa_decode.yaml 记录逐字节一致）
#
# 公开算子入口（pa_decode.py:18）：
#
#   paged_attention_decode(output, query, key_cache, value_cache, seq_lens,
#                          block_tables, attn_scale, max_seq_len, compute_type,
#                          k_scale, v_scale, num_seq_partitions=0,
#                          alibi_slopes=None) -> None     # void，就地写 output
#
# 调用约定来自 aiter 官方测试 op_tests/triton_tests/test_pa_decode.py
# （sha256 57d236538f9668b59afeb6f8d5d8aa3d38c5b89ad75a14ea21e6e2d8cfbf8361）：
# 测试把 max_context_len = max(context_lens) 传入，非量化路径传
# k_scale=v_scale=torch.tensor([1.0])（numel()==1 才走非量化分支，pa_decode.py:47）。
#
# 本题（1021）与 1002 是同一入口的**互斥调度子集**，适配器的差别只在分派断言：
#   pa_decode.py:42-46
#     max_num_partitions = ceil(max_seq_len / _SEQ_PARTITION_SIZE)   # 1024
#     use_v1 = max_seq_len <= 8192 and (max_num_partitions == 1 or num_seqs*num_q_heads > 512)
# 1021 的工况是 max(seq_lens) > 8192 → use_v1 必为 False → 走 v2 两 pass：
#   MHA (query_grp_sz == 1): _paged_attn_decode_v2_wo_dot_kernel +
#                            _paged_attn_decode_v2_wo_dot_reduce_kernel
#   GQA (query_grp_sz  > 1): _paged_attn_decode_v2_w_dot_kernel +
#                            _paged_attn_decode_v2_w_dot_reduce_kernel
# 本适配器显式断言分派落在 v2，落到 v1 就 raise（绝不静默按错路径记基线）。
#
# 布局说明：官方测试里 key_cache 的原生布局是 [num_blks, H_KV, D//x, KV_BLK_SZ, x]，
# 经 permute(0,1,3,2,4).flatten(3,4) 得到 kernel 消费布局
# [num_blks, H_KV, KV_BLK_SZ, D]。本评测集 make_inputs 产出的（也是 reference
# 使用的）正是后者 → 直接传入 key_cache / value_cache，无需再转换。
# 输出 out 为单张量 [num_seqs, num_q_heads, head_size]，与 reference 同形同 dtype，
# 无需打包（reference 的 forward 只返回一个张量）。
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。

import math

import torch
import triton.language as tl

# 与 aiter/ops/triton/pa_decode.py 的 _SEQ_PARTITION_SIZE 保持一致（pa_decode.py:15）
_SEQ_PARTITION_SIZE = 1024
# pa_decode.py:44 的 v1 触发阈值
_V1_MAX_SEQ_LEN = 8192

_COMPUTE_TYPES = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（v2 分区两 pass 路径）。

    inputs    : [query, key_cache, value_cache, block_tables, seq_lens]
                query       [B, H_Q, D]                     fp16/bf16
                key_cache   [num_blks, H_KV, bs, D]        同 query dtype
                value_cache [num_blks, H_KV, bs, D]        同 query dtype
                block_tables[B, max_blks_per_seq]          int32
                seq_lens    [B]                            int32
    init_kwargs: {"head_size": int}（Model.__init__(head_size, scale=None) 的按名取值；
                 scale 只在显式给出时覆盖默认 1/sqrt(head_size)）

    返回 (out, ctx)；out 为 [B, H_Q, D] 输出张量，dtype 与 query 一致。
    """
    from aiter.ops.triton.pa_decode import paged_attention_decode

    query, key_cache, value_cache, block_tables, seq_lens = inputs
    dtype = query.dtype

    # —— 构造参数：按名取值，缺失即 raise（不猜） ——
    if "head_size" not in init_kwargs:
        raise KeyError(f"init_kwargs 缺少 head_size：{sorted(init_kwargs)}")
    head_size = int(init_kwargs["head_size"])
    scale_arg = init_kwargs.get("scale", None)

    # —— dtype：题面 io 只声明 fp16/bf16（compute_type 决定 kernel 内 tl.dot 的精度）——
    if dtype not in _COMPUTE_TYPES:
        raise ValueError(f"1021 只覆盖非量化 KV；不支持的 query dtype: {dtype}")
    compute_type = _COMPUTE_TYPES[dtype]

    # —— 输入必须是 kernel 假定的紧凑布局（末维 stride==1），只做 layout 规整 ——
    query = query.contiguous()
    key_cache = key_cache.contiguous()
    value_cache = value_cache.contiguous()
    seq_lens_i32 = seq_lens.to(torch.int32).contiguous()
    block_tables_i32 = block_tables.to(torch.int32).contiguous()

    B, H_Q, D = query.shape
    H_KV = key_cache.shape[1]
    block_size = key_cache.shape[2]

    # —— shape 自洽性：aiter 从张量取 shape，head_size 只用于 scale；不一致必须暴露 ——
    if D != head_size or key_cache.shape[3] != head_size or value_cache.shape[3] != head_size:
        raise ValueError(
            f"head_size 不一致：init_kwargs={head_size}, query.shape={tuple(query.shape)}, "
            f"key_cache.shape={tuple(key_cache.shape)}, value_cache.shape={tuple(value_cache.shape)}"
        )
    if H_Q % H_KV != 0:
        raise ValueError(f"num_q_heads({H_Q}) 必须是 num_kv_heads({H_KV}) 的整数倍")
    if value_cache.shape[:3] != key_cache.shape[:3]:
        raise ValueError(
            f"key/value cache 形状不匹配：{tuple(key_cache.shape)} vs {tuple(value_cache.shape)}"
        )

    max_context_len = int(seq_lens_i32.max().item())
    if max_context_len < 1:
        raise ValueError(f"seq_lens 必须 >= 1，实测 max={max_context_len}")
    if int(block_tables_i32.shape[1]) * block_size < max_context_len:
        raise ValueError(
            f"block_tables 容量不足：{block_tables_i32.shape[1]} 块 x {block_size} "
            f"< max(seq_lens)={max_context_len}"
        )

    # —— aiter 的 v1/v2 分派规则（pa_decode.py:42-46），确认走的是本题准入的 v2 ——
    max_num_partitions = (max_context_len + _SEQ_PARTITION_SIZE - 1) // _SEQ_PARTITION_SIZE
    use_v1 = max_context_len <= _V1_MAX_SEQ_LEN and (
        max_num_partitions == 1 or B * H_Q > 512
    )
    if use_v1:
        raise RuntimeError(
            f"该 shape 会使 aiter 分派到 v1 路径（max_context_len={max_context_len}, "
            f"max_num_partitions={max_num_partitions}, B*H_Q={B * H_Q}），"
            "超出 1021 的准入范围（v2 分区两 pass, max(seq_lens) > 8192）"
        )

    # —— v2 kernel 用整数除法把分区映射到 KV 块（kv_blk_start =
    #    seq_part_idx * (SEQ_PARTITION_SZ // KV_BLK_SZ)），块长必须整除分区长度 ——
    if _SEQ_PARTITION_SIZE % block_size != 0:
        raise ValueError(
            f"block_size={block_size} 不能整除 _SEQ_PARTITION_SIZE={_SEQ_PARTITION_SIZE}，"
            "v2 kernel 的分区->块映射不成立"
        )

    scale = 1.0 / math.sqrt(head_size) if scale_arg is None else float(scale_arg)
    dev = query.device if device is None else device

    group = H_Q // H_KV
    out = torch.empty_like(query)

    # 非量化路径判据是 k_scale.numel() == 1（pa_decode.py:47），此处即 1；
    # num_seq_partitions 在源码中形参存在但未使用（pa_decode.py:30 TODO），
    # 与官方测试一致传 0。
    paged_attention_decode(
        out,
        query,
        key_cache,
        value_cache,
        seq_lens_i32,
        block_tables_i32,
        scale,
        max_context_len,
        compute_type,
        k_scale=torch.tensor([1.0], device=dev),
        v_scale=torch.tensor([1.0], device=dev),
        num_seq_partitions=0,
        alibi_slopes=None,
    )
    torch.cuda.synchronize()

    ctx = {
        "path": "v2_wo_dot" if group == 1 else "v2_w_dot",
        "num_seqs": B,
        "num_q_heads": H_Q,
        "num_kv_heads": H_KV,
        "query_group_size": group,
        "head_size": head_size,
        "block_size": block_size,
        "max_context_len": max_context_len,
        "max_num_partitions": max_num_partitions,
        "seq_partition_size": _SEQ_PARTITION_SIZE,
        "attn_scale": scale,
        "compute_type": str(compute_type),
        "source": "aiter/ops/triton/pa_decode.py@paged_attention_decode",
    }
    return out, ctx
