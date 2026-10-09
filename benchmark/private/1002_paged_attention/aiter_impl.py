# aiter_impl.py — 1002_paged_attention 的 aiter 官方实现适配器
#
# 迁移自 dev 分支（换到上游 main 基座后 1002 仍存在，perf case 名一致）。
# 适配器契约已统一为「按名取参」：run(inputs, init_kwargs: dict, device)——
# 构造参数往往稀疏给出，位置式取值在 append 时会错位（见 audit_model_class.py
# case_init_kwargs 的说明）。
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
#
# 调用约定来自 aiter 官方测试 op_tests/triton_tests/test_pa_decode.py
# （sha256 57d236538f9668b59afeb6f8d5d8aa3d38c5b89ad75a14ea21e6e2d8cfbf8361）：
#
#   paged_attention_decode(output, query, key_cache, value_cache,
#                          context_lens, block_tables, attn_scale, max_context_len,
#                          compute_type, k_scale, v_scale, num_seq_partitions, alibi_slopes)
#
# 布局说明：官方测试里 key_cache 原生布局是 [num_blks, H_KV, D//x, KV_BLK_SZ, x]，
# 经 permute(0,1,3,2,4).flatten(3,4) 得到 key_cache_tri 的
# [num_blks, H_KV, KV_BLK_SZ, D]。本评测集 make_inputs 产出（及 reference 使用）的
# 正是后者，故直接传入 key_cache / value_cache，无需再转换。

import math

import torch
import triton.language as tl

# 与 aiter/ops/triton/pa_decode.py 的 _SEQ_PARTITION_SIZE 保持一致
_SEQ_PARTITION_SIZE = 1024


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现。

    inputs    : [query, key_cache, value_cache, block_tables, seq_lens]
                query [B, H_Q, D] / cache [num_blks, H_KV, bs, D] / bt [B, max_blk] i32 / sl [B]
    init_kwargs: {"head_size": int}

    返回 (out, ctx)；out 为 [B, H_Q, D] 输出张量。
    """
    from aiter.ops.triton.pa_decode import paged_attention_decode

    query, key_cache, value_cache, block_tables, seq_lens = inputs
    head_size = int(init_kwargs["head_size"])
    dtype = query.dtype

    B, H_Q, D = query.shape
    seq_lens_i32 = seq_lens.to(torch.int32).contiguous()
    block_tables_i32 = block_tables.to(torch.int32).contiguous()
    max_context_len = int(seq_lens_i32.max().item())

    # 与 aiter 的 v1/v2 分派规则一致，确认走的是准入记录声明的 v1 路径
    max_num_partitions = (max_context_len + _SEQ_PARTITION_SIZE - 1) // _SEQ_PARTITION_SIZE
    use_v1 = max_context_len <= 8192 and (max_num_partitions == 1 or B * H_Q > 512)
    if not use_v1:
        raise RuntimeError(
            f"该 shape 会使 aiter 分派到 v2 路径（max_context_len={max_context_len}, "
            f"max_num_partitions={max_num_partitions}, B*H_Q={B * H_Q}），"
            "超出 1002 的准入范围（v1, max_seq_len<=8192）"
        )

    compute_type = tl.float16 if dtype == torch.float16 else tl.bfloat16
    out = torch.empty_like(query)

    paged_attention_decode(
        out,
        query,
        key_cache,
        value_cache,
        seq_lens_i32,
        block_tables_i32,
        1.0 / math.sqrt(head_size),
        max_context_len,
        compute_type,
        k_scale=torch.tensor([1.0], device=device),
        v_scale=torch.tensor([1.0], device=device),
        num_seq_partitions=0,
        alibi_slopes=None,
    )
    return out, {"path": "v1" if use_v1 else "v2", "max_context_len": max_context_len}
