# aiter_impl.py — 1010_chunked_pa_prefill 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) -> (out, ctx) —— 按名取参。
#
# 来源（aiter pinned 检出 .dcu_runs/aiter_pinned/，commit c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#   device_kernel : aiter/ops/triton/chunked_pa_prefill.py
#                   sha256 eb64f05270eca4385ab91c83ca837143f2f3900df732ccc29c5f284d020886ca
#                   （宿主入口 chunked_prefill_paged_decode :441；decode 内核
#                    _kernel_paged_attention_2d :33 + paged_attention_2d 宿主 :336）
#   official_test : op_tests/triton_tests/test_chunked_pa_prefill.py
#                   sha256 4f1b79ced5816c74415d214052cc75e60d44d41f461e73460f6b5f5a401bab16
#                   （语义真值 compute_golden_attention :264 = 逐 token 因果 softmax，
#                    与 tasks/1010/reference.py 的 forward 同语义）
#
# 入口签名（chunked_pa_prefill.py:441-458；官方测试 test:634-649 的调用口径）：
#
#   chunked_prefill_paged_decode(query, key, value, output, kv_cache_dtype,
#                                key_cache, value_cache, block_table,
#                                query_start_loc, seq_lens, max_query_len,
#                                k_scale, v_scale, alibi_slopes=None,
#                                sliding_window=None, sm_scale=None)
#       -> None（结果就地写入调用方预分配的 output；无返回值）
#
# 路径分派（wrapper :467-503，与题面「chunked prefill / decode 混排」语义一一对应）：
#   * max_query_len > 1  → context_attention_fwd(skip_decode=True, pa_prefill.py:684)
#                          处理 query_len > 1 的 chunked prefill 序列：前缀段走分页
#                          cache（无因果掩码，全部是历史位置）+ 本轮新 token 段走
#                          key/value（因果掩码）；内核内 skip_decode 让 q_len==1 的
#                          序列提前 return（pa_prefill.py:95-96）。
#   * 随后总是调用 paged_attention_2d(filter_by_query_len=True) → SKIP_PREFILL=True
#     （chunked_pa_prefill.py:434、502），只处理 query_len == 1 的 decode 序列，
#     在整段 seq_len 上做在线 softmax（query 位置 = seq_len-1，故等价于因果掩码）。
#   两者输出写入同一 output，拼起来即 reference 对每个 token 的完整因果 softmax。
#
# 布局说明：本评测集 make_inputs 产出（及 reference 使用）的布局与 aiter 期望
# **逐维一致**，无需 permute/reshape：
#   query [num_tokens, H_Q, D] / key,value [num_tokens, H_KV, D]  ←→ aiter q/k/v
#   key_cache [num_blocks, H_KV, D//x, block_size, x]（x=8）     ←→ aiter key_cache
#   value_cache [num_blocks, H_KV, D, block_size]                ←→ aiter value_cache
#   block_tables [num_seqs, max_blocks] / query_start_loc / seq_lens ←→ 同名参数
# 输出单张量 [num_tokens, H_Q, D]、dtype 与 query 一致（torch.empty_like(query)），
# 与 reference.forward 的返回同形同 dtype，**无需打包**。
#
# sm_scale：wrapper 的缺省值是 1/sqrt(query.shape[1])（即 1/sqrt(num_heads)，
# chunked_pa_prefill.py:459-460 与 paged_attention_2d :352-353），与 attention
# 标准定义 1/sqrt(head_size) 不符——准入记录
# benchmark/sources/1010_chunked_pa_prefill.yaml 判定该缺省为继承自上游的历史笔误，
# 以官方测试/golden 为准。故本适配器**显式传 sm_scale**，取值与 reference.Model
# 一致（init_kwargs["scale"] 缺省时 1/sqrt(head_size)），绝不走 wrapper 的缺省。
#
# k_scale / v_scale：题面是非量化 KV（fp16/bf16），io 里没有量化 scale 张量。
# 内核只在 K/V cache dtype 为 fp8 时才乘这两个 scale（:174-177、:204-207），故传
# 标量 1.0；若 cache dtype 不是半精度则 raise（不静默按 1.0 反量化算错）。
#
# autotune config：paged_attention_2d 会查
#   $AITER_TRITON_CONFIGS_PATH/paged_attention_2d/
#     paged_attention_2d-device={arch}-CACHE_BLOCK_SIZE={bs}-HEAD_SIZE_PADDED={hsp}
#     -SLIDING_WINDOW=0-USE_ALIBI_SLOPES=False-HEAD_DIM_PAD_REQ={bool}-kv_dtype=auto.json
# （chunked_pa_prefill.py:284-333）。缺文件**不会失败**——只 warning 后退回默认
# config {'num_warps':4,'num_stages':1,'USE_MATRIX_LOAD':False} + 现算 BLOCK_SIZE
# （:380-392），但会退化为未调优配置（pinned commit 只带 device=gfx938 的 4 个 json，
# 目标 gfx936 无对应文件）。context_attention_fwd 侧的 config 查询已被上游注释停用
# （pa_prefill.py:715-718 硬编码 config=None），无此依赖。

import math

import torch


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（chunked_prefill_paged_decode 混合批次入口）。

    inputs     : [query, key, value, key_cache, value_cache, block_tables,
                  query_start_loc, seq_lens]（顺序同 reference.make_inputs）
                 query [num_tokens, H_Q, D] / key,value [num_tokens, H_KV, D]（fp16/bf16）
                 key_cache [num_blocks, H_KV, D//x, block_size, x]，x = 8 通道打包
                 value_cache [num_blocks, H_KV, D, block_size]
                 block_tables [num_seqs, max_blocks_per_seq] i32
                 query_start_loc [num_seqs+1] i32（query_len 前缀和，首元素 0）
                 seq_lens [num_seqs] i32（ctx_len + query_len）
    init_kwargs: {"head_size": int, "scale": float | None}（= reference.Model 的构造参数）
    device     : 目标设备（张量已在 device 上）

    返回 (out, ctx)；out 为与 query 同形同 dtype 的 [num_tokens, H_Q, D] 张量。
    """
    from aiter.ops.triton.chunked_pa_prefill import chunked_prefill_paged_decode

    (query, key, value, key_cache, value_cache,
     block_tables, query_start_loc, seq_lens) = inputs

    # ---- 构造参数（按名取参；缺 head_size 直接 raise，绝不猜）---------------
    if "head_size" not in init_kwargs:
        raise KeyError(
            "1010 适配器需要 init_kwargs['head_size']（reference.Model 的构造参数；"
            "scale 缺省 1/sqrt(head_size)）"
        )
    head_size = int(init_kwargs["head_size"])
    # 与 reference.Model.__init__ 同式：float(scale) if scale is not None else 1/sqrt(head_size)
    scale = init_kwargs.get("scale", None)
    sm_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_size)

    # ---- shape / dtype 合法性（题目 invariants；违反即 raise，不静默用错）----
    if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
        raise ValueError(
            f"query/key/value 必须是 3 维 [num_tokens, heads, head_size]，实际 "
            f"{tuple(query.shape)} / {tuple(key.shape)} / {tuple(value.shape)}"
        )
    if key_cache.dim() != 5 or value_cache.dim() != 4:
        raise ValueError(
            f"key_cache 必须是 5 维、value_cache 必须是 4 维，实际 "
            f"{tuple(key_cache.shape)} / {tuple(value_cache.shape)}"
        )
    num_tokens, num_heads, head_dim = query.shape
    num_kv_heads = int(key_cache.shape[1])
    x = int(key_cache.shape[4])
    block_size = int(value_cache.shape[3])
    if (head_dim != head_size or key.shape[2] != head_size
            or value.shape[2] != head_size or key.shape[1] != num_kv_heads
            or value.shape[1] != num_kv_heads):
        raise ValueError(
            f"head_size/头数不自洽：init head_size={head_size}，query{tuple(query.shape)} "
            f"key{tuple(key.shape)} value{tuple(value.shape)} key_cache{tuple(key_cache.shape)}"
        )
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"H_Q({num_heads}) 必须是 H_KV({num_kv_heads}) 的整数倍（GQA 整除，题面 invariant）"
        )
    num_queries_per_kv = num_heads // num_kv_heads
    # K 通道打包：key_cache [nblk, H_KV, D//x, bs, x]，x 必须整除 head_size
    if x <= 0 or head_dim % x != 0 or int(key_cache.shape[2]) * x != head_dim:
        raise ValueError(
            f"key_cache 通道打包布局不自洽：shape={tuple(key_cache.shape)}，head_size={head_dim}，"
            f"要求 D//x * x == D（x=key_cache.shape[4]）"
        )
    if int(key_cache.shape[3]) != block_size or int(value_cache.shape[2]) != head_dim:
        raise ValueError(
            f"key_cache/value_cache block_size 或头维不一致：key_cache{tuple(key_cache.shape)} "
            f"value_cache{tuple(value_cache.shape)}"
        )
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"本题为非量化 KV 的半精度路径，query.dtype 需为 float16/bfloat16，实际 {query.dtype}"
        )
    if (key.dtype != query.dtype or value.dtype != query.dtype
            or key_cache.dtype != query.dtype or value_cache.dtype != query.dtype):
        raise ValueError(
            f"query/key/value/cache dtype 必须一致：{query.dtype} / {key.dtype} / "
            f"{value.dtype} / {key_cache.dtype} / {value_cache.dtype}；"
            "若 cache 为 fp8，题面 io 未提供 k_scale/v_scale，无法对齐 reference，拒绝按 1.0 反量化"
        )

    # ---- 长度张量：题面给 int32，aiter 内核按元素取值（与官方测试 dtype 无碍）----
    seq_lens_i32 = seq_lens.to(torch.int32).contiguous()
    block_tables_i32 = block_tables.to(torch.int32).contiguous()
    qsl_i32 = query_start_loc.to(torch.int32).contiguous()

    num_seqs = int(seq_lens_i32.numel())
    if num_seqs == 0 or int(block_tables_i32.shape[0]) != num_seqs:
        raise ValueError(
            f"block_tables 第一维({tuple(block_tables_i32.shape)})必须等于 num_seqs({num_seqs})"
        )
    if int(qsl_i32.numel()) != num_seqs + 1:
        raise ValueError(
            f"query_start_loc 长度必须为 num_seqs+1={num_seqs + 1}，实际 {int(qsl_i32.numel())}；"
            "context_attention_fwd 内部 assert batch + 1 == len(b_start_loc)（pa_prefill.py:744）"
        )

    # query_len 逐序列取自前缀和差（题面 invariant：query_start_loc 为 query_len 前缀和）
    qsl_list = [int(v) for v in qsl_i32.tolist()]
    query_lens = [qsl_list[i + 1] - qsl_list[i] for i in range(num_seqs)]
    if min(query_lens) < 1:
        raise ValueError(
            f"query_len 必须 >= 1（题面 invariant：seq_lens[b]=ctx_len+query_len>=1），实际 {query_lens}；"
            "q_len==0 会让 decode 路径（SKIP_PREFILL 只跳过 >1）越界写入"
        )
    if qsl_list[0] != 0 or qsl_list[-1] != int(num_tokens):
        raise ValueError(
            f"query_start_loc 必须首元素 0、末元素 num_tokens({int(num_tokens)})，实际 "
            f"[0]={qsl_list[0]} [-1]={qsl_list[-1]}"
        )
    max_query_len = max(query_lens)
    ctx_lens = [int(s) - int(q) for s, q in zip(seq_lens_i32.tolist(), query_lens)]
    if min(ctx_lens) < 0:
        raise ValueError(f"ctx_len = seq_lens - query_len 必须 >= 0，实际 {ctx_lens}")

    # ---- layout 归一（题目已是 aiter 期望布局，contiguous 为无操作保护）----
    query_c = query.contiguous()
    key_c = key.contiguous()
    value_c = value.contiguous()
    key_cache_c = key_cache.contiguous()
    value_cache_c = value_cache.contiguous()

    out = torch.empty_like(query_c)

    # 非量化路径：内核仅在 cache 为 fp8 时使用这两个 scale（:174-177、:204-207）
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=query.device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=query.device)

    # 混合批次入口：内部按 max_query_len 分派 prefill / decode 两条路径（:467-503），
    # alibi_slopes / sliding_window 缺省 None（题面无 ALiBi、无滑窗）。
    chunked_prefill_paged_decode(
        query_c,
        key_c,
        value_c,
        out,
        "auto",
        key_cache_c,
        value_cache_c,
        block_tables_i32,
        qsl_i32,
        seq_lens_i32,
        max_query_len,
        k_scale,
        v_scale,
        alibi_slopes=None,
        sliding_window=None,
        sm_scale=sm_scale,
    )

    torch.cuda.synchronize()

    paths = []
    if max_query_len > 1:
        paths.append("context_attention_fwd(skip_decode=True)")   # chunked prefill 序列
    paths.append("paged_attention_2d(filter_by_query_len=True)")  # decode 序列

    ctx = {
        "impl": "aiter",
        "module": "aiter.ops.triton.chunked_pa_prefill",
        "path": "chunked_prefill_paged_decode",
        "sub_paths": paths,
        "num_seqs": num_seqs,
        "num_decode_seqs": sum(1 for q in query_lens if q == 1),
        "num_prefill_seqs": sum(1 for q in query_lens if q > 1),
        "num_tokens": int(num_tokens),
        "num_q_heads": int(num_heads),
        "num_kv_heads": num_kv_heads,
        "query_group_size": num_queries_per_kv,
        "head_size": head_dim,
        "block_size": block_size,
        "x": x,
        "num_blocks": int(key_cache.shape[0]),
        "max_blocks_per_seq": int(block_tables.shape[1]),
        "max_query_len": max_query_len,
        "max_seq_len": int(seq_lens_i32.max().item()),
        "min_ctx_len": min(ctx_lens),
        "sm_scale": sm_scale,
        "sm_scale_source": "init_kwargs['scale']" if scale is not None else "1/sqrt(head_size)",
        "kv_cache_dtype": "auto",
        "k_scale": 1.0,
        "v_scale": 1.0,
        "dtype": str(query.dtype),
        "autotune_config_note": (
            "paged_attention_2d 会查 AITER_TRITON_CONFIGS_PATH 下 "
            "paged_attention_2d-device={arch}-CACHE_BLOCK_SIZE=%d-HEAD_SIZE_PADDED=%d"
            "-SLIDING_WINDOW=0-USE_ALIBI_SLOPES=False-HEAD_DIM_PAD_REQ=%s-kv_dtype=auto.json；"
            "缺文件仅 warning 后回退默认 config（正确性不受影响，性能可能次优）"
            % (block_size, 1 << (head_dim - 1).bit_length(),
               head_dim != (1 << (head_dim - 1).bit_length()))
        ),
        "device": str(device),
    }
    return out, ctx
