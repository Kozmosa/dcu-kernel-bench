# aiter_impl.py — 1030_unified_attention 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) —— 按名取参（构造参数稀疏
# 给出时位置式会错位，见 audit_model_class.py::case_init_kwargs 的说明）。
#
# 来源（aiter pinned 检出 .dcu_runs/aiter_pinned/，
# commit c39fff8c77df4e80617649e92fa3c2615f2c43d1，与
# sources/1030_unified_attention.yaml 的 sha256 逐字节一致）：
#   device_kernel : aiter/ops/triton/unified_attention.py
#                   sha256 e42c894394b712e0de309fd0209606751286e21537d632f9fa7da57565b3a89f（已核对）
#                   （kernel_unified_attention_2d :77-432 / kernel_unified_attention_3d
#                    :436-803 / reduce_segments :807-898；宿主入口 unified_attention :1037）
#   official_test : op_tests/triton_tests/test_unified_attention.py
#                   sha256 2ceb60ab73672042d4a974bce0eae74ab7f7249544f1ebb61c755e3a54bc2e29（已核对）
#                   （ref_paged_attn :35-113 为语义唯一权威，调用口径 :283-308）
#
# 入口签名（unified_attention.py:1037-1065，全部可按关键字传）：
#
#   unified_attention(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k,
#                     max_seqlen_k, softmax_scale, causal, window_size, block_table,
#                     softcap, q_descale, k_descale, v_descale, seq_threshold_3D=None,
#                     num_par_softmax_segments=None, softmax_segm_output=None,
#                     softmax_segm_max=None, softmax_segm_expsum=None,
#                     alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None,
#                     mm_prefix_range=None, use_alibi_sqrt=False)
#       -> None（结果就地写入 out，:1066-1067 断言 causal 且 q_descale is None）
#
#   q            [num_query_tokens, num_q_heads, head_size]   （ragged，行连续拼接）
#   k, v         [num_blks, blk_size, num_kv_heads, head_size]（题目分页池就是这个布局）
#   out          [num_query_tokens, num_q_heads, head_size]，与 q 同 dtype（:1041）
#   cu_seqlens_q [num_seqs + 1] int32；seqused_k [num_seqs] int32（= 题目 seq_lens）
#   block_table  [num_seqs, max_num_blocks_per_seq] int32（= 题目 block_tables）
#   window_size  (left, right)，宿主只取 window_size[0]：
#                SLIDING_WINDOW constexpr = 1 + window_size[0]（:1110、:1204）
#
# 布局说明：题目 io 的 key_cache / value_cache 原生就是 [num_blks, blk_size, H_kv, D]，
# 正是 2D kernel 注释里的 k/v 布局（:80-81），**无需 permute**（3D kernel 头注释
# :442-443 写的是 vLLM 的另一套布局，但 kernel 只按传入 stride 寻址，本题不必走 3D）。
# 输出是单个 [total_q, H_Q, D] 张量，与 reference.forward 同形同 dtype，**无打包**。
#
# 语义对齐（与 sources/1030_unified_attention.yaml 的准入说明一致）：
#   - causal 右对齐：context_len = seq_len - cur_batch_query_len（:201），
#     可见条件 seq_offset <= context_len + query_pos（:315-316）等价于 reference 的
#     torch.triu(diagonal=kv_len-q_len+1) 掩码；
#   - 滑动窗口：seq_mask &= (query_abs_pos - seq_offset) < SLIDING_WINDOW（:320-321），
#     配合 :1110 的 SLIDING_WINDOW = 1 + window_size[0]，取 window_size[0] = sw - 1
#     时与 reference 的 `~torch.triu(diagonal=kv_len-(q_len+sw)+1)` 逐位等价；
#     sw <= 0 → window_size = (-1, -1) → SLIDING_WINDOW = 0 → 关闭窗口（与 reference 的
#     `if sliding_window > 0` 一致）；sw = 1 也正确（只剩 j = T 一列）。
#   - GQA：kv_head_idx 取自 program_id(1)，query_offset_1 = kv_head*nqpk + offs_m % nqpk
#     （:140、:165），即 GQA 的 h -> h // nqpk 映射，与 reference 的
#     repeat_interleave(group, dim=1) 同义；
#   - softcap：S = c*tanh(S/c)（apply_softcap :45-50，:352-353）在掩码前作用于 QK logit；
#   - sinks：M 直接以 sinks[h] 初始化（:185-192）、L 以 1.0 初始化（:194），
#     即 sink 列 logit = sinks[h]、value = 0；use_sinks=False → sinks=None → USE_SINKS=False；
#     ⚠️ 已知的次要差异：2D kernel 不对 sink logit 施加 softcap，而 reference/task.yaml
#     的语义（sources 准入记录已裁定以官方测试为准）对 sink 一并 softcap，两者差
#     |s|^3/(3c^2) 量级（c=50 时 < 2e-4），远小于 2e-2 容差；本题 3 个 perf case 的
#     softcap 全为 0.0，基线上该差异不存在；
#   - 在线 softmax fp32（:348-417）、acc 除以 L 后按 out 的 dtype 写回（:417、:428-432）；
#   - 未被引用的物理块槽位不会被读：block table 索引被 tile_mask = seq_offset <
#     max_seq_prefix_len <= seq_len 掩住（:264-268），只读前 ceil(kv_len/block_size) 项。
#
# ⚠️ 路径选择（dispatch）——本适配器恒走 2D kernel：
#   unified_attention.py:1128-1136 的分派条件是「未给分段缓冲 或 max_seqlen_q > 1 或
#   num_seqs > seq_threshold_3D」→ 2D。本适配器不传 seq_threshold_3D /
#   num_par_softmax_segments / softmax_segm_*（全 None），因此**恒走 2D**：
#     ① 本题的题眼（prefill/decode 混合 ragged batch）只能走 2D；
#     ② 官方测试 :284-307 对纯 decode batch 另有一条 seq_threshold_3D=8 的 3D +
#        reduce_segments 分派（调用方自行分配 [seq_threshold_3D, H_Q, num_segments,
#        head_size_padded] fp32 缓冲），那是**调用方**的调优选择、非入口默认；
#        本适配器取入口默认（无分段缓冲），保证 2D/3D 逐位同源的掩码逻辑只有一份。
#     ctx["dispatch"] 记录了该选择与启用 3D 所需的三个缓冲，便于复核方按需切换。
#
# ⚠️ autotune config（needs_autotune_config=True）：2D 路径的 tile 参数来自
#   _get_unified_attention_2d_config（:949-990）→
#   {aiter/ops/triton/configs/unified_attention}/unified_attention_2d-device=<triton arch>
#   -bs=<bs>-hs=<hs>-sw=<1+window_size[0]>-alibi=0-qq=0-softcap=<0|1>-sinks=<0|1>-mm=0
#   -fp8=0-hsp=<next_pow2(hs)>-pad=<0|1>-kv=auto.json（文件名拼装见 :967-982）。
#   **该文件在 pinned commit c39fff8c 下不存在**：该目录只有 8 个 hs=192 的配置
#   （gfx936/gfx938 各 4 个，即 [bs=16, hs=192, sw=128, sinks=1] 与
#   [bs=32, hs=192, sw=0, sinks=0] 的 2D/3D 各一，例如
#   unified_attention_2d-device=gfx936-bs=16-hs=192-sw=128-alibi=0-qq=0-softcap=0-
#   sinks=1-mm=0-fp8=0-hsp=256-pad=1-kv=auto.json），
#   本题 perf case 的 hs=128 / sw∈{0,512,1024} / softcap=0 / sinks=1 组合一个都不覆盖。
#   _load_ranked_kernel_config 对缺文件是**容错**返回 None（:937-940），宿主随即
#   :1157-1163 回落到默认 config {BLOCK_M, TILE_SIZE=32, num_warps=4, num_stages=2}
#   并打一条 logger.warning——**数值正确性不受影响，只是绝对性能不是官方 tuned 值**，
#   即基线偏保守（同 1017 的处理口径）。拿到对应 JSON 后无需改本文件即可自动切回。
#
# 其他：causal 恒 True（题目就是因果右对齐）；k/v descale、alibi_slopes、qq_bias、
# mm_prefix_range、output_scale(fp8) 全为 None —— 题目 io 不提供这些张量，属准入记录
# 明确排除的后续变体；block_size % 64 == 0 且 head_size == 256 且环境装了
# flash_attn.varlen_fwd_unified 时会命中 :1137-1142 的旁路（走 flash_attn 而非本文件
# 准入的纯 Triton kernel），本适配器对这种组合直接 raise，绝不静默换实现。

import math

import torch


def _next_power_of_2(n: int) -> int:
    """仅用于 ctx 记录 aiter 的 HEAD_SIZE_PADDED 推导（宿主 :1195）。"""
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（unified_attention.unified_attention，2D kernel 路径）。

    inputs     : [query, key_cache, value_cache, block_tables, seq_lens, cu_seqlens_q,
                  sinks]（顺序同 reference.make_inputs / Model.forward）
                 query [total_q, H_Q, D]（ragged，fp16/bf16）
                 key_cache/value_cache [num_blocks, block_size, H_kv, D]（分页池，含未引用垃圾块）
                 block_tables [num_seqs, max_blocks_per_seq] int32（逻辑块 -> 物理块）
                 seq_lens [num_seqs] int32（有效 KV 长度 kv_s >= q_s）
                 cu_seqlens_q [num_seqs + 1] int32（query 长度前缀和，首元素 0）
                 sinks [H_Q]（use_sinks=False 时被忽略，aiter 收 None）
    init_kwargs: {"head_size": int, "sliding_window": int, "softcap": float,
                  "use_sinks": bool}（Model.__init__ 参数；scale 由 head_size 推出，
                  reference 里没有可覆盖的 scale 入参）
    device     : 目标设备（张量已在 device 上，仅用于 ctx 记录）

    返回 (out, ctx)；out 为 [total_q, H_Q, D] 连续张量，dtype 同 query（就地写入的空张量）。
    """
    # aiter 一律函数内 import（顶层 import 很重，真机部署走最小导入垫片）
    from aiter.ops.triton.unified_attention import unified_attention

    if len(inputs) != 7:
        raise ValueError(
            "1030_unified_attention 期望 7 个输入（query,key_cache,value_cache,"
            f"block_tables,seq_lens,cu_seqlens_q,sinks），实际 {len(inputs)} 个"
        )
    (query, key_cache, value_cache, block_tables, seq_lens,
     cu_seqlens_q, sinks) = inputs

    # ---- 构造参数（按名取参；缺/越界一律 raise，绝不静默用错）---------------
    head_size_kw = init_kwargs.get("head_size", None)
    if head_size_kw is None:
        raise ValueError(
            "init_kwargs 缺少 head_size：reference 的 scale = 1/sqrt(head_size) 与 kernel 的 "
            "HEAD_SIZE 都由它决定（io.init_inputs 声明了 head_size，正常路径必然给出）"
        )
    head_size = int(head_size_kw)
    if head_size < 1 or head_size > 256:
        raise ValueError(f"head_size={head_size} 越界（题目不变式 1 <= head_size <= 256）")

    sliding_window_kw = init_kwargs.get("sliding_window", 0)
    sliding_window = 0 if sliding_window_kw is None else int(sliding_window_kw)

    softcap_kw = init_kwargs.get("softcap", 0.0)
    softcap = 0.0 if softcap_kw is None else float(softcap_kw)
    if not math.isfinite(softcap) or softcap < 0.0:
        raise ValueError(
            f"softcap={softcap!r} 非法：reference 只在 softcap > 0 时启用，aiter 用 "
            "USE_SOFTCAP=(softcap > 0)，负值/NaN 无法映射，故 raise"
        )

    use_sinks = bool(init_kwargs.get("use_sinks", True))

    # ---- shape / dtype 合法性（题目全域约束，违反即 raise）-----------------
    if query.dim() != 3:
        raise ValueError(f"query 必须是 3 维 [total_q, H_Q, D]，实际 {tuple(query.shape)}")
    if key_cache.dim() != 4 or value_cache.dim() != 4:
        raise ValueError(
            f"key_cache/value_cache 必须是 4 维 [num_blocks, block_size, H_kv, D]，实际 "
            f"{tuple(key_cache.shape)} / {tuple(value_cache.shape)}"
        )
    if tuple(value_cache.shape) != tuple(key_cache.shape):
        raise ValueError(
            f"key_cache/value_cache shape 必须一致，实际 {tuple(key_cache.shape)} / "
            f"{tuple(value_cache.shape)}"
        )
    if block_tables.dim() != 2:
        raise ValueError(
            f"block_tables 必须是 2 维 [num_seqs, max_blocks_per_seq]，实际 "
            f"{tuple(block_tables.shape)}"
        )
    if seq_lens.dim() != 1 or cu_seqlens_q.dim() != 1:
        raise ValueError(
            f"seq_lens/cu_seqlens_q 必须是 1 维，实际 {tuple(seq_lens.shape)} / "
            f"{tuple(cu_seqlens_q.shape)}"
        )

    total_q, num_q_heads, head_size_q = (int(x) for x in query.shape)
    num_blocks, block_size, num_kv_heads, head_size_k = (int(x) for x in key_cache.shape)
    num_seqs, max_blocks_per_seq = (int(x) for x in block_tables.shape)

    if head_size_q != head_size or head_size_k != head_size:
        raise ValueError(
            f"init 的 head_size({head_size}) 与 query/key_cache 的头维不符：query "
            f"{head_size_q} / key_cache {head_size_k}"
        )
    if int(seq_lens.numel()) != num_seqs:
        raise ValueError(
            f"seq_lens 元素数({int(seq_lens.numel())}) 必须等于 num_seqs({num_seqs})"
        )
    if int(cu_seqlens_q.numel()) != num_seqs + 1:
        raise ValueError(
            f"cu_seqlens_q 元素数({int(cu_seqlens_q.numel())}) 必须是 num_seqs + 1"
            f"({num_seqs + 1})"
        )
    if num_kv_heads < 1 or num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads({num_q_heads}) 必须是 num_kv_heads({num_kv_heads}) 的整数倍（GQA 整除）"
        )
    if min(total_q, num_blocks, block_size, num_seqs, max_blocks_per_seq) < 1:
        raise ValueError(
            f"total_q/num_blocks/block_size/num_seqs/max_blocks_per_seq 必须 >= 1，实际 "
            f"({total_q}, {num_blocks}, {block_size}, {num_seqs}, {max_blocks_per_seq})"
        )
    dtype = query.dtype
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"aiter unified_attention 走 tl.dot 半精度/fp32 路径，dtype 需为 "
            f"float16/bfloat16/float32，实际 {dtype}（fp8 KV 需要 k/v descale 张量，"
            "题目 io 不提供，属准入记录排除的变体）"
        )
    if key_cache.dtype != dtype or value_cache.dtype != dtype:
        raise ValueError(
            f"query/key_cache/value_cache dtype 必须一致：{dtype} / {key_cache.dtype} / "
            f"{value_cache.dtype}"
        )

    # ---- 索引张量归一 + 一次性取回全部宿主标量 -----------------------------
    # run() 在计时循环内被反复调用，逐个 .item() 是多次 device->host 流同步，会虚高
    # 基线；此处把所有校验量 stack 起来只同步一次。
    bt_i32 = block_tables.to(torch.int32).contiguous()
    sl_i32 = seq_lens.to(torch.int32).contiguous()
    cu_i32 = cu_seqlens_q.to(torch.int32).contiguous()

    q_lens = cu_i32[1:] - cu_i32[:-1]
    num_kv_blocks = (sl_i32 + (block_size - 1)) // block_size
    valid_blk = (
        torch.arange(max_blocks_per_seq, device=query.device)[None, :] < num_kv_blocks[:, None]
    )
    bt_valid = torch.where(valid_blk, bt_i32, torch.zeros_like(bt_i32))
    bt_bad = ((bt_i32 < 0) | (bt_i32 >= num_blocks)) & valid_blk

    (cu_first, cu_last, q_len_min, max_seqlen_q, kv_len_min, max_seqlen_k,
     bt_max, bt_bad_any, kv_ge_q_all) = [
        int(v) for v in torch.stack([
            cu_i32[0], cu_i32[-1], q_lens.min(), q_lens.max(),
            sl_i32.min(), sl_i32.max(), bt_valid.max(),
            bt_bad.any().to(torch.int32),
            (sl_i32 >= q_lens).all().to(torch.int32),
        ]).tolist()
    ]

    if cu_first != 0:
        raise ValueError(f"cu_seqlens_q[0] 必须为 0（题目不变式），实际 {cu_first}")
    if cu_last != total_q:
        raise ValueError(
            f"cu_seqlens_q[-1]({cu_last}) 必须等于 query.shape[0]({total_q})"
        )
    if q_len_min < 1:
        raise ValueError(f"每段 query 长度必须 >= 1（题目不变式），实际最小 {q_len_min}")
    if not kv_ge_q_all:
        raise ValueError(
            "存在 seq_lens[s] < query_len[s] 的序列（题目不变式要求 kv_len >= q_len）；"
            "aiter 的 context_len = seq_len - query_len 会变负、掩码语义与 reference 偏离"
        )
    if bt_bad_any:
        raise ValueError(
            f"block_tables 前 ceil(kv_len/block_size) 项里存在越界编号（合法范围 "
            f"[0, {num_blocks})），最大合法项 {bt_max}"
        )
    if max_seqlen_k > max_blocks_per_seq * block_size:
        raise ValueError(
            f"max(seq_lens)={max_seqlen_k} 超过块表容量 "
            f"{max_blocks_per_seq} * {block_size} = {max_blocks_per_seq * block_size}"
        )

    # ---- 参数映射（reference.Model.__init__ -> aiter 入口）------------------
    # reference: self.scale = 1/sqrt(head_size)，无外部 scale 入参
    scale = 1.0 / math.sqrt(head_size)
    # 官方测试 :284：window_size = (sliding_window - 1, 0)；宿主只用 [0]，
    # SLIDING_WINDOW = 1 + window_size[0] = sliding_window（:1110、:1204）。
    # sw <= 0 关闭窗口（reference 是 `if self.sliding_window > 0`）。
    window_size = (-1, -1) if sliding_window <= 0 else (sliding_window - 1, 0)
    sliding_window_val = 0 if sliding_window <= 0 else sliding_window

    # flash_attn 旁路（:1137-1142）：只有 bs%64==0 且 hs==256 才可能命中，先短路再确认，
    # 避免在热路径上 import flash_attn。
    if block_size % 64 == 0 and head_size == 256:
        try:
            from flash_attn import varlen_fwd_unified  # noqa: F401
            has_varlen_fwd_unified = True
        except Exception:
            has_varlen_fwd_unified = False
        if has_varlen_fwd_unified and getattr(torch.version, "hip", None) is not None:
            raise RuntimeError(
                "该 shape 命中 unified_attention.py:1137-1142 的 use_fa_unified_2d 旁路"
                "（ROCM + 已装 flash_attn.varlen_fwd_unified + block_size%64==0 + "
                "head_size==256），会走 flash_attn 而非本文件准入的纯 Triton kernel；"
                "1030 的准入范围不含该组合，故 raise 而不静默换实现"
            )

    # ---- 输入归一（题目已是 aiter 期望的布局，仅保证连续 + int32）----------
    q_c = query.contiguous()
    k_c = key_cache.contiguous()
    v_c = value_cache.contiguous()

    sinks_arg = None
    if use_sinks:
        if sinks is None:
            raise ValueError("use_sinks=True 但 inputs 里的 sinks 为 None")
        if sinks.dim() != 1:
            raise ValueError(f"sinks 必须是 1 维 [num_q_heads]，实际 {tuple(sinks.shape)}")
        if int(sinks.shape[0]) < num_q_heads:
            raise ValueError(
                f"sinks 元素数({int(sinks.shape[0])}) 必须 >= num_q_heads({num_q_heads})"
            )
        # reference: sinks[:num_q_heads]；aiter 断言 sinks.shape[0] == q.shape[1]（:1069-1070）
        sinks_arg = sinks[:num_q_heads].to(dtype=dtype).contiguous()

    out = torch.empty_like(q_c)

    # ---- 调用官方入口（全关键字，口径对齐官方测试 :285-308）-----------------
    # seq_threshold_3D/num_par_softmax_segments/softmax_segm_* 全 None ⇒
    # :1128-1136 判定为 2D 路径（kernel_unified_attention_2d）。
    unified_attention(
        q=q_c,
        k=k_c,
        v=v_c,
        out=out,
        cu_seqlens_q=cu_i32,
        max_seqlen_q=max_seqlen_q,
        seqused_k=sl_i32,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=bt_i32,
        softcap=softcap,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=None,
        num_par_softmax_segments=None,
        softmax_segm_output=None,
        softmax_segm_max=None,
        softmax_segm_expsum=None,
        alibi_slopes=None,
        output_scale=None,
        qq_bias=None,
        sinks=sinks_arg,
        mm_prefix_range=None,
        use_alibi_sqrt=False,
    )

    torch.cuda.synchronize()

    # ---- ctx（供复核：走了哪条路径 + 关键 shape + config 来源）-------------
    num_queries_per_kv = num_q_heads // num_kv_heads
    block_m = 16 if num_queries_per_kv <= 16 else _next_power_of_2(num_queries_per_kv)
    block_q = block_m // num_queries_per_kv
    tile_size = 32  # _get_tile_size(is_prefill=True) 恒为 32（:928-930）
    head_size_padded = _next_power_of_2(head_size)
    config_suffix = (
        f"-bs={block_size}-hs={head_size}-sw={sliding_window_val}"
        f"-alibi=0-qq=0-softcap={int(softcap > 0)}-sinks={int(use_sinks)}"
        f"-mm=0-fp8=0-hsp={head_size_padded}-pad={int(head_size != head_size_padded)}"
        f"-kv=auto.json"
    )

    ctx = {
        "impl": "aiter",
        "module": "aiter.ops.triton.unified_attention",
        "symbol": "unified_attention",
        "path": "2d_unified_prefill_decode",
        "kernel": "kernel_unified_attention_2d",
        "kernel_cache": True,
        "dispatch": {
            "seq_threshold_3D": None,
            "num_par_softmax_segments": None,
            "segm_buffers": None,
            "note": "不传分段缓冲 ⇒ unified_attention.py:1128-1136 恒走 2D kernel；"
                    "纯 decode batch 的 3D + reduce_segments 路径（官方测试 :284-307 的 "
                    "seq_threshold_3D=8 口径）需要调用方自备 softmax_segm_output/max/expsum "
                    "三个 fp32 缓冲，属调用方调优选择，本适配器取入口默认",
        },
        "num_seqs": num_seqs,
        "total_q": total_q,
        "min_seqlen_q": q_len_min,
        "max_seqlen_q": max_seqlen_q,
        "min_seqlen_k": kv_len_min,
        "max_seqlen_k": max_seqlen_k,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "num_queries_per_kv": num_queries_per_kv,
        "head_size": head_size,
        "head_size_padded": head_size_padded,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "max_blocks_per_seq": max_blocks_per_seq,
        "block_table_max_index": bt_max,
        "block_q": block_q,
        "grid": (total_q // block_q + num_seqs, num_kv_heads),
        "scale": scale,
        "causal": True,
        "sliding_window": sliding_window,
        "window_size": window_size,
        "aiter_sliding_window_constexpr": sliding_window_val,
        "softcap": softcap,
        "use_sinks": use_sinks,
        "dtype": str(dtype),
        "out_shape": tuple(out.shape),
        "autotune_config": {
            "dir": "aiter/ops/triton/configs/unified_attention/（unified_attention.py:933-934）",
            "expected_file_pattern": "unified_attention_2d-device=<triton arch>"
                                     + config_suffix,
            "present_in_pinned_checkout": False,
            "fallback_used": {"BLOCK_M": block_m, "TILE_SIZE": tile_size,
                              "num_warps": 4, "num_stages": 2},
            "note": "pinned commit 下该 JSON 不存在（目录只有 hs=192 的四个配置），"
                    "宿主 :986-989 打 warning 后回落默认 config：数值正确、性能非官方 tuned 值",
        },
        "device": str(device),
    }
    return out, ctx
