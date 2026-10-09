# aiter_impl.py — 1022_pa_prefill 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) —— 按名取参。
#
# 来源（aiter pinned 检出 .dcu_runs/aiter_pinned/，commit c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#   device_kernel : aiter/ops/triton/pa_prefill.py
#                   sha256 6433a1c80d0c484540518e130eead5cc10c839a4adcecff721d335b19aa1b69c
#                   _fwd_kernel       pa_prefill.py:28   （无 ALiBi 偏置路径）
#                   _fwd_kernel_alibi pa_prefill.py:364  （ALiBi 偏置路径，ALiBi_slopes != None）
#                   宿主入口 context_attention_fwd  pa_prefill.py:685
#   official_test : op_tests/triton_tests/test_pa_prefill.py
#                   sha256 9916c7fec8fd3a126da35cb23a181d6931b68d030976c5c9396469510a5d67e0
#                   （调用口径 test_pa_prefill.py:361 / 444；输入构造 input_helper:173）
#
# 入口签名（宿主函数 pa_prefill.py:685-701；结果**就地**写入 o，无返回值）：
#
#   context_attention_fwd(q, k, v, o, kv_cache_dtype, k_cache, v_cache, b_loc,
#                         b_start_loc, b_seq_len, max_input_len, k_scale, v_scale,
#                         alibi_slopes=None, sliding_window=None, sm_scale=None,
#                         skip_decode=False) -> None
#
#   q            [num_tokens, H, D]              fp16/bf16，token 主序拼接（无 padding）
#   k, v         [num_tokens, Hkv, D]            同 q dtype；**只含本轮新 token**
#   o            [num_tokens, H, D]              调用方预分配，同 q 形状/dtype
#   k_cache      [num_blocks, Hkv, D//x, block_size, x]    K 通道打包（x = shape[4] = 8）
#   v_cache      [num_blocks, Hkv, D, block_size]
#   b_loc        = block_tables   [num_seqs, max_blocks_per_seq]
#   b_start_loc  = query_start_loc[num_seqs+1]（query_len 前缀和，首元素 0）
#   b_seq_len    = seq_lens       [num_seqs]
#   max_input_len：grid 第 2 维上界 = max(query_lens)，宿主做
#                  triton.cdiv(max_input_len, BLOCK_M)（pa_prefill.py:745）；官方测试传
#                  MAX_SEQ_LEN（即 query_len 采样上界，test_pa_prefill.py:200/232）
#   k_scale/v_scale：非量化路径**从不被读取**——只在 k_load/v_load 为 fp8 时
#                  tl.load（pa_prefill.py:195-198、265-268）；本题非量化，传 1.0 占位
#   alibi_slopes [H] fp32：None -> _fwd_kernel；非 None -> _fwd_kernel_alibi
#   sliding_window / sm_scale / skip_decode：见下方调用点
#
# 布局说明：题面 io（= reference.py 的缓存布局）与 aiter 期望的布局**逐位一致**——
# K 通道打包 [nblk, Hkv, D//x, bs, x] 与 V [nblk, Hkv, D, bs]，q/k/v 均为
# [tokens, heads, D] 拼接形态；**无需 permute/reshape**，直接传入（.contiguous()
# 仅作兜底，已经是连续张量时零开销）。输出是单个 [num_tokens, H, D] 张量，
# 与 reference.forward 同形同 dtype，**无需打包**。
#
# 语义对齐（与 benchmark/sources/1022_pa_prefill.yaml 的准入裁定一致）：
#   - 前缀 K/V 经块表间接寻址：bn = B_Loc[b, (start_n+offs_n)//block_size]
#     （pa_prefill.py:154-174），仅读前缀段；前缀之后的槽位/未引用块不参与计算
#     （掩码 (start_n+offs_n) < cur_batch_ctx_len）。
#   - **统一在线 softmax**：前缀循环（135-274）与本轮新 token 循环（287-343）共享
#     同一份 m_i / l_i / acc，最后 acc/l_i 一次性归一（345）；不是「两段各自
#     softmax 后相加」。与 reference.py:131-141 的单次 softmax 口径一致。
#   - 因果掩码只作用于新 token 段：offs_m >= start_n + offs_n（305-306）；前缀
#     位置恒 <= 当前 query 位置，故无需掩码（与 reference 的 l > pos 置 -inf 等价，
#     因为前缀段 l < ctx_len <= pos 永不触发）。
#   - GQA：cur_kv_head = cur_head // num_queries_per_kv（85、416），
#     num_queries_per_kv = q.shape[1] // k.shape[1]（742）。
#   - ALiBi：alibi = (k_idx - q_idx) * slope，q_idx = ctx_len + m（455-456、544-545），
#     掩码 (alibi <= 0) & (q_idx < seq_len)（495-497、570-572）。slope 恒正
#     （reference._get_alibi_slopes），故 alibi <= 0 等价于 k_idx <= q_idx，与
#     reference.py:136-138 的 rel = l - pos <= 0 完全同义（l == pos 处偏置 0）。
#   - 中间累加 fp32、输出 store 时 cast 回 o.dtype（345、353-360）。
#
# 无需 autotune config：config 文件查找在本 commit 里是**死代码**——
# `config = get_context_attention_fwd_config(...)` 已被注释（pa_prefill.py:715），
# 宿主固定用 config = {'num_warps': 4, 'num_stages': 1, 'USE_MATRIX_LOAD': False}
# （无 alibi）/ {'num_warps': 4, 'num_stages': 1}（alibi）（pa_prefill.py:718-722）。
# 两个 kernel 都没有 @triton.autotune；get_context_attention_fwd_config /
# get_context_attention_fwd_config_filepath（631-682）无人调用，
# 因此**不需要** AITER_TRITON_CONFIGS_PATH 下的 context_attention_fwd/*.json。
# BLOCK_M/BLOCK_N 由宿主按 dtype/元素大小算出（724-732），也无需外部 JSON。
#
# ⚠️ 调用方注意（与适配器无关的既有缺口）：本任务 io 的 alibi_slopes 是 optional，
# make_inputs 在 use_alibi=False 时该位置返回 **None**（reference.py:298），而
# record_baseline.py:170 / audit_model_class.py:179,213 的
# `[t.to(device) for t in case_inputs(...)]` 会对 None 抛 AttributeError。
# 1022 的 perf case 1/2 与 hidden 的多个 case 都是 use_alibi=False，
# 采集/终审前需让这些列表推导跳过 None（本适配器已能容忍 8 元或 9 元 inputs）。

import math

import torch

# 与 pa_prefill.py:22-23 保持一致（ctx 记录用）
_BASE_BLOCK = 128
_NUM_WARPS = 4


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（pa_prefill.context_attention_fwd）。

    inputs     : [query, key, value, key_cache, value_cache, block_tables,
                  query_start_loc, seq_lens, alibi_slopes]
                  （顺序同 reference.make_inputs；alibi_slopes 可为 None，
                    缺失时按 None 处理）
    init_kwargs: {"head_size": int, "scale": float | None}
                  head_size -> sm_scale 缺省值；scale 显式给出时按 reference 直传
    device     : 目标设备（张量已在 device 上；仅用于占位张量与 ctx 记录）

    返回 (out, ctx)；out 为 [num_tokens, num_heads, head_size] 连续张量，
    dtype 与 query 一致（与 reference.forward 同形同 dtype，单张量不打包）。
    """
    try:
        from aiter.ops.triton.pa_prefill import context_attention_fwd
    except ImportError as exc:  # pragma: no cover - 仅真机环境缺失时触发
        raise RuntimeError(
            "无法导入 aiter.ops.triton.pa_prefill.context_attention_fwd：真机需 DTK + aiter"
            "（部署侧最小导入垫片，见 record_baseline.py 的 --aiter-root）；"
            "本机无 GPU / 未装 aiter，未做运行验证"
        ) from exc

    # ---- 输入解包（容错：允许调用方过滤掉 None 的 alibi_slopes）---------------
    if len(inputs) not in (8, 9):
        raise ValueError(
            f"inputs 应为 8 或 9 个元素（query,key,value,key_cache,value_cache,"
            f"block_tables,query_start_loc,seq_lens[,alibi_slopes]），实际 {len(inputs)}"
        )
    query, key, value, key_cache, value_cache, block_tables, query_start_loc, seq_lens = inputs[:8]
    alibi_slopes = inputs[8] if len(inputs) > 8 else None

    # ---- 维度先校验（下面要按位置读 shape，维数不对会变成 IndexError）---------
    if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
        raise ValueError(
            f"query/key/value 必须是 3 维 [num_tokens, heads, head_size]，实际 "
            f"{query.dim()} / {key.dim()} / {value.dim()} 维"
        )
    if key_cache.dim() != 5 or value_cache.dim() != 4:
        raise ValueError(
            f"key_cache 必须是 5 维 [nblk,Hkv,D//x,bs,x]、value_cache 必须是 4 维 "
            f"[nblk,Hkv,D,bs]，实际 {key_cache.dim()} / {value_cache.dim()} 维"
        )

    # ---- 构造参数（按名取参；越界 raise，绝不静默用错）------------------------
    num_tokens, num_heads, head_size = query.shape
    num_kv_heads = key_cache.shape[1]
    block_size = value_cache.shape[3]
    x = key_cache.shape[4]

    head_size_init = init_kwargs.get("head_size", None)
    if head_size_init is None:
        # reference.get_init_inputs() 的缺省就是 head_size，正常不会走到这里
        head_size_init = head_size
    head_size_init = int(head_size_init)
    if head_size_init <= 0:
        raise ValueError(f"init_kwargs['head_size'] 必须为正整数，实际 {head_size_init}")

    # sm_scale **一律显式传入**：reference 的 scale 只由 __init__ 的 head_size 决定
    # （reference.py:80-82），而 aiter 在 sm_scale=None 时用 q.shape[-1] 兜底
    # （pa_prefill.py:739-740）。本评测集全部 case 里二者相等（1/sqrt(D)），
    # 显式传值使它在二者不等的 case 上同样正确，而不是依赖这个巧合。
    scale = init_kwargs.get("scale", None)
    if scale is None:
        sm_scale = 1.0 / math.sqrt(head_size_init)   # 与 Model 缺省值同式
    else:
        sm_scale = float(scale)
    if not math.isfinite(sm_scale):
        raise ValueError(f"sm_scale 必须有限，实际 {sm_scale!r}（init_kwargs['scale']={scale!r}）")

    # ---- shape / dtype 合法性（题目全域约束，违反即 raise）-------------------
    if tuple(key.shape) != (num_tokens, num_kv_heads, head_size) or tuple(value.shape) != tuple(key.shape):
        raise ValueError(
            f"key/value 必须是 [num_tokens, num_kv_heads, head_size]="
            f"{(num_tokens, num_kv_heads, head_size)}，实际 {tuple(key.shape)} / {tuple(value.shape)}"
        )
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads({num_heads}) 必须是 num_kv_heads({num_kv_heads}) 的整数倍")
    if head_size % x != 0:
        raise ValueError(
            f"head_size({head_size}) 必须是 x=key_cache.shape[4]({x}) 的整数倍"
            "（reference.py:98 的同名断言；K 通道打包布局）"
        )
    if (
        key_cache.shape[0] != value_cache.shape[0]
        or key_cache.shape[2] != head_size // x
        or key_cache.shape[3] != block_size
        or key_cache.shape[4] != x
        or value_cache.shape[2] != head_size
    ):
        raise ValueError(
            f"分页缓存布局不合题目约定：key_cache{tuple(key_cache.shape)}（应为 "
            f"[nblk,{num_kv_heads},{head_size // x},{block_size},{x}]）、"
            f"value_cache{tuple(value_cache.shape)}（应为 [nblk,{num_kv_heads},{head_size},{block_size}]）"
        )
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"aiter pa_prefill 走半精度 tl.dot 路径，query.dtype 需为 float16/bfloat16，实际 {query.dtype}"
        )
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError(f"query/key/value dtype 必须一致：{query.dtype} / {key.dtype} / {value.dtype}")
    if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
        raise ValueError(
            f"非量化 KV 路径要求缓存与 query 同 dtype（fp8e4m3/fp8e5m2 反量化路径本题不覆盖）："
            f"key_cache={key_cache.dtype} / value_cache={value_cache.dtype} / query={query.dtype}"
        )
    if block_tables.dim() != 2:
        raise ValueError(f"block_tables 必须是 2 维 [num_seqs, max_blocks_per_seq]，实际 {block_tables.dim()} 维")
    if query_start_loc.numel() != seq_lens.numel() + 1:
        raise ValueError(
            f"query_start_loc 必须是 num_seqs+1={seq_lens.numel() + 1} 个元素（query_len 前缀和），"
            f"实际 {query_start_loc.numel()}"
        )
    if block_tables.shape[0] != seq_lens.numel():
        raise ValueError(
            f"block_tables 的序列数 {block_tables.shape[0]} 与 seq_lens 长度 {seq_lens.numel()} 不一致"
        )

    if alibi_slopes is not None:
        if alibi_slopes.dim() != 1 or alibi_slopes.numel() != num_heads:
            raise ValueError(
                f"alibi_slopes 必须是 [num_heads={num_heads}] 的 1 维张量，实际 "
                f"{tuple(alibi_slopes.shape)}"
            )
        alibi = alibi_slopes.to(torch.float32).contiguous()
    else:
        alibi = None

    # ---- 索引张量 dtype + int32 溢出保护 -------------------------------------
    # 题面 io 与 reference.forward 都用 int32（reference.py:87-89），官方测试用 int64
    # （test_pa_prefill.py:226-231），二者在本 kernel 下等价（kernel 不做 dtype 断言）。
    # 按题面口径传 int32；但 kernel 内的块表偏移 = b_loc * stride_k_cache_bs +
    # 块内偏移（pa_prefill.py:163-174），若上界超出 int32（地址算术会截断）则退化为
    # int64 —— 宁可慢一点也不静默算错。
    max_block = int(block_tables.max().item()) if block_tables.numel() else 0
    cache_span = max(int(key_cache.stride(0)), int(value_cache.stride(0)))
    addr_bound = (max_block + 1) * cache_span + max(
        int(key_cache.numel()), int(value_cache.numel()), int(query.numel())
    )
    index_dtype = torch.int32 if addr_bound < 2**31 else torch.int64

    b_loc = block_tables.to(index_dtype).contiguous()
    b_start_loc = query_start_loc.to(index_dtype).contiguous()
    b_seq_len = seq_lens.to(index_dtype).contiguous()

    # 一致性：num_tokens 必须等于 query_start_loc[-1]（kernel 用 q_start + m 寻址 q/o，
    # 不等会越界读写）
    if int(b_start_loc[-1].item()) != num_tokens:
        raise ValueError(
            f"query_start_loc[-1]={int(b_start_loc[-1].item())} 与 num_tokens={num_tokens} 不一致"
            "（query_start_loc 必须是 query_len 前缀和）"
        )
    q_lens = b_start_loc[1:] - b_start_loc[:-1]
    if q_lens.numel() == 0 or int(q_lens.min().item()) < 1:
        raise ValueError(f"每个序列的 query_len 必须 >= 1，实际 {q_lens.tolist()}")
    max_input_len = int(q_lens.max().item())

    q_c = query.contiguous()
    k_c = key.contiguous()
    v_c = value.contiguous()
    k_cache_c = key_cache.contiguous()
    v_cache_c = value_cache.contiguous()
    out = torch.empty_like(q_c)

    dev = device if device is not None else q_c.device
    # 非量化路径不读 k_scale/v_scale，仅为占位（官方测试同款：test_pa_prefill.py:273）
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)

    context_attention_fwd(
        q_c,
        k_c,
        v_c,
        out,
        "auto",          # kv_cache_dtype：非 fp8（pa_prefill.py:747 的 fp8 分支不触发）
        k_cache_c,
        v_cache_c,
        b_loc,
        b_start_loc,
        b_seq_len,
        max_input_len,
        k_scale,
        v_scale,
        alibi_slopes=alibi,
        sliding_window=0,        # 本题不含滑窗；0 = disable（pa_prefill.py:711-712）
        sm_scale=sm_scale,
        skip_decode=False,       # 必须 False：SKIP_DECODE 会整体跳过 query_len==1 的序列
                                 # （pa_prefill.py:95-96），而 q_len==1 是本题合法输入
                                 # （hidden_zero_ctx 的 query_lens=[1,16,48]）
    )
    torch.cuda.synchronize()

    # ctx 仅作记录：grid 由宿主按 triton.cdiv(max_input_len, BLOCK_M) 算出
    # （pa_prefill.py:745）；BLOCK_M = BASE_BLOCK（fp32 才取一半，本题上方已挡掉），
    # BLOCK_N 由宿主按 dtype/元素大小算出（724-732）。
    block_m = _BASE_BLOCK
    ctx = {
        "impl": "aiter",
        "module": "aiter.ops.triton.pa_prefill",
        "symbol": "context_attention_fwd",
        "path": "alibi" if alibi is not None else "no_alibi",
        "kernel": "_fwd_kernel_alibi" if alibi is not None else "_fwd_kernel",
        "num_seqs": int(b_seq_len.numel()),
        "num_tokens": int(num_tokens),
        "num_heads": int(num_heads),
        "num_kv_heads": int(num_kv_heads),
        "query_group_size": int(num_heads // num_kv_heads),
        "head_size": int(head_size),
        "head_size_init": int(head_size_init),
        "block_size": int(block_size),
        "x": int(x),
        "max_input_len": max_input_len,
        "sm_scale": sm_scale,
        "index_dtype": str(index_dtype),
        "dtype": str(q_c.dtype),
        "config": (
            {"num_warps": _NUM_WARPS, "num_stages": 1}
            if alibi is not None
            else {"num_warps": _NUM_WARPS, "num_stages": 1, "USE_MATRIX_LOAD": False}
        ),
        "grid": (int(b_seq_len.numel()), int(num_heads), -(-max_input_len // block_m)),
        "device": str(dev),
    }
    return out, ctx
