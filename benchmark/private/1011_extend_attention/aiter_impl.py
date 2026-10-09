# aiter_impl.py — 1011_extend_attention 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) —— 按名取参。
#
# 来源（aiter pinned 检出 .dcu_runs/aiter_pinned/，
# commit c39fff8c77df4e80617649e92fa3c2615f2c43d1，与 sources/1011_extend_attention.yaml
# 的 sha256 逐字节一致）：
#   device_kernel : aiter/ops/triton/extend_attention.py
#                   sha256 a7af325028a59f714fed9d79ca36fe0ccc5ff0cbae094f6aaab0db8ad0590dc9
#                   （v1 主 kernel _fwd_kernel 于同文件 :51；宿主入口 extend_attention_fwd :1650）
#   official_test : op_tests/triton_tests/test_extend_attention.py
#                   sha256 daa7af5cc70a14035a829eefce193fc074d37788dc08c86f61e4894e5bc9162b
#
# 入口签名（extend_attention.py:1650-1675）：
#
#   extend_attention_fwd(q_extend, k_extend, v_extend, o_extend, k_buffer, v_buffer,
#                        qo_indptr, kv_indptr, kv_indices, custom_mask, is_causal,
#                        mask_indptr, max_len_extend, sm_scale=None, logit_cap=0.0,
#                        skip_prefix_custom_mask=True, config=None, k_scale=None,
#                        v_scale=None, sliding_window_size=-1, sinks=None,
#                        window_kv_offsets=None, xai_temperature_len=-1,
#                        force_v2_prefill=False)
#       -> None（结果就地写入 o_extend）
#
# 官方测试的调用口径与本题逐位对应（test_extend_attention.py:331-347）：
#   q/k/v/o 均为 [total_tokens, heads, head_dim] 连续张量；qo_indptr/kv_indptr/
#   kv_indices 为 int32；custom_mask=None、mask_indptr=None、logit_cap=0.0；
#   max_len_extend = 批内最大 extend 长度（同测试 :195、:688 的取值方式）。
#
# 布局：题目 io 与 reference.py 用的正是 aiter 期望的 BSHD 拼接形态
# （q/k/v/buffer 头维在中间、token 在前），**无需 permute/reshape**；k_buffer /
# v_buffer 就是 KV 池本身，kernel 内按 kv_indices 逐 token 间接寻址
# （_fwd_kernel :147-159 的 offs_kv_loc * stride_buf_kbs + cur_kv_head * stride_buf_kh），
# 未被引用的池行不会被读。输出单张量 [total_extend, num_q_heads, head_dim_v]，
# 与 reference.forward 的返回同形同 dtype，无需打包。
#
# 语义对齐（与 sources/1011_extend_attention.yaml 的准入说明一致）：
#   - 两段混合：stage 1 遍历 prefix（pool gather，无 causal 掩码，prefix 全可见，
#     :143-210），stage 2 遍历本步 extend（:212-291）；与 reference 的
#     k_full = cat(prefix, extend) 等价；
#   - GQA：kv_group_num = H_Q // H_KV，cur_kv_head = cur_head // kv_group_num
#     （宿主 :1715、kernel :97），与 reference 的 repeat_interleave(group, dim=1)
#     给出的映射 h -> h // group 一致；
#   - causal：IS_CAUSAL constexpr，仅在 stage 2 生效，掩码
#     (cur_block_m*BLOCK_M + offs_m) >= (start_n + offs_n)（:265-270），即
#     extend 段第 t 个 query 可见前 t+1 个 extend token —— 与 reference 的
#     pos_k > pos_q 掩码相同；causal=False 时两段均全可见（:191-192、:271-273）；
#   - sm_scale 缺省 1/sqrt(head_dim_q)（宿主 :1712 `sm_scale or 1.0/(Lq**0.5)`），
#     与 reference.py:91 的缺省完全一致；显式 sm_scale 也由宿主直接接受；
#   - 在线 softmax fp32 累加、输出按 o_extend.dtype 写回（:139-141、:293-310）；
#   - 非 2 次幂 head_dim：BLOCK_DMODEL/BLOCK_DV = next_power_of_2(Lq/Lv) 且
#     mask_d/mask_dv 掩码（宿主 :1690-1707、kernel :113-114）；Lq ∈ {192, 288, 576}
#     走 BLOCK_DPE 拆分（q/k 的头维切成两块做两次 tl.dot，等价于整维点积）。
#
# 走的是 v1 kernel（_fwd_kernel）：本适配器不传 k_scale/v_scale，宿主
# :1721 `use_v2 = k_scale is not None or v_scale is not None` 判定为 False，
# 于是 :1838-1839 直接取 default_config（BLOCK_M/BLOCK_N = 32，
# extend_attention.py:1346-1355），**不查 AITER_TRITON_CONFIGS_PATH 下的 autotune
# JSON**（那些 JSON 只在 v2 / k_scale·v_scale 路径上按 :1222-1294 的基础查找，以及
# :1462-1515 的 v3 查找被读取，且 _load_config* 对文件缺失是容错返回空配置）。
# 因此本题不需要 autotune config 文件；v2/v2_decode 路径依赖的
# k_scale/v_scale/sinks/window_kv_offsets 本题 io 也不提供，属于准入记录里
# 明确排除的后续变体。
#
# 其他：logit_cap 恒 0（题面显式声明未启用）、无自定义 mask、无滑动窗口、
# 无 attention sink、无 xai_temperature，全部取 aiter 的缺省关闭值。

import math

import torch


def _next_power_of_2(n: int) -> int:
    """仅用于 ctx 记录 aiter 的 BLOCK 推导（宿主 :1705/:1707）。"""
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（extend_attention.extend_attention_fwd, v1 kernel）。

    inputs     : [q_extend, k_extend, v_extend, k_buffer, v_buffer,
                  qo_indptr, kv_indptr, kv_indices]（顺序同 reference.make_inputs）
                 q_extend [total_extend, H_Q, Dq] / k_extend [total_extend, H_KV, Dq]
                 v_extend [total_extend, H_KV, Dv]（fp16/bf16）
                 k_buffer/v_buffer [num_pool, H_KV, Dq/Dv]——prefix KV 池，含未引用垃圾行
                 qo_indptr/kv_indptr [num_seqs+1] int32（各段长度前缀和）
                 kv_indices [total_prefix] int32（prefix token 的池行号）
    init_kwargs: {"causal": bool}（Model.__init__ 的另一参数 sm_scale 缺省 None，
                 与 aiter 的缺省 1/sqrt(head_dim_q) 一致，一般不出现在 kwargs 里）
    device     : 目标设备（张量已在 device 上，仅用于 ctx 记录）

    返回 (out, ctx)；out 为 [total_extend, H_Q, Dv] 连续张量，dtype 同 q_extend。
    """
    from aiter.ops.triton.extend_attention import extend_attention_fwd

    if len(inputs) != 8:
        raise ValueError(
            f"1011_extend_attention 期望 8 个输入"
            f"（q_extend,k_extend,v_extend,k_buffer,v_buffer,qo_indptr,kv_indptr,"
            f"kv_indices），实际 {len(inputs)} 个"
        )
    (q_extend, k_extend, v_extend, k_buffer, v_buffer,
     qo_indptr, kv_indptr, kv_indices) = inputs

    # ---- 构造参数（按名取参；越界 raise，绝不静默用错）---------------------
    is_causal = bool(init_kwargs.get("causal", True))
    sm_scale = init_kwargs.get("sm_scale", None)
    if sm_scale is not None:
        sm_scale = float(sm_scale)
        if sm_scale == 0.0:
            # reference: scale = sm_scale if sm_scale is not None else 1/sqrt(Dq)
            # → 0.0 得到均匀 softmax；aiter 宿主 :1712 是 `sm_scale or 1.0/(Lq**0.5)`，
            # 会把 0.0 当缺省值静默换成 1/sqrt(head_dim_q)。无法表达，故 raise。
            raise ValueError(
                "sm_scale=0.0 与 aiter 无法对齐：宿主 extend_attention.py:1712 用 "
                "`sm_scale or 1.0/(Lq**0.5)`，0.0 会被当作缺省值换成 "
                "1/sqrt(head_dim_q)，而 reference 会给出均匀权重"
            )

    # ---- shape / dtype 合法性（题目全域约束，违反即 raise）----------------
    for name, t in (("q_extend", q_extend), ("k_extend", k_extend),
                    ("v_extend", v_extend), ("k_buffer", k_buffer),
                    ("v_buffer", v_buffer)):
        if t.dim() != 3:
            raise ValueError(f"{name} 必须是 3 维 [tokens, heads, head_dim]，"
                             f"实际 {tuple(t.shape)}")
    if qo_indptr.dim() != 1 or kv_indptr.dim() != 1 or kv_indices.dim() != 1:
        raise ValueError(
            f"qo_indptr/kv_indptr/kv_indices 必须是 1 维，实际 "
            f"{tuple(qo_indptr.shape)} / {tuple(kv_indptr.shape)} / "
            f"{tuple(kv_indices.shape)}"
        )
    if qo_indptr.numel() != kv_indptr.numel() or qo_indptr.numel() < 2:
        raise ValueError(
            f"qo_indptr/kv_indptr 必须是等长的 num_seqs+1 前缀和，实际 "
            f"{qo_indptr.numel()} / {kv_indptr.numel()}"
        )

    total_extend, H_Q, Dq = q_extend.shape
    H_KV = k_extend.shape[1]
    Dv = v_extend.shape[-1]
    if k_extend.shape[0] != total_extend or k_extend.shape[-1] != Dq:
        raise ValueError(
            f"k_extend 必须是 [total_extend, H_KV, head_dim_q]：q{tuple(q_extend.shape)} "
            f"k{tuple(k_extend.shape)}"
        )
    if v_extend.shape[0] != total_extend or v_extend.shape[1] != H_KV:
        raise ValueError(
            f"v_extend 必须是 [total_extend, H_KV, head_dim_v]：k{tuple(k_extend.shape)} "
            f"v{tuple(v_extend.shape)}"
        )
    if H_KV == 0 or H_Q % H_KV != 0:
        raise ValueError(f"H_Q({H_Q}) 必须是 H_KV({H_KV}) 的整数倍（GQA 整除）")
    if k_buffer.shape[1] != H_KV or v_buffer.shape[1] != H_KV:
        raise ValueError(
            f"k_buffer/v_buffer 的 KV 头数必须等于 k_extend 的 {H_KV}，实际 "
            f"{k_buffer.shape[1]} / {v_buffer.shape[1]}"
        )
    if k_buffer.shape[-1] != Dq or v_buffer.shape[-1] != Dv:
        raise ValueError(
            f"k_buffer/v_buffer 头维必须为 (Dq={Dq}, Dv={Dv})，实际 "
            f"{k_buffer.shape[-1]} / {v_buffer.shape[-1]}"
        )
    if k_buffer.shape[0] != v_buffer.shape[0]:
        raise ValueError(
            f"k_buffer/v_buffer 的池行数必须一致，实际 {k_buffer.shape[0]} / "
            f"{v_buffer.shape[0]}"
        )
    # tl.dot 的 M/N/K 均需 >= 16：第一处 dot 的 K 是 BLOCK_DMODEL=next_pow2(Dq)，
    # 第二处 dot 的 N 是 BLOCK_DV=next_pow2(Dv)（宿主 :1705/:1707）
    if _next_power_of_2(Dq) < 16 or _next_power_of_2(Dv) < 16:
        raise ValueError(
            f"head_dim_q({Dq}) / head_dim_v({Dv}) 使 BLOCK 小于 16，tl.dot 无法表达"
        )
    if q_extend.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"aiter extend_attention 走半精度 tl.dot 路径，dtype 需为 "
            f"float16/bfloat16，实际 {q_extend.dtype}"
        )
    if not (k_extend.dtype == v_extend.dtype == k_buffer.dtype == v_buffer.dtype
            == q_extend.dtype):
        raise ValueError(
            f"q/k/v/buffer dtype 必须一致，实际 {q_extend.dtype} / {k_extend.dtype} / "
            f"{v_extend.dtype} / {k_buffer.dtype} / {v_buffer.dtype}"
        )

    # ---- layout / dtype 归一（题目已是 aiter 期望的形态，此处仅保证连续）----
    q_c = q_extend.contiguous()
    k_c = k_extend.contiguous()
    v_c = v_extend.contiguous()
    k_buf_c = k_buffer.contiguous()
    v_buf_c = v_buffer.contiguous()
    qo_i32 = qo_indptr.to(torch.int32).contiguous()
    kv_i32 = kv_indptr.to(torch.int32).contiguous()
    kv_idx_i32 = kv_indices.to(torch.int32).contiguous()

    # ---- indptr / 池行号一致性 + grid 上界 -------------------------------
    ext_lens = qo_i32[1:] - qo_i32[:-1]
    pre_lens = kv_i32[1:] - kv_i32[:-1]
    idx_max = (kv_idx_i32.max() if kv_idx_i32.numel() > 0
               else kv_idx_i32.new_zeros(()).to(torch.int32))
    # 一次性取回全部宿主标量：run() 在计时循环内被反复调用，逐个 .item() 是多次
    # device→host 流同步，会虚高基线；此处只做一次。
    (qo_start, qo_end, kv_start, kv_end,
     ext_min, max_len_extend, pre_min, pre_max, k_idx_max) = [
        int(v) for v in torch.stack([
            qo_i32[0], qo_i32[-1], kv_i32[0], kv_i32[-1],
            ext_lens.min(), ext_lens.max(), pre_lens.min(), pre_lens.max(), idx_max,
        ]).tolist()
    ]
    num_pool = int(k_buffer.shape[0])
    if qo_start != 0 or kv_start != 0:
        raise ValueError(f"qo_indptr/kv_indptr 必须从 0 开始，实际 {qo_start} / {kv_start}")
    if qo_end != total_extend:
        raise ValueError(
            f"qo_indptr[-1]({qo_end}) 必须等于 q_extend.shape[0]({total_extend})"
        )
    if ext_min < 1:
        raise ValueError(f"每段 extend 长度必须 >= 1（题目不变式），实际最小 {ext_min}")
    if pre_min < 0:
        raise ValueError(f"每段 prefix 长度必须 >= 0，实际最小 {pre_min}")
    if kv_end != int(kv_idx_i32.numel()):
        raise ValueError(
            f"kv_indices 元素数({int(kv_idx_i32.numel())}) 必须等于 kv_indptr[-1]({kv_end})"
        )
    if kv_end > num_pool:
        raise ValueError(f"kv_indptr[-1]({kv_end}) 超出 KV 池行数({num_pool})")
    if kv_end > 0 and k_idx_max >= num_pool:
        raise ValueError(
            f"kv_indices 最大值({k_idx_max}) 越界（池行数 {num_pool}）"
        )

    # grid 第三维上界（宿主 :1852 grid = (batch, head_num, cdiv(max_len_extend,
    # BLOCK_M))）：kernel 内再按各序列的 cur_seq_len_extend 掩码越界 query 行
    # （_fwd_kernel :111），故取批内最大 extend 长度即可（同官方测试
    # test_extend_attention.py:195/:688）
    num_seqs = qo_i32.numel() - 1
    out = torch.empty((total_extend, H_Q, Dv), dtype=q_c.dtype, device=q_c.device)

    extend_attention_fwd(
        q_c,
        k_c,
        v_c,
        out,
        k_buf_c,
        v_buf_c,
        qo_i32,
        kv_i32,
        kv_idx_i32,
        None,             # custom_mask（题面无自定义 mask）
        is_causal,        # is_causal -> IS_CAUSAL constexpr
        None,             # mask_indptr
        max_len_extend,   # max_len_extend -> grid 第三维上界
        sm_scale=sm_scale,          # None -> 宿主取 1/sqrt(head_dim_q)
        logit_cap=0.0,              # 题面：logit cap 未启用（恒 0）
        skip_prefix_custom_mask=True,
        config=None,                # None -> v1 取 default_config，不查 autotune JSON
        k_scale=None,               # k/v_scale 均为 None => use_v2 False => v1 kernel
        v_scale=None,
        sliding_window_size=-1,     # 题面无滑动窗口
        sinks=None,                 # 题面无 attention sink
        window_kv_offsets=None,
        xai_temperature_len=-1,     # 题面无 xai temperature
    )

    torch.cuda.synchronize()

    # BLOCK 推导（宿主 :1690-1707），仅用于 ctx 记录，不影响调用本身
    if Dq == 576:
        block_dmodel, block_dpe = 512, 64
    elif Dq == 288:
        block_dmodel, block_dpe = 256, 32
    elif Dq == 192:
        block_dmodel, block_dpe = 128, 64
    else:
        block_dmodel, block_dpe = _next_power_of_2(Dq), 0
    block_dv = _next_power_of_2(Dv)
    block_m = 32  # default_config["BLOCK_M"]（extend_attention.py:1346-1355）

    ctx = {
        "impl": "aiter",
        "module": "aiter.ops.triton.extend_attention",
        "symbol": "extend_attention_fwd",
        "path": "v1_prefill_kernel",
        "kernel": "_fwd_kernel",
        "kernel_cache": False,
        "autotune_config": None,   # v1 路径用 default_config，无需 config JSON
        "batch": int(num_seqs),
        "total_extend": int(total_extend),
        "total_prefix": int(kv_end),
        "num_pool": num_pool,
        "max_len_extend": int(max_len_extend),
        "min_extend_len": int(ext_min),
        "max_prefix_len": int(pre_max),
        "num_q_heads": int(H_Q),
        "num_kv_heads": int(H_KV),
        "query_group_size": int(H_Q // H_KV),
        "head_dim_q": int(Dq),
        "head_dim_v": int(Dv),
        "is_causal": is_causal,
        "sm_scale": float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(Dq),
        "logit_cap": 0.0,
        "custom_mask": None,
        "sliding_window_size": -1,
        "block": {"BLOCK_M": block_m, "BLOCK_DMODEL": block_dmodel,
                  "BLOCK_DPE": block_dpe, "BLOCK_DV": block_dv},
        "grid": (int(num_seqs), int(H_Q), -(-max_len_extend // block_m)),
        "dtype": str(q_c.dtype),
        "device": str(device),
    }
    return out, ctx
