# aiter_impl.py — 1024_prefill_attention 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) —— 按名取参。
#
# 来源（aiter pinned 检出 .dcu_runs/aiter_pinned/，commit c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#   device_kernel : aiter/ops/triton/prefill_attention.py
#                   sha256 c037a49e6159d947ffff61c87c8935dc50b3e546e13dd6d78f8975e4221d2cf5
#   official_test : op_tests/triton_tests/test_prefill_attention.py
#                   sha256 e966e02a11dddbeadd1b9ca4fc7940ac82bb950a830db32085e47a1a5b6f6ec5
#
# 入口签名（宿主函数 prefill_attention.py:175，官方测试 test_prefill_attention.py:94 的调用口径）：
#
#   context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, max_input_len, is_causal=True)
#       -> None（结果就地写入 o）
#
#   q, k, v : [b*s, head, head_dim]（连续，fp16/bf16）
#   o       : [b*s, head, head_dim]，同 dtype，由调用方预分配
#   b_start_loc / b_seq_len : [b] int32；max_input_len : int（grid 沿序列内位置维的上界）
#
# 布局说明：aiter 期望的正是题目 io 的 BSHD 拼接形态 [total_tokens, heads, head_dim]，
# 与 reference.py 的 q/k/v 逐位一致，**无需 permute/reshape**；o 用 empty_like(q)
# 预分配为同形同 dtype 连续张量即可。输出为单张量，与 reference.forward 的返回
# （单个 [total_tokens, num_q_heads, head_dim]）同形同 dtype，无需打包。
#
# 语义对齐（与 sources/1024_prefill_attention.yaml 的准入说明一致）：
#   - 连续（非分页）KV、ragged 变长 batch：kernel 内用 b_start_loc/b_seq_len 取
#     序列起点与长度，越界行/列由掩码处理（prefill_attention.py:69-116）；
#   - GQA：kv_group_num = q.shape[1] // k.shape[1]，cur_kv_head = cur_head // kv_group_num
#     （prefill_attention.py:67、193）；
#   - causal：IS_CAUSAL constexpr，掩码 offs_m >= start_n + offs_n（prefill_attention.py:123-133）；
#   - fp32 在线 softmax 累加、输出 cast 回 o.dtype（prefill_attention.py:98-172）；
#   - 非 2 次幂 head_dim：BLOCK_DMODEL = next_power_of_2(Lk) 且 mask_d = offs_d < Lk
#     对越界通道掩码（prefill_attention.py:86、216）。
#
# sm_scale 说明：宿主函数**内部硬编码** sm_scale = 1.0 / sqrt(q.shape[-1])
# （prefill_attention.py:189-191），不接受外部传入。题目 Model 的 sm_scale 缺省值
# 同为 1/sqrt(head_dim)（reference.py:84），故缺省路径精确一致；init_kwargs 若给出
# 与之不等的显式 sm_scale，本适配器 raise 而不是静默用错 scale——本题
# io.init_inputs 只声明了 causal，正常路径不会触发。
#
# 其他说明：本题走的是 prefill_attention.py 的**非分页**入口。pa_prefill.py 里同名的
# context_attention_fwd 是分页/分块 KV 变体（签名要求 kv_cache_dtype/k_cache/v_cache/
# b_loc/k_scale/v_scale，且依赖 AITER_TRITON_CONFIGS_PATH 的 autotune JSON，
# pa_prefill.py:684-701、631-682），题目 io 不提供这些张量，故不采用。
# prefill_attention.py 无 @triton.autotune、无 config 文件查找，不需要 autotune config。

import math

import torch


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（prefill_attention.context_attention_fwd）。

    inputs     : [q, k, v, b_start_loc, b_seq_len]（顺序同 reference.make_inputs）
                 q [total_tokens, H_Q, D] / k,v [total_tokens, H_KV, D]（fp16/bf16）
                 b_start_loc, b_seq_len [num_seqs] int32（排他前缀和 / 各序列长度）
    init_kwargs: {"causal": bool}（Model.__init__ 的 sm_scale 缺省，见文件头说明）
    device     : 目标设备（张量已在 device 上，仅用于 ctx 记录与兜底）

    返回 (out, ctx)；out 为与 q 同形同 dtype 的 [total_tokens, H_Q, D] 连续张量。
    """
    from aiter.ops.triton.prefill_attention import context_attention_fwd

    q, k, v, b_start_loc, b_seq_len = inputs

    # ---- 构造参数（按名取参，越界 raise，绝不静默用错）----------------------
    is_causal = bool(init_kwargs.get("causal", True))

    # ---- shape / dtype 合法性（题目全域约束，违反即 raise）-----------------
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(
            f"q/k/v 必须是 3 维 [total_tokens, heads, head_dim]，实际 "
            f"{tuple(q.shape)} / {tuple(k.shape)} / {tuple(v.shape)}"
        )
    total_tokens, H_Q, D = q.shape
    H_KV = k.shape[1]
    if k.shape[-1] != D or v.shape[-1] != D or v.shape[1] != H_KV:
        raise ValueError(
            f"Q/K/V 头维必须相同且 K/V 头数一致：q{tuple(q.shape)} k{tuple(k.shape)} "
            f"v{tuple(v.shape)}"
        )
    if H_Q % H_KV != 0:
        raise ValueError(f"H_Q({H_Q}) 必须是 H_KV({H_KV}) 的整数倍（GQA 整除）")
    if D < 16:
        raise ValueError(f"head_dim({D}) 必须 >= 16（tl.dot 的最小 K 维，题目不变式）")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"aiter prefill_attention 走半精度 tl.dot 路径，dtype 需为 float16/bfloat16，"
            f"实际 {q.dtype}"
        )
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(f"Q/K/V dtype 必须一致：{q.dtype} / {k.dtype} / {v.dtype}")
    if b_start_loc.numel() != b_seq_len.numel() or b_seq_len.numel() == 0:
        raise ValueError(
            f"b_start_loc/b_seq_len 必须等长且非空，实际 "
            f"{b_start_loc.numel()} / {b_seq_len.numel()}"
        )

    # sm_scale：aiter 宿主函数硬编码 1/sqrt(head_dim)，与 Model 缺省值一致；
    # 显式给出不等值时无法表达，raise（不静默退化）。
    sm_scale = init_kwargs.get("sm_scale", None)
    default_scale = 1.0 / math.sqrt(D)
    if sm_scale is not None and not math.isclose(
        float(sm_scale), default_scale, rel_tol=1e-6, abs_tol=0.0
    ):
        raise ValueError(
            f"aiter context_attention_fwd 内部硬编码 sm_scale=1/sqrt(head_dim)="
            f"{default_scale!r}（prefill_attention.py:189-191），无法表达显式 "
            f"sm_scale={sm_scale!r}；本题 io.init_inputs 只声明 causal，正常路径不触发"
        )

    # ---- layout 归一（题目已是 aiter 期望的 BSHD 拼接形态）------------------
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    b_start_loc_i32 = b_start_loc.to(torch.int32).contiguous()
    b_seq_len_i32 = b_seq_len.to(torch.int32).contiguous()

    # grid 上界：题目不给 max_seq_len，按 reference.py 的约定由 b_seq_len.max() 推得
    # （kernel 内再按各序列 L_i 掩码越界行/列，b_start_loc/b_seq_len 逐序列生效）
    max_input_len = int(b_seq_len_i32.max().item())

    out = torch.empty_like(q_c)

    context_attention_fwd(
        q_c,
        k_c,
        v_c,
        out,
        b_start_loc_i32,
        b_seq_len_i32,
        max_input_len,
        is_causal,
    )

    torch.cuda.synchronize()

    # 仅用于 ctx 记录：宿主函数按 _is_hip 选 BLOCK（prefill_attention.py:184-187），
    # 不影响调用本身。
    try:
        from aiter.ops.triton import prefill_attention as _pa

        block_hint = 128 if getattr(_pa, "_is_hip", False) else 64
    except Exception:  # pragma: no cover - 仅 ctx 记录，失败不影响结果
        block_hint = None

    ctx = {
        "impl": "aiter",
        "module": "aiter.ops.triton.prefill_attention",
        "path": "context_attention_fwd",
        "batch": int(b_seq_len_i32.numel()),
        "num_q_heads": int(H_Q),
        "num_kv_heads": int(H_KV),
        "query_group_size": int(H_Q // H_KV),
        "head_dim": int(D),
        "total_tokens": int(total_tokens),
        "max_input_len": max_input_len,
        "block_hint": block_hint,
        "is_causal": is_causal,
        "sm_scale": default_scale,
        "dtype": str(q.dtype),
        "grid": (int(b_seq_len_i32.numel()), int(H_Q), -(-max_input_len // block_hint))
        if block_hint
        else None,
        "device": str(device),
    }
    return out, ctx
