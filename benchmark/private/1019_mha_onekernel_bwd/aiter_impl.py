# aiter_impl.py — 1019_mha_onekernel_bwd 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) —— 按名取参。
#
# 来源（aiter pinned 检出 .dcu_runs/aiter_pinned/，commit c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#   device_kernel  : aiter/ops/triton/mha_onekernel_bwd.py
#                    sha256 866d5ad20ea8f0a4dc83a646404890659e765c665eaed960ac7972e67a4e5220
#                    （_bwd_preprocess + bwd_kernel_causal/noncausal：dkdv 与 dq 两段
#                      内层循环同 kernel，对应题面 io 的 one-kernel 反向）
#   official_test  : op_tests/triton_tests/test_mha.py
#                    sha256 92e59f0977cab90c27707f5426000cd482dc0dabdd1f71fa4b8a5b0042737beb
#                    （test_mha_backward，FUSED=False 走本文件的 one-kernel 反向；
#                      mha_set_use_fused_bwd_kernel(False)，test_mha.py:477-601）
#   official_caller: aiter/ops/triton/mha.py:1190（FlashAttnFunc.backward 的 FUSED=False 分支）
#
# 入口签名（宿主函数 mha_onekernel_bwd.py:1546-1573）：
#
#   flash_attn_onekernel_backward(
#       do, q, k, v, o, softmax_lse, dq, dk, dv, dbias,
#       sm_scale, alibi_slopes, causal, cu_seqlens_q, cu_seqlens_k,
#       max_seqlen_q, max_seqlen_k, dropout_p,
#       philox_seed=0, philox_offset=0,
#       descale_q=None, descale_k=None, descale_v=None, descale_do=None,
#       USE_INT64_STRIDES=False, config=None) -> delta
#
#   **返回值是 delta（fp32），真正的输出 dq/dk/dv 由调用方预分配、kernel 就地写入**
#   （mha_onekernel_bwd.py:1083/893/898、1522/1405/1410）。dbias 非 None 直接 raise
#   （:1574-1575），本适配器传 None（题面无 bias）。
#
# 布局：稠密定长 bshd，与题目 io **逐位一致，无需 permute**——
#   q/o/do/dq [batch, seqlen_q, num_q_heads, head_dim]
#   k/v/dk/dv [batch, seqlen_k, num_kv_heads, head_dim]
#   宿主函数按 q.stride() 组装 (b,h,m,d) 四元组交给 kernel：q_strides =
#   (stride(0), stride(2), stride(1), stride(3))（mha_onekernel_bwd.py:1624-1634），
#   故只需连续张量。
#   softmax_lse 必须是**连续 fp32** 的 [batch, num_q_heads, seqlen_q]：宿主用
#   delta = zeros_like(softmax_lse) 并把 delta.stride() 当作 lse 的 stride 传给
#   M/Delta（:1646-1652、1710-1719），非连续 lse 会 stride 与 delta 不匹配。
#
# 语义对齐（与 sources/1019_mha_onekernel_bwd.yaml 的准入说明一致）：
#   - delta 由**给定的 o** 与 do 逐元素乘加（_bwd_preprocess，:100），与
#     reference.py:105 的 delta = sum(o*do) 一致；
#   - USE_EXP2=True，pT = exp2(qkT*scale*log2e - lse*log2e) = exp(qkT*scale - lse)
#     （:229-232、1744），题面 lse 是自然对数 log-sum-exp，口径一致；
#   - causal 右下对齐：causal_mask = (offs_m - (seqlen_q - seqlen_k)) >= offs_n
#     （:238、434），即 query i 只看 key j <= i + seqlen_k - seqlen_q，与
#     reference.py:98-102 的 triu(diagonal=N-M+1) 一致；causal=0 走
#     bwd_kernel_noncausal（:1088、1753-1805，全不掩码）；
#   - dK/dQ 末端乘 sm_scale、dV 不乘：dk *= sm_scale（:897）、dq *= sm_scale
#     （:1082）、dV 直接 store（:893），与 reference.py:109-113 一致；
#   - GQA：GROUP_SIZE = HQ // HK（:663、1283），组内 hqid 循环累加到同一 hkid
#     的 dk/dv（:721、1316），与 reference.py:89、109-112 的 h//group 一致；
#   - causal 且 seqlen_q > seqlen_k 时的全掩码行：kernel 对整块全掩码的 tile 直接
#     return（:908-913），**不写 dq**，故 dq 必须预分配为 0（官方 caller 亦用
#     zeros_like(q)，mha.py:1157）；这些行在 reference 里 p/ds 恒 0、dq 全 0，一致。
#
# 输出打包（上游 bd991d0 的单张量协议，1 维 float32）：reference.forward 返回
#   packed = cat([dq.to(k.dtype).to(float32).flatten(),
#                 dk.to(k.dtype).to(float32).flatten(),
#                 dv.to(v.dtype).to(float32).flatten()])
#   （reference.py:115-119，形状 [numel(dq)+numel(dk)+numel(dv)]，dtype float32）。
# 适配器按**同样顺序**把就地写好的 dq/dk/dv 展平后 cast 再 cat；三段与 reference 的
# dq/dk/dv 逐位同形同序（各自 [batch, seq, heads, head_dim] 行优先）
# （官方 caller 也是 zeros_like(q)/empty_like(k)/empty_like(v)，mha.py:1157）。
#
# head_dim 限制：宿主把 HEAD_DIM=head_sz、ACTUAL_HEAD_DIM=next_power_of_2(head_sz)
# 传给 kernel（:1739-1740、1792-1793），而 kernel 用 HEAD_DIM 做
# tl.arange(0, HEAD_DIM)（:662、1282）——即 head_sz 必须是 2 的幂，否则 arange 非
# 2 次幂直接编译失败；非 2 次幂时 PADDED_HEAD 的 mask 也失效（ACTUAL_HEAD_DIM >
# head_sz，越界通道不被掩码）。故本适配器对 head_dim 做 2 次幂校验并 raise，
# 绝不静默跑错。本题 perf case 的 head_dim 只有 64/128，均为 2 次幂。
#
# autotune config 依赖（needs_autotune_config）：宿主在 config=None 时读
#   AITER_TRITON_CONFIGS_PATH/{dev}-MHA-DEFAULT.json 的 "bkwd_onekernel" 键
#   （_get_config，:1533-1543；调用点 :1642-1643），该键需含
#   preprocess_kernel.PRE_BLOCK（:1657、1671）与 onekernel.{BLOCK_M1, BLOCK_N1,
#   BLOCK_M2, BLOCK_N2, BLK_SLICE_FACTOR}（:1693-1698、1751、1804）。
#   该 JSON **不在** aiter 源码树里（本 pinned 检出及其他 aiter 副本均无，
#   `git ls-files "*MHA-DEFAULT*"` 为空，只有 3 处 py 引用），部署侧缺失即
#   FileNotFoundError。因此本适配器**优先**调用官方 _get_config()（部署侧若已
#   提供该 JSON 就用官方调优参数），读不到时退化为文件内联的保守分块，并在
#   ctx["config_source"] 里显式标注来源，绝不静默换参。
#
# 不允许的算子：本文件不含任何 torch 高层计算，只用 contiguous / zeros_like /
# empty_like / reshape / cat 做布局与打包，核心计算全在 aiter kernel 内。

import math

import torch

# 内联兜底分块（**仅**在官方 {dev}-MHA-DEFAULT.json 读不到时使用，ctx 里会标注）。
# 取值 = kernel 头注释里的默认块：BLOCK_M1=32 / BLOCK_N1=128 / BLOCK_M2=128 /
# BLOCK_N2=32（mha_onekernel_bwd.py:1155-1158），BLK_SLICE_FACTOR=2 使
# MASK_BLOCK_M1 = 32 // 2 = 16、MASK_BLOCK_N2 = 32 // 2 = 16（:773、966），满足两条
# static_assert：_bwd_dkdv_inner 的 BLOCK_N1 % MASK_BLOCK_M1 == 0（:172）、
# _bwd_dq_inner 的 BLOCK_M2 % BLOCK_N2 == 0（:369）；且所有 tl.dot 的 M/N/K 维
# 均 >= 16（head_dim >= 16 由下方校验保证）。PRE_BLOCK=64 对应预处理阶段的
# [BLOCK_M, head_dim] 行块（:70、1671-1672）。
_FALLBACK_CONFIG = {
    "preprocess_kernel": {"PRE_BLOCK": 64},
    "onekernel": {
        "BLOCK_M1": 32,
        "BLOCK_N1": 128,
        "BLOCK_M2": 128,
        "BLOCK_N2": 32,
        "BLK_SLICE_FACTOR": 2,
    },
}


def _resolve_config():
    """取 aiter 官方 one-kernel 反向的 triton 分块配置。

    返回 (config, source)；source 写进 ctx，明确基线用的是官方 JSON 还是内联兜底。
    """
    try:
        from aiter.ops.triton.mha_onekernel_bwd import _get_config

        config = _get_config()
        return dict(config), "aiter:{dev}-MHA-DEFAULT.json#bkwd_onekernel"
    except Exception as exc:  # noqa: BLE001 - FileNotFoundError/KeyError/ImportError
        return dict(_FALLBACK_CONFIG), f"inline_fallback:{type(exc).__name__}"


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（mha_onekernel_bwd.flash_attn_onekernel_backward）。

    inputs     : [q, k, v, o, lse, do]（顺序同 reference.make_inputs）
                 q/o/do [batch, seqlen_q, num_q_heads, head_dim] fp16/bf16
                 k/v    [batch, seqlen_k, num_kv_heads, head_dim] 与 q 同 dtype
                 lse    [batch, num_q_heads, seqlen_q] float32（自然对数 log-sum-exp）
    init_kwargs: {"head_dim": int, "causal": 0/1}（Model.__init__ 的参数；
                 scale 未声明时缺省 1/sqrt(head_dim)）
    device     : 目标设备（张量已在 device 上，仅用于 ctx 记录）

    返回 (out, ctx)；out 为 reference 同形的单张量
    [numel(dq) + numel(dk) + numel(dv)]、dtype 同输入。
    """
    # aiter 顶层 import 很重：一律函数内 import（真机部署走最小导入垫片）。
    from aiter.ops.triton.mha_onekernel_bwd import flash_attn_onekernel_backward

    if len(inputs) != 6:
        raise ValueError(
            f"1019 需要 6 个输入 [q, k, v, o, lse, do]，实际 {len(inputs)} 个"
        )
    q, k, v, o, lse, do = inputs

    # ---- 构造参数（按名取参，越界 raise，绝不静默用错）----------------------
    causal_raw = init_kwargs.get("causal", 1)
    causal = bool(int(causal_raw))
    scale = init_kwargs.get("scale", None)

    # ---- shape / dtype 合法性（题目全域约束，违反即 raise）-----------------
    for name, t in (("q", q), ("k", k), ("v", v), ("o", o), ("do", do)):
        if t.dim() != 4:
            raise ValueError(
                f"{name} 必须是 4 维 [batch, seq, heads, head_dim]，实际 {tuple(t.shape)}"
            )
    if lse.dim() != 3:
        raise ValueError(
            f"lse 必须是 3 维 [batch, num_q_heads, seqlen_q]，实际 {tuple(lse.shape)}"
        )

    B, M, H_Q, D = q.shape
    Bk, N, H_K, Dk = k.shape
    if Bk != B or Dk != D:
        raise ValueError(
            f"k 的 batch/head_dim 必须与 q 一致：q{tuple(q.shape)} k{tuple(k.shape)}"
        )
    if tuple(v.shape) != (B, N, H_K, D):
        raise ValueError(
            f"v 必须与 k 同形 {tuple(k.shape)}，实际 {tuple(v.shape)}"
        )
    if tuple(o.shape) != (B, M, H_Q, D) or tuple(do.shape) != (B, M, H_Q, D):
        raise ValueError(
            f"o/do 必须与 q 同形 {tuple(q.shape)}，实际 {tuple(o.shape)} / {tuple(do.shape)}"
        )
    if tuple(lse.shape) != (B, H_Q, M):
        raise ValueError(
            f"lse 必须是 [batch, num_q_heads, seqlen_q] = {(B, H_Q, M)}，"
            f"实际 {tuple(lse.shape)}"
        )
    if H_Q % H_K != 0:
        raise ValueError(f"num_q_heads({H_Q}) 必须是 num_kv_heads({H_K}) 的整数倍（GQA）")
    if M < 1 or N < 1:
        raise ValueError(f"seqlen_q/seqlen_k 必须 >= 1，实际 {M}/{N}")

    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"aiter one-kernel 反向走半精度 tl.dot 路径，q 需为 float16/bfloat16，"
            f"实际 {q.dtype}"
        )
    for name, t in (("k", k), ("v", v), ("o", o), ("do", do)):
        if t.dtype != q.dtype:
            raise ValueError(f"{name}.dtype({t.dtype}) 必须与 q.dtype({q.dtype}) 一致")
    if lse.dtype != torch.float32:
        raise ValueError(
            f"lse 必须是 float32（宿主用 zeros_like(softmax_lse) 当 delta，"
            f"mha_onekernel_bwd.py:1646），实际 {lse.dtype}"
        )

    # head_dim：宿主把 HEAD_DIM=head_sz 直接当 tl.arange 上界（mha_onekernel_bwd.py:
    # 662/1282、1739-1740），非 2 次幂会编译失败/掩码失效；tl.dot 的最小 K 维为 16。
    if D != (1 << (D.bit_length() - 1)):
        raise ValueError(
            f"aiter one-kernel 反向要求 head_dim 是 2 的幂（HEAD_DIM 被当作 "
            f"tl.arange 上界，mha_onekernel_bwd.py:1739-1740），实际 {D}"
        )
    if D < 16 or D > 256:
        raise ValueError(f"head_dim({D}) 必须落在 [16, 256]（tl.dot K 维下限 / 题面上限）")

    # init_kwargs 的 head_dim 只用于 sm_scale 缺省；与 q.shape[-1] 不一致则题面
    # 语义本身就矛盾，raise 而不是猜。
    init_head_dim = init_kwargs.get("head_dim", D)
    if int(init_head_dim) != D:
        raise ValueError(
            f"init_kwargs['head_dim']={init_head_dim} 与 q.shape[-1]={D} 不一致"
            "（reference 用 init 的 head_dim 推 sm_scale，无法同时满足）"
        )

    # sm_scale：Model 缺省 1/sqrt(head_dim)（reference.py:73），与输入生成器一致
    # （make_inputs 用 head_dim ** -0.5，reference.py:199）。
    sm_scale = (
        float(scale) if scale is not None else 1.0 / math.sqrt(int(init_head_dim))
    )

    # ---- layout 归一（题目已是 aiter 期望的稠密 bshd，无需 permute）----------
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    o_c = o.contiguous()
    do_c = do.contiguous()
    lse_f32 = lse.contiguous()  # 必须连续：宿主拿 delta.stride() 当它的 stride

    # ---- 输出预分配（同官方 caller：dq 置 0、dk/dv 空，dtype 同输入）--------
    # dq 必须置 0：causal 且 seqlen_q > seqlen_k 时，整块全掩码的 tile 直接 return
    # 不写 dq（mha_onekernel_bwd.py:908-913），那些行的真值就是 0。
    dq = torch.zeros_like(q_c)
    dk = torch.empty_like(k_c)
    dv = torch.empty_like(v_c)

    config, config_source = _resolve_config()

    flash_attn_onekernel_backward(
        do_c,                    # do
        q_c,                     # q
        k_c,                     # k
        v_c,                     # v
        o_c,                     # o（delta = sum(o*do)，_bwd_preprocess:100）
        lse_f32,                 # softmax_lse（自然对数，USE_EXP2 路径）
        dq,                      # 就地写入
        dk,                      # 就地写入
        dv,                      # 就地写入
        None,                    # dbias（题面无 bias；非 None 宿主直接 raise）
        sm_scale,                # sm_scale（reference: dK/dQ 乘它、dV 不乘）
        None,                    # alibi_slopes（题面无 ALiBi）
        causal,                  # causal=1 -> bwd_kernel_causal，0 -> noncausal
        None,                    # cu_seqlens_q（稠密定长，非 varlen）
        None,                    # cu_seqlens_k
        max_seqlen_q=M,          # 稠密：即 q 的 seqlen
        max_seqlen_k=N,
        dropout_p=0.0,           # 题面无 dropout
        philox_seed=0,
        philox_offset=0,
        descale_q=None,          # 题面非量化（fp8 descale 不使用）
        descale_k=None,
        descale_v=None,
        descale_do=None,
        USE_INT64_STRIDES=False,  # 与官方 caller 默认一致（mha.py:1187/1211）
        config=config,
    )

    torch.cuda.synchronize()

    # ---- 打包成单张量（与 reference.py:115-119 逐句同构）-------------------
    # 上游 bd991d0 的单张量协议：各梯度先 cast 回输入 dtype（输出 dtype 约定），
    # 再转 float32，最后沿第 0 维拼接成一维 float32。
    out = torch.cat(
        [
            dq.reshape(-1).to(k.dtype).to(torch.float32),
            dk.reshape(-1).to(k.dtype).to(torch.float32),
            dv.reshape(-1).to(v.dtype).to(torch.float32),
        ],
        dim=0,
    )

    numel_dq, numel_dk, numel_dv = dq.numel(), dk.numel(), dv.numel()
    block_n1 = int((config.get("onekernel") or {}).get("BLOCK_N1", 128))
    ctx = {
        "impl": "aiter",
        "module": "aiter.ops.triton.mha_onekernel_bwd",
        "path": "bwd_kernel_causal" if causal else "bwd_kernel_noncausal",
        "batch": int(B),
        "seqlen_q": int(M),
        "seqlen_k": int(N),
        "num_q_heads": int(H_Q),
        "num_kv_heads": int(H_K),
        "query_group_size": int(H_Q // H_K),
        "head_dim": int(D),
        "dtype": str(q.dtype),
        "causal": causal,
        "sm_scale": sm_scale,
        "packed_numel": int(numel_dq + numel_dk + numel_dv),
        "numel_dq": int(numel_dq),
        "numel_dk": int(numel_dk),
        "numel_dv": int(numel_dv),
        # 宿主 grid = (num_k_heads, cdiv(max(seqlen_q, seqlen_k), BLOCK_N1), batch)
        # （mha_onekernel_bwd.py:1691-1698）
        "grid": (int(H_K), -(-max(M, N) // block_n1), int(B)),
        "config_source": config_source,
        "config": config,
        "device": str(device),
    }
    return out, ctx
