# aiter_impl.py — 2005_moe_activation 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
#
# 适配器契约（按名取参，见 audit_model_class.py::case_init_kwargs 的说明）：
#     run(inputs, init_kwargs: dict, device) -> (out, ctx)
#
# ── 来源 ────────────────────────────────────────────────────────────────────
# aiter 源文件：aiter/ops/triton/moe_activation.py
#   commit  c39fff8c77df4e80617649e92fa3c2615f2c43d1
#   sha256  f9bb1757302b85dbf1c98b9d8f605007fdb40f102589fd1c937feaf05f985e9e
#           （与 benchmark/sources/2005_moe_activation.yaml 记录一致，已核对）
# 官方测试：op_tests/test_swiglu_variant_compare.py
#   sha256  d1cba22c0dfc8905d1dd1b5e97f97c66c2d454726e9cf0dfb0917f610294e06c
#   → run_triton_swiglu_variant() 是 mode 0/1/2 的调用权威；本适配器的鉴权
#     入口即该文件用的三个 host 包装函数。其余变体入口见
#     op_tests/triton_autotune/fused_moe/tune_act_and_mul.py 的 smoke test。
#
# ── 入口签名（host 包装函数，全部为 in-place：out 预分配后由 kernel 写入）──
#   moe_activation.triton_silu_and_mul(out, input)                       # input=[M,2N] -> out=[M,N], chunked
#   moe_activation.triton_gelu_and_mul(out, input)                        # 同上，GELU 精确 erf
#   moe_activation.triton_gelu_tanh_and_mul(out, input)                   # 同上，GELU tanh 近似
#   moe_activation.triton_swiglu_silu_clamp_mul(out, inp, limit)          # mode 1
#   moe_activation.triton_swiglu_gpt_oss_sigmoid_alpha(out, inp, alpha, limit)  # mode 0
#   moe_activation.triton_swiglu_step_and_mul(out, inp, limit)            # mode 2
#   moe_activation.triton_swiglu_oai_and_mul(out, inp, alpha, limit)      # interleaved 布局
#   moe_activation.triton_silu_no_mul(out, inp)                           # [M,N] -> [M,N]
#   moe_activation.triton_gelu_no_mul(out, inp)
#   moe_activation.triton_gelu_tanh_no_mul(out, inp)
#   moe_activation.triton_relu2(out, inp)
#
# ── 题面变体名 → aiter 入口的映射（语义逐条对照，见 task.yaml / reference.py）──
#   gated（chunked 布局，gate = x[:, :N]，up = x[:, N:]）：
#     "silu" + alpha + limit → triton_swiglu_gpt_oss_sigmoid_alpha（官方 mode 0
#                              = gate'*sigmoid(alpha*gate')*(up'+1)）
#     "silu" + limit（无 alpha）→ triton_swiglu_silu_clamp_mul（官方 mode 1
#                              = SiLU(gate')*up'）
#     "silu" 无 alpha 无 limit → triton_silu_and_mul（SiLU(gate)*up）
#     "gelu"                     → triton_gelu_and_mul
#     "gelu_tanh"                → triton_gelu_tanh_and_mul
#     "swiglustep"               → triton_swiglu_step_and_mul（官方 mode 2
#                              = clamp(SiLU(gate),max=limit)*up'）
#   gated（interleaved 布局，gate = x[:, 0::2]，up = x[:, 1::2]）：
#     "swiglu_interleaved"       → triton_swiglu_oai_and_mul
#     ⚠️ 题面把 aiter 的 "swigluoai" 更名为 "swiglu_interleaved" 以规避溯源词
#        （见 sources/2005_moe_activation.yaml 的 note），故此处显式映射，
#        不会把题面名直接喂给 aiter 的 dispatch。
#   非 gated（[M,n] -> [M,n]）：
#     "silu_no_mul"        → triton_silu_no_mul
#     "gelu_no_mul"        → triton_gelu_no_mul
#     "gelu_tanh_no_mul"   → triton_gelu_tanh_no_mul（aiter dispatch 表
#                            _NO_MUL_ACTIVATIONS 用的是 "gelu_tanh_no_mul"，
#                            而非 gated 侧的 "gelu_tanh"；这里只调 host 包装
#                            函数，不经过字符串 dispatch，故无歧义）
#     "relu2"              → triton_relu2
#
# ── layout ─────────────────────────────────────────────────────────────────
# 本题输入就是 aiter 期望的 2D contiguous [M, n]，无需 permute/reshape；
# reference 输出也是单张量同形（gated: [M, n/2]；非 gated: [M, n]），与各 host
# 包装函数的 out shape 一致，不需要额外打包。行数统一取 numel / n，使 [1, n]
# 与非 contiguous 输入都不会触发包装函数里的 2D 断言。
#
# ── autotune config ────────────────────────────────────────────────────────
# 全部 host 包装函数用硬编码启发式 get_triton_*_config(M, N) 取 BLOCK_SIZE/
# num_warps，**不依赖 @triton.autotune，也不读 AITER_TRITON_CONFIGS_PATH**，
# 因此无需任何外部 JSON 配置（needs_autotune_config = false）。
#
# ⚠️ 本文件未在真机运行验证（本机无 GPU、无 aiter）。仅做过 py_compile 静态检查。
# 已知上游风险一处：get_triton_swiglu_oai_interleaved_config 的注释声明
# BLOCK_SIZE_D 必须保持 64，否则 interleaved 地址模式会算错——适配器不覆盖
# 该配置，perf case perf_m4096_n16384_interleaved_bf16 走的就是官方 BS64。

import torch


# 题面变体名 → aiter 标识（仅用于校验与 ctx 记录）
_GATED = ("silu", "gelu", "gelu_tanh", "swiglustep", "swiglu_interleaved")
_NOMUL = ("silu_no_mul", "gelu_no_mul", "gelu_tanh_no_mul", "relu2")
# interleaved 变体在 aiter 侧的规范名（仅供 ctx 溯源）
_AITER_ACTIVATION_NAME = {"swiglu_interleaved": "swigluoai"}


def _resolve_activation(activation, alpha, limit):
    """复刻 reference.py::Model.__init__ 的校验与缺省语义。

    非法组合一律 raise（绝不静默用错参数）；返回 (name, is_gated, alpha, limit)，
    其中 alpha/limit 是完成缺省填充后的 float 或 None。
    """
    name = str(activation).lower()
    if name not in _GATED and name not in _NOMUL:
        raise ValueError(
            "activation 必须是 " + "/".join(_GATED + _NOMUL) + " 之一，收到 " + repr(activation)
        )
    if name in ("gelu", "gelu_tanh") and (alpha is not None or limit is not None):
        raise ValueError("gelu / gelu_tanh 不接受 alpha / limit")
    if name == "swiglustep" and alpha is not None:
        raise ValueError("swiglustep 不接受 alpha")
    if name == "silu" and alpha is not None and limit is None:
        raise ValueError("silu 设置 alpha 时必须同时设置 limit")
    if name == "swiglustep" and limit is None:
        limit = 7.0
    if name == "swiglu_interleaved":
        alpha = 1.702 if alpha is None else alpha
        limit = 7.0 if limit is None else limit
    return (
        name,
        name in _GATED,
        None if alpha is None else float(alpha),
        None if limit is None else float(limit),
    )


def _launch(name, out, inp, alpha, limit, moe):
    """按变体分派到 aiter 官方 host 包装函数（in-place 写 out）。"""
    if name == "silu":
        if alpha is not None:
            if limit is None:
                raise ValueError("silu 设置 alpha 时必须同时设置 limit")
            moe.triton_swiglu_gpt_oss_sigmoid_alpha(out, inp, alpha, limit)
        elif limit is not None:
            moe.triton_swiglu_silu_clamp_mul(out, inp, limit)
        else:
            moe.triton_silu_and_mul(out, inp)
    elif name == "gelu":
        moe.triton_gelu_and_mul(out, inp)
    elif name == "gelu_tanh":
        moe.triton_gelu_tanh_and_mul(out, inp)
    elif name == "swiglustep":
        if limit is None:
            raise ValueError("swiglustep 需要 limit（题面缺省 7.0）")
        moe.triton_swiglu_step_and_mul(out, inp, limit)
    elif name == "swiglu_interleaved":
        if alpha is None or limit is None:
            raise ValueError("swiglu_interleaved 需要 alpha 与 limit（题面缺省 1.702 / 7.0）")
        moe.triton_swiglu_oai_and_mul(out, inp, alpha, limit)
    else:
        raise ValueError("非 gated 变体不应走到 gated 分派：" + name)


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现。

    inputs     : [x] 单张量，[M, n]，contiguous，dtype 为 float16/float32/bfloat16
                 （评测器搬到设备并 cast 成 fp32；此处对齐 reference 的
                 `out.to(x.dtype)`，以**传入 dtype** 为输出 dtype）
    init_kwargs: {"activation": str, "alpha": float|None, "limit": float|None}

    返回 (out, ctx)；out 与 reference 同形同 dtype 的单个张量：
      gated 变体 [M, n/2]，非 gated 变体 [M, n]。
    """
    from aiter.ops.triton import moe_activation as moe

    if not isinstance(inputs, (list, tuple)) or len(inputs) != 1:
        raise ValueError("2005_moe_activation 只接受单个输入张量 x，收到 " + repr(type(inputs)))

    x = inputs[0]
    if x.dim() != 2:
        raise ValueError("输入必须是 (M, n) 的 2D 张量，收到 " + str(tuple(x.shape)))

    name, is_gated, alpha, limit = _resolve_activation(
        init_kwargs.get("activation", "silu"),
        init_kwargs.get("alpha"),
        init_kwargs.get("limit"),
    )

    m, n = int(x.shape[0]), int(x.shape[1])
    if is_gated:
        if n % 2 != 0:
            raise ValueError("gated 变体要求最后一维 n 为偶数，收到 n=" + str(n))
        d = n // 2
    else:
        d = n
    if m < 1 or d < 1:
        raise ValueError("需 M >= 1 且输出宽度 >= 1，收到 M=" + str(m) + " D=" + str(d))

    # aiter host 包装函数断言 input/out 均 contiguous；行数用 numel//n 取，
    # 使 [1, n] 这类情形也不会触发 ".view(M, ...)" 的形状断言。
    x2d = x.contiguous().view(m, n)
    out = torch.empty((m, d), dtype=x.dtype, device=x.device)
    out2d = out.view(m, d)

    if is_gated:
        _launch(name, out2d, x2d, alpha, limit, moe)
    elif name == "silu_no_mul":
        moe.triton_silu_no_mul(out2d, x2d)
    elif name == "gelu_no_mul":
        moe.triton_gelu_no_mul(out2d, x2d)
    elif name == "gelu_tanh_no_mul":
        moe.triton_gelu_tanh_no_mul(out2d, x2d)
    elif name == "relu2":
        moe.triton_relu2(out2d, x2d)
    else:  # pragma: no cover - _resolve_activation 已挡住未知名
        raise ValueError("未支持的 activation：" + name)

    torch.cuda.synchronize()

    ctx = {
        "aiter_module": "aiter.ops.triton.moe_activation",
        "activation": name,
        "aiter_activation_name": _AITER_ACTIVATION_NAME.get(name, name),
        "gated": is_gated,
        "layout": "interleaved" if name == "swiglu_interleaved" else ("chunked" if is_gated else "flat"),
        "alpha": alpha,
        "limit": limit,
        "M": m,
        "n": n,
        "out_width": d,
        "dtype": str(x.dtype),
    }
    return out, ctx
