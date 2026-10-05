# aiter_impl.py — 2007_rmsnorm 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用（契约：run(inputs, init_kwargs,
# device) -> (out, ctx)），采集离线终审用的 aiter 基线。
#
# 来源（本地 pinned 检出 .dcu_runs/aiter_pinned，commit c39fff8c）：
#   aiter/ops/triton/rmsnorm.py
#     sha256 f15078cb15058f927a010194b9d2d339e2c0c7d04139399ed4636399495c1d35
#   op_tests/triton_tests/test_rmsnorm.py（调用约定的权威）
#     sha256 4bb6bdcf71821197729e264c39128113942fc26dfe045691b4c20e2fe7b01737
#
# 本题语义 = 官方测试 test_rmsnorm / test_fused_add_rmsnorm 的非量化前向路径，
# 入口签名（aiter/ops/triton/rmsnorm.py 行号按上述 commit）：
#
#   rms_norm(input, weight, epsilon) -> y                    # :1174（_RMSNorm :1121）
#       input (M, N) / weight (N,) / epsilon float
#       返回单张量 y，形状 (M, N)，dtype 与 input 相同（_rmsnorm_forward :1000
#       内部 y = torch.empty_like(x)）。对应 variant="rmsnorm"。
#
#   rmsnorm2d_fwd_with_add(out, input, residual_in, residual_out, weight, epsilon)
#       -> out                                              # :1189（_RMSNorm2dFwdWithAdd :1146）
#       out / input / residual_in / residual_out 均 (M, N)，weight (N,)，
#       epsilon float。kernel 内先 residual_out = input + residual_in（在输入
#       dtype 下相加并原样写出），再对 residual_out 做 RMSNorm 写入 out。
#       对应 variant="fused_add"。
#
# 官方测试的调用方式（test_rmsnorm.py::run_triton :52）：
#   residual is None     -> output = rms_norm(input, weight, eps)
#   residual is not None -> residual_out = torch.empty_like(input)
#                           output = torch.empty_like(input)
#                           output = rmsnorm2d_fwd_with_add(
#                               output, input, residual, residual_out, weight, eps)
#
# 布局：两个入口都直接吃 2D contiguous (M, N) / (N,)，不需要 permute / flatten
# （kernel 只做行 stride，见 :1017 x.stride(0) 与 :1054 x.stride(0)）。题面
# make_inputs 产出的正是该布局，唯一需要的重排是输出打包（见下）。
#
# 输出打包：题面 reference 的 forward 返回**单张量**——fused_add 时
# (M, 2N) = cat([y, residual_out], dim=-1)（前 N 列 y、后 N 列 residual_out），
# rmsnorm 时 (M, N) = y。aiter 的 fused_add 入口把两段结果分别写进两个独立
# 缓冲并只返回 out=y，故适配器在 kernel 之外用 torch.cat 做同样的列拼接
# （打包是允许的 torch 用途；不把 residual_out 写成 packed 的列切片，是为了
# 避免非 16 元素对齐的视图基址撞上 kernel 内 tl.multiple_of(..., (16,))
# 的对齐断言，:580 / :795）。
#
# 注意：本机无 GPU、未安装 aiter，本文件只做静态（py_compile）检查，
# 未在真机运行验证。

import math

import torch

_VARIANTS = ("rmsnorm", "fused_add")
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _use_blocked(x: torch.Tensor) -> bool:
    """复刻 aiter rmsnorm.py 的 block_size/use_blocked（:16 / :20），仅用于 ctx。

    block_size = min(65536 // element_size, next_power_of_2(N))；
    use_blocked = N > block_size —— 大 N 走分块归约分支（:138 / :197）。
    """
    blk = min(65536 // x.element_size(), 1 << (x.shape[1] - 1).bit_length())
    return x.shape[1] > blk


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现。

    inputs     : [x, weight, residual]（make_inputs 的顺序，均已搬到 device）
                 x (M, N) / weight (N,) / residual (M, N)，fp16/bf16
    init_kwargs: {"variant": "rmsnorm"|"fused_add", "eps": float}

    返回 (out, ctx)：out 与题面 reference 同形同 dtype——fused_add 为
    (M, 2N)（前 N 列 y、后 N 列 residual_out），rmsnorm 为 (M, N)。
    """
    # aiter 一律函数内 import：顶层 import 会拉起整个 aiter（很重），
    # 真机部署走最小导入垫片
    from aiter.ops.triton.rmsnorm import rms_norm, rmsnorm2d_fwd_with_add

    variant = init_kwargs.get("variant", "fused_add")
    eps = float(init_kwargs.get("eps", 1e-5))

    # 越界即 raise，绝不静默用错参数（variant 决定走哪个 aiter 入口，
    # eps 直接进 kernel 的 rsqrt(mean + eps)）
    if variant not in _VARIANTS:
        raise ValueError(
            f"variant 必须是 {_VARIANTS} 之一，收到 {variant!r}"
            "（题面只有非量化的标准 / 残差融合两条前向路径）"
        )
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps 必须是有限正数，收到 {eps!r}")

    if len(inputs) != 3:
        raise ValueError(f"本题 make_inputs 返回 3 个张量 (x, weight, residual)，收到 {len(inputs)}")

    x, weight, residual = inputs

    if x.dim() != 2:
        raise ValueError(f"x 必须是 2D (M, N)，收到 shape={tuple(x.shape)}")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"x dtype={x.dtype} 不在 aiter rmsnorm 支持范围 {_SUPPORTED_DTYPES}")

    m, n = x.shape
    if n < 1 or m < 1:
        raise ValueError(f"需 M >= 1 且 N >= 1，收到 M={m}, N={n}")
    if weight.numel() != n:
        raise ValueError(f"weight 元素数 {weight.numel()} 与 N={n} 不符")
    if weight.dtype != x.dtype:
        raise ValueError(f"weight dtype={weight.dtype} 与 x dtype={x.dtype} 不符")

    # layout：两个入口都只吃 contiguous 2D（kernel 用 x.stride(0)/out.stride(0)
    # 做行 stride，列方向按 col_offsets 连续寻址），已 contiguous 时是空操作
    x = x.contiguous()
    weight = weight.contiguous()

    if variant == "rmsnorm":
        # 题面 invariant：variant=rmsnorm 时 residual 为垃圾输入，不参与计算，
        # 因此本分支完全不碰 residual
        y = rms_norm(x, weight, eps)
        if y is None or y.shape != (m, n) or y.dtype != x.dtype:
            raise RuntimeError(
                f"aiter rms_norm 返回异常：shape={None if y is None else tuple(y.shape)} "
                f"dtype={None if y is None else y.dtype}（期望 ({m}, {n}) / {x.dtype}）"
            )
        out = y
        entry = "rms_norm"
    else:
        if residual.shape != x.shape:
            raise ValueError(f"residual shape={tuple(residual.shape)} 与 x shape={tuple(x.shape)} 不符")
        if residual.dtype != x.dtype:
            raise ValueError(f"residual dtype={residual.dtype} 与 x dtype={x.dtype} 不符")
        residual = residual.contiguous()

        y = torch.empty_like(x)
        residual_out = torch.empty_like(x)
        # kernel 内 residual_out = x + residual（输入 dtype）并直通写出，
        # 再以该值做 RMSNorm 得 y
        ret = rmsnorm2d_fwd_with_add(y, x, residual, residual_out, weight, eps)
        if ret is not None:
            y = ret
        if y.shape != (m, n) or y.dtype != x.dtype:
            raise RuntimeError(
                f"aiter rmsnorm2d_fwd_with_add 返回异常：shape={tuple(y.shape)} "
                f"dtype={y.dtype}（期望 ({m}, {n}) / {x.dtype}）"
            )
        if residual_out.shape != (m, n) or residual_out.dtype != x.dtype:
            raise RuntimeError(
                f"residual_out 异常：shape={tuple(residual_out.shape)} dtype={residual_out.dtype}"
            )
        # 打包成题面 reference 的单张量输出（列拼接，与 reference 的
        # torch.cat([y, residual_out], dim=-1) 一致）
        out = torch.cat([y, residual_out], dim=-1)
        entry = "rmsnorm2d_fwd_with_add"

    torch.cuda.synchronize()

    ctx = {
        "path": entry,
        "variant": variant,
        "eps": eps,
        "m": m,
        "n": n,
        "dtype": str(x.dtype).replace("torch.", ""),
        "out_shape": tuple(out.shape),
        "use_blocked": _use_blocked(x),
    }
    return out, ctx
