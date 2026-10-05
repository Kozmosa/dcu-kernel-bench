# aiter_impl.py — 2006_norm（fused add LayerNorm 前向）的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线
# （先与 reference 按 task.yaml 容差比对通过才记录）。
#
# ── 来源 ────────────────────────────────────────────────────────────────────
#   repo    : OpenDAS/aiter @ c39fff8c77df4e80617649e92fa3c2615f2c43d1
#   file    : aiter/ops/triton/norm.py
#   sha256  : 4248394b4cabaf0114be87b36d9026489445fa3f2bed8426594378e7fa6bfa0c
#             （与 benchmark/sources/2006_norm.yaml 的准入记录逐位一致）
#   语义依据: op_tests/triton_tests/test_layernorm.py
#             sha256 c1e8bf41e98ee422e2436a7cd326bfa42a0e30e3fd686b42ceb17603bdd54dbd
#             （test_fused_add_layernorm 的 run_torch 为语义唯一权威）
#
# ── 公开算子入口（真正的 host 入口，不是 kernel）────────────────────────────
#   norm.py:1072
#   def layernorm2d_fwd_with_add(out, input, residual_in, residual_out,
#                                weight, bias, epsilon, x_bias=None) -> out
#
#   该入口是 torch.autograd.Function 包装（norm.py:1014
#   _Layernorm2dFwdWithAdd）：两个输出张量 out / residual_out 都由**调用方预
#   分配**、由 kernel 就地写入；返回值就是传入的 out（同一块显存）。
#   底层 kernel = _fused_add_layernorm_kernel（norm.py:136），host 分发函数 =
#   _layernorm_forward_with_add（norm.py:607）：grid=(M,)，一个 program 处理
#   一整行；BLOCK_SIZE = min(65536 // x.element_size(), next_power_of_2(N))，
#   N > BLOCK_SIZE 时在 kernel 内走分块循环（掩码只作用在最后一块）。
#   无 @triton.autotune、不读 AITER_TRITON_CONFIGS_PATH → 不需要任何 config JSON。
#
#   注意区分：aiter/ops/norm.py:53 的同名函数是 hip/C++ 绑定（@compile_ops，
#   返回 None，void 语义），本题按 sources 准入记录取的 device_kernel 是纯
#   Triton 实现，故一律走 aiter/ops/triton/norm.py 这一支。
#
# ── 官方测试的调用约定（test_layernorm.py:65-69）────────────────────────────
#   residual_out = torch.empty_like(input)
#   output = torch.empty_like(input)
#   output = layernorm2d_fwd_with_add(output, input, residual, residual_out,
#                                     weight, bias, eps, x_bias)     # 7 位置参 + x_bias
#   （x_bias 该函数内未使用，本题没有对应张量，固定传 None。）
#
# ── 布局 ────────────────────────────────────────────────────────────────────
#   全程 2D (M, N) 行主序 contiguous。kernel 用 x.stride(0) 定位 residual_in /
#   residual_out 的行、用 y.stride(0) 定位 out 的行（norm.py:634-635），因此
#   x / residual / residual_out / out 四者都必须是 contiguous 的 (M, N)。
#   题面 make_inputs 产出的就是该布局（无 bshd/sbhd/varlen 概念），无需 permute
#   或 flatten，只需 defensive 的 .contiguous()。
#
# ── 与题面 reference 的语义对齐 ─────────────────────────────────────────────
#   kernel 先算 s = x + residual 并**就地写回** residual_out（输入 dtype），
#   随后方差/归一化再**重读** residual_out（norm.py:192 / 213）——与 reference
#   「统计以舍入回输入 dtype 的 residual_out 为基准」逐句对应；mean、有偏 var、
#   rsqrt(var + eps)、仿射全在 fp32 中求值，y 与 residual_out 落回输入 dtype。
#
# ── 输出打包 ────────────────────────────────────────────────────────────────
#   题面契约是单张量 (M, 2N)（前 N 列 y、后 N 列 residual_out），而 aiter 入口
#   是「两个就地写入的张量 + 返回 out」。适配器用允许的打包操作
#   torch.cat([out, residual_out], dim=-1) 复现同样的拼接口径（两段 dtype 一致，
#   都是输入 dtype），不做任何数值重算。
#
# 已知遗留（真机观察项，非适配器可解）：官方测试 get_vals 里 (8192, 8192) /
# (4096, 8192) 因 triton-internal#843 被注释掉（test_layernorm.py:106-123）；
# perf case perf_m8192_n8192_f16 正落在该 shape 上。若真机复现该上游缺陷，
# record_baseline 会以「与 reference 不一致」拒绝记录基线（不会静默记错），
# 此时需要换 case 或等 aiter 修复——适配器本身无替代入口可选。

import math

import torch


def _next_power_of_2(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现。

    inputs     : [x, residual, weight, bias]
                 x/residual (M, N)、weight/bias (N,)，均已搬到 device 上，
                 dtype 为 float16 / bfloat16 / float32（四者同 dtype）。
    init_kwargs: {"eps": float}（Model.__init__ 的 eps，默认 1e-5）

    返回 (packed, ctx)：packed 为单张量 (M, 2N)，与 reference 输出同形同 dtype。
    """
    from aiter.ops.triton.norm import layernorm2d_fwd_with_add

    if len(inputs) != 4:
        raise ValueError(
            f"2006_norm 期望 4 个输入 [x, residual, weight, bias]，收到 {len(inputs)} 个"
        )
    x, residual, weight, bias = inputs

    # 构造参数按名取，缺省与 get_init_inputs() 一致（eps=1e-5）
    eps = float(init_kwargs.get("eps", 1e-5))
    if not math.isfinite(eps):
        raise ValueError(f"2006_norm: eps 必须有限，收到 {init_kwargs.get('eps')!r}")

    supported = (torch.float16, torch.bfloat16, torch.float32)
    if x.dtype not in supported:
        raise TypeError(f"2006_norm: x.dtype={x.dtype} 不在 aiter 入口支持范围 {supported}")
    for name, t in (("residual", residual), ("weight", weight), ("bias", bias)):
        if t.dtype != x.dtype:
            raise TypeError(
                f"2006_norm: io 要求 {name}.dtype == x.dtype，"
                f"实际 {t.dtype} vs {x.dtype}"
            )

    # 布局转换：kernel 只用 stride(0) 定位行，要求四个张量都是行主序 contiguous
    x = x.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    if x.dim() != 2:
        raise ValueError(f"2006_norm: x 期望 2D (M, N)，收到 {tuple(x.shape)}")
    M, N = x.shape
    if M < 1 or N < 1:
        raise ValueError(f"2006_norm: 需 M >= 1 且 N >= 1，收到 M={M}, N={N}")
    if residual.shape != x.shape:
        raise ValueError(
            f"2006_norm: residual 必须与 x 同形，收到 {tuple(residual.shape)} vs {tuple(x.shape)}"
        )
    if weight.numel() != N or bias.numel() != N:
        raise ValueError(
            f"2006_norm: weight/bias 应各含 N={N} 个元素，"
            f"收到 {weight.numel()} / {bias.numel()}"
        )
    weight = weight.reshape(N)
    bias = bias.reshape(N)

    # 两个输出由调用方预分配（与官方测试一致）
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)

    # 只取前向结果：官方入口的 backward 路径（_layernorm_backward）不在本题范围，
    # 关掉梯度记录避免多余的计算图/保存开销（对前向数值无影响）。
    with torch.no_grad():
        out = layernorm2d_fwd_with_add(
            out,
            x,
            residual,
            residual_out,
            weight,
            bias,
            epsilon=eps,
            x_bias=None,
        )

    # 打包成题面契约的单张量 (M, 2N)：前 N 列 y、后 N 列 residual_out
    packed = torch.cat([out, residual_out], dim=-1)

    torch.cuda.synchronize()

    block_size = min(65536 // x.element_size(), _next_power_of_2(N))
    return packed, {
        "aiter_entry": "aiter.ops.triton.norm.layernorm2d_fwd_with_add",
        "path": "fused_add_layernorm_fwd/_fused_add_layernorm_kernel",
        "M": int(M),
        "N": int(N),
        "dtype": str(x.dtype),
        "eps": eps,
        "block_size": int(block_size),
        "blocked": bool(N > block_size),
        "packed_shape": list(packed.shape),
    }
