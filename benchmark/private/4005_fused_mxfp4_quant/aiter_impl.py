# aiter_impl.py — 4005_fused_mxfp4_quant 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线；
# 契约与样例 private/1002_paged_attention/aiter_impl.py 一致：
#     run(inputs, init_kwargs: dict, device) -> (out, ctx)
# 其中 init_kwargs 是 task.yaml `io.init_inputs` 声明并按名绑定的构造参数
# （本题：dtype / eps），绝不按位置取值。
#
# 来源（aiter pinned 检出 .dcu_runs/aiter_pinned，commit c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#   aiter/ops/triton/fused_mxfp4_quant.py
#       sha256 8f6919e1b872e5b9934a481387d4bc00e27432970c2130823d3cc24a797b33be
#       入口（host 侧公开算子，非 kernel、非 helper）：
#         fused_rms_mxfp4_quant(inp1, inp1_weight, inp1_epsilon,
#                               inp2=None, inp2_weight=None, inp2_epsilon=0.0,
#                               res1=None)
#       —— 源文件 :108。它把 res1 相加（FIRST_INPUT_RES）、RMSNorm（_rmsmorm_op,
#       :9）、mxfp4 量化（aiter.ops.triton.quant._mxfp4_quant_op, quant.py:364）
#       全部放进一个 Triton kernel _fused_rms_mxfp4_quant_kernel(:19)，
#       grid=(M,)，逐行处理。
#   op_tests/triton_tests/test_fused_mxfp4_quant.py（官方测试，调用口径权威）
#       sha256 81f1172815d59bc38b1495a1bc70320a28a1794bf798be1ac1404baf12c92ad0
#       :118 无残差、:122 带残差的调用形态：残留直通/第二路 RMSNorm 均为可选，
#       本题只用 residual 分支 + skip_second。
#
# 语义对齐（本题 reference.py 即官方测试 calculate_target_w_torch +
# torch_dynamic_mxfp4_quant 的逐位移植，见 sources/4005_fused_mxfp4_quant.yaml）：
#   * 残差：kernel 内 inp1.to(fp32) + res1.to(fp32)（:52-:65）
#          == reference 的 s = inp1.to(fp32) + res1.to(fp32)。
#   * RMSNorm：row*row → sum → rsqrt(sum/n_cols + eps) → row * norm_factor * weight
#          （_rmsmorm_op :10-:14）== reference _rmsnorm（含 /n_cols 与 eps=1e-6）。
#   * 量化：amax 位型 +0x200000 & 0xFF800000 向上圆整到 2 的幂、floor(log2)-2
#          clamp[-127,127]、e8m0 = e+127、round-half-up（(E<<2|M>>21)+1>>1、
#          饱和 min(·,7)）、低 4 位放偶下标（tl.split 的 evens/odds，
#          quant.py:404-:433）—— 与 reference._mxfp4_quant 逐位同构。
#
# 布局/契约说明（本题无 layout 重排）：
#   * 题目输入 inp1 (M,N1) / weight1 (N1,) / res1 (M,N1) 已是 aiter 期望的
#     行主序 2-D / 1-D 形态（官方测试用的是 strided 切片行，本评测集简化为
#     contiguous，故只需 .contiguous() 兜底，无 permute）。
#   * aiter 返回**元组**：res1 提供且 inp2 为 None 时
#     `return (out1_fp4, out1_bs), out_res1`（fused_mxfp4_quant.py:201-:203）。
#     本题输出契约是单张量：reference.forward 末尾
#     `torch.cat([codes, scales], dim=1)`，即
#     out[:, :N1//2] = packed e2m1 码字、out[:, N1//2:] = e8m0 块 scale，
#     故本适配器把 (out1_fp4, out1_bs) 按同一顺序 cat 回单张量
#     (M, N1//2 + ceil(N1/32))，dtype uint8：
#       out1_fp4 : (M, N1//2)              torch.uint8（kernel :79 按行存储）
#       out1_bs  : (M, ceil(N1/32))        torch.uint8（host 侧以
#                  torch.empty((K, M)).T 生成，行内 stride=M，kernel 用
#                  out1_bs_row_stride/col_stride 正确寻址；cat 会拷贝成连续）
#   * out_res1（残差直通，inp1.dtype）与第二路 RMSNorm 输出本题不消费，
#     与准入记录一致（被单张量契约排除）。
#
# 依赖：无 @triton.autotune、无 AITER_TRITON_CONFIGS_PATH —— 入口只做一次
# triton.next_power_of_2 的形状计算后直接 launch，故不需要任何 autotune
# config JSON（needs_autotune_config=false）。

import torch


# 官方 fused_mxfp4_quant.py 内固定值（:139，量化块大小，沿最后一维）
_MXFP4_QUANT_BLOCK_SIZE = 32
# task.yaml io.init_inputs 声明的 dtype 取值（Model.__init__ 用 getattr(torch, dtype)）
_ALLOWED_DTYPES = {"bfloat16", "float16", "float32"}


def _declared_dtype(init_kwargs):
    """解析 init_kwargs['dtype']（str 或 torch.dtype），越界即 raise。"""
    dtype = init_kwargs.get("dtype", "bfloat16")
    if isinstance(dtype, str):
        if dtype not in _ALLOWED_DTYPES:
            raise ValueError(
                f"4005 适配器：init_kwargs['dtype']={dtype!r} 不在 "
                f"{sorted(_ALLOWED_DTYPES)} 内（与 task.yaml io.init_inputs 契约不符）"
            )
        return getattr(torch, dtype)
    if isinstance(dtype, torch.dtype):
        if not dtype.is_floating_point:
            raise ValueError(f"4005 适配器：init_kwargs['dtype']={dtype} 不是浮点 dtype")
        return dtype
    raise TypeError(f"4005 适配器：init_kwargs['dtype'] 类型非法: {type(dtype)!r}")


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方 fused_rms_mxfp4_quant（residual 分支）。

    inputs      : [inp1 (M, N1), weight1 (N1,), res1 (M, N1)]
                  同 dtype（bfloat16/float16；评测器若已 cast 成 fp32 亦无损）
    init_kwargs : {"dtype": "bfloat16"|"float16", "eps": 1e-6}
    device      : torch.device（张量已由调用方搬到该设备）

    返回 (out, ctx)：out 为单张量 uint8 (M, N1//2 + ceil(N1/32))，
    前 N1//2 列为 packed e2m1 码字、其后为 e8m0 块 scale 字节。
    """
    # aiter 一律函数内 import（顶层 import 很重；真机部署走最小导入垫片）
    from aiter.ops.triton.fused_mxfp4_quant import fused_rms_mxfp4_quant
    from triton import next_power_of_2

    if len(inputs) != 3:
        raise ValueError(
            f"4005 适配器：inputs 应为 [inp1, weight1, res1]，实际 {len(inputs)} 个"
        )
    inp1, weight1, res1 = inputs

    dt = _declared_dtype(init_kwargs)
    eps = float(init_kwargs.get("eps", 1e-6))
    if not (eps >= 0.0) or eps != eps or eps == float("inf"):
        raise ValueError(f"4005 适配器：eps={eps!r} 非法（需为非负有限值）")

    # ---- 形状/布局校验：越界即 raise，绝不静默用错参数 ----
    if inp1.dim() != 2:
        raise ValueError(f"4005 适配器：inp1 应为 (M, N1) 2-D，实际 {tuple(inp1.shape)}")
    if res1.shape != inp1.shape:
        raise ValueError(
            f"4005 适配器：res1{tuple(res1.shape)} 与 inp1{tuple(inp1.shape)} 形状不一致"
        )
    if weight1.dim() != 1:
        raise ValueError(f"4005 适配器：weight1 应为 (N1,) 1-D，实际 {tuple(weight1.shape)}")
    M, N1 = int(inp1.shape[0]), int(inp1.shape[1])
    if M < 1:
        raise ValueError(f"4005 适配器：M={M} 越界（要求 M >= 1）")
    if N1 < 2 or N1 % 2 != 0:
        raise ValueError(f"4005 适配器：N1={N1} 越界（要求 N1 为偶数且 >= 2）")
    if int(weight1.numel()) != N1:
        raise ValueError(
            f"4005 适配器：weight1 元素数 {int(weight1.numel())} != N1={N1}"
        )
    for name, t in (("inp1", inp1), ("weight1", weight1), ("res1", res1)):
        if not t.is_floating_point():
            raise ValueError(f"4005 适配器：{name}.dtype={t.dtype} 不是浮点 dtype")
    if not (inp1.dtype == weight1.dtype == res1.dtype):
        raise ValueError(
            "4005 适配器：inp1/weight1/res1 dtype 必须一致，实际 "
            f"{inp1.dtype}/{weight1.dtype}/{res1.dtype}"
        )

    # 题面为 contiguous 契约（task.yaml layout: contiguous）；.contiguous() 兜底
    # 而不改变数值（aiter kernel 按显式 row_stride 寻址，但 weight1 是纯 1-D 索引）。
    inp1 = inp1.contiguous()
    weight1 = weight1.contiguous()
    res1 = res1.contiguous()

    # 官方入口参数序（源文件 :108）：(inp1, inp1_weight, inp1_epsilon,
    #                                     inp2, inp2_weight, inp2_epsilon, res1)
    # inp2 全为 None → SKIP_SECOND_INPUT=True；res1 非 None → FIRST_INPUT_RES=True。
    ret = fused_rms_mxfp4_quant(inp1, weight1, eps, None, None, None, res1)
    if not (isinstance(ret, tuple) and len(ret) == 2 and isinstance(ret[0], tuple)):
        raise RuntimeError(
            f"4005 适配器：aiter 返回结构异常（期望 ((out1_fp4, out1_bs), out_res1)），"
            f"实际 {type(ret)!r}"
        )
    (out1_fp4, out1_bs), out_res1 = ret

    scale_cols = (N1 + _MXFP4_QUANT_BLOCK_SIZE - 1) // _MXFP4_QUANT_BLOCK_SIZE
    if tuple(out1_fp4.shape) != (M, N1 // 2) or out1_fp4.dtype != torch.uint8:
        raise RuntimeError(
            f"4005 适配器：out1_fp4 形状/dtype 异常: {tuple(out1_fp4.shape)} {out1_fp4.dtype}"
        )
    if tuple(out1_bs.shape) != (M, scale_cols) or out1_bs.dtype != torch.uint8:
        raise RuntimeError(
            f"4005 适配器：out1_bs 形状/dtype 异常: {tuple(out1_bs.shape)} {out1_bs.dtype}"
        )

    # 单张量打包：码字在前、scale 字节在后（与 reference.forward 的 cat 顺序一致）。
    out = torch.cat([out1_fp4, out1_bs], dim=1).contiguous()
    if tuple(out.shape) != (M, N1 // 2 + scale_cols) or out.dtype != torch.uint8:
        raise RuntimeError(
            f"4005 适配器：打包输出形状/dtype 异常: {tuple(out.shape)} {out.dtype}"
        )

    torch.cuda.synchronize()
    return out, {
        "path": "fused_rms_mxfp4_quant(residual=True, skip_second=True)",
        "aiter_module": "aiter.ops.triton.fused_mxfp4_quant",
        "aiter_symbol": "fused_rms_mxfp4_quant",
        "kernel": "_fused_rms_mxfp4_quant_kernel",
        "M": M,
        "N1": N1,
        # aiter host 侧同式：BLOCK_SIZE = max(next_power_of_2(N1), 32)（:141/:150）
        "block_size": max(next_power_of_2(N1), _MXFP4_QUANT_BLOCK_SIZE),
        "codes_cols": N1 // 2,
        "scale_cols": scale_cols,
        "eps": eps,
        "input_dtype": str(inp1.dtype),
        "declared_dtype": str(dt),
        "out_dtype": str(out.dtype),
        "out_shape": tuple(out.shape),
        "unused_outputs": ["out_res1"],
        "note": "aiter 元组 (out1_fp4, out1_bs) 按 cat(dim=1) 打包为单张量 uint8",
    }
