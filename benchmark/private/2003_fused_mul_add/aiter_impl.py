# aiter_impl.py — 2003_fused_mul_add 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) —— 按名取参（构造参数稀疏
# 给出时位置式会错位，见 audit_model_class.py::case_init_kwargs 的说明）。
#
# 来源（aiter pinned 检出 .dcu_runs/aiter_pinned/，commit c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#   device_kernel : aiter/ops/triton/fused_mul_add.py
#                   sha256 fcb58763975e047fd5becd85e3355df2035ba4f375ed57aae6ad9edd418c94b2（已核对）
#                   宿主入口 fused_mul_add 在 :52-131，device kernel
#                   _fused_mul_add_kernel 在 :7-49。
#   official_test : op_tests/triton_tests/test_fused_mul_add.py
#                   sha256 f98792e05f7719e24b68bc100a501aab9ed22b48b28c8114d8284ccd8a809fb1（已核对）
#                   语义唯一权威 ref_mul_add 在 :32-33。
#
# 入口签名（fused_mul_add.py:52-57，官方测试 :56-59 的调用口径）：
#
#   fused_mul_add(x, a, b, out=None) -> out
#       x          torch.Tensor，任意形状，**必须 contiguous**
#       a, b       float | int | contiguous Tensor，numel ∈ {1, N}（N = x.numel()）
#       返回       out，与 x 同形同 dtype
#
# 布局说明：本题是纯逐元素算子，x 的维度布局无关紧要——kernel 只按扁平偏移
# 计算（fused_mul_add.py:23 `pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)`，
# N = x.numel()），因此 [16,50,4186] / [4096,8192] / [1] 等任意形状都直接传入，
# **无需 permute / flatten**；只做 contiguous 归一（aiter 在 :75-85 对 x/a/b 的
# contiguous 与 numel 都有 assert）。out 为 empty_like(x) 即与 x 同形同 dtype，
# 与 reference 的 `(a * x.float() + b).to(x.dtype)` 逐元素同义。
#
# 关键：必须显式传 out（out-of-place）。aiter 在 out=None 时(fused_mul_add.py:87-88)
# 会**原地改写 x**（x_triton = x 路径，即官方测试的 output=False 分支）；那既违反
# 题面「out-of-place，不得原地改写任何输入」（reference.py:53），也会让
# record_baseline 的重复计时逐次叠加乘加而记出错误基线。故传 out=empty_like(x)。
#
# dtype/形态对齐：题面 x ∈ {fp16, bf16, fp32}，a/b 与 x 同 dtype 且 contiguous
# （task.yaml io.invariants）。kernel 内部把 x/a/b 都 .to(tl.float32) 后乘加
# （fused_mul_add.py:29/37/45），输出 cast 回 out 的 element_ty（:48），与
# reference「中间 float32、输出 cast 回 x.dtype」一致。
#
# 无 autotune 依赖：fused_mul_add.py 内没有 @triton.autotune、不读
# AITER_TRITON_CONFIGS_PATH，tile/warps 是源码内写死的
# BLOCK_SIZE_N = max(min(next_pow2(N), 32), 1024)（恒为 1024）与 num_warps=4，
# 故 needs_autotune_config = False。
#
# ⚠️ 链路备注（与本适配器无关的评测器侧缺口）：record_baseline.py:170 用
# `[t.to(device) for t in case_inputs(...)]` 搬输入，对 Python float/int 形态的
# a/b 会 AttributeError。本题 perf case perf_16M_scalar_bf16 的 a/b 都是
# scalar_float，因此该 case 在 record_baseline 修好之前到不了本适配器。本适配器
# 自身对三种形态（标量 / 单元素张量 / 同形张量）都已支持，无需改动。

import torch


def _next_pow2(n: int) -> int:
    """等价 triton.next_power_of_2(n)（n >= 1），仅用于 ctx 记录 aiter 的 tile。"""
    return 1 << (n - 1).bit_length()


def _classify_operand(v, name: str, numel: int, shape, dtype, device):
    """把 a/b 归一成 aiter 入口接受的形态，越界即 raise（绝不静默用错参数）。

    返回 (归一后的值, 形态标签)。三类形态与 reference.py:40-44 的算子契约一一对应：
      scalar_float/scalar_int —— Python 标量原样透传（不得物化成同形大张量）；
      tensor1                 —— numel == 1 的连续张量，kernel 侧 tl.load(a_ptr) 广播；
      tensor_full             —— 与 x 同形（numel == N）的连续张量。
    """
    if isinstance(v, bool):
        # bool 是 int 的子类，会被 aiter 当成标量吞掉；题面不产生该形态，直接拦。
        raise TypeError(f"{name} 为 bool，题面 a/b 只有 float/int/张量三种形态")
    if isinstance(v, (int, float)):
        return v, "scalar_int" if isinstance(v, int) else "scalar_float"
    if not torch.is_tensor(v):
        raise TypeError(f"{name} 必须是 float/int 或 torch.Tensor，实际 {type(v)}")

    v = v.to(device)
    if v.dtype != dtype:
        raise ValueError(f"{name}.dtype={v.dtype} 与 x.dtype={dtype} 不一致（题目不变式：同 dtype）")
    n = v.numel()
    if n == 1:
        return v.contiguous(), "tensor1"          # 形状任意（含 (1,)），广播
    if n == numel:
        if tuple(v.shape) != tuple(shape):
            # numel 相等但形状不同：kernel 按扁平偏移取值，与 reference 的广播语义
            # 在一般情况下不等价（且 reference 输出形状也会偏离 x），必须拦下。
            raise ValueError(
                f"{name} numel={n} 与 x 相同但形状 {tuple(v.shape)} != x 形状 {tuple(shape)}；"
                "题面同形形态要求 shape 与 x 完全一致"
            )
        return v.contiguous(), "tensor_full"
    raise ValueError(
        f"{name}.numel()={n} 越界：aiter fused_mul_add 只接受 numel ∈ {{1, N={numel}}}"
        "（fused_mul_add.py:79/84 的 assert）"
    )


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（aiter.ops.triton.fused_mul_add.fused_mul_add）。

    inputs     : [x, a, b]（顺序同 tasks/2003_fused_mul_add/reference.py::make_inputs）
                 x 为 device 上的连续浮点张量（任意形状，含非 2 次幂）；
                 a/b 各自为 Python float/int、numel==1 张量、或与 x 同形张量。
    init_kwargs: 必须为空 {}——Model.__init__ 无参数（reference.py:72-74），
                 task.yaml io.init_inputs 声明为 []，逐元素算子无超参数。
    device     : 目标设备（张量已在 device 上，此处仅用于归一/ctx 记录）。

    返回 (out, ctx)；out 为与 x 同形同 dtype 的新张量（out-of-place，不改写输入）。
    """
    from aiter.ops.triton.fused_mul_add import fused_mul_add  # 顶层 import 很重，函数内 import

    # ---- 构造参数（本题无超参：任何多余键都说明调用方与题面不一致，直接 raise）----
    if init_kwargs:
        raise ValueError(
            f"2003_fused_mul_add 无构造参数（Model.__init__(self) 无参、io.init_inputs: []），"
            f"但收到 init_kwargs={sorted(init_kwargs)}"
        )

    if len(inputs) != 3:
        raise ValueError(f"inputs 必须是 [x, a, b] 三个元素，实际 {len(inputs)} 个")

    x, a, b = inputs

    # ---- x 合法性（task.yaml io.invariants：任意形状、连续、浮点、numel >= 1）----
    if not torch.is_tensor(x):
        raise TypeError(f"x 必须是 torch.Tensor，实际 {type(x)}")
    if not x.is_cuda:
        raise RuntimeError(f"x 必须在 DCU(CUDA) 设备上（aiter triton kernel 要求），实际 {x.device}")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"x.dtype={x.dtype} 不在题面域 {{float16, bfloat16, float32}}（reference 的 "
            "float32 中间计算假定浮点输入）"
        )
    numel = x.numel()
    if numel < 1:
        raise ValueError(f"x.numel()={numel} 非法（题面不变式 numel >= 1）")

    x_c = x.to(device).contiguous()
    a_c, a_kind = _classify_operand(a, "a", numel, x.shape, x.dtype, device)
    b_c, b_kind = _classify_operand(b, "b", numel, x.shape, x.dtype, device)

    # ---- 官方入口：显式 out（out-of-place），与 reference 同形同 dtype -------------
    out = torch.empty_like(x_c)
    fused_mul_add(x_c, a_c, b_c, out)

    torch.cuda.synchronize()

    # 复刻 aiter 的 launch 参数（fused_mul_add.py:113-114），仅作 ctx 记录
    block_size_n = max(min(_next_pow2(numel), 32), 1024)

    ctx = {
        "impl": "aiter",
        "module": "aiter.ops.triton.fused_mul_add",
        "path": "fused_mul_add",
        "semantics": "out = a*x + b (elementwise, single kernel single pass)",
        "x_shape": tuple(x_c.shape),
        "numel": numel,
        "a_kind": a_kind,
        "b_kind": b_kind,
        "dtype": str(x_c.dtype),
        "out_shape": tuple(out.shape),
        "out_dtype": str(out.dtype),
        "out_of_place": True,
        "block_size_n": block_size_n,
        "grid": ((numel + block_size_n - 1) // block_size_n,),
        "need_mask": numel % block_size_n != 0,
        "device": str(device),
    }
    return out, ctx
