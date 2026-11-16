# aiter_impl.py — 2002_add_swiglu 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) -> (out, ctx)，按名取参。
#
# ── 来源 ─────────────────────────────────────────────────────────────────────
# aiter 本地 pinned 检出：.dcu_runs/aiter_pinned/
# commit c39fff8c77df4e80617649e92fa3c2615f2c43d1
#   aiter/ops/triton/add_swiglu.py
#     sha256 adcdff8045250f2ade55202fd4499345b27547e39deea4f0585c78af9556132a
#   op_tests/test_add_swiglu_training.py
#     sha256 719df4e5cfe33b6a10f67ec5df214ad89864400373bce8461cdd92e0f0ce080a
# （与 benchmark/sources/2002_add_swiglu.yaml 记录一致）
#
# ── 公开入口（host 侧）───────────────────────────────────────────────────────
#   from aiter.ops.triton.add_swiglu import add_swiglu
#   add_swiglu(base: torch.Tensor, delta: torch.Tensor) -> torch.Tensor
#     base / delta: (..., 2*D) 连续、同形、同 dtype 的 fp16/bf16 CUDA(HIP) 张量，
#                   ndim >= 2，D > 0（add_swiglu.py:83-109，逐项校验后 raise）
#     返回：        (..., D)，与输入同 dtype（纯前向：silu((a+b)[..., :D]) * (a+b)[..., D:]）
#   该签名与官方测试的唯一调用方式一致（test_add_swiglu_training.py:8 导入、
#   :30-32 调用）。
#
# ── 反向路径 ─────────────────────────────────────────────────────────────────
# aiter 没有独立的反向 host 入口：一阶反向只由 torch.autograd.Function
# `_AddSwiGLU`（add_swiglu.py:51-80）提供，其 backward 里启动 @triton.jit
# `_backward`（add_swiglu.py:32-48）。官方测试同样走 autograd——`out.backward(grad)`
# 之后读 `x.grad` / `y.grad`（test_add_swiglu_training.py:31-32）。因此本适配器用
# `torch.autograd.grad(out, base, grad_out)` 触发**同一条** autograd 路径：dgate /
# dup 完全由 aiter 的 `_backward` Triton kernel 算出，torch 只负责把上游梯度送进
# aiter 的 Function，没有任何 torch 高层算子参与核心计算。
#
# _backward 的写出布局（add_swiglu.py:35-48）：DX（= empty_like(base)，形状
# (num_rows, 2D)）的前半区存 dgate、后半区存 dy*silu = dup，即
#   dx = [ dgate | dup ]，dgate/dup 均为 (num_rows, D)
# 与题面 reference 的两路梯度逐元素一致（base 与 delta 以相同系数进入和式，
# 两路梯度相同）：
#   dgate = (grad_out * up) * (sig + gate*sig*(1-sig))      (= (g*up)*silu'(gate))
#   dup   = grad_out * silu(gate)
# 故打包为 fused = cat([out, dx], dim=-1) 即 reference 的
# cat([out, dgate, dup], dim=-1) → (num_rows, 3D)，dtype 与 base 相同。
#
# ── 布局 ─────────────────────────────────────────────────────────────────────
# 本题输入 (num_rows, width) 已是 aiter 要求的 (..., 2D) 连续布局，无需 permute；
# grad_out (num_rows, D) 与 aiter 前向输出 out 同形，可直接作 grad_outputs。
# 唯一需要的 torch 操作是连续性保证（contiguous 对已连续张量为 no-op）与最终的
# cat 打包。
#
# ── init_kwargs ──────────────────────────────────────────────────────────────
# 本题 io.init_inputs 为 []（算子无超参，D 由输入宽度推断），故 init_kwargs 为空；
# 出现任何未声明的构造参数即 raise（不静默忽略）。

import torch


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方 add_swiglu（前向 + 一阶解析反向），按题面契约打包。

    inputs     : [base (num_rows, width), delta (num_rows, width),
                  grad_out (num_rows, width // 2)]，均为连续 fp16/bf16
    init_kwargs: {}（本题无构造参数；出现额外键即 raise）
    返回        : (fused (num_rows, 3*width//2), ctx)
    """
    from aiter.ops.triton.add_swiglu import add_swiglu

    base, delta, grad_out = inputs
    init_kwargs = dict(init_kwargs or {})
    if init_kwargs:
        raise ValueError(
            f"2002_add_swiglu 无构造参数（io.init_inputs 为 []），收到未声明的 "
            f"init_kwargs={sorted(init_kwargs)}"
        )

    # ── 合法性校验：越界就 raise，绝不静默用错参数 ──────────────────────────
    if base.ndim != 2 or delta.ndim != 2 or grad_out.ndim != 2:
        raise ValueError(
            f"2002_add_swiglu 期望 2D 输入，收到 base{tuple(base.shape)} "
            f"delta{tuple(delta.shape)} grad_out{tuple(grad_out.shape)}"
        )
    num_rows, width = int(base.shape[0]), int(base.shape[1])
    if width == 0 or width % 2 != 0:
        raise ValueError(f"width 必须为非零偶数（D = width // 2 > 0），收到 width={width}")
    if delta.shape != base.shape:
        raise ValueError(f"delta 形状 {tuple(delta.shape)} 与 base {tuple(base.shape)} 不一致")
    half_width = width // 2
    if tuple(grad_out.shape) != (num_rows, half_width):
        raise ValueError(
            f"grad_out 形状 {tuple(grad_out.shape)} 与 (num_rows, width//2)="
            f"({num_rows}, {half_width}) 不一致"
        )
    if base.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"aiter add_swiglu 仅支持 fp16/bf16，收到 {base.dtype}")
    if delta.dtype != base.dtype or grad_out.dtype != base.dtype:
        raise ValueError(
            f"base/delta/grad_out dtype 必须一致，收到 {base.dtype}/{delta.dtype}/{grad_out.dtype}"
        )

    # aiter 要求连续输入；已连续时为 no-op，不做任何数据搬运语义改变。
    base_c = base.detach().contiguous()
    delta_c = delta.detach().contiguous()
    grad_out_c = grad_out.detach().contiguous()

    # 前向走 aiter 公开入口；把 base 变成可求导的叶子以触发 aiter 的
    # _AddSwiGLU.backward（delta 不需要求导：两路梯度逐元素相同，一次反向即得）。
    base_leaf = base_c.requires_grad_(True)
    out = add_swiglu(base_leaf, delta_c)      # (num_rows, D)，dtype = base.dtype
    if out.shape != (num_rows, half_width) or out.dtype != base.dtype:
        raise RuntimeError(
            f"aiter add_swiglu 输出异常：shape={tuple(out.shape)} dtype={out.dtype}，"
            f"期望 ({num_rows}, {half_width}) / {base.dtype}"
        )

    # aiter 的一阶反向（_backward Triton kernel）：dx = [dgate | dup]，(num_rows, width)
    (dx,) = torch.autograd.grad(out, base_leaf, grad_out_c, create_graph=False)
    if tuple(dx.shape) != (num_rows, width) or dx.dtype != base.dtype:
        raise RuntimeError(
            f"aiter 反向输出异常：shape={tuple(dx.shape)} dtype={dx.dtype}，"
            f"期望 ({num_rows}, {width}) / {base.dtype}"
        )

    # 打包成题面单张量契约：fused = [out | dgate | dup]（reference.py:72）。
    # out.detach()：反向已由上面的 autograd.grad 取走，返回值只需数值，不带图。
    fused = torch.cat([out.detach(), dx], dim=-1).contiguous()

    torch.cuda.synchronize()

    ctx = {
        "impl": "aiter",
        "path": "aiter.ops.triton.add_swiglu.add_swiglu (+) + _AddSwiGLU.autograd grads (_backward)",
        "aiter_module": "aiter.ops.triton.add_swiglu",
        "aiter_symbol": "add_swiglu",
        "commit": "c39fff8c77df4e80617649e92fa3c2615f2c43d1",
        "num_rows": num_rows,
        "width": width,
        "half_width": half_width,
        "dtype": str(base.dtype),
        "out_shape": tuple(out.shape),
        "dx_shape": tuple(dx.shape),
        "fused_shape": tuple(fused.shape),
        "packing": "cat([out, dx], dim=-1); dx[:, :D]=dgate, dx[:, D:]=dup",
        "device": str(device),
    }
    return fused, ctx
