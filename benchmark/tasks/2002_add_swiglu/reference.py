# 2002_add_swiglu — 训练用 Add + SwiGLU 融合（前向激活 + 一阶解析反向），
# KernelBench 兼容 model_class 题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。评测真值为 Model 的 float32 中间
# 计算、输出 cast 回输入 dtype；容差见 task.yaml。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=2, problem_id=2002, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """训练用 Add + SwiGLU 融合算子：前向激活 + 一阶解析反向，单次调用。

    算子语义：
      base 与 delta 逐元素相加后按最后一维对半拆分（记 D = width // 2）：
        gate = (base + delta)[..., :D]，up = (base + delta)[..., D:]
      前向为普通（非裁剪、非交错）SwiGLU：
        out = SiLU(gate) * up，其中 SiLU(g) = g * sigmoid(g)
      反向给定上游梯度 grad_out（形状与 out 相同），SiLU 的导数为
        SiLU'(g) = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
      base 与 delta 以相同系数进入和式，两路输入梯度逐元素相同：
        grad[..., :D] = (grad_out * up) * SiLU'(gate)
        grad[..., D:] = grad_out * SiLU(gate)
      本题把前向与反向合并为一次 forward 调用，返回单张量 fused（3D 列）：
        fused[..., 0:D)   = out（激活输出）
        fused[..., D:2D)  = grad 的 gate 半区（对 base 与对 delta 相同）
        fused[..., 2D:3D) = grad 的 up 半区
      边界行为：width 为非零偶数；num_rows 可为 0（空批次，输出 (0, 3D)）；
      输入张量不被修改；无广播、无隐式 dtype 提升；评测真值以 float32 中间
      计算为准，容差覆盖半精度舍入路径差异，无需逐位复现某条 eager 舍入链。

    输入输出规格：
      base     (num_rows, width)      连续 float16/bfloat16，width 非零偶数
      delta    (num_rows, width)      与 base 同形状、同 dtype、同 device
      grad_out (num_rows, width // 2) 与 base 同 dtype，out 的上游梯度
      返回      (num_rows, 3 * (width // 2)) 与 base 同 dtype，布局见上

    实现约束（违规判负）：
      - 核心计算（加法、SiLU/sigmoid、两条梯度公式）必须在提交文件内以
        Triton kernel 完成；torch 张量操作仅限输出显存分配与布局处理。
      - 禁止调用 aiter / torch.nn.functional.silu / torch.sigmoid /
        torch.matmul / torch.bmm / torch.einsum 等现成实现。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    """

    def __init__(self):
        super().__init__()

    def forward(self, base, delta, grad_out):
        # 输入全为浮点（上游可能整体 cast 成 float32），内部统一 float32 计算
        x = base.to(torch.float32)
        y = delta.to(torch.float32)
        g = grad_out.to(torch.float32)

        gate, up = (x + y).chunk(2, dim=-1)
        sig = torch.sigmoid(gate)
        silu = gate * sig
        out = silu * up

        # 一阶解析反向（对 base 与对 delta 相同）
        dgate = (g * up) * (sig + gate * sig * (1.0 - sig))
        dup = g * silu

        fused = torch.cat([out, dgate, dup], dim=-1)
        return fused.to(base.dtype)


def get_init_inputs():
    return []   # 算子无超参，D 由输入宽度推断


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    num_rows = 2048
    width = 8192      # D = 4096

    base = torch.randn(num_rows, width).to(torch.float16)
    delta = torch.randn(num_rows, width).to(torch.float16)
    grad_out = torch.randn(num_rows, width // 2).to(torch.float16)
    return [base, delta, grad_out]


def make_inputs(num_rows: int, width: int, dtype: str = "float16", seed: int = 0):
    """按公开/隐藏/性能案例描述生成输入（确定性，CPU 生成后由评测器搬运到设备）。

    base/delta 形状 (num_rows, width)，grad_out 形状 (num_rows, width // 2)。
    """
    assert isinstance(num_rows, int) and num_rows >= 0
    assert isinstance(width, int) and width > 0 and width % 2 == 0, "width 必须为非零偶数"
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    base = torch.randn(num_rows, width, generator=gen).to(dt)
    delta = torch.randn(num_rows, width, generator=gen).to(dt)
    grad_out = torch.randn(num_rows, width // 2, generator=gen).to(dt)
    return base, delta, grad_out
