# 2003_fused_mul_add — 逐元素融合乘加 out = a*x + b（KernelBench 兼容 model_class 题目）。
#
# 本文件必须自包含：评测器以 exec(source) 加载题目，禁止本地模块导入，
# 参考实现（Model.forward）与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=2, problem_id=2003, backend=triton。

import torch
import torch.nn as nn

# a / b 的四种取值形态（与算子契约一致）：
#   scalar_float —— Python float 标量；scalar_int —— Python int 标量；
#   tensor1      —— 单元素张量（numel == 1）；tensor_full —— 与 x 同形的张量。
_OPERAND_KINDS = ("scalar_float", "scalar_int", "tensor1", "tensor_full")


def _draw_operand(kind, shape, dt, gen):
    """按形态生成 a / b：标量取 randn*100（对齐官方算子的量级习惯），张量直接 randn。

    gen 为 None 时消费全局 RNG（get_inputs 路径），否则消费给定 Generator
    （make_inputs 确定性路径）。
    """
    if kind == "tensor_full":
        return torch.randn(*shape, dtype=dt, generator=gen)
    base = torch.randn(1, dtype=dt, generator=gen)
    if kind == "scalar_float":
        return float(base.item() * 100)
    if kind == "scalar_int":
        return int(base.item() * 100)
    if kind == "tensor1":
        return base.reshape(1)
    raise ValueError(f"未知形态: {kind}")


class Model(nn.Module):
    """逐元素融合乘加（fused multiply-add）：out = a * x + b。

    ① 算子语义：对 x 的每个元素 i，out[i] = a_i * x[i] + b_i。a 与 b 各自
    独立地取以下三种形态之一（可任意组合，如 a 为标量、b 为同形张量）：
      1. Python float / int 标量：a_i = a（值广播到 x 的每个元素）；
      2. 单元素张量（numel == 1，形状任意，如 (1,)）：a_i = 其唯一元素（广播）；
      3. 与 x 同形的张量（numel == x.numel()）：a_i = a 的对应元素。
    x 为任意维数、任意形状（含非 2 次幂、素数规模）的浮点张量。中间计算在
    float32 中完成（x / a / b 先提升到 float32 再乘加），最终结果 cast 回
    x 的 dtype。边界行为：本算子无整数张量输入、无索引/长度类语义；a、b
    同为标量时输出仍与 x 同形。

    ② 输入输出规格：
      - x：float16 / bfloat16 / float32 张量，任意形状，contiguous；
      - a、b：上述三种形态之一；为张量时 contiguous 且 dtype 与 x 一致；
      - 返回：与 x 同形、同 dtype 的新张量（out-of-place，不得原地改写任何输入）。

    ③ 实现约束（违规判负）：
      - 核心计算（乘加）必须在提交文件内完成：自写 Triton kernel，单个
        kernel、单次读入-计算-写回 pass 完成，禁止拆成先乘后加两个 kernel；
      - a / b 为标量或单元素张量时作为广播参数在 kernel 内参与计算，不得
        物化为与 x 同形的临时大张量；
      - ModelNew 的 __init__ 与 forward 签名不可更改；
      - 中间计算用 float32；输出 cast 回 x 的 dtype。

    ④ 禁用列表与目标硬件：
      - 禁止以 ATen 捷径直接完成本题计算：torch.addcmul（含 .addcmul /
        .addcmul_ 方法）、torch.add、torch.mul，以及 .add( / .mul( /
        .add_( / .mul_( 等张量方法与原地形式；亦禁止调用任何预编译
        算子库提供的同功能入口（终审静态审计对 docstring/注释之外的代码
        做子串匹配）；
      - 目标硬件：海光 DCU（gfx936），实现语言 Triton。
    """

    def __init__(self):
        super().__init__()
        # 逐元素算子无超参数：block 尺寸与广播策略均属实现细节，由提交方自定。

    def forward(self, x, a, b):
        # 评测器会把输入张量统一 cast 成评测精度（triton 后端为 fp32）再传入；
        # a / b 为 Python 标量时原样透传。本算子无整数张量输入，无需恢复。
        return (a * x.to(torch.float32) + b).to(x.dtype)


def get_init_inputs():
    return []


def get_inputs():
    # 固定 shape 族（与 public_cases.json 一致）；a / b 的形态由全局 RNG 从
    # 四种形态独立等概率抽取——评测器 set_seed 后调用本函数，多轮 correctness
    # trial 因此覆盖标量 / 单元素 / 同形张量的不同组合。
    shape = (2048, 4096)
    dt = torch.float16
    a_kind = _OPERAND_KINDS[torch.randint(0, len(_OPERAND_KINDS), (1,)).item()]
    b_kind = _OPERAND_KINDS[torch.randint(0, len(_OPERAND_KINDS), (1,)).item()]
    x = torch.randn(*shape, dtype=dt)
    a = _draw_operand(a_kind, shape, dt, None)
    b = _draw_operand(b_kind, shape, dt, None)
    return [x, a, b]


def make_inputs(shape, dtype="float16", a_kind="tensor_full", b_kind="tensor1", seed=0):
    """确定性 case 生成器：torch.Generator().manual_seed(seed)，CPU 生成。

    字段与隐藏/性能 case 一一对应（shape 列表、dtype 字符串、a_kind/b_kind
    形态名、seed），返回 (x, a, b)。仅在评测端运行。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    shape = tuple(shape)
    x = torch.randn(*shape, dtype=dt, generator=gen)
    a = _draw_operand(a_kind, shape, dt, gen)
    b = _draw_operand(b_kind, shape, dt, gen)
    return x, a, b
