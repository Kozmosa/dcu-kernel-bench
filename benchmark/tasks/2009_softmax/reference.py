# 2009_softmax —— 2D 行级 online softmax（单 kernel 两遍扫描）的 model_class
# （KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=2, problem_id=2009, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """2D 行级 softmax（数值稳定的 online 两遍扫描形态）。

    对 (M, N) 输入的每一行（最后一维）独立计算
        m[i]     = max_j x[i, j]
        z[i, j]  = exp(x[i, j] - m[i])
        y[i, j]  = z[i, j] / sum_j z[i, j]
    行间完全独立，逐行归一化后每行元素之和为 1。参考实现全程在 float32 中
    做 max / exp / 求和，输出 cast 回输入 dtype。
    目标实现形态（与题面同语义的计算组织方式）：单个 Triton kernel 内对每行
    做两遍分块扫描——第一遍（online）维护运行最大值 m 与重标定部分和
    s <- s * exp(m_old - m_new) + sum_block exp(x - m_new)，第二遍写出
    exp(x - m) / s；行长超出单个 block 的部分由循环分块处理。

    边界行为：N 为任意正整数——N=1 时该行输出恒为 1；N 非 2 的幂、以及
    超长行（N 超出单 block 上限——按 64KB 共享缓冲计，fp16/bf16 为
    65536/element_size = 32768 个元素）都必须用掩码 / 分块循环正确处理，
    行尾越界元素不得参与 max 与求和归约；输入为常规 randn 量级
    （无 inf/NaN）。

    输入输出规格：
        x   float16 / bfloat16 (M, N)，contiguous，M >= 1，N >= 1
        y   同 x dtype (M, N)，逐行 softmax 结果
        评测器会把全部输入 cast 成 fp32 传入（fp16/bf16 -> fp32 无损）；
        本算子无整数张量输入，无需恢复。

    实现约束（违规判负）：
        - 核心计算（行内 max 归约、exp 求和归约、逐元素归一化）必须在提交
          文件内以 Triton kernel 完成。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 中间累加（max / sum / exp）在 float32 中进行；输出 cast 回输入
          dtype。

    禁用列表：aiter / torch.softmax / torch.nn.functional.softmax /
    torch.nn.Softmax / torch.log_softmax / torch.nn.functional.log_softmax /
    torch.nn.LogSoftmax / torch.special.softmax / torch.special.log_softmax。
    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    """

    def __init__(self):
        super().__init__()
        # softmax 前向无超参数：block 尺寸、行内分块策略均属实现细节，
        # 由提交方自定。

    def forward(self, x):
        # 评测器把输入 cast 成评测精度（triton 后端为 fp32）后传入；统一在
        # float32 中计算（数值稳定：先减行 max 再 exp），输出 cast 回入口
        # dtype（在线为 fp32、离线为原 dtype）
        dt = x.dtype
        xf = x.to(torch.float32)
        m = xf.max(dim=-1, keepdim=True).values
        e = torch.exp(xf - m)
        y = e / e.sum(dim=-1, keepdim=True)
        return y.to(dt)


def get_init_inputs():
    return []   # 无超参数


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    m, n = 1024, 8192
    x = torch.randn(m, n).to(torch.float16)
    return [x]


def make_inputs(m: int, n: int, dtype: str = "float16", seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：x = torch.randn(m, n, generator=gen).to(dtype)，来自
    同一 torch.Generator(seed)，同 seed 下逐位可复现。
    返回 (x,) 单元组：离线终审按 inputs = [t.to(device) for t in inputs]
    逐元素消费后 Model(*inputs) 位置展开，裸张量会被迭代成行向量。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert m >= 1 and n >= 1, "需 M >= 1 且 N >= 1"
    x = torch.randn(m, n, generator=gen).to(dt)
    return (x,)
