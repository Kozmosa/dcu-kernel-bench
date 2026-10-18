# 8001_cumsum —— Cumsum（inclusive 前缀和）的 model_class（KernelBench
# 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=2, problem_id=8001, backend=cuda（HIP）。

import torch
import torch.nn as nn


class Model(nn.Module):
    """Cumsum（inclusive 前缀和）：沿指定维的顺序累加。

    算子定义：对 n 维输入 x 与 scan 维 dim，输出 out 与 x 同形：
        out[i_0, ..., i_k, ...] = sum_{j <= i_dim} x[i_0, ..., j, ...]
    即 dim 维上的 inclusive 前缀和（含当前元素），其余维度各条链独立
    累加。__init__ 参数 dim 指定 scan 维（torch 惯例：非负从 0 起，负数
    从末尾计数，如 -1 为最后一维），缺省 2。

    数值契约：累加一律在 float32 中进行——float16/bfloat16 输入先 cast
    到 float32 累加，再 cast 回输入 dtype（与生产 scan 库的
    compute-in-fp32 口径一致）。

    输入输出规格：
        x     float32 / float16 / bfloat16，contiguous，ndim ∈ [1, 4]，
              各维 >= 1，scan 维长度 L ∈ [1, 65536]，总元素数 <= 2^25；
              值域为常规 randn 量级（无 inf/NaN）
        out   单张量，与 x 同形状、同 dtype、contiguous；forward 必须返回
              单个张量（评测器按单输出的 .shape / allclose 比较）
        在线评测器会把全部输入 cast 成 fp32 传入（fp16/bf16 -> fp32 无损）；
        离线终审按声明 dtype 原生传入——实现须对两种传入都正确。

    边界行为：L=1 时 out 等于 x（cast 回输入 dtype）；L 非 2 的幂、素数
    长度、超出单 block 容量的长轴都必须用掩码 + 分块扫描正确处理，越界
    lane 不得进入前缀和；dim 为负数时按 torch 惯例解释。

    实现约束（违规判负）：
        - 核心 scan（前缀和）必须在提交文件内以 HIP kernel 手写完成，
          禁止调用任何库级 scan 原语，禁止用 FFT / 矩阵乘 / 卷积等
          数学等价路径绕过。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 累加在 float32 中进行，输出 cast 回输入 dtype。
        - forward 返回单张量 out（评测器按单输出比较）。

    禁用列表（静态审计剥除 docstring 与注释后按字面子串匹配）：
    torch.cumsum / .cumsum( / torch.cumulative_sum / .cumulative_sum( /
    aten.cumsum / aten.cumulative_sum /
    torch.fft / torch.matmul / torch.mm / torch.bmm / torch.einsum /
    conv1d( / numpy / hipdnn。
    目标硬件：海光 DCU（gfx936/gfx938），实现语言 HIP（经 hipcc 编译；在 CUDA_CU_SRC / CUDA_CPP_SRC 锚点区编写，设备侧必须 include <hip/hip_runtime.h>，本机无 <cuda_runtime.h>）。
    """

    def __init__(self, dim: int = 2):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x):
        # dim 语义按 torch 惯例（负数从末尾计数），必须指向既有维度
        ndim = x.dim()
        assert -ndim <= self.dim < ndim, "dim 需指向输入的既有维度"
        # 数值契约：float32 中累加，输出 cast 回输入 dtype
        y = torch.cumsum(x.to(torch.float32), dim=self.dim)
        return y.to(x.dtype)


def get_init_inputs():
    return [2]   # dim


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    b, c, h, w = 8, 16, 512, 128
    x = torch.randn(b, c, h, w)   # float32 主 dtype
    return [x]


def make_inputs(shape, dim, dtype="float32", seed=0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    shape 为输入形状列表（ndim 1-4），dim 为 scan 维（同时是构造 Model 的
    init 超参，此处仅做合法性校验），全部来自同一 torch.Generator(seed)，
    同 seed 下逐位可复现；消费顺序仅 x 一个张量（torch.randn 后 cast 到
    dtype）。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    ndim = len(shape)
    assert 1 <= ndim <= 4, "需 1 <= ndim <= 4"
    assert all(s >= 1 for s in shape), "各维长度需 >= 1"
    d = int(dim)
    assert -ndim <= d < ndim, "dim 需指向既有维度"
    L = shape[d % ndim]
    assert 1 <= L <= 65536, "scan 维长度需在 [1, 65536]"
    numel = 1
    for s in shape:
        numel *= s
    assert numel <= 2 ** 25, "总元素数需 <= 2^25"
    x = torch.randn(*shape, generator=gen).to(dt)
    return (x,)
