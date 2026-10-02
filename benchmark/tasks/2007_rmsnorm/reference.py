# 2007_rmsnorm —— RMSNorm 前向家族（标准 / 残差融合）的 model_class
# （KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=2, problem_id=2007, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """RMSNorm 前向家族：标准 RMSNorm 与残差融合（fused residual-add）RMSNorm。

    对 (M, N) 输入的每一行，归一化在 float32 中按如下定义计算：
        mean_sq = sum_i x_i^2 / N
        rsigma  = rsqrt(mean_sq + eps)
        y_i     = x_i * rsigma * w_i
    w 为逐通道缩放向量。__init__ 参数 variant 选择两条路径之一：
        "fused_add"：先在输入 dtype 下求 residual_out = x + residual（残差流
        原样直通输出），再对 residual_out 施加上述 RMSNorm；
        "rmsnorm"  ：直接对 x 施加 RMSNorm；residual 不参与计算（内容任意，
        可为垃圾数据）。
    forward 返回单张量 out（同 x dtype）：fused_add 时 (M, 2N)，前 N 列为
    y、后 N 列为 residual_out（两段逻辑输出沿最后一维列拼接）；rmsnorm 时
    (M, N)，即 y。
    边界行为：全零行 mean_sq = 0 时 rsigma = rsqrt(eps)（eps 保证不除零）；
    N 任意（含 1、非 2 的幂、非 16 对齐、超出单 block 归约上限的大 N）都必须
    用掩码 / 分块归约正确处理，行尾越界元素不得参与平方和。

    输入输出规格：
        x        float16 / bfloat16 (M, N)，contiguous，M >= 1，N >= 1
        weight   同 x dtype (N,)，contiguous
        residual 同 x dtype (M, N)，contiguous（variant="rmsnorm" 时忽略）
        out      单张量、同 x dtype：fused_add 时 (M, 2N)（前 N 列 y、后 N 列
                 residual_out），rmsnorm 时 (M, N)（即 y）
        评测器会把全部输入 cast 成 fp32 传入（fp16/bf16 -> fp32 无损）。

    实现约束（违规判负）：
        - 核心计算（残差加法、逐行平方和归约、rsqrt、逐元素缩放）必须在提交
          文件内以 Triton kernel 完成。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 平方和累加与归一化在 float32 中进行；y cast 回输入 dtype；
          residual_out 在输入 dtype 下相加并原样写出。
        - forward 返回单张量 out（评测器按单输出比较）；fused_add 的
          residual_out 列块须与 x + residual 在输入 dtype 下逐位一致。

    禁用列表：aiter / torch.rms_norm / torch.nn.functional.rms_norm /
    torch.nn.RMSNorm / torch.layer_norm / torch.nn.functional.layer_norm /
    torch.nn.LayerNorm / torch.norm / torch.linalg.norm /
    torch.linalg.vector_norm。
    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    """

    def __init__(self, variant: str = "fused_add", eps: float = 1e-5):
        super().__init__()
        assert variant in ("rmsnorm", "fused_add"), "variant 必须是 rmsnorm / fused_add 之一"
        self.variant = variant
        self.eps = float(eps)

    def forward(self, x, weight, residual):
        # variant="fused_add"：残差流在输入 dtype 下相加并原样直通输出；
        # variant="rmsnorm"：直接对 x 归一化，residual 不参与计算
        # （用条件表达式分流，避免多子句复合语句）
        fused = self.variant == "fused_add"
        norm_input = x + residual if fused else x
        residual_out = norm_input

        # RMSNorm 主体：float32 中间计算，输出 cast 回输入 dtype
        xf = norm_input.to(torch.float32)
        wf = weight.to(torch.float32)
        mean_sq = (xf * xf).mean(dim=-1, keepdim=True)
        y = (xf * torch.rsqrt(mean_sq + self.eps) * wf).to(x.dtype)

        # 单张量输出：fused_add 把 y 与 residual_out 沿最后一维列拼接
        # （前 N 列 y、后 N 列 residual_out）
        return torch.cat([y, residual_out], dim=-1) if fused else y


def get_init_inputs():
    return ["fused_add", 1e-5]   # variant, eps


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    m, n = 1024, 4096
    x = torch.randn(m, n).to(torch.float16)
    weight = torch.randn(n).to(torch.float16)
    residual = torch.randn(m, n).to(torch.float16)
    return [x, weight, residual]


def make_inputs(m: int, n: int, variant: str = "fused_add",
                dtype: str = "float16", seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：x -> weight -> residual（均 torch.randn 后 cast 到 dtype），
    全部来自同一 torch.Generator(seed)，同 seed 下逐位可复现。
    variant 同时是构造 Model 的 init 超参，此处仅做合法性校验。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert variant in ("rmsnorm", "fused_add"), "variant 必须是 rmsnorm / fused_add 之一"
    assert m >= 1 and n >= 1, "需 M >= 1 且 N >= 1"
    x = torch.randn(m, n, generator=gen).to(dt)
    weight = torch.randn(n, generator=gen).to(dt)
    residual = torch.randn(m, n, generator=gen).to(dt)
    return x, weight, residual
