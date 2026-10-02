# 2006_norm —— 残差融合 LayerNorm 前向（fused add LayerNorm）的
# model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=2, problem_id=2006, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """残差融合 LayerNorm（fused add LayerNorm）前向。

    输入 x 与 residual 逐元素相加得残差和 s（加法在 float32 中完成，随后
    把 s 舍入回输入 dtype 存储）；再对 s 的最后一维做 LayerNorm 并施加
    仿射变换（全部按 float32 求值）：
        mean_i  = (1/N) * sum_j s[i, j]
        var_i   = (1/N) * sum_j (s[i, j] - mean_i)^2      （有偏方差）
        rstd_i  = 1 / sqrt(var_i + eps)                   （eps 由 __init__ 给出）
        y[i, j] = (s[i, j] - mean_i) * rstd_i * weight[j] + bias[j]
    统计以舍入回输入 dtype 的 s 为基准；y 最后舍入回输入 dtype。
    边界行为：N 为任意正整数（含 1、非 2 次幂、超长行），掩码外的列不参与
    统计；var = 0 时由 eps 保证 rstd 有限；输入无 inf/NaN。

    输出为单张量 out (M, 2N)，dtype 与输入一致：
        out[:, :N] = y           LayerNorm 输出
        out[:, N:] = s           residual_out（舍入到输入 dtype 的残差和）

    输入输出规格：
        x        (M, N)  float16 / bfloat16 / float32，contiguous，M >= 1，N >= 1
        residual (M, N)  同 x
        weight   (N,)    同 x（逐通道缩放）
        bias     (N,)    同 x（逐通道偏移）
        （评测器会把输入 cast 成 fp32 传入，fp16/bf16 -> fp32 无损）

    实现约束（违规判负）：
        - 核心计算（残差加、mean/var 统计、归一化、仿射）必须在提交文件内
          以 Triton kernel 完成。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 中间累加用 float32；y 与 residual_out 均舍入回输入 dtype。

    禁用列表：torch.nn.functional.layer_norm / torch.layer_norm /
    torch.nn.LayerNorm。
    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x, residual, weight, bias):
        # 评测器把全部输入 cast 成 fp32；fp16/bf16 -> fp32 无损，统一在
        # float32 计算，输出 cast 回入口 dtype（在线为 fp32、离线为原 dtype）
        dt = x.dtype
        x = x.to(torch.float32)
        residual = residual.to(torch.float32)
        w = weight.to(torch.float32)
        b = bias.to(torch.float32)

        # ① 残差相加：float32 求和，存储值舍入回输入 dtype
        residual_out = (x + residual).to(dt)

        # ② LayerNorm 统计与归一化：以舍入后的 residual_out 为基准（与
        #    kernel 先写回再重读的语义一致），全程 float32
        z = residual_out.to(torch.float32)
        mean = z.mean(dim=-1, keepdim=True)
        var = (z - mean).pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + self.eps)
        y = ((z - mean) * rstd) * w + b
        y = y.to(dt)

        # 双输出拼接为单张量：前 N 列 y、后 N 列 residual_out
        return torch.cat([y, residual_out], dim=-1)


def get_init_inputs():
    return [1e-5]  # eps


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性。消费顺序：x / residual /
    # weight / bias（randn / randn / rand / rand）
    m, n = 128, 1024
    x = torch.randn(m, n).to(torch.float16)
    residual = torch.randn(m, n).to(torch.float16)
    weight = torch.rand(n).to(torch.float16)
    bias = torch.rand(n).to(torch.float16)
    return [x, residual, weight, bias]


def make_inputs(m: int, n: int, dtype: str = "float16", seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：x（randn）、residual（randn）、weight（rand）、bias（rand），
    全部来自同一 torch.Generator(seed)，同 seed 下逐位可复现。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert m >= 1 and n >= 1, "需 M >= 1 且 N >= 1"
    x = torch.randn(m, n, generator=gen).to(dt)
    residual = torch.randn(m, n, generator=gen).to(dt)
    weight = torch.rand(n, generator=gen).to(dt)
    bias = torch.rand(n, generator=gen).to(dt)
    return (x, residual, weight, bias)
