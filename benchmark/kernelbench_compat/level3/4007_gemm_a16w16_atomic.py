# 4007_gemm_a16w16_atomic — 无 bias 16 位浮点矩阵乘（split-K 原子归约变体）
# 的 model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，参考实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=4007, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """无 bias 的 16 位浮点矩阵乘（split-K 原子归约变体的语义核心）。

    算子语义：
      x 形状 [M, K]，weight 形状 [N, K]（线性层权重布局），输出 out 形状
      [M, N]：
        out[m, n] = sum_{k=0}^{K-1} x[m, k] * weight[n, k]
      等价于 torch.nn.functional.linear(x, weight, bias=None)。M、N、K 均为
      任意正整数（含 1 与非 2 次幂；K 不要求与任何分块大小对齐，K 方向越界
      的部分按 0 参与累加）。原算子沿 K 维分片：各分片独立做 float32 乘累
      加，分片部分和经原子加合并进同一输出张量，因此输出整体只经历一次
      dtype 收缩——float32 累加结果直接 cast 回输入 dtype，不存在逐分片的
      半精度中间舍入。本参考实现按同一语义在 float32 中一次性完成全部累加。

    输入输出规格：
      forward(x, weight) -> out
        x      [M, K]，float16 或 bfloat16，行主连续
        weight [N, K]，与 x 同 dtype，行主连续
        out    [M, N]，与 x 同 dtype
      __init__(in_features, out_features)：in_features 对应 K，
      out_features 对应 N，仅作实例元数据；forward 一律以运行期张量的实际
      形状为准，不与 init 参数做强一致检查。

    实现约束（违规判负）：
      - 核心计算（乘累加主循环）必须在提交文件内以 Triton kernel 完成，
        禁止调用 torch.matmul / torch.bmm / torch.mm / torch.addmm /
        torch.einsum / torch.nn.functional.linear / F.linear 等 ATen
        GEMM 捷径与现成算子库。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：行主 TN 布局、无 bias、单输出张量、float16 与 bfloat16 两种
    输入 dtype；split-K 的分片数与并行策略由实现自行决定，以输出容差判定。
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, weight):
        # KernelBench 评测器会把全部输入张量 cast 成评测 precision（triton
        # 后端为 fp32）；本题输入无整型张量，fp32 输入走同一 float32 累加
        # 路径即可，输出 dtype 跟随运行期输入
        x32 = x.to(torch.float32)
        w32 = weight.to(torch.float32)
        assert x32.shape[1] == w32.shape[1], "x 与 weight 的 K 维必须一致"
        # [M, K] @ [K, N] -> [M, N]，float32 一次性累加
        out = torch.mm(x32, w32.t())
        return out.to(x.dtype)


def get_init_inputs():
    return [4096, 1280]  # in_features(K)，out_features(N)


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    m = 64
    n = 1280
    k = 4096
    x = torch.randn(m, k).to(torch.float16)
    weight = torch.randn(n, k).to(torch.float16)
    return [x, weight]


def make_inputs(m: int, n: int, k: int, dtype: str = "float16", seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。"""
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    x = torch.randn(m, k, generator=gen).to(dt)
    weight = torch.randn(n, k, generator=gen).to(dt)
    return x, weight
