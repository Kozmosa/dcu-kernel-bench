# 4001_batched_gemm_a8w8 — batch 维 int8 量化 GEMM（per-token × per-channel
# scale 反量化，可选 bias）的 model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=4001, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """batch 维 int8 量化 GEMM：int8 激活 × int8 权重，外乘 scale 反量化。

    对 batch 内每个矩阵独立计算（数学定义）：
      acc[b, m, n] = sum_k xq[b, m, k] * wq[b, n, k]          （float32 累加）
      out[b, m, n] = acc[b, m, n] * xs[b, m, 0] * ws[b, 0, n] （+ bias[b, 0, n]）

    即 out[b] = xq[b] @ wq[b]^T 后做反量化。wq 以行主 (B, N, K) 存放，计算
    等价于转置 GEMM（权重 n 行为一列输出通道）。xs 是 per-token scale：M 维
    每行一个，(B, M, 1) float32；ws 是 per-channel scale：N 维每列一个，
    (B, 1, N) float32；两者外乘为 (B, M, N) 反量化矩阵。bias (B, 1, N) 可选，
    与输出同 dtype；有 bias 时遵循量化 GEMM kernel 的融合顺序：先把 float32
    的 scale 结果 cast 到输出 dtype，再加 bias。

    输入输出规格：
      xq      int8      (B, M, K)   激活码字
      wq      int8      (B, N, K)   权重码字（行主，等价转置 GEMM）
      xs      float32   (B, M, 1)   per-token scale
      ws      float32   (B, 1, N)   per-channel scale
      bias    输出 dtype (B, 1, N)  可选；use_bias=False 时传入占位张量并被忽略
      out     输出 dtype (B, M, N)  dtype 由 __init__ 的 out_dtype 指定
                              （bfloat16 / float16），B/M/N/K >= 1

    实现约束（违规判负）：
      - 核心计算（int8 GEMM 累加）必须在提交文件内完成，禁止调用
        torch.matmul / torch.bmm / torch.mm / torch.addmm / torch.einsum /
        torch._int_mm / F.linear / functional.linear，禁止调用任何现成
        算子库/闭源 kernel 封装直接完成本计算。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 到 out_dtype。
      - 评测器会把全部输入 cast 成 fp32 后传入 forward；码字是 [-20, 20)
        内的小整数、bias 值来自低精度量化，fp32 均精确表示，入口处
        .to(torch.int8) / .to(out_dtype) 无损恢复。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：TN 布局（x 行主 (B,M,K)、w 行主 (B,N,K)）、无 splitK；K 不整
    除 tile 大小时由 kernel 内掩码处理尾部。K 维逐块累加时以 float32 保持
    精度，与参考实现的 fp32 GEMM 结果在容差内一致。
    """

    def __init__(self, use_bias: bool = True, out_dtype: str = "bfloat16"):
        super().__init__()
        self.use_bias = bool(use_bias)
        self.out_dtype = getattr(torch, out_dtype)

    def forward(self, xq, wq, x_scale, w_scale, bias):
        # 注意：forward 顶层保持单行语句（含三元表达式）。框架 loader 逐条
        # 提取顶层语句生成 ModelNew scaffold，多行 if/else 块会破坏 scaffold
        # 缩进（loader 对语句段统一加 8 空格，块内相对缩进会错位）。
        xq = xq.to(torch.int8)
        wq = wq.to(torch.int8)
        bias = bias.to(self.out_dtype) if self.use_bias else None

        B, M, K = xq.shape
        N = wq.shape[1]
        assert K == wq.shape[2], "xq 与 wq 的 K 维必须一致"

        # int8 码字提升 float32 后做 GEMM，累加全程 float32
        a = xq.to(torch.float32)
        b = wq.to(torch.float32)
        acc = torch.matmul(a, b.transpose(1, 2))   # (B, M, N) float32

        # per-token 与 per-channel scale 外乘反量化（广播）
        acc = acc * x_scale                        # (B, M, 1) 广播到 M 维
        acc = acc * w_scale                        # (B, 1, N) 广播到 N 维

        # 融合顺序与量化 GEMM kernel 一致：先 cast 到输出 dtype 再加 bias
        acc = acc.to(self.out_dtype) if self.use_bias else acc
        acc = acc + bias if self.use_bias else acc
        return acc.to(self.out_dtype)


def get_init_inputs():
    return [True, "bfloat16"]   # use_bias；out_dtype


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性。码字值域与 scale/bias 量级
    # 同生成器 make_inputs 保持一致。
    b, m, n, k = 4, 128, 256, 512

    xq = torch.randint(-20, 20, (b, m, k), dtype=torch.int8)
    wq = torch.randint(-20, 20, (b, n, k), dtype=torch.int8)
    x_scale = torch.rand(b, m, 1, dtype=torch.float32) + 1e-6
    w_scale = torch.rand(b, 1, n, dtype=torch.float32) + 1e-6
    bias = (torch.rand(b, 1, n, dtype=torch.float32) * 10).to(torch.bfloat16)
    return [xq, wq, x_scale, w_scale, bias]


def make_inputs(b: int, m: int, n: int, k: int, use_bias: bool = True,
                out_dtype: str = "bfloat16", seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：xq → wq → x_scale → w_scale → bias，全部来自同一
    torch.Generator(seed)，同 seed 下逐位可复现。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, out_dtype)

    xq = torch.randint(-20, 20, (b, m, k), generator=gen, dtype=torch.int8)
    wq = torch.randint(-20, 20, (b, n, k), generator=gen, dtype=torch.int8)
    x_scale = torch.rand(b, m, 1, generator=gen, dtype=torch.float32) + 1e-6
    w_scale = torch.rand(b, 1, n, generator=gen, dtype=torch.float32) + 1e-6
    bias = (torch.rand(b, 1, n, generator=gen, dtype=torch.float32) * 10).to(dt)
    return xq, wq, x_scale, w_scale, bias
