# 4004_batched_gemm_bf16 — batch 维 bf16×bf16 GEMM（可选 bias）的 model_class
# （KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=4004, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """batch 维 bf16 GEMM：对 batch 内每个矩阵独立计算 x[b] @ weight[b]^T。

    数学定义（对 batch 内每个 b 独立，B/M/N/K >= 1）：
      acc[b, m, n] = sum_k x[b, m, k] * weight[b, n, k]    （float32 累加）
      out[b, m, n] = acc[b, m, n] + bias[b, 0, n]          （可选 bias）
    weight 以行主 (B, N, K) 存放，计算等价于转置 GEMM（每个输出通道对应
    weight 的一行）。bias 融合顺序遵循 GEMM kernel 惯例：float32 累加结果
    先 cast 到输入 dtype（bfloat16），再加 bias（bfloat16 算术），最后 cast
    到输出 dtype。无 bias 时直接 cast。边界行为：B/M/N/K 任意 >= 1（含
    1×1×1 最小规模），维度允许非 2 次幂；K 不整除 tile 大小时由 kernel 内
    掩码补零处理尾部，数学结果不变。

    输入输出规格：
      x      bfloat16  (B, M, K)  激活矩阵
      weight bfloat16  (B, N, K)  权重矩阵（行主，转置 GEMM 语义）
      bias   bfloat16  (B, 1, N)  可选；use_bias=False 时传入占位张量并被忽略
      out    输出 dtype (B, M, N) dtype 由 __init__ 的 out_dtype 指定
                            （bfloat16 / float16）

    实现约束（违规判负）：
      - 核心计算（bf16 GEMM 的乘累加与 bias 融合）必须在提交文件内完成，
        禁止调用 torch.matmul / torch.bmm / torch.mm / torch.addmm /
        torch.baddbmm / torch.einsum / F.linear / functional.linear /
        aiter。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；bias 分支按上述融合顺序；输出 cast 到 out_dtype。
      - 评测器会把全部输入 cast 成 fp32 后传入 forward；输入值（小整数码值
        与 bfloat16 可表示值）在 fp32 中均精确表示，入口处
        .to(torch.bfloat16) 无损恢复。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：TN 布局（x 行主 (B,M,K)、weight 行主 (B,N,K)）、无 splitK、
    无任何量化成分（纯 bf16 输入、无 scale 反量化）。
    """

    def __init__(self, use_bias: bool = True, out_dtype: str = "bfloat16"):
        super().__init__()
        self.use_bias = bool(use_bias)
        self.out_dtype = getattr(torch, out_dtype)

    def forward(self, x, weight, bias):
        # 评测器把全部输入 cast 成 fp32；bfloat16 值域 fp32 精确表示，无损恢复
        x = x.to(torch.bfloat16)
        weight = weight.to(torch.bfloat16)
        # 单行条件（forward 语句需可被逐条提取重排，避免 if/else 复合块）
        bias = bias.to(torch.bfloat16) if self.use_bias else None

        B, M, K = x.shape
        N = weight.shape[1]
        assert K == weight.shape[2], "x 与 weight 的 K 维必须一致"

        # bf16 提升到 float32 后做 GEMM，累加全程 float32
        acc = torch.matmul(x.to(torch.float32), weight.to(torch.float32).transpose(1, 2))

        if bias is not None:
            # 与 GEMM kernel 的融合顺序一致：先 cast 到输入 dtype 再加 bias
            acc = acc.to(x.dtype) + bias
        return acc.to(self.out_dtype)


def get_init_inputs():
    return [True, "bfloat16"]   # use_bias；out_dtype


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性。值域量级与生成器
    # make_inputs 保持一致（randint(-20,20)、rand*10）。
    b, m, n, k = 8, 128, 256, 512

    x = torch.randint(-20, 20, (b, m, k), dtype=torch.bfloat16)
    weight = torch.randint(-20, 20, (b, n, k), dtype=torch.bfloat16)
    bias = torch.rand(b, 1, n, dtype=torch.bfloat16) * 10
    return [x, weight, bias]


def make_inputs(b: int, m: int, n: int, k: int, use_bias: bool = True,
                out_dtype: str = "bfloat16", seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：x → weight → bias，全部来自同一 torch.Generator(seed)，
    同 seed 下逐位可复现。out_dtype 仅描述输出 dtype（Model 构造参数），
    不参与输入生成——输入恒为 bfloat16。
    """
    gen = torch.Generator().manual_seed(seed)

    x = torch.randint(-20, 20, (b, m, k), generator=gen, dtype=torch.bfloat16)
    weight = torch.randint(-20, 20, (b, n, k), generator=gen, dtype=torch.bfloat16)
    bias = torch.rand(b, 1, n, generator=gen, dtype=torch.bfloat16) * 10
    return x, weight, bias
