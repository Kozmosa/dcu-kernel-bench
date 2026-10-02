# 4009_gemm_a8w8 — 2D 8bit 量化 GEMM（int8 / fp8 激活 × 权重，per-token ×
# per-channel 标量 scale 反量化，可选 bias）的 model_class（KernelBench 兼容）
# 题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=4009, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """8bit 量化 GEMM：int8 / fp8 激活 × 同 dtype 权重，外乘 scale 反量化。

    数学定义：给定量化激活 x (M, K)、量化权重 w (N, K)（行主存放，输出通道
    为行，等价转置 GEMM）、per-token scale x_scale (M, 1) float32、
    per-channel scale w_scale (1, N) float32、可选 bias (1, N) float32：
      acc[m, n] = sum_k x[m, k] * w[n, k]                （float32 累加）
      out[m, n] = acc[m, n] * x_scale[m, 0] * w_scale[0, n]（外乘反量化）
      out[m, n] = out[m, n] + bias[0, n]                 （use_bias=True 时，
                                                    反量化结果在 float32 域相加）
    最后 cast 到 out_dtype 输出。数值由量化码字与 scale 共同决定：
    反量化元素 = 码字 × (x_scale 与 w_scale 外积) 对应位置乘积。

    输入输出规格：
      x        in_dtype   (M, K)  行主激活码字；in_dtype ∈ {int8, fp8e4m3,
                          fp8e5m2}（e4m3 = 4bit 指数 3bit 尾数，e5m2 = 5bit
                          指数 2bit 尾数），由 __init__ 的 in_dtype 指定
      w        in_dtype   (N, K)  行主权重码字（输出通道为行）
      x_scale  float32    (M, 1)  per-token scale
      w_scale  float32    (1, N)  per-channel scale
      bias     float32    (1, N)  use_bias=False 时为占位张量，被忽略
      out      out_dtype  (M, N)  out_dtype ∈ {bfloat16, float16}；M/N/K >= 1，
                          允许非 2 次幂与 K 不整除 tile（尾部由掩码补零）

    实现约束（违规判负）：
      - 核心计算（8bit GEMM 累加与反量化）必须在提交文件内完成，禁止调用
        torch.matmul / torch.bmm / torch.mm / torch.addmm / torch.einsum /
        torch._int_mm / torch._scaled_mm / F.linear / functional.linear。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；反量化与加 bias 在 float32 域完成后 cast 到
        out_dtype。
      - 评测器会把全部输入 cast 成 fp32 后传入 forward：int8 / fp8 码字与
        scale 均可被 fp32 精确表示，入口处 .to(in_dtype) 无损恢复量化码字。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：2D GEMM（无 batch 维）、x 行主 (M,K) / w 行主 (N,K)、标量
    （per-token × per-channel）scale；不含 128×128 分块 scale、batch 维与
    输出缓冲预分配变体。
    """

    def __init__(self, in_dtype="fp8e4m3", out_dtype="bfloat16", use_bias=True):
        super().__init__()
        quant = {"fp8e4m3": torch.float8_e4m3fn,
                 "fp8e5m2": torch.float8_e5m2,
                 "int8": torch.int8}
        out = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        self.in_dtype = quant[in_dtype]
        self.out_dtype = out[out_dtype]
        self.use_bias = bool(use_bias)

    def forward(self, x, w, x_scale, w_scale, bias):
        # 评测器把全部输入 cast 成 fp32；量化码字无损恢复回 in_dtype，
        # scale 恢复回 float32
        x = x.to(self.in_dtype)
        w = w.to(self.in_dtype)
        x_scale = x_scale.to(torch.float32)
        w_scale = w_scale.to(torch.float32)

        M, K = x.shape
        N = w.shape[0]
        assert K == w.shape[1], "x 与 w 的 K 维必须一致"

        # 码字提升 float32 后做 GEMM（x @ w^T），累加全程 float32
        acc = torch.matmul(x.to(torch.float32), w.to(torch.float32).t())

        # (M,1) x (1,N) 外积为 (M,N) 反量化矩阵后相乘
        out = acc * torch.matmul(x_scale, w_scale)

        if self.use_bias:
            out = out.to(torch.float32) + bias.to(torch.float32)
        return out.to(self.out_dtype)


def get_init_inputs():
    return ["fp8e4m3", "bfloat16", True]   # in_dtype；out_dtype；use_bias


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性。量化方式与 make_inputs
    # 一致：行 absmax 对齐码字值域上限后 cast。
    m, n, k = 256, 512, 1024
    qmax = torch.finfo(torch.float8_e4m3fn).max

    x = torch.randn((m, k), dtype=torch.float32)
    w = torch.randn((n, k), dtype=torch.float32)
    x_scale = x.abs().amax(dim=1, keepdim=True) / qmax
    x = (x / x_scale).to(torch.float8_e4m3fn)
    w_scale = w.abs().amax(dim=1, keepdim=True).T.contiguous() / qmax
    w = (w / w_scale.T).to(torch.float8_e4m3fn)
    bias = torch.rand([1, n], dtype=torch.float32) * 10
    return [x, w, x_scale, w_scale, bias]


def make_inputs(m: int, n: int, k: int, in_dtype: str = "fp8e4m3",
                out_dtype: str = "bfloat16", use_bias: bool = True,
                seed: int = 0):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器
    搬运到设备）。量化路径与官方测试生成器一致：行 absmax 缩放到码字值域
    上限后 cast 到 in_dtype。消费顺序固定：x → w → bias，全部来自同一
    torch.Generator(seed)，同 seed 下逐位可复现。use_bias 只控制 Model 的
    bias 生效开关，bias 张量始终生成（use_bias=False 时被 forward 忽略）。
    """
    gen = torch.Generator().manual_seed(seed)
    quant = {"fp8e4m3": torch.float8_e4m3fn, "fp8e5m2": torch.float8_e5m2,
             "int8": torch.int8}
    assert out_dtype in ("bfloat16", "float16")
    dt = quant[in_dtype]
    qmax = (torch.finfo(dt) if dt.is_floating_point else torch.iinfo(dt)).max

    x = torch.randn((m, k), generator=gen, dtype=torch.float32)
    w = torch.randn((n, k), generator=gen, dtype=torch.float32)
    x_scale = x.abs().amax(dim=1, keepdim=True) / qmax
    x = (x / x_scale).to(dt)
    w_scale = w.abs().amax(dim=1, keepdim=True).T.contiguous() / qmax
    w = (w / w_scale.T).to(dt)
    bias = torch.rand([1, n], generator=gen, dtype=torch.float32) * 10
    return x, w, x_scale, w_scale, bias
