# 4005_fused_mxfp4_quant — 残差 + RMSNorm + MXFP4 动态量化融合算子
# （为下游 mxfp4 GEMM 供数）的 model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=4005, backend=triton。

import torch
import torch.nn as nn


def _rmsnorm(x, weight, eps):
    # 官方语义：float32 中按行做 RMS 归一化后乘权重
    row_norm = torch.sum(x * x, dim=-1)
    norm_factor = torch.rsqrt((row_norm / x.shape[1]) + eps).reshape(-1, 1)
    return x * norm_factor * weight.reshape(1, -1)


def _mxfp4_quant(x):
    # MXFP4 动态量化（块大小 32，even 圆整）。逐位语义见 Model 的文档字符串：
    # 位级算法与参考实现一致，输入为 float32 的 (M, N) 矩阵。
    MXFP4_QUANT_BLOCK_SIZE = 32
    x_shape = x.shape
    if x.shape[-1] % MXFP4_QUANT_BLOCK_SIZE != 0:
        shape = list(x_shape)
        shape = shape[:-1] + [
            ((shape[-1] - 1 + MXFP4_QUANT_BLOCK_SIZE) // MXFP4_QUANT_BLOCK_SIZE)
            * MXFP4_QUANT_BLOCK_SIZE
        ]
        shape = tuple(shape)
        x_padded = torch.zeros(shape, dtype=x.dtype)
        x_padded[..., : x.shape[-1]] = x
    else:
        x_padded = x

    # 块 scale：amax 向上圆整到 2 的幂，取 floor(log2)-2
    x_padded = x_padded.reshape(
        -1, x_padded.shape[-1] // MXFP4_QUANT_BLOCK_SIZE, MXFP4_QUANT_BLOCK_SIZE
    ).to(torch.float32)
    amax, _ = torch.max(torch.abs(x_padded), dim=-1)
    amax = amax.view(torch.int32)
    amax = (amax + 0x200000) & 0xFF800000
    amax = amax.view(torch.float32)
    scale_e8m0_unbiased = torch.log2(amax).floor() - 2
    scale_e8m0_unbiased = torch.clamp(scale_e8m0_unbiased, min=-127, max=127)
    quant_scale = torch.exp2(-scale_e8m0_unbiased)

    # 量化后的值
    qx = x_padded * quant_scale.unsqueeze(-1)

    # blockscale_e8m0
    bs_e8m0 = scale_e8m0_unbiased.to(torch.uint8) + 127

    # e2m1 舍入：round-half-up，超出饱和（对 (E<<2 | M>>21) + 1 >> 1 取 min(·,7)）
    qx = qx.view(torch.int32)
    s = qx & 0x80000000
    e = (qx >> 23) & 0xFF
    m = qx & 0x7FFFFF

    E8_BIAS = 127
    E2_BIAS = 1
    adjusted_exponents = E8_BIAS - e - 1
    m = torch.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)
    e = torch.where(e > E8_BIAS - E2_BIAS, e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

    combined_val = (((e << 2) | (m >> 21)) + 1) >> 1
    e2m1_tmp = torch.where(combined_val < 0x7, combined_val, 0x7)
    e2m1_value = (((s >> 28) & 0xF) | e2m1_tmp).to(torch.uint8)

    # 每字节打包 2 个 4-bit 码字：偶数下标在低 4 位
    x_mxfp4 = e2m1_value[..., ::2] | (e2m1_value[..., 1::2] << 4)
    x_mxfp4 = torch.flatten(x_mxfp4, -2, -1)

    # 去掉补零产生的多余字节
    if x.shape[-1] % MXFP4_QUANT_BLOCK_SIZE != 0:
        x_mxfp4 = x_mxfp4[..., : x.shape[-1] // 2]

    mxfp4_shape = list(x_shape)
    mxfp4_shape = tuple(mxfp4_shape[:-1] + [mxfp4_shape[-1] // 2])
    x_mxfp4 = x_mxfp4.reshape(mxfp4_shape)
    return x_mxfp4, bs_e8m0


class Model(nn.Module):
    """残差相加 + RMSNorm + MXFP4 动态量化的融合算子（mxfp4 GEMM 前处理）。

    数学定义（全部中间量 float32；输入为 bfloat16/float16 值域的 randn）：
      1. s = inp1 + res1                                        逐元素相加
      2. norm[m, j] = s[m, j] * rsqrt( (1/N1) * Σ_j s[m, j]² + eps ) * weight1[j]
         eps = 1e-6，沿最后一维归一化
      3. 对 norm 每一行按 32 元素分块做 MXFP4 动态量化（定义见下），得到
         packed e2m1 码字（每字节 2 个）与 e8m0 块 scale 字节。

    MXFP4 量化（量化块大小固定 32，沿最后一维；行尾不足 32 的块补零参与）：
      a. amax = max |x|（块内，含补零）
      b. amax 向上圆整到 2 的幂：把 amax 的 IEEE754 位型按 int32 解释，
         加 0x200000 后与 0xFF800000 按位与，再按 float32 解释
         （amax 已是 2 的幂时不变，否则进到下一个 2 的幂）
      c. e = clamp( floor(log2(amax 圆整)) - 2, -127, 127 )
      d. scale 字节 = e + 127（uint8 的 e8m0 编码）
      e. q = x * 2^(-e)，舍入到 e2m1 码表 {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}：
         round-half-up（0.5 向上），超出 ±6 饱和到 ±6；等价位级实现为对
         fp32 位型的 (E<<2 | M>>21) + 1 >> 1 再取 min(·, 7)，符号位取 bit31
      f. 打包：第 2i 与第 2i+1 个码字合入同一字节，偶数下标在低 4 位

    输入输出规格：
      inp1    (M, N1)  bfloat16/float16  主输入
      weight1 (N1,)    同 inp1           RMSNorm 权重
      res1    (M, N1)  同 inp1           残差
      out     (M, N1//2 + ceil(N1/32)) uint8，单张量输出契约：
              out[:, :N1//2]        packed e2m1 码字（每字节 2 个，低 4 位偶下标）
              out[:, N1//2:]        e8m0 块 scale 字节（每行 ceil(N1/32) 个）
      约束：M ≥ 1；N1 为偶数且 N1 ≥ 2。
      边界行为：全零块 amax=0，按 log2(0)=-inf 处理，clamp 后 e=-127，
      scale 字节为 0，码字为 0。

    实现约束（违规判负）：
      - RMSNorm 与量化（amax/scale 计算、e2m1 舍入、打包）的核心计算必须在
        提交文件内完成，禁止调用 torch.nn.functional.rms_norm /
        torch.nn.RMSNorm，禁止使用 torch 内置 float4/mxfp4 量化或转换 API
        （含 torch.float4_e2m1fn_x2、torch.quantize_per_tensor 等现成量化
        算子），禁止改包装任何第三方算子库的现成实现。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间计算用 float32；输出为 uint8，码字与 scale 字节都必须与参考实现
        逐位一致（精确匹配，不容差）。
      - 评测器把全部输入 cast 成 fp32 后传入 forward（bfloat16/float16 的值
        在 fp32 中精确表示），直接以其值计算即可。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：单路残差 + RMSNorm + MXFP4 动态量化（块大小 32、even 圆整），
    不含下游 GEMM 主计算。
    """

    def __init__(self, dtype: str = "bfloat16", eps: float = 1e-6):
        super().__init__()
        self.dtype = getattr(torch, dtype)
        self.eps = float(eps)

    def forward(self, inp1, weight1, res1):
        # 输入既可能是原始 bfloat16/float16，也可能是评测器 cast 后的 fp32
        #（值无损），统一提升 float32 计算
        s = inp1.to(torch.float32) + res1.to(torch.float32)
        norm = _rmsnorm(s, weight1.to(torch.float32), self.eps)
        codes, scales = _mxfp4_quant(norm)
        return torch.cat([codes, scales], dim=1)


def get_init_inputs():
    return ["bfloat16"]   # dtype；eps 缺省 1e-6


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性。量级与 make_inputs 一致。
    m, n1 = 64, 512
    inp1 = torch.randn(m, n1, dtype=torch.bfloat16)
    weight1 = torch.randn(n1, dtype=torch.bfloat16)
    res1 = torch.randn(m, n1, dtype=torch.bfloat16)
    return [inp1, weight1, res1]


def make_inputs(m: int, n1: int, dtype: str = "bfloat16", seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：inp1 → weight1 → res1，全部来自同一 torch.Generator(seed)，
    同 seed 下逐位可复现。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    inp1 = torch.randn(m, n1, generator=gen, dtype=dt)
    weight1 = torch.randn(n1, generator=gen, dtype=dt)
    res1 = torch.randn(m, n1, generator=gen, dtype=dt)
    return inp1, weight1, res1
