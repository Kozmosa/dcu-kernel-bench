# 2001_activation —— 门控激活 × SwiGLU 分半乘 × 动态 MXFP4 量化融合算子的
# model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=2, problem_id=2001, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """门控激活 × SwiGLU 分半乘 × 动态 MXFP4 量化融合算子。

    输入 x 形状 (M, N)（N 为 4 的倍数）。沿最后一维对半劈开 a = x[:, :N/2]、
    b = x[:, N/2:]，先在 float32 中求门控积 out = act(a) * b，act 由 __init__
    参数 activation 指定（数学定义，全部按 float32 求值）：
        silu      : u * sigmoid(u)
        gelu      : 0.5 * u * (1 + erf(u * 0.7071067811865476))
        gelu_tanh : 0.5 * u * (1 + tanh(0.7978845608028654 * (u + 0.044715 * u^3)))

    随后把 out 沿最后一维切成连续 32 元素块，逐块动态量化为 MXFP4（全程
    float32 位运算）：
      1. N/2 不是 32 的倍数时，最后一个块右端补零到 32 个元素，补零参与 amax。
      2. amax = max|块|；对 amax 的 float32 位模式做 (bits + 0x200000) &
         0xFF800000（even_round：尾数清零、向上取整到 2 的幂网格）。
      3. 无偏 scale = clamp(floor(log2(amax)) - 2, -127, 127)；e8m0 字节 =
         无偏 scale + 127，落在 [0, 254]（含义 2^(字节 - 127)）。全零块
         （含补零段使 amax 为 0）经 log2(0) = -inf 钳到 -127，字节为 0。
      4. q = 块 * 2^(-无偏 scale)，逐元素舍入到 e2m1 格点
         {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}。位级公式：取 q 的 float32
         位段 s（符号位）/ e（偏置指数）/ m（23 位尾数），
             e' = max(e, 126) - 126
             m' = (e < 127) ? (0x400000 | (m >> 1)) >> (126 - e) : m （非正规折算）
             code = (((e' << 2) | (m' >> 21)) + 1) >> 1，再饱和到 0x7
             e2m1 码字（4 位）= (s ? 8 : 0) | code
      5. 相邻两个 e2m1 码字打包进一个 uint8：偶数下标占低 4 位，奇数下标占
         高 4 位。

    输出为单张 uint8 张量（两段沿最后一维拼接）：
        out (M, N/4 + ceil(N/64)) uint8
        前 N/4 列        ：MXFP4 打包码字（每字节两个 e2m1 值）
        后 ceil(N/64) 列 ：逐 32 元素块的 e8m0 scale 字节
    两段均为整数码字，要求精确匹配。

    输入输出规格：
        x    float16 / bfloat16 (M, N)，contiguous，N % 4 == 0，M >= 1
             （评测器会把输入 cast 成 fp32 传入，fp16/bf16 -> fp32 无损）
        out  uint8 (M, N/4 + ceil(N/64))

    实现约束（违规判负）：
        - 核心计算（激活、分半乘、逐块 scale 与 e2m1 量化、码字打包）必须在
          提交文件内以 Triton kernel 完成；输出为 uint8 码字，无浮点输出。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 激活、乘法与量化位运算一律在 float32 中进行，与本参考实现逐位一致
          （码字与 scale 字节精确相等，无容差）。
        - N/2 非 32 倍数时按上文补零语义处理块尾。

    禁用列表：torch.nn.functional.silu / torch.nn.functional.gelu /
    torch.nn.SiLU / torch.nn.GELU / torch.sigmoid / torch.erf / torch.tanh。
    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    """

    def __init__(self, activation: str = "silu"):
        super().__init__()
        assert activation in ("silu", "gelu", "gelu_tanh"), "activation 必须是 silu / gelu / gelu_tanh 之一"
        self.activation = activation

    def forward(self, x):
        # 评测器把全部输入 cast 成 fp32；fp16/bf16 -> fp32 无损，统一在 float32 计算
        x = x.to(torch.float32)
        M, N = x.shape
        assert N % 4 == 0, "N 必须是 4 的倍数"
        d = N // 2
        a, b = x.split([d, d], dim=-1)
        # 门控积 act(a) * b（三条互斥分支，__init__ 已断言 activation 合法）
        out = nn.functional.silu(a) * b
        if self.activation == "gelu":
            out = nn.functional.gelu(a) * b
        if self.activation == "gelu_tanh":
            out = nn.functional.gelu(a, approximate="tanh") * b

        # ---- 动态 MXFP4 量化：逐 32 元素块、e8m0 block scale，全程 float32 ----
        BLOCK = 32
        n_blocks = (d + BLOCK - 1) // BLOCK
        blocks = out
        if d % BLOCK != 0:
            # 块尾补零（补零参与 amax）
            padded = torch.zeros(M, n_blocks * BLOCK, dtype=torch.float32)
            padded[:, :d] = out
            blocks = padded
        blocks = blocks.reshape(M, n_blocks, BLOCK)

        # even_round：对 amax 的 float32 位模式 (bits + 0x200000) & 0xFF800000
        amax, _ = torch.max(torch.abs(blocks), dim=-1)
        amax = amax.view(torch.int32)
        amax = (amax + 0x200000) & 0xFF800000
        amax = amax.view(torch.float32)
        # e8m0 无偏 scale 与字节（字节 = 无偏 scale + 127，落在 [0, 254]）
        scale_unbiased = torch.clamp(torch.log2(amax).floor() - 2, min=-127, max=127)
        bs_e8m0 = (scale_unbiased + 127).to(torch.uint8)
        quant_scale = torch.exp2(-scale_unbiased)
        qx = blocks * quant_scale.unsqueeze(-1)

        # e2m1 舍入（round-to-nearest、饱和到 ±6），位级公式与题面一致
        qx = qx.contiguous().view(torch.int32)
        s = qx & 0x80000000
        e = (qx >> 23) & 0xFF
        m = qx & 0x7FFFFF
        adjusted = torch.clamp(127 - e - 1, min=0)   # e >= 127 的分支会被 where 丢弃
        m = torch.where(e < 127, (0x400000 | (m >> 1)) >> adjusted, m)
        e = torch.where(e > 126, e, 126) - 126
        combined = (((e << 2) | (m >> 21)) + 1) >> 1
        e2m1_tmp = torch.where(combined < 0x7, combined, 0x7)
        e2m1_value = (((s >> 28) & 0xF) | e2m1_tmp).to(torch.uint8)

        # 相邻码字打包：偶数下标低 4 位、奇数下标高 4 位
        x_fp4 = e2m1_value[..., ::2] | (e2m1_value[..., 1::2] << 4)
        x_fp4 = x_fp4.reshape(M, -1)[:, : d // 2]

        # 单张量输出：前 d//2 列 FP4 码字、后 n_blocks 列 e8m0 scale 字节
        return torch.cat([x_fp4, bs_e8m0], dim=-1)


def get_init_inputs():
    return ["silu"]   # activation


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    m, n = 128, 1024
    x = torch.randn(m, n).to(torch.float16)
    return [x]


def make_inputs(m: int, n: int, activation: str = "silu",
                dtype: str = "float16", seed: int = 0,
                name=None, seq_lens=None):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：x（torch.randn 后 cast 到 dtype），全部来自同一
    torch.Generator(seed)，同 seed 下逐位可复现。
    name 为 case 名字段直传兼容（忽略）；seq_lens 为通用终审工具的占位
    调用约定（本题输入无序列长度概念，忽略）。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert activation in ("silu", "gelu", "gelu_tanh"), "activation 必须是 silu / gelu / gelu_tanh 之一"
    assert m >= 1 and n % 4 == 0, "需 M >= 1 且 N % 4 == 0"
    x = torch.randn(m, n, generator=gen).to(dt)
    return (x,)
