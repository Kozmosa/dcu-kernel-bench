# 5008_conv_bias_act —— Conv+Bias+逐元素激活融合前向（relu / swish / prelu）
# 的 model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=5008, backend=cuda（HIP）。

import torch
import torch.nn as nn


class Model(nn.Module):
    """Conv + Bias + 逐元素激活融合前向家族：relu / swish / prelu 三变体。

    对输入 x (N, C, H, W) 与卷积核 weight (K, C, R, S) 做 2D cross-correlation
    前向（深度学习库 conv_fprop 惯例语义，非数学卷积的核翻转），加逐通道偏置
    bias (1, K, 1, 1)（沿 batch 与空间维广播），再施加逐元素激活。卷积超参
    三变体共享：stride=(1, 1)、dilation=(1, 1)、对称零 padding=(1, 1)（上/下/
    左/右各补 1，padding 值恒为零）。逐元素定义：
        z[b, k, i, j] = ( sum_{c, u, v} x[b, c, i+u-1, j+v-1] *
                          weight[k, c, u, v] ) + bias[0, k, 0, 0]
        out[b, k, i, j] = act(z[b, k, i, j])
    其中 i ∈ [0, H_out)、j ∈ [0, W_out)，H_out = H - R + 3、W_out = W - S + 3
    （stride/padding/dilation 固定后的标准卷积输出公式）。激活按 __init__ 参数
    variant 三选一（z 为加 bias 后的卷积结果）：
        "relu"  ：out = max(z, 0)
        "swish" ：out = z * sigmoid(z) = z / (1 + exp(-z))
        "prelu" ：out = z（z > 0 时），否则 slope * z（标量斜率，等价于
                  negative_slope=slope 的 leaky-ReLU；slope 是 forward 输入
                  张量而非模块参数）
    forward 返回单张量 out (N, K, H_out, W_out)，dtype 与 x 一致（卷积天然
    单输出，无多输出打包分段）。

    输入输出规格：
        x      float16 / bfloat16 / float32 (N, C, H, W)，contiguous，
               N/C/H/W >= 1
        weight 同 x dtype (K, C, R, S)，contiguous，K/R/S >= 1，输入通道维
               等于 C
        bias   同 x dtype (1, K, 1, 1)，contiguous（逐通道偏置，广播到全
               batch 与空间位置）
        slope  同 x dtype 的 0 维标量张量；仅 variant="prelu" 参与计算
               （缺省 0.01，可变）；variant="relu"/"swish" 时内容任意
               （垃圾数据），不得影响输出
        out    单张量、同 x dtype (N, K, H_out, W_out)
        输入为常规 randn 量级（无 inf/NaN）；评测器会把全部输入 cast 成
        fp32 传入（fp16/bf16 -> fp32 无损；本题输出 dtype 始终与传入 x 的
        dtype 一致）。

    实现约束（违规判负）：
        - 核心计算（含 padding 的索引/掩码展开与点积累加、bias 加法、逐元素
          激活）必须全部在提交文件内以 HIP kernel 完成并融合，禁止以任何
          形式的库卷积或 ATen GEMM（matmul/mm/bmm/addmm/baddbmm/einsum/
          tensordot/unfold/linear 等，含借 conv3d 深度维 unsqueeze、
          conv_transpose2d 翻转核改写等一步等价路由）替代。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 卷积点积累加、bias 加法与激活在 float32 中进行，输出单次 cast 回
          输入 dtype（不得以 fp16/bf16 低精度累加，也不得多次往返 cast）；
          padding 须在 kernel 内以边界掩码/索引处理，不得在 host 侧物化
          padded 副本代替 kernel 内处理。
        - forward 返回单张量 out（评测器按单输出比较 shape/dtype/allclose）。

    禁用列表：torch.nn.Conv2d / torch.nn.functional.conv2d / F.conv2d /
    torch.conv2d / aten.conv2d / aten.convolution /
    torch.nn.functional.conv3d / torch.conv3d / torch.nn.Conv3d /
    torch.nn.functional.conv_transpose2d / torch.conv_transpose2d /
    torch.nn.ConvTranspose2d /
    torch.nn.functional.unfold / torch.nn.Unfold / .unfold( /
    torch.matmul / .matmul( / torch.mm / torch.bmm / torch.addmm /
    torch.baddbmm / torch.einsum / torch.tensordot /
    torch.nn.functional.linear / torch.nn.Linear /
    miopen / hipdnn。
    目标硬件：海光 DCU（gfx936/gfx938），实现语言 HIP（经 hipcc 编译；在 CUDA_CU_SRC / CUDA_CPP_SRC 锚点区编写，设备侧必须 include <hip/hip_runtime.h>，本机无 <cuda_runtime.h>）。
    """

    # 卷积超参三变体共享（样例参数集）
    _STRIDE = (1, 1)
    _PADDING = (1, 1)   # 对称零 padding：上下/左右各补 1
    _DILATION = (1, 1)

    def __init__(self, variant: str = "relu"):
        super().__init__()
        assert variant in ("relu", "swish", "prelu"), \
            "variant 必须是 relu / swish / prelu 之一"
        self.variant = variant

    def forward(self, x, weight, bias, slope):
        # conv 在 float32 中累加（io dtype 的输入先无损升位），bias 以
        # (1, K, 1, 1) 广播相加
        z = torch.nn.functional.conv2d(
            x.to(torch.float32), weight.to(torch.float32), bias=None,
            stride=self._STRIDE, padding=self._PADDING, dilation=self._DILATION,
        )
        z = z + bias.to(torch.float32)

        # 激活三分支（用 if/elif 分流，避免多子句复合语句）
        if self.variant == "relu":
            z = torch.relu(z)
        elif self.variant == "swish":
            z = z * torch.sigmoid(z)
        else:   # prelu：z > 0 直通，否则 slope * z
            sf = slope.to(torch.float32)
            z = torch.where(z > 0, z, z * sf)

        # 输出单次 cast 回输入 dtype
        return z.to(x.dtype)


def get_init_inputs():
    return ["relu"]   # variant


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    n, c, h, w = 2, 16, 32, 32
    k, r, s = 16, 3, 3
    x = torch.randn(n, c, h, w).to(torch.float16)
    weight = torch.randn(k, c, r, s).to(torch.float16)
    bias = torch.randn(1, k, 1, 1).to(torch.float16)
    slope = torch.tensor(0.01).to(torch.float16)   # 0 维标量张量，不消费 RNG
    return [x, weight, bias, slope]


def make_inputs(n: int, c: int, h: int, w: int, k: int, r: int, s: int,
                variant: str = "relu", dtype: str = "float16",
                slope: float = 0.01, seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：x -> weight -> bias（均 torch.randn 后 cast 到 dtype），
    全部来自同一 torch.Generator(seed)，同 seed 下逐位可复现；slope 由 case
    字段直接物化为 0 维标量张量（cast 到 dtype），不消费 RNG。
    variant 同时是构造 Model 的 init 超参，此处校验其合法性、dtype 声明域
    与 shape 域（各维 >= 1 且输出维 >= 1，即 R <= H+2 且 S <= W+2）。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert dt in (torch.float16, torch.bfloat16, torch.float32), \
        "本题 dtype 域仅声明 float16 / bfloat16 / float32"
    assert variant in ("relu", "swish", "prelu"), \
        "variant 必须是 relu / swish / prelu 之一"
    assert min(n, c, h, w, k, r, s) >= 1, "N/C/H/W/K/R/S 均需 >= 1"
    assert r <= h + 2 and s <= w + 2, "卷积核超出 padding 后的输入，输出维为负"
    x = torch.randn(n, c, h, w, generator=gen).to(dt)
    weight = torch.randn(k, c, r, s, generator=gen).to(dt)
    bias = torch.randn(1, k, 1, 1, generator=gen).to(dt)
    slope_t = torch.tensor(float(slope)).to(dt)
    return x, weight, bias, slope_t
