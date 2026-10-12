# 5001_conv2d_fwd —— 标准 NCHW 2D 卷积前向（对称 / 非对称 padding）的
# model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=5001, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """标准 NCHW 2D 卷积前向家族：对称 padding（standard）与非对称 padding
    （asymmetric）。

    对输入 x (N, C, H, W) 与卷积核 weight (K, C, R, S) 做 2D cross-correlation
    前向（深度学习库 conv_fprop 惯例语义，非数学卷积的核翻转）。逐元素定义：
        out[b, k, i, j] = sum_{c, u, v}
            x[b, c, i*stride_h + u*dil_h - pad_top, j*stride_w + v*dil_w -
              pad_left] * weight[k, c, u, v]
    越界索引处取零（零 padding）。无 bias；stride=(1, 1)、dilation=(1, 1)
    恒定。__init__ 参数 variant 选择 padding 方式：
        "standard"   ：对称零 padding (pad_h, pad_w) = (1, 1)（上下/左右
                       各补 1）；
        "asymmetric" ：非对称零 padding，pre_padding = (2, 2)（补在高的
                       上侧、宽的左侧），post_padding = (1, 1)（补在高的
                       下侧、宽的右侧）。
    输出尺寸：H_out = floor((H + pad_pre_h + pad_post_h - (R-1)*dil_h - 1)
    / stride_h) + 1（W_out 同理），代入本题固定参数即：
        standard   ：H_out = H - R + 3（W_out = W - S + 3）
        asymmetric ：H_out = H - R + 4（W_out = W - S + 4）
    forward 返回单张量 out (N, K, H_out, W_out)，float32（卷积天然单输出，
    无多输出打包分段）。

    输入输出规格：
        x      float32 (N, C, H, W)，contiguous，N/C/H/W >= 1
        weight float32 (K, C, R, S)，contiguous，K/R/S >= 1，输入通道维
               等于 C
        out    单张量 float32 (N, K, H_out, W_out)
        输入为常规 randn 量级（无 inf/NaN）；评测器会把全部输入 cast 成
        fp32 传入（本题输入本就是 fp32，cast 无损）。

    实现约束（违规判负）：
        - 核心计算（含 padding 的索引/掩码展开与点积累加）必须在提交文件
          内以 Triton kernel 完成，禁止以任何形式的库卷积或 ATen GEMM
          （matmul/mm/bmm/addmm/baddbmm/einsum/tensordot/unfold/linear 等，
          含借 conv3d 深度维 unsqueeze、conv_transpose2d 翻转核改写等一步
          等价路由）替代。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 全程 float32 计算；padding 值恒为零，须在 kernel 内以边界掩码/
          索引处理，不得在 host 侧物化 padded 副本代替 kernel 内处理。
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
    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    """

    # 本题固定卷积超参（variant 间共享，样例参数集）：
    _STRIDE = (1, 1)
    _DILATION = (1, 1)
    _PAD_SYMMETRIC = (1, 1)                 # standard：对称 padding
    _PRE_PADDING = (2, 2)                   # asymmetric：上/左侧
    _POST_PADDING = (1, 1)                  # asymmetric：下/右侧

    def __init__(self, variant: str = "standard"):
        super().__init__()
        assert variant in ("standard", "asymmetric"), "variant 必须是 standard / asymmetric 之一"
        self.variant = variant

    def forward(self, x, weight):
        # variant="asymmetric"：pre=(2,2)/post=(1,1) 是 (高, 宽) 顺序，F.pad 的
        # 顺序是 (宽左, 宽右, 高上, 高下)，两维取值两两相同，即 (2, 1, 2, 1)；
        # variant="standard"：对称 padding 直接交给 conv（用条件表达式分流，
        # 避免多子句复合语句）
        if self.variant == "asymmetric":
            xp = torch.nn.functional.pad(x, (2, 1, 2, 1))
            return torch.nn.functional.conv2d(
                xp, weight, bias=None, stride=self._STRIDE,
                padding=(0, 0), dilation=self._DILATION,
            )
        return torch.nn.functional.conv2d(
            x, weight, bias=None, stride=self._STRIDE,
            padding=self._PAD_SYMMETRIC, dilation=self._DILATION,
        )


def get_init_inputs():
    return ["standard"]   # variant


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    n, c, h, w = 4, 16, 56, 56
    k, r, s = 4, 3, 3
    x = torch.randn(n, c, h, w)
    weight = torch.randn(k, c, r, s)
    return [x, weight]


def make_inputs(n: int, c: int, h: int, w: int, k: int, r: int, s: int,
                variant: str = "standard", dtype: str = "float32",
                seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：x -> weight（均 torch.randn，float32），全部来自同一
    torch.Generator(seed)，同 seed 下逐位可复现。
    variant 同时是构造 Model 的 init 超参，此处校验其合法性与 shape 域
    （各维 >= 1 且输出维 >= 1）；dtype 本题声明域仅 float32。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert dt == torch.float32, "本题 dtype 域仅声明 float32"
    assert variant in ("standard", "asymmetric"), "variant 必须是 standard / asymmetric 之一"
    assert min(n, c, h, w, k, r, s) >= 1, "N/C/H/W/K/R/S 均需 >= 1"
    pad_pre_h, pad_post_h, pad_pre_w, pad_post_w = (
        (1, 1, 1, 1) if variant == "standard" else (2, 1, 2, 1)
    )
    h_out = (h + pad_pre_h + pad_post_h - (r - 1) - 1) // 1 + 1
    w_out = (w + pad_pre_w + pad_post_w - (s - 1) - 1) // 1 + 1
    assert h_out >= 1 and w_out >= 1, "卷积核超出 padding 后的输入，输出维为负"
    x = torch.randn(n, c, h, w, generator=gen)
    weight = torch.randn(k, c, r, s, generator=gen)
    return x, weight
