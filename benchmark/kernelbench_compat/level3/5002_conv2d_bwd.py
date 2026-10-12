# 5002_conv2d_bwd —— 标准 2D 卷积反向（dgrad / wgrad）的 model_class
# （KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=5002, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """标准 2D 卷积反向：对输入求梯度（dgrad）与对权值求梯度（wgrad）。

    语义（NCHW、groups=1、无 bias，与 PyTorch 前向 conv2d 严格一致）：
        前向 y = conv2d(x, weight, stride=(stride_h,stride_w),
                        padding=(pad_h,pad_w), dilation=(dil_h,dil_w))，
        输出尺寸
            out_h = floor((H + 2*pad_h - dil_h*(R-1) - 1) / stride_h) + 1
        （out_w 同理）；给定上游梯度 dy = dL/dy（形状
        (N, out_c, out_h, out_w)）：
            dgrad：dx = dL/dx，形状 (N, C, H, W)
            wgrad：dw = dL/dw，形状 (out_c, C, R, S)
    __init__ 参数 variant 选择返回的方向：
        "dgrad"：返回 dx；此时 x 的数值不参与输出（仅形状约定）；
        "wgrad"：返回 dw；此时 weight 的数值不参与输出（仅形状约定）。
    forward(dy, weight, x) 为固定三输入签名，各张量含义：
        dy     (N, out_c, out_h, out_w)   上游梯度，两方向都消费
        weight (out_c, C, R, S)           卷积核（out_c 输出通道 / C 输入通道）
        x      (N, C, H, W)               前向输入
    forward 返回单张量 out——两方向均天然单输出，无拼接打包：
        dgrad → (N, C, H, W)；wgrad → (out_c, C, R, S)；dtype 同输入
        （本题为 float32）。

    输入输出规格：
        dy / weight / x 均为 float32、contiguous、NCHW 排布；各维 >= 1；
        R/S 允许 1（1x1 卷积）；H/W 允许 1、奇数、非 2 的幂、非 16 对齐；
        stride/dilation >= 1、padding >= 0，padding 为前向 conv 的四向对称
        (pad_h, pad_w) 语义（非对称 pre/post padding 不在本题域内）；一致性约束
            H + 2*pad_h >= dil_h*(R-1) + 1，W + 2*pad_w >= dil_w*(S-1) + 1，
        且 dy 的空间尺寸必须等于前向公式的 out_h/out_w。
        输入值域为常规 randn 量级（无 inf/NaN）。评测器会把全部输入 cast
        成 fp32 传入。

    实现约束（违规判负）：
        - 核心计算必须在提交文件内以 Triton kernel 完成：dgrad 的散射累加
          （dy 对卷积窗按 stride/dilation 反卷积累加）或 wgrad 的相关累加
          （x 窗口与 dy 的互相关求和），禁止回退到 torch 卷积或 autograd。
        - ModelNew 的 __init__ 与 forward 签名不可更改；forward 返回单张量
          out（评测器按单输出比较）。
        - 数值约定：float32 输入直接以 float32 累加，不做半精度下转。

    禁用列表：aiter / torch.nn.functional.conv2d / F.conv2d / torch.conv2d /
    torch.nn.Conv2d / torch.nn.functional.conv_transpose2d /
    F.conv_transpose2d / torch.nn.ConvTranspose2d /
    torch.nn.grad.conv2d_input / torch.nn.grad.conv2d_weight /
    aten.conv2d / aten.convolution / aten._convolution /
    aten.conv_transpose2d / aten.convolution_backward（含 torch.ops.aten.*、
    at::.* 等一切拼写） / torch.autograd / .backward( / unfold /
    F.fold / torch.nn.functional.fold / im2col / col2im / as_strided；
    闭源库 MIOpen / hipDNN / cuDNN / rocBLAS / hipBLASLt 同样禁止（全局黑名单）。
    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    """

    def __init__(self, variant: str = "dgrad", stride_h: int = 1, stride_w: int = 1, pad_h: int = 1, pad_w: int = 1, dil_h: int = 1, dil_w: int = 1):
        # 注意：__init__/forward 的 def 签名必须保持单行（框架 loader 按首行
        # 提取签名生成 ModelNew scaffold）。
        super().__init__()
        assert variant in ("dgrad", "wgrad"), "variant 必须是 dgrad / wgrad 之一"
        self.variant = variant
        self.stride_h, self.stride_w = int(stride_h), int(stride_w)
        self.pad_h, self.pad_w = int(pad_h), int(pad_w)
        self.dil_h, self.dil_w = int(dil_h), int(dil_w)

    def forward(self, dy, weight, x):
        # 对前向 conv2d 构图后用 autograd 反传：保证 dgrad / wgrad 与前向
        # 卷积语义严格一致（reference 不受禁用列表约束；数学上 dx 与 x 的
        # 数值无关、dw 与 weight 的数值无关，autograd 自然给出该定义）。
        xg = x.detach().requires_grad_(True)
        wg = weight.detach().requires_grad_(True)
        y = torch.nn.functional.conv2d(
            xg, wg, None,
            (self.stride_h, self.stride_w),
            (self.pad_h, self.pad_w),
            (self.dil_h, self.dil_w),
        )
        if self.variant == "dgrad":
            (dx,) = torch.autograd.grad(y, (xg,), dy)
            return dx
        (dw,) = torch.autograd.grad(y, (wg,), dy)
        return dw


def get_init_inputs():
    return ["dgrad", 1, 1, 1, 1, 1, 1]   # variant, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w


def get_inputs():
    # 固定 shape 族（样例口径：stride=1、pad=1、dilation=1，故 dy 与 x 同为
    # (4, 64/32, 16, 16) 族的对应件）；随机部分消费全局 RNG——评测器在 set_seed
    # 后调用本函数，多轮 correctness trial 因此获得输入多样性
    n, c, h, w = 4, 32, 16, 16
    out_c, r, s = 64, 3, 3
    dy = torch.randn(n, out_c, h, w).to(torch.float32)
    weight = torch.randn(out_c, c, r, s).to(torch.float32)
    x = torch.randn(n, c, h, w).to(torch.float32)
    return [dy, weight, x]


def make_inputs(n: int, c: int, h: int, w: int, out_c: int, r: int, s: int,
                stride_h: int = 1, stride_w: int = 1, pad_h: int = 1, pad_w: int = 1,
                dil_h: int = 1, dil_w: int = 1, variant: str = "dgrad",
                dtype: str = "float32", seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：dy -> weight -> x（均 torch.randn 后 cast 到 dtype），全部
    来自同一 torch.Generator(seed)，同 seed 下逐位可复现。dy 的空间尺寸由
    前向公式从 x/weight/卷积属性推出（out = floor((in + 2*pad - dil*(k-1) - 1)
    / stride) + 1），保证三张量对任意 variant 始终自洽。variant 与各卷积属性
    同时是构造 Model 的 init 超参，此处仅做合法性校验。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert variant in ("dgrad", "wgrad"), "variant 必须是 dgrad / wgrad 之一"
    assert min(n, c, h, w, out_c, r, s) >= 1, "各维需 >= 1"
    assert min(stride_h, stride_w, dil_h, dil_w) >= 1 and min(pad_h, pad_w) >= 0, \
        "stride/dilation >= 1 且 padding >= 0"
    eff_r, eff_s = dil_h * (r - 1) + 1, dil_w * (s - 1) + 1
    assert h + 2 * pad_h >= eff_r and w + 2 * pad_w >= eff_s, "前向卷积尺寸不自洽"
    out_h = (h + 2 * pad_h - eff_r) // stride_h + 1
    out_w = (w + 2 * pad_w - eff_s) // stride_w + 1
    dy = torch.randn(n, out_c, out_h, out_w, generator=gen).to(dt)
    weight = torch.randn(out_c, c, r, s, generator=gen).to(dt)
    x = torch.randn(n, c, h, w, generator=gen).to(dt)
    return dy, weight, x
