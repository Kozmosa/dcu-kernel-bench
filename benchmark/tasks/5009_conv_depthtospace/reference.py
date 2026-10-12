# 5009_conv_depthtospace —— Conv+Bias+DepthToSpace 融合（conv_bias_dts）与
# 纯 DepthToSpace 像素重排（dts）的 model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=2, problem_id=5009, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """Conv+Bias+DepthToSpace 融合（conv_bias_dts）与纯 DepthToSpace 像素
    重排（dts）前向家族。

    DepthToSpace（D2S）把 (N, K, H, W) 特征图的通道维重排进空间维：K 须被
    block_size^2 整除，记 C = K // block_size^2，输出 (N, C, H*bs, W*bs)
    （bs = block_size）。重排模式由 __init__ 参数 mode 二选一，按对
    v = (N, K, H, W) 的六维 view + 转置 + reshape 定义：
        "CRD"：v.reshape(N, C, bs, bs, H, W).permute(0, 1, 4, 2, 5, 3)
                .reshape(N, C, H*bs, W*bs)
                （与 torch.nn.functional.pixel_shuffle(v, bs) 逐位等价——
                该 ATen 原语在禁用列表，此处仅注明等价关系，不得直接调用）
        "DCR"：v.reshape(N, bs, bs, C, H, W).permute(0, 3, 4, 1, 5, 2)
                .reshape(N, C, H*bs, W*bs)
    两模式输出同形，仅通道到空间块位置的映射不同。逐元素公式：
        out[b, c, h*bs + r, w*bs + d] = v[b, idx(c, r, d), h, w]
        CRD：idx = c*bs^2 + r*bs + d；DCR：idx = (r*bs + d)*C + c
        （c ∈ [0, C)，r/d ∈ [0, bs)，h ∈ [0, H)，w ∈ [0, W)）

    融合路径按 __init__ 参数 variant 二选一：
        "conv_bias_dts"：对 x (N, C_in, H, W) 与卷积核 weight (K, C_in, 3, 3)
            做 2D cross-correlation 前向（深度学习库 conv_fprop 惯例语义，
            非数学卷积的核翻转），加逐通道偏置 bias (1, K, 1, 1)（沿 batch
            与空间维广播），再对卷积结果施加上述 D2S。卷积超参固定：
            stride=(1, 1)、对称零 padding=(1, 1)（上下/左右各补 1，padding
            值恒为零）、dilation=(1, 1)，故卷积输出空间维恒为 (H, W)，最终
            输出 (N, K//bs^2, H*bs, W*bs)。forward(x, weight, bias) 三参
            全用。
        "dts"：无卷积的纯数据重排（超分/像素重排场景）：x 本身即
            (N, K, H, W) 的待重排特征图，forward(x) 单参调用，weight/bias
            不传（可选参数缺省 None，即使传入也不得影响输出）；重排为纯
            索引搬运、逐位无损。
    卷积点积累加与 bias 加法全程 float32 中间计算（dts 分支同样先无损升位
    float32 再重排），输出单次 cast 回 x 的 dtype。

    输入输出规格：
        x      float16 / bfloat16 / float32 4D contiguous；
               variant="conv_bias_dts" 时 (N, C_in, H, W)、C_in >= 1，
               variant="dts" 时 (N, K, H, W)、K % bs^2 == 0
        weight 同 x dtype (K, C_in, 3, 3)，contiguous（仅 conv_bias_dts）
        bias   同 x dtype (1, K, 1, 1)，contiguous（仅 conv_bias_dts）
        out    单张量、同 x dtype (N, K//bs^2, H*bs, W*bs)（卷积与 D2S 均
               天然单输出，无多输出打包分段）
        N/K/H/W >= 1（允许非 2 的幂、素数与边界值，含 N=1、H=W=1）；
        K % block_size^2 == 0；block_size ∈ {2, 3, 4}；mode ∈ {"CRD", "DCR"}；
        输入为常规 randn 量级（无 inf/NaN）；评测器会把全部输入 cast 成
        fp32 传入（fp16/bf16 -> fp32 无损；本题输出 dtype 始终与传入 x 的
        dtype 一致）。

    实现约束（违规判负）：
        - 核心计算（含 padding 的索引/掩码展开与点积累加、bias 加法、六维
          重排索引映射）必须全部在提交文件内以 Triton kernel 完成并融合，
          禁止以任何形式的库卷积、ATen GEMM（matmul/mm/bmm/addmm/baddbmm/
          einsum/tensordot/unfold/fold/linear 等）、pixel_shuffle 系原语
          （含 pixel_unshuffle 往返），或借 conv3d 深度维 unsqueeze、
          conv_transpose2d 翻转核改写等一步等价路由替代。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 卷积点积累加与 bias 加法在 float32 中进行，输出单次 cast 回输入
          dtype（不得以 fp16/bf16 低精度累加，也不得多次往返 cast）；
          padding 须在 kernel 内以边界掩码/索引处理，不得在 host 侧物化
          padded 副本代替 kernel 内处理。
        - forward 返回单张量 out（评测器按单输出比较 shape/dtype/allclose）。

    禁用列表：torch.nn.Conv2d / torch.nn.functional.conv2d / F.conv2d /
    torch.conv2d / aten.conv2d / aten.convolution /
    torch.nn.functional.conv3d / torch.conv3d / torch.nn.Conv3d /
    torch.nn.functional.conv_transpose2d / torch.conv_transpose2d /
    torch.nn.ConvTranspose2d /
    torch.nn.functional.unfold / torch.nn.Unfold / .unfold( /
    torch.nn.functional.fold / torch.nn.Fold / .fold( /
    torch.nn.functional.pixel_shuffle / torch.pixel_shuffle / pixel_shuffle /
    torch.nn.PixelShuffle / PixelShuffle /
    torch.nn.functional.pixel_unshuffle / torch.pixel_unshuffle /
    pixel_unshuffle / torch.nn.PixelUnshuffle / PixelUnshuffle /
    torch.matmul / .matmul( / torch.mm / torch.bmm / torch.addmm /
    torch.baddbmm / torch.einsum / torch.tensordot /
    torch.nn.functional.linear / torch.nn.Linear /
    miopen / hipdnn。
    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    """

    # 卷积超参固定（两 variant 的 D2S 语义共享同一参数集）
    _STRIDE = (1, 1)
    _PADDING = (1, 1)   # 对称零 padding：上下/左右各补 1
    _DILATION = (1, 1)

    def __init__(self, variant: str = "conv_bias_dts", mode: str = "CRD",
                 block_size: int = 2):
        super().__init__()
        assert variant in ("conv_bias_dts", "dts"), \
            "variant 必须是 conv_bias_dts / dts 之一"
        assert mode in ("CRD", "DCR"), "mode 必须是 CRD / DCR 之一"
        assert block_size in (2, 3, 4), "block_size 须为 2 / 3 / 4"
        self.variant = variant
        self.mode = mode
        self.block_size = block_size

    def _depth_to_space(self, v):
        """六维 view + 转置 + reshape：与 mode/block_size 一一对应的纯索引
        重排（无数值运算），排列定义逐维对应融合图的首 reshape → transpose
        → 次 reshape。"""
        bs = self.block_size
        n, k, h, w = v.shape
        c = k // (bs * bs)
        if self.mode == "CRD":
            t = v.reshape(n, c, bs, bs, h, w).permute(0, 1, 4, 2, 5, 3)
        else:   # DCR
            t = v.reshape(n, bs, bs, c, h, w).permute(0, 3, 4, 1, 5, 2)
        return t.reshape(n, c, h * bs, w * bs)

    def forward(self, x, weight=None, bias=None):
        # 输入无损升位 float32，输出单次 cast 回 x 的 dtype
        xf = x.to(torch.float32)
        if self.variant == "conv_bias_dts":
            z = torch.nn.functional.conv2d(
                xf, weight.to(torch.float32), bias=None,
                stride=self._STRIDE, padding=self._PADDING, dilation=self._DILATION,
            )
            z = z + bias.to(torch.float32)
            return self._depth_to_space(z).to(x.dtype)
        else:   # dts：纯重排，weight/bias 不参与
            return self._depth_to_space(xf).to(x.dtype)


def get_init_inputs():
    return ["conv_bias_dts", "CRD", 2]   # variant, mode, block_size


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    n, c_in, h, w = 2, 8, 64, 64
    k = 16
    x = torch.randn(n, c_in, h, w).to(torch.float16)
    weight = torch.randn(k, c_in, 3, 3).to(torch.float16)
    bias = torch.randn(1, k, 1, 1).to(torch.float16)
    return [x, weight, bias]


def make_inputs(n: int, c_in: int, k: int, h: int, w: int,
                variant: str = "conv_bias_dts", mode: str = "CRD",
                block_size: int = 2, dtype: str = "float16", seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：conv_bias_dts 为 x -> weight -> bias（均 torch.randn 后
    cast 到 dtype），dts 仅消费 x；全部来自同一 torch.Generator(seed)，同
    seed 下逐位可复现。variant/mode/block_size 同时是构造 Model 的 init
    超参，此处校验其合法性、dtype 声明域与 shape 域（N/K/H/W >= 1、
    K % block_size^2 == 0、block_size ∈ {2,3,4}；variant=dts 时 c_in 不
    参与、case 约定填 0）。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert dt in (torch.float16, torch.bfloat16, torch.float32), \
        "本题 dtype 域仅声明 float16 / bfloat16 / float32"
    assert variant in ("conv_bias_dts", "dts"), \
        "variant 必须是 conv_bias_dts / dts 之一"
    assert mode in ("CRD", "DCR"), "mode 必须是 CRD / DCR 之一"
    assert block_size in (2, 3, 4), "block_size 须为 2 / 3 / 4"
    assert min(n, k, h, w) >= 1, "N/K/H/W 均需 >= 1"
    assert k % (block_size * block_size) == 0, "K 须被 block_size^2 整除"
    if variant == "conv_bias_dts":
        assert c_in >= 1, "conv_bias_dts 需 C_in >= 1"
        x = torch.randn(n, c_in, h, w, generator=gen).to(dt)
        weight = torch.randn(k, c_in, 3, 3, generator=gen).to(dt)
        bias = torch.randn(1, k, 1, 1, generator=gen).to(dt)
        return x, weight, bias
    x = torch.randn(n, k, h, w, generator=gen).to(dt)
    return x,
