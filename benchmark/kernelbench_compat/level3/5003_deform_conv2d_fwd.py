# 5003_deform_conv2d_fwd —— 可变形 2D 卷积前向（modulated / plain 两 variant）
# 的 model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=5003, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """可变形 2D 卷积前向（Deformable Convolution）：采样网格 + 手写双线性插值。

    语义（NCHW，中间计算全程 float32）：
      对每个输出位置 (oi, oj) 与卷积核位置 p = i*S + j（i∈[0,R)、j∈[0,S)，行主
      序），常规卷积感受野给出的基础采样坐标为
          y0 = oi * stride_h + i * dilation_h - padding_h
          x0 = oj * stride_w + j * dilation_w - padding_w
      叠加逐（输出位置 × 卷积核位置）的偏移后，在输入图上按实数坐标做双线性
      插值采样：
          yc = y0 + offset[n, 2*p,     oi, oj]     # 偶数通道 = y 方向偏移
          xc = x0 + offset[n, 2*p + 1, oi, oj]     # 奇数通道 = x 方向偏移
          v(n, c, p, oi, oj) = bilinear(输入图, yc, xc)
      __init__ 参数 variant 选择两条路径之一：
          "modulated"（缺省，调制型）：采样值再乘以逐位置调制掩码
              v *= mask[n, p, oi, oj]
          "plain"（无掩码型）：mask 不参与计算（内容任意，可为垃圾数据）。
      输出为单张量：
          out[n, k, oi, oj] = bias[k] + Σ_{c, p} v(n, c, p, oi, oj) * weight[k, c, i, j]
      计算完成后 cast 回输入 dtype。

    双线性插值定义（与通行可变形卷积算子逐点同值）：
      令 iy = floor(yc)、ix = floor(xc)、wy = yc - iy、wx = xc - ix，取四个角点
      (iy, ix)、(iy, ix+1)、(iy+1, ix)、(iy+1, ix+1)，权重依次为
      (1-wy)(1-wx)、(1-wy)wx、wy(1-wx)、wy*wx；某角点的行或列越界（<0 或
      >dim-1）时该角点贡献记 0、其余角点照常加权——即零填充语义：坐标部分
      越界仍按剩余有效角点加权，完全越界（yc <= -1、yc >= H-1 边界带之外等）
      时采样值为 0。全程不做 clamp / 最近邻回拉。

    输入输出规格（张量均 contiguous，NCHW 布局）：
      x        float32 / float16 / bfloat16 (N, C, H, W)
      offset   同 x dtype (N, 2*R*S, H_out, W_out)：每个卷积核位置 2 个通道
               （偶数通道 y 偏移、奇数通道 x 偏移，p = i*S + j 行主序）
      weight   同 x dtype (K, C, R, S)
      bias     同 x dtype (K,)
      mask     同 x dtype (N, R*S, H_out, W_out)（variant="plain" 时忽略）
      out      单张量、同 x dtype (N, K, H_out, W_out)
      H_out = (H + 2*padding_h - dilation_h*(R-1) - 1) // stride_h + 1，
      W_out 同理。评测器会把全部输入 cast 成 fp32 传入（低精度输入无损）。

    实现约束（违规判负）：
      - 核心计算（采样网格构建、双线性插值采样、mask 调制、与 weight 的加权
        归约）必须在提交文件内以 Triton kernel 手写完成。
      - forward 返回单张量 out（评测器按单输出比较）。
      - 插值与加权归约在 float32 中进行，输出 cast 回输入 dtype。
      - ModelNew 的 __init__ 与 forward 签名不可更改。

    禁用列表：torchvision（含 torchvision.ops.deform_conv2d）、
    torch.nn.functional.conv2d（F.conv2d）、torch.nn.Conv2d、
    torch.nn.functional.grid_sample（F.grid_sample）、
    torch.nn.functional.unfold（F.unfold）、torch.matmul / torch.bmm /
    torch.mm / torch.einsum / torch.tensordot、miopen、hipdnn、aiter。
    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    """

    def __init__(self, variant: str = "modulated", stride=(1, 1), padding=(1, 1),
                 dilation=(1, 1)):
        super().__init__()
        assert variant in ("modulated", "plain"), "variant 必须是 modulated / plain 之一"
        self.variant = variant
        self.stride = (int(stride[0]), int(stride[1]))
        self.padding = (int(padding[0]), int(padding[1]))
        self.dilation = (int(dilation[0]), int(dilation[1]))

    def forward(self, x, offset, weight, bias, mask):
        n, c, h, w = x.shape
        k_out = weight.shape[0]
        r, s = weight.shape[2], weight.shape[3]
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation
        h_out = (h + 2 * ph - (dh * (r - 1) + 1)) // sh + 1
        w_out = (w + 2 * pw - (dw * (s - 1) + 1)) // sw + 1
        assert offset.shape[1] == 2 * r * s and mask.shape[1] == r * s, \
            "offset/mask 通道数须为 2*R*S / R*S"
        assert offset.shape[2] == h_out and offset.shape[3] == w_out \
            and mask.shape[2] == h_out and mask.shape[3] == w_out, \
            "offset/mask 空间尺寸须为 (H_out, W_out)"

        xf = x.to(torch.float32)
        off = offset.to(torch.float32)
        wf = weight.to(torch.float32)
        bf = bias.to(torch.float32)

        # 每卷积核位置的基础感受野坐标（未加偏移），行主序 p = i*S + j
        # device=dev 必需：这些坐标要与已在设备上的 off 相加，漏 device 会在
        # CUDA/DCU 上抛 "Expected all tensors to be on the same device"（CPU 上无感）
        dev = xf.device
        base_y = (torch.arange(h_out, dtype=torch.float32, device=dev) * sh - ph).view(1, 1, h_out, 1) \
            + torch.arange(r, dtype=torch.float32, device=dev).view(r, 1, 1, 1) * dh      # (r, 1, h_out, 1)
        base_x = (torch.arange(w_out, dtype=torch.float32, device=dev) * sw - pw).view(1, 1, 1, w_out) \
            + torch.arange(s, dtype=torch.float32, device=dev).view(1, s, 1, 1) * dw      # (1, s, 1, w_out)

        # 叠加逐位置偏移（偶数通道 dy、奇数通道 dx）→ 采样坐标 (n, R*S, H_out, W_out)
        y_coord = (base_y + off[:, 0::2].reshape(n, r, s, h_out, w_out))
        x_coord = (base_x + off[:, 1::2].reshape(n, r, s, h_out, w_out))
        y_coord = y_coord.reshape(n, r * s, h_out, w_out)
        x_coord = x_coord.reshape(n, r * s, h_out, w_out)

        # 手写双线性插值：四角点采样、越界角点贡献记 0（零填充语义）
        iy0 = torch.floor(y_coord).to(torch.long)
        ix0 = torch.floor(x_coord).to(torch.long)
        wy = y_coord - iy0.to(torch.float32)
        wx = x_coord - ix0.to(torch.float32)
        iy1 = iy0 + 1
        ix1 = ix0 + 1
        n_idx = torch.arange(n, device=xf.device).view(n, 1, 1, 1, 1)
        c_idx = torch.arange(c, device=xf.device).view(1, c, 1, 1, 1)

        def corner(iy, ix, wgt):
            # 单角点采样：越界角点权重置 0；clamp 仅防索引越界（乘 0 抵消）
            valid = ((iy >= 0) & (iy <= h - 1) & (ix >= 0) & (ix <= w - 1)).to(torch.float32)
            vals = xf[n_idx, c_idx, iy.clamp(0, h - 1).unsqueeze(1),
                      ix.clamp(0, w - 1).unsqueeze(1)]
            return vals * valid.unsqueeze(1) * wgt.unsqueeze(1)

        sampled = (corner(iy0, ix0, (1.0 - wy) * (1.0 - wx))
                   + corner(iy0, ix1, (1.0 - wy) * wx)
                   + corner(iy1, ix0, wy * (1.0 - wx))
                   + corner(iy1, ix1, wy * wx))            # (n, c, R*S, H_out, W_out)

        # 调制掩码（plain variant 下 mask 不参与计算）
        if self.variant == "modulated":
            sampled = sampled * mask.to(torch.float32).unsqueeze(1)

        # 与 weight 的加权归约（float32），加 bias，cast 回输入 dtype
        y = torch.matmul(wf.reshape(k_out, c * r * s),
                         sampled.reshape(n, c * r * s, h_out * w_out))
        y = y + bf.view(1, k_out, 1)
        return y.view(n, k_out, h_out, w_out).to(x.dtype)


def get_init_inputs():
    return ["modulated", [1, 1], [1, 1], [1, 1]]   # variant, stride, padding, dilation


def get_inputs():
    # 固定 shape 族（N=4, C=16, H=W=32, K=32, R=S=3, stride=padding=dilation=(1,1)，
    # variant=modulated，float32）；随机部分消费全局 RNG——评测器在 set_seed 后调用
    # 本函数，多轮 correctness trial 因此获得输入多样性
    n, c, h, w, k, r, s = 4, 16, 32, 32, 32, 3, 3
    ph = pw = 1
    dh = dw = 1
    h_out = (h + 2 * ph - (dh * (r - 1) + 1)) // 1 + 1   # = 32
    w_out = (w + 2 * pw - (dw * (s - 1) + 1)) // 1 + 1   # = 32
    x = torch.randn(n, c, h, w).to(torch.float32)
    offset = torch.randn(n, 2 * r * s, h_out, w_out).to(torch.float32)
    weight = torch.randn(k, c, r, s).to(torch.float32)
    bias = torch.randn(k).to(torch.float32)
    mask = torch.randn(n, r * s, h_out, w_out).to(torch.float32)
    return [x, offset, weight, bias, mask]


def make_inputs(n: int, c: int, h: int, w: int, k: int, r: int = 3, s: int = 3,
                variant: str = "modulated", dtype: str = "float32", seed: int = 0,
                stride=(1, 1), padding=(1, 1), dilation=(1, 1)):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：x -> offset -> weight -> bias -> mask（均 fp32 torch.randn 后
    cast 到 dtype），全部来自同一 torch.Generator(seed)，同 seed 下逐位可复现。
    variant/stride/padding/dilation 同时是构造 Model 的 init 超参，此处用于推导
    H_out/W_out 并做合法性校验。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert variant in ("modulated", "plain"), "variant 必须是 modulated / plain 之一"
    assert n >= 1 and c >= 1 and h >= 1 and w >= 1 and k >= 1 and r >= 1 and s >= 1, \
        "全部维度须 >= 1"
    sh, sw = int(stride[0]), int(stride[1])
    ph, pw = int(padding[0]), int(padding[1])
    dh, dw = int(dilation[0]), int(dilation[1])
    assert sh >= 1 and sw >= 1 and dh >= 1 and dw >= 1, "stride/dilation 须 >= 1"
    assert ph >= 0 and pw >= 0, "padding 须 >= 0"
    h_out = (h + 2 * ph - (dh * (r - 1) + 1)) // sh + 1
    w_out = (w + 2 * pw - (dw * (s - 1) + 1)) // sw + 1
    assert h_out >= 1 and w_out >= 1, "H_out/W_out 须 >= 1（padding 需覆盖膨胀核）"
    x = torch.randn(n, c, h, w, generator=gen).to(dt)
    offset = torch.randn(n, 2 * r * s, h_out, w_out, generator=gen).to(dt)
    weight = torch.randn(k, c, r, s, generator=gen).to(dt)
    bias = torch.randn(k, generator=gen).to(dt)
    mask = torch.randn(n, r * s, h_out, w_out, generator=gen).to(dt)
    return x, offset, weight, bias, mask
