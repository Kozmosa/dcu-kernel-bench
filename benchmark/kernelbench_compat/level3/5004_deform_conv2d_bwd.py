# 5004_deform_conv2d_bwd —— 可变形 2D 卷积反向（DCNv2 dgrad / wrw）的
# model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=5004, backend=cuda（HIP）。

import torch
import torch.nn as nn


def _bilinear_zero_pad(x, y_coord, x_coord):
    """双线性采样（整数 = 像素中心），边界外贡献为 0。

    x: (n, c, h, w)；y_coord / x_coord: (n, oh, ow) 浮点采样坐标（H/W 坐标系）。
    返回 (n, c, oh, ow)。对 y_coord / x_coord 可导：坐标梯度只经插值分数
    （frac 项）回传，floor/索引路径视为常数——即标准零填充双线性 Theano
    /DCNv2 约定的次梯度。
    """
    n, c, h, w = x.shape
    oh, ow = y_coord.shape[1], y_coord.shape[2]
    y0 = torch.floor(y_coord)
    x0 = torch.floor(x_coord)
    y1 = y0 + 1
    x1 = x0 + 1
    ty = (y_coord - y0).unsqueeze(1)   # (n, 1, oh, ow)
    tx = (x_coord - x0).unsqueeze(1)

    def gather(yc, xc):
        valid = ((yc >= 0) & (yc <= h - 1) & (xc >= 0) & (xc <= w - 1)).unsqueeze(1)
        yi = yc.clamp(0, h - 1).long()
        xi = xc.clamp(0, w - 1).long()
        # 索引张量自带 batch 维，直接高级索引会把索引维前置成 5-D；
        # 用展平 + gather 保持 (n, c, oh, ow)，对 x 可导
        idx = (yi * w + xi).reshape(n, 1, oh * ow).expand(n, c, oh * ow)
        got = x.reshape(n, c, h * w).gather(2, idx).reshape(n, c, oh, ow)
        return got * valid.to(x.dtype)

    return ((1 - ty) * (1 - tx) * gather(y0, x0)
            + (1 - ty) * tx * gather(y0, x1)
            + ty * (1 - tx) * gather(y1, x0)
            + ty * tx * gather(y1, x1))


class Model(nn.Module):
    """可变形 2D 卷积反向（DCNv2，带 mask 调制）：对输入求梯度（dgrad）与对
    权值求梯度（wrw）。

    语义（NCHW、offset group = 1、无 bias，与标准 DCNv2 前向严格一致）：
        前向 y = deform_conv2d(x, offset, weight, mask, stride=(sh,sw),
                               padding=(ph,pw), dilation=(dh,dw))
        输出尺寸 out_h = floor((H + 2*ph - dh*(R-1) - 1) / sh) + 1（out_w 同理）
        第 m 个核采样点按行主序 m = pm*S + pn（pm = m // S，pn = m % S），在
        输出位置 (i, j) 处：
            采样坐标  Y = i*sh - ph + dh*pm + offset[b, 2*m,   i, j]
                      X = j*sw - pw + dw_*pn + offset[b, 2*m+1, i, j]
            采样值  V[b, :, i, j, m] = bilinear(x, Y, X)：整数坐标即像素中心，
                双线性插值；采样点落在 [0,H-1]x[0,W-1] 之外的部分贡献为 0。
            y[b, ko, i, j] = sum_m mask[b, m, i, j] *
                                 sum_ci weight[ko, ci, pm, pn] * V[b, ci, i, j, m]
        offset 通道与核采样点按行主序一一对应：通道 2*m 为 H 方向偏移、
        2*m+1 为 W 方向偏移（DCNv2 唯一通行约定；本题 offset group = 1）。
    给定上游梯度 dy = dL/dy（形状 (N, out_c, out_h, out_w)）：
        dgrad：dx = dL/dx (N,C,H,W)、doffset = dL/doffset (N,2*R*S,out_h,out_w)、
               dmask = dL/dmask (N,R*S,out_h,out_w)
        wrw：  dw = dL/dw (out_c,C,R,S)
    __init__ 参数 variant 选择返回的方向：
        "dgrad"：返回三个梯度按分段协议打包成的单个一维张量；
        "wrw"：  返回 dw；此时 weight 的数值不参与输出（仅形状约定）。
    forward(dy, weight, x, offset, mask) 为固定五输入签名，各张量含义：
        dy     (N, out_c, out_h, out_w)     上游梯度，两方向都消费
        weight (out_c, C, R, S)             卷积核（out_c 输出通道 / C 输入通道）
        x      (N, C, H, W)                 前向输入
        offset (N, 2*R*S, out_h, out_w)     采样偏移
        mask   (N, R*S, out_h, out_w)       调制掩码
    forward 返回单张量 out（float32）——dgrad 为一维拼接打包（分段协议）：
        out = cat([dx.reshape(-1), doffset.reshape(-1), dmask.reshape(-1)])
        长度 = N*C*H*W + N*2*R*S*out_h*out_w + N*R*S*out_h*out_w，段序固定
        dx | doffset | dmask，各段为对应 4-D 张量的 C 序展平，段序不得交换；
        wrw → (out_c, C, R, S)，天然单输出无打包。

    输入输出规格：
        dy / weight / x / offset / mask 均为 float32、contiguous、NCHW 排布；
        各维 >= 1；R/S 允许 1（1x1 核）；H/W 允许 1、奇数、非 2 的幂、
        非 16 对齐；stride/dilation >= 1、padding >= 0，padding 为前向的四向
        对称 (ph, pw) 语义；一致性约束
            H + 2*ph >= dh*(R-1) + 1，W + 2*pw >= dw_*(S-1) + 1，
        且 dy/offset/mask 的空间尺寸必须等于前向公式的 out_h/out_w，offset
        通道数 = 2*R*S、mask 通道数 = R*S。
        输入值域为常规 randn 量级（无 inf/NaN）。评测器会把全部输入 cast
        成 fp32 传入。

    实现约束（违规判负）：
        - 核心计算必须在提交文件内以 HIP kernel 完成：dgrad 的反向双线性
          采样散射（对 Y/X 的解析偏导经核权值与 mask 链回 dx/doffset/dmask，
          边界外贡献置 0）或 wrw 的采样值相关累加（x 采样窗与 dy 的互相关
          求和），禁止回退到 torch autograd / 卷积原语 / grid_sample 完成
          核心计算。
        - ModelNew 的 __init__ 与 forward 签名不可更改；forward 返回单张量
          out（评测器按单输出比较）；dgrad 三段须按上述分段协议打包，段序
          不得交换。
        - 数值约定：float32 输入直接以 float32 累加，不做半精度下转；dgrad
          三个梯度段与 wrw 输出均为 float32。

    禁用列表：aiter / torchvision / torch.nn.functional.conv2d / F.conv2d /
    torch.conv2d / torch.nn.Conv2d / torch.nn.functional.conv_transpose2d /
    F.conv_transpose2d / torch.nn.ConvTranspose2d /
    torch.nn.grad.conv2d_input / torch.nn.grad.conv2d_weight /
    aten.convolution_backward / deform_conv2d / deform_conv2d_backward /
    grid_sample / F.grid_sample / torch.nn.functional.grid_sample /
    torch.autograd / .backward( / unfold / F.fold /
    torch.nn.functional.fold / im2col / col2im / as_strided；
    闭源库 MIOpen / hipDNN / cuDNN / rocBLAS / hipBLASLt 同样禁止（全局黑名单）。
    目标硬件：海光 DCU（gfx936/gfx938），实现语言 HIP（经 hipcc 编译；在 CUDA_CU_SRC / CUDA_CPP_SRC 锚点区编写，设备侧必须 include <hip/hip_runtime.h>，本机无 <cuda_runtime.h>）。
    """

    def __init__(self, variant: str = "dgrad", stride_h: int = 1, stride_w: int = 1,
                 pad_h: int = 1, pad_w: int = 1, dil_h: int = 1, dil_w: int = 1):
        super().__init__()
        assert variant in ("dgrad", "wrw"), "variant 必须是 dgrad / wrw 之一"
        self.variant = variant
        self.stride_h, self.stride_w = int(stride_h), int(stride_w)
        self.pad_h, self.pad_w = int(pad_h), int(pad_w)
        self.dil_h, self.dil_w = int(dil_h), int(dil_w)

    def _deform_conv2d_forward(self, x, offset, weight, mask):
        """手写 DCNv2 前向（标准 torch 可导算子组合），供 autograd 反传。"""
        n, c, h, w = x.shape
        k, _, r, s = weight.shape
        sh, sw = self.stride_h, self.stride_w
        ph, pw = self.pad_h, self.pad_w
        dh, dwl = self.dil_h, self.dil_w
        oh, ow = offset.shape[2], offset.shape[3]
        out = torch.zeros(n, k, oh, ow, dtype=x.dtype, device=x.device)
        for m in range(r * s):
            pm, pn = m // s, m % s
            base_y = torch.arange(oh, dtype=x.dtype, device=x.device) * sh - ph + dh * pm
            base_x = torch.arange(ow, dtype=x.dtype, device=x.device) * sw - pw + dwl * pn
            yo = base_y.view(1, oh, 1) + offset[:, 2 * m]      # (n, oh, ow)
            xo = base_x.view(1, 1, ow) + offset[:, 2 * m + 1]  # (n, oh, ow)
            val = _bilinear_zero_pad(x, yo, xo)                # (n, c, oh, ow)
            wc = weight[:, :, pm, pn]                          # (k, c)
            contrib = torch.einsum("oc,ncij->noij", wc, val)
            out = out + mask[:, m].unsqueeze(1) * contrib
        return out

    def forward(self, dy, weight, x, offset, mask):
        # 对手写 DCNv2 前向构图后用 autograd 反传：保证 dgrad / wrw 与前向
        # 语义严格一致（reference 不受禁用列表约束；数学上 dw 与 weight 的
        # 数值无关，autograd 自然给出该定义）。
        xg = x.detach().requires_grad_(True)
        og = offset.detach().requires_grad_(True)
        mg = mask.detach().requires_grad_(True)
        wg = weight.detach().requires_grad_(True)
        y = self._deform_conv2d_forward(xg, og, wg, mg)
        if self.variant == "dgrad":
            dx, doffset, dmask = torch.autograd.grad(y, (xg, og, mg), dy)
            # 单张量输出：三段一维拼接，段序 dx | doffset | dmask
            return torch.cat([dx.reshape(-1), doffset.reshape(-1), dmask.reshape(-1)])
        (dw,) = torch.autograd.grad(y, (wg,), dy)
        return dw


def get_init_inputs():
    return ["dgrad", 1, 1, 1, 1, 1, 1]   # variant, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w


def get_inputs():
    # 固定 shape 族（样例口径：3x3 核、stride=1、dilation=1；pad 取 1 覆盖填充
    # 路径，尺寸为样例的适度放宽）；随机部分消费全局 RNG——评测器在 set_seed
    # 后调用本函数，多轮 correctness trial 因此获得输入多样性
    n, c, h, w = 2, 16, 16, 16
    out_c, r, s = 32, 3, 3
    out_h = (h + 2 * 1 - 1 * (3 - 1) - 1) // 1 + 1
    out_w = (w + 2 * 1 - 1 * (3 - 1) - 1) // 1 + 1
    dy = torch.randn(n, out_c, out_h, out_w).to(torch.float32)
    weight = torch.randn(out_c, c, r, s).to(torch.float32)
    x = torch.randn(n, c, h, w).to(torch.float32)
    offset = torch.randn(n, 2 * r * s, out_h, out_w).to(torch.float32)
    mask = torch.randn(n, r * s, out_h, out_w).to(torch.float32)
    return [dy, weight, x, offset, mask]


def make_inputs(n: int, c: int, h: int, w: int, out_c: int, r: int, s: int,
                stride_h: int = 1, stride_w: int = 1, pad_h: int = 1, pad_w: int = 1,
                dil_h: int = 1, dil_w: int = 1, variant: str = "dgrad",
                dtype: str = "float32", seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：dy -> weight -> x -> offset -> mask（均 torch.randn 后 cast
    到 dtype），全部来自同一 torch.Generator(seed)，同 seed 下逐位可复现。
    dy/offset/mask 的空间尺寸由前向公式从 x/weight/卷积属性推出
    （out = floor((in + 2*pad - dil*(k-1) - 1) / stride) + 1），保证五张量对
    任意 variant 始终自洽。variant 与各卷积属性同时是构造 Model 的 init 超参，
    此处仅做合法性校验。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert variant in ("dgrad", "wrw"), "variant 必须是 dgrad / wrw 之一"
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
    offset = torch.randn(n, 2 * r * s, out_h, out_w, generator=gen).to(dt)
    mask = torch.randn(n, r * s, out_h, out_w, generator=gen).to(dt)
    return dy, weight, x, offset, mask
