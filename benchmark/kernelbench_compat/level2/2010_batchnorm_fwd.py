# 2010_batchnorm_fwd —— BatchNorm 前向家族（inference / training）的
# model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=2, problem_id=2010, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """BatchNorm 前向家族：推理仿射（inference）与训练批统计（training）。

    输入 x (N, C, H, W)，NCHW contiguous；scale / bias / mean / variance 均为
    (1, C, 1, 1)，沿通道维广播。统计维度：per-channel，对每个通道 c 在
    (N, H, W) 三个维度上求统计，每通道元素数 m = N*H*W。__init__ 参数
    variant 选择两条路径（eps 缺省 1e-5，momentum 缺省 0.1，均仅作标量
    超参）：

        "inference"：均值与方差作为输入给定，逐元素仿射（float32 中间计算）：
            y = scale * (x - mean) / sqrt(variance + eps) + bias
        "training"：批统计 + 仿射 + running 更新（float32 全程）：
            batch_mean_c = sum_{n,h,w} x[n,c,h,w] / m                    （有偏批均值）
            batch_var_c  = sum_{n,h,w} (x[n,c,h,w] - batch_mean_c)^2 / m （有偏批方差）
            saved_mean         = batch_mean                              （shape (1,C,1,1)）
            saved_inv_variance = 1 / sqrt(batch_var + eps)               （shape (1,C,1,1)）
            y = scale * (x - saved_mean) * saved_inv_variance + bias
            next_running_mean = (1 - momentum) * mean + momentum * batch_mean
            next_running_var  = (1 - momentum) * variance
                                + momentum * batch_var * m / (m - 1)
            其中 forward 第 4/5 参在 training 下为上一时刻的 running 统计
            （prev_running_mean / prev_running_var）；running 方差更新使用
            无偏批方差（m/(m-1) 修正，故要求 m >= 2）——这是 PyTorch /
            cuDNN / MIOpen 训练 BN 的标准约定。

    forward 形参对两 variant 统一为 (x, scale, bias, mean, variance)：
    inference 的第 4/5 参为给定 mean / variance，training 的第 4/5 参为
    prev_running_mean / prev_running_var。

    单张量输出协议（forward 必须返回单个张量，评测器按单输出比较）：
        "inference"：返回 y 本体，shape (N, C, H, W)，dtype 与 x 相同；
        "training" ：返回一维 float32 打包张量，长度 N*C*H*W + 4*C，五个
                     逻辑输出各以 float32 cast、reshape(-1) 后按样例输出
                     顺序 torch.cat，分段协议（半开区间）：
            [0,            N*C*H*W)      y
            [N*C*H*W,      N*C*H*W + C)  saved_mean
            [N*C*H*W + C,  N*C*H*W + 2C) saved_inv_variance
            [N*C*H*W + 2C, N*C*H*W + 3C) next_running_mean
            [N*C*H*W + 3C, N*C*H*W + 4C) next_running_var

    输入输出规格：
        x        float32（training 恒为 float32）/ float16 / bfloat16
                 （仅 inference），(N, C, H, W)，contiguous，N/C/H/W >= 1
        scale    同 x dtype (1, C, 1, 1)，contiguous
        bias     同 x dtype (1, C, 1, 1)，contiguous
        mean     同 x dtype (1, C, 1, 1)，contiguous（inference：给定均值，
                 任意实值；training：prev_running_mean）
        variance 同 x dtype (1, C, 1, 1)，contiguous，恒为正（inference：
                 给定方差；training：prev_running_var）
        out      inference：(N, C, H, W)，同 x dtype；training：一维
                 float32，长度 N*C*H*W + 4*C（五段打包协议如上）
        输入为常规 randn/rand 量级（无 inf/NaN）；评测器会把全部输入
        cast 成 fp32 传入（fp16/bf16 -> fp32 无损）。

    实现约束（违规判负）：
        - 核心计算（per-channel 归约统计、rsqrt、仿射、running 更新）必须
          在提交文件内以 Triton kernel 完成。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 统计、归一化与 running 更新全程 float32 中间计算；inference 的 y
          cast 回 x dtype；training 五段均为 float32。
        - forward 返回单张量 out（评测器按单输出比较 shape/dtype/allclose）。

    禁用列表：torch.nn.BatchNorm1d / torch.nn.BatchNorm2d /
    torch.nn.BatchNorm3d / torch.nn.SyncBatchNorm /
    torch.nn.functional.batch_norm / torch.batch_norm /
    torch.native_batch_norm / miopen / hipdnn。
    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    """

    def __init__(self, variant: str = "training", eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        assert variant in ("inference", "training"), "variant 必须是 inference / training 之一"
        assert eps > 0, "eps 需为正"
        assert 0.0 <= momentum <= 1.0, "momentum 需在 [0, 1]"
        self.variant = variant
        self.eps = float(eps)
        self.momentum = float(momentum)

    def forward(self, x, scale, bias, mean, variance):
        eps = self.eps
        xf = x.to(torch.float32)
        sf = scale.to(torch.float32)
        bf = bias.to(torch.float32)
        mf = mean.to(torch.float32)
        vf = variance.to(torch.float32)

        if self.variant == "inference":
            # 给定 mean/variance 的逐元素仿射，(1,C,1,1) 沿通道广播
            inv_std = torch.rsqrt(vf + eps)
            y = (xf - mf) * inv_std * sf + bf
            return y.to(x.dtype)

        # training：per-channel 有偏批统计（对 (N,H,W) 归约），y 用批统计归一化
        m = xf.shape[0] * xf.shape[2] * xf.shape[3]
        assert m >= 2, "training 变体要求 N*H*W >= 2（running 方差的无偏修正因子 m/(m-1)）"
        batch_mean = xf.mean(dim=(0, 2, 3), keepdim=True)                        # (1,C,1,1) 有偏均值
        batch_var = ((xf - batch_mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)   # (1,C,1,1) 有偏方差
        saved_inv_variance = torch.rsqrt(batch_var + eps)
        y = (xf - batch_mean) * saved_inv_variance * sf + bf

        # running 统计更新：均值用批均值，方差用无偏批方差（m/(m-1) 修正）
        batch_var_unbiased = batch_var * (m / (m - 1))
        next_running_mean = (1.0 - self.momentum) * mf + self.momentum * batch_mean
        next_running_var = (1.0 - self.momentum) * vf + self.momentum * batch_var_unbiased

        # 单张量输出：五段按样例输出顺序一维拼接（全 float32）
        return torch.cat([
            y.to(torch.float32).reshape(-1),
            batch_mean.to(torch.float32).reshape(-1),
            saved_inv_variance.to(torch.float32).reshape(-1),
            next_running_mean.to(torch.float32).reshape(-1),
            next_running_var.to(torch.float32).reshape(-1),
        ])


def get_init_inputs():
    return ["training", 1e-5, 0.1]   # variant, eps, momentum


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性。生成规则与 make_inputs 一致
    # （消费顺序 x -> scale -> bias -> mean -> variance，此处消费全局 RNG）
    n, c, h, w = 4, 16, 56, 56
    x = torch.randn(n, c, h, w)
    scale = torch.rand(1, c, 1, 1) + 0.5
    bias = torch.randn(1, c, 1, 1)
    mean = torch.randn(1, c, 1, 1)              # training：prev_running_mean
    variance = torch.rand(1, c, 1, 1) + 0.25    # training：prev_running_var
    return [x, scale, bias, mean, variance]


def make_inputs(n: int, c: int, h: int, w: int, variant: str = "training",
                dtype: str = "float32", seed: int = 0,
                eps: float = 1e-5, momentum: float = 0.1):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：x -> scale -> bias -> mean -> variance。x/bias/mean 用
    torch.randn，scale 用 torch.rand + 0.5（正缩放域 [0.5, 1.5)），variance
    用 torch.rand + 0.25（恒正域 [0.25, 1.25)），全部来自同一
    torch.Generator(seed)，同 seed 下逐位可复现。
    variant / eps / momentum 同时是构造 Model 的 init 超参，此处仅做合法性
    校验；training 变体的 dtype 域仅声明 float32（打包输出协议），且要求
    N*H*W >= 2。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert variant in ("inference", "training"), "variant 必须是 inference / training 之一"
    assert min(n, c, h, w) >= 1, "N/C/H/W 均需 >= 1"
    assert eps > 0 and 0.0 <= momentum <= 1.0, "eps 需为正，momentum 需在 [0, 1]"
    if variant == "training":
        assert dt == torch.float32, "training 变体的 dtype 域仅声明 float32"
        assert n * h * w >= 2, "training 变体要求 N*H*W >= 2"
    x = torch.randn(n, c, h, w, generator=gen).to(dt)
    scale = (torch.rand(1, c, 1, 1, generator=gen) + 0.5).to(dt)
    bias = torch.randn(1, c, 1, 1, generator=gen).to(dt)
    mean = torch.randn(1, c, 1, 1, generator=gen).to(dt)
    variance = (torch.rand(1, c, 1, 1, generator=gen) + 0.25).to(dt)
    return x, scale, bias, mean, variance
