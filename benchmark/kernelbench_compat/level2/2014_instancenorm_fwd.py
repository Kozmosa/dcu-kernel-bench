# 2014_instancenorm_fwd —— InstanceNorm 前向（per-(N,C) 空间统计 + 逐通道
# 仿射）的 model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=2, problem_id=2014, backend=cuda（HIP）。

import torch
import torch.nn as nn


class Model(nn.Module):
    """InstanceNorm 前向：per-(N,C) 空间统计归一化 + 逐通道仿射（单 variant）。

    输入 x (N, C, H, W)，NCHW contiguous；scale / bias 为 (1, C, 1, 1) 的
    逐通道仿射参数，沿通道维广播；eps 缺省 1e-5。对每个 (n, c) 平面在空间
    维 (H, W) 上求统计（每平面元素数 HW = H*W），在 float32 中：
        mean_nc    = sum_{h,w} x[n,c,h,w] / HW                  （有偏均值）
        var_nc     = sum_{h,w} (x[n,c,h,w] - mean_nc)^2 / HW    （有偏方差）
        inv_std_nc = 1 / sqrt(var_nc + eps)
        y[n,c,h,w] = scale[c] * (x[n,c,h,w] - mean_nc) * inv_std_nc + bias[c]
    instance norm 无 train / inference 语义差：归一化统计恒取自输入本身。
    有偏口径（除以 HW 而非 HW-1），eps 在 rsqrt 内部（唯一用途即除零保护，
    不参与任何无偏修正）。

    边界行为：HW == 1（空间单元素）时 var = 0、x - mean = 0，此时
    inv_std = 1/sqrt(eps)、y = bias（eps 保证不除零）；N/C/H/W 任意（含
    1、非 2 的幂、非对齐），大 HW 需分块归约，行尾越界元素不得参与统计。

    输入输出规格：
        x     float32 / float16 / bfloat16 (N, C, H, W)，contiguous，
              N/C/H/W >= 1
        scale 同 x dtype (1, C, 1, 1)，contiguous
        bias  同 x dtype (1, C, 1, 1)，contiguous
        out   单张量 y 本体：(N, C, H, W)、同 x dtype（统计与仿射全程
              float32 中间计算后 cast 回 x dtype），无打包
        输入为常规 randn/rand 量级（无 inf/NaN）；评测器会把全部输入
        cast 成 fp32 传入（fp16/bf16 -> fp32 无损）。

    实现约束（违规判负）：
        - 核心计算（per-(N,C) 空间归约统计、rsqrt、逐通道仿射）必须在提交
          文件内以 HIP kernel 完成。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 统计、rsqrt 与仿射全程 float32 中间计算；y cast 回 x dtype。
        - forward 返回单张量 out（评测器按单输出比较 shape/dtype/allclose）。

    禁用列表：torch.nn.InstanceNorm1d / torch.nn.InstanceNorm2d /
    torch.nn.InstanceNorm3d / torch.nn.functional.instance_norm /
    F.instance_norm / torch.instance_norm / native_instance_norm /
    torch.nn.GroupNorm / nn.GroupNorm / torch.nn.functional.group_norm /
    F.group_norm / torch.group_norm / native_group_norm /
    torch.nn.BatchNorm1d / torch.nn.BatchNorm2d / torch.nn.BatchNorm3d /
    torch.nn.SyncBatchNorm / torch.nn.functional.batch_norm /
    F.batch_norm / torch.batch_norm / torch.native_batch_norm /
    miopen / hipdnn。
    （GroupNorm 取 num_groups=C、BatchNorm 经 (1, N*C, H, W) 重排均能一步
    完成本题核心计算，故与 instance_norm 族一并禁用。）
    目标硬件：海光 DCU（gfx936/gfx938），实现语言 HIP（经 hipcc 编译；在 CUDA_CU_SRC / CUDA_CPP_SRC 锚点区编写，设备侧必须 include <hip/hip_runtime.h>，本机无 <cuda_runtime.h>）。
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        assert eps > 0, "eps 需为正"
        self.eps = float(eps)

    def forward(self, x, scale, bias):
        # per-(N,C) 空间统计：沿 (H, W) 归约，float32 中间计算
        # （有偏均值 / 有偏方差）
        xf = x.to(torch.float32)
        mean = xf.mean(dim=(2, 3), keepdim=True)                   # (N,C,1,1)
        var = ((xf - mean) ** 2).mean(dim=(2, 3), keepdim=True)    # (N,C,1,1) 有偏方差
        inv_std = torch.rsqrt(var + self.eps)

        # 逐通道仿射：(1,C,1,1) 沿通道广播，float32 计算后 cast 回 x dtype
        sf = scale.to(torch.float32)
        bf = bias.to(torch.float32)
        y = (xf - mean) * inv_std * sf + bf
        return y.to(x.dtype)


def get_init_inputs():
    return [1e-5]   # eps


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性。生成规则与 make_inputs 一致
    # （消费顺序 x -> scale -> bias，此处消费全局 RNG）
    n, c, h, w = 16, 32, 64, 64
    x = torch.randn(n, c, h, w)
    scale = torch.rand(1, c, 1, 1) + 0.5
    bias = torch.randn(1, c, 1, 1)
    return [x, scale, bias]


def make_inputs(n: int, c: int, h: int, w: int,
                dtype: str = "float32", seed: int = 0, eps: float = 1e-5):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：x -> scale -> bias。x/bias 用 torch.randn，scale 用
    torch.rand + 0.5（正缩放域 [0.5, 1.5)），全部来自同一
    torch.Generator(seed)，同 seed 下逐位可复现。
    eps 同时是构造 Model 的 init 超参，此处仅做合法性校验（不参与输入生成）。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert min(n, c, h, w) >= 1, "N/C/H/W 均需 >= 1"
    assert eps > 0, "eps 需为正"
    assert dt in (torch.float32, torch.float16, torch.bfloat16), \
        "dtype 需为 float32 / float16 / bfloat16"
    x = torch.randn(n, c, h, w, generator=gen).to(dt)
    scale = (torch.rand(1, c, 1, 1, generator=gen) + 0.5).to(dt)
    bias = torch.randn(1, c, 1, 1, generator=gen).to(dt)
    return x, scale, bias
