# 2012_groupnorm_fwd —— GroupNorm 前向家族（标准 / Swish 融合）的
# model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=2, problem_id=2012, backend=cuda（HIP）。

import torch
import torch.nn as nn


class Model(nn.Module):
    """GroupNorm 前向家族：标准 GroupNorm 与 GroupNorm+Swish 融合。

    输入 x (N, C, H, W)，NCHW contiguous；scale / bias 为 (C,) 逐通道仿射
    参数；__init__ 参数 groups = G（整除 C，1 <= G <= C）把通道分成 G 组，
    通道 c 属于组 c // (C//G)；eps 缺省 1e-5。归一化统计按 (n, g) 组计算：
    把 x[n] 沿通道维按组切分为 G 段（组 g 占通道 [g*(C//G), (g+1)*(C//G))，
    组内元素数 D = (C//G)*H*W），在 float32 中：
        mean_ng     = sum_D x / D                      （有偏均值）
        var_ng      = sum_D (x - mean_ng)^2 / D        （有偏方差）
        inv_var_ng  = 1 / sqrt(var_ng + eps)
        gn[n,c,h,w] = (x[n,c,h,w] - mean_ng) * inv_var_ng * scale[c] + bias[c]
                      （其中组号 g = c // (C//G)）
    __init__ 参数 variant 选择两条路径：
        "groupnorm"      ：y = gn
        "groupnorm_swish"：y = gn * sigmoid(gn)（Swish / SiLU）
    两 variant 均同时输出逐组统计 mean 与 inv_var（每个 (n, g) 一组，
    (N, G) 布局，float32）。

    边界行为：组内方差为 0（如 D == 1，组内只有单个元素）时
    inv_var = 1/sqrt(eps)（eps 保证不除零），此时 gn = bias；G=1 为全通道
    归一化、G=C 为逐通道归一化，均合法；N/C/H/W 任意（含 1、非 2 的幂、
    非对齐），组内大 D 需分块归约，行尾越界元素不得参与统计。

    单张量输出协议（forward 必须返回单个张量，评测器按单输出比较）：
    返回一维 float32 打包张量，长度 N*C*H*W + 2*N*G，三段按序紧排
    （半开区间；各段以 float32、reshape(-1) 后 torch.cat）：
        [0,               N*C*H*W)           y
        [N*C*H*W,         N*C*H*W + N*G)     mean    （(N, G) 行主序展平）
        [N*C*H*W + N*G,   N*C*H*W + 2*N*G)   inv_var （(N, G) 行主序展平）
    输出恒为 float32：y 段不做低精度量化；fp16/bf16 仅是输入量化档（输入
    按该 dtype 提供后 cast 成 float32 参与全程计算）。

    输入输出规格：
        x      float32 / float16 / bfloat16 (N, C, H, W)，contiguous，
               N/C/H/W >= 1
        scale  同 x dtype (C,)，contiguous
        bias   同 x dtype (C,)，contiguous
        out    一维 float32，长度 N*C*H*W + 2*N*G（三段打包协议如上）
        输入为常规 randn 量级（无 inf/NaN）；评测器会把全部输入 cast 成
        fp32 传入（fp16/bf16 -> fp32 无损）。

    实现约束（违规判负）：
        - 核心计算（组内 mean/有偏 var 归约、rsqrt、逐通道仿射、Swish）
          必须在提交文件内以 HIP kernel 完成。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 统计、rsqrt、仿射与 Swish 全程 float32 中间计算；输出三段均为
          float32。
        - forward 返回单张量 out（评测器按单输出比较 shape/dtype/allclose）。

    禁用列表：torch.nn.GroupNorm / torch.nn.functional.group_norm /
    F.group_norm / torch.group_norm / native_group_norm / miopen / hipdnn。
    目标硬件：海光 DCU（gfx936/gfx938），实现语言 HIP（经 hipcc 编译；在 CUDA_CU_SRC / CUDA_CPP_SRC 锚点区编写，设备侧必须 include <hip/hip_runtime.h>，本机无 <cuda_runtime.h>）。
    """

    def __init__(self, variant: str = "groupnorm", groups: int = 2, eps: float = 1e-5):
        super().__init__()
        assert variant in ("groupnorm", "groupnorm_swish"), "variant 必须是 groupnorm / groupnorm_swish 之一"
        assert int(groups) >= 1, "groups 需为正整数"
        assert eps > 0, "eps 需为正"
        self.variant = variant
        self.groups = int(groups)
        self.eps = float(eps)

    def forward(self, x, scale, bias):
        n, c, h, w = x.shape
        g = self.groups
        rep = c // g
        assert 1 <= g <= c and c % g == 0, "groups 需整除通道数且 1 <= groups <= C"

        # 组内统计：按 (N, G, D) 沿 D 归约，float32 中间计算（有偏均值/有偏方差）
        xf = x.to(torch.float32)
        xg = xf.reshape(n, g, -1)
        mean = xg.mean(dim=-1)                                   # (N, G)
        var = (xg - mean.unsqueeze(-1)).pow(2).mean(dim=-1)      # (N, G) 有偏方差
        inv_var = torch.rsqrt(var + self.eps)                    # (N, G)

        # 组统计广播回通道：通道 c 属于组 c // rep
        mean_c = mean.repeat_interleave(rep, dim=1).reshape(n, c, 1, 1)
        inv_c = inv_var.repeat_interleave(rep, dim=1).reshape(n, c, 1, 1)
        sf = scale.to(torch.float32).reshape(1, c, 1, 1)
        bf = bias.to(torch.float32).reshape(1, c, 1, 1)
        gn = (xf - mean_c) * inv_c * sf + bf

        # variant 分流：groupnorm_swish 对归一化结果施加 Swish（gn * sigmoid(gn)）
        y = gn * torch.sigmoid(gn) if self.variant == "groupnorm_swish" else gn

        # 单张量输出：y | mean | inv_var 三段一维拼接（全 float32）
        return torch.cat([y.reshape(-1), mean.reshape(-1), inv_var.reshape(-1)])


def get_init_inputs():
    return ["groupnorm", 2, 1e-5]   # variant, groups, eps


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性。生成规则与 make_inputs 一致
    # （消费顺序 x -> scale -> bias，此处消费全局 RNG）
    n, c, h, w = 2, 16, 512, 512
    x = torch.randn(n, c, h, w)
    scale = torch.randn(c)
    bias = torch.randn(c)
    return [x, scale, bias]


def make_inputs(batch: int, channels: int, height: int, width: int,
                groups: int, variant: str = "groupnorm",
                dtype: str = "float32", eps: float = 1e-5, seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：x -> scale -> bias（均 torch.randn 后 cast 到 dtype），
    全部来自同一 torch.Generator(seed)，同 seed 下逐位可复现。
    variant / groups / eps 同时是构造 Model 的 init 超参，此处仅做合法性
    校验；输出打包协议恒为 float32，dtype 仅决定输入的量化档。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert variant in ("groupnorm", "groupnorm_swish"), "variant 必须是 groupnorm / groupnorm_swish 之一"
    assert min(batch, channels, height, width) >= 1, "batch/channels/height/width 均需 >= 1"
    assert 1 <= groups <= channels and channels % groups == 0, "groups 需整除 channels 且 1 <= groups <= channels"
    assert eps > 0, "eps 需为正"
    x = torch.randn(batch, channels, height, width, generator=gen).to(dt)
    scale = torch.randn(channels, generator=gen).to(dt)
    bias = torch.randn(channels, generator=gen).to(dt)
    return x, scale, bias
