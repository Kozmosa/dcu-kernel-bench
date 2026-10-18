# 2013_groupnorm_bwd —— GroupNorm 训练反向（dx / dgamma / dbeta 全梯度）的
# model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem-id=2013, backend=cuda（HIP）。

import torch
import torch.nn as nn


class Model(nn.Module):
    """GroupNorm 训练反向：给定上游梯度 dy、前向输入 x、缩放参数 scale 与
    训练前向保存的逐组统计量 mean / inv_var，计算 GroupNorm 前向
        y[n,c,h,w] = xhat[n,c,h,w] * scale[c] + bias[c]，
        xhat[n,c,h,w] = (x[n,c,h,w] - mean_ng) * inv_var_ng，
        组号 g = c // (C//G)
    对 x / scale / bias 的完整梯度。__init__ 参数 groups = G（整除 C，
    1 <= G <= C）；eps 仅在前向统计量口径中出现（inv_var 已含 eps 的保护，
    反向数学本身不含 eps）。mean / inv_var 为一维 float32、长度 N*G，是
    (N, G) 的行主序展平，恒为前向保存值：按给定值原样消费，禁止由 x 重新
    计算。dbeta 梯度与 bias 的取值无关，故 forward 签名不含 bias。
    数学定义（全程 float32 中间计算；组 (n, g) 含 rep = C//G 个通道、
    D = rep*H*W 个元素）：
        xhat_nc   = (x_nc - mean_ng) * inv_var_ng
        dgamma[c] = sum_{n,h,w} dy * xhat                      （逐通道）
        dbeta[c]  = sum_{n,h,w} dy                             （逐通道）
        记 S_ng   = sum_{c'∈g,h,w} dy * scale_{c'}
           T_ng   = sum_{c'∈g,h,w} dy * xhat * scale_{c'}
        dx_nc     = (inv_var_ng / D) * (D * dy_nc * scale_c - S_ng
                                         - xhat_nc * T_ng)
    （S_ng / T_ng 为对组 (n,g) 内全部通道与空间位置的归约，即数据梯度中的
    完整链式项；逐通道缩放 scale_c 已并入两路归约。）

    单张量输出协议（forward 必须返回单个张量，评测器按单输出比较）：
    返回一维 float32 打包张量，长度 N*C*H*W + 2*C，三段按序紧排（各段
    float32、reshape(-1) 后 torch.cat）：
        [0, N*C*H*W)                   dx      （(N, C, H, W) 行主序展平）
        [N*C*H*W, N*C*H*W + C)         dgamma  （(C,)）
        [N*C*H*W + C, N*C*H*W + 2*C)   dbeta   （(C,)）
    输出恒为 float32：fp16/bf16 仅是输入量化档（输入按该 dtype 提供后
    cast 成 float32 参与全程计算），梯度不做低精度量化。

    输入输出规格：
        x       float32 / float16 / bfloat16 (N, C, H, W)，NCHW contiguous，
                N/C/H/W >= 1
        dy      同 x dtype (N, C, H, W)，contiguous（上游梯度 dL/dy）
        scale   同 x dtype (C,)，contiguous
        mean    float32 (N*G,)，(N, G) 行主序展平（前向保存的逐组均值）
        inv_var float32 (N*G,)，(N, G) 行主序展平，元素恒为正
        out     一维 float32，长度 N*C*H*W + 2*C（三段打包协议如上）
        输入为常规 randn 量级（无 inf/NaN）；评测器会把全部输入 cast 成
        fp32 传入（fp16/bf16 -> fp32 无损）。

    实现约束（违规判负）：
        - 核心计算（xhat 重构、组内两路归约 S/T、逐通道参数归约
          dgamma/dbeta、dx 的逐元素链式乘加）必须在提交文件内以 HIP
          kernel 完成；任意规模的组内归约（D 可远大于单 block 容量）都需
          掩码 / 分块归约，元素一个不漏。
        - 禁止借助 torch.autograd / autograd.grad / .backward() 反传求梯度，
          禁止调用 group_norm 原语族与任何闭源库原语（MIOpen / hipDNN 等）。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 归约与中间计算全程 float32；输出三段均为 float32，按上述分段
          协议紧排。

    边界行为：N/C/H/W 均可取 1（D 可为 1，此时 xhat = 0、dx = 0、
    dgamma = 0、dbeta = dy）；G=1 为全通道归一化反向、G=C 为逐通道归一化
    反向，均合法；允许非 2 的幂与不对齐维度；统计量形状恒为 (N*G,)。
    禁用列表：aiter / torch.nn.GroupNorm / nn.GroupNorm /
    torch.nn.functional.group_norm / F.group_norm / torch.group_norm /
    native_group_norm / torch.ops.aten.native_group_norm /
    torch.ops.aten.group_norm / autograd / .backward( / miopen / hipdnn。
    目标硬件：海光 DCU（gfx936/gfx938），实现语言 HIP（经 hipcc 编译；在 CUDA_CU_SRC / CUDA_CPP_SRC 锚点区编写，设备侧必须 include <hip/hip_runtime.h>，本机无 <cuda_runtime.h>）。
    """

    def __init__(self, groups: int = 2, eps: float = 1e-5):
        super().__init__()
        assert int(groups) >= 1, "groups 需为正整数"
        assert eps > 0, "eps 需为正"
        self.groups = int(groups)
        self.eps = float(eps)

    def forward(self, x, dy, scale, mean, inv_var):
        n, c, h, w = x.shape
        g = self.groups
        rep = c // g
        assert 1 <= g <= c and c % g == 0, "groups 需整除通道数且 1 <= groups <= C"
        assert mean.numel() == n * g and inv_var.numel() == n * g, "mean/inv_var 需为长度 N*G 的一维统计量"

        # 语义金标：对"2012 同口径 GroupNorm 前向"用 autograd 反传。在线评测
        # 器在 no_grad 上下文中调用 reference forward，故此段显式 enable_grad。
        with torch.enable_grad():
            xf = x.to(torch.float32).detach().requires_grad_(True)
            sf = scale.to(torch.float32).detach().requires_grad_(True)
            bf = torch.zeros(c, dtype=torch.float32, device=xf.device,
                             requires_grad=True)  # dbeta 与 bias 取值无关，取 0

            # 2012 同口径前向（组内有偏统计 -> rsqrt(有偏方差+eps) -> 通道广播
            # -> 逐通道仿射），统计量的导数结构取自该前向图
            xg = xf.reshape(n, g, -1)
            mean_t = xg.mean(dim=-1)                                  # (N, G)
            var_t = (xg - mean_t.unsqueeze(-1)).pow(2).mean(dim=-1)   # (N, G) 有偏方差
            inv_t = torch.rsqrt(var_t + self.eps)

            # 恒等拼接（值 / 导数分离）：mean_eff、inv_eff 的取值恒等于题面给定
            # 的保存统计量（t - t.detach() 恒为 0），梯度则经由上方前向图回传
            # ——即"在给定保存统计量处对 2012 前向反传"。评测用例的统计量均按
            # 该前向口径由 x 确定性导出，此时与保存统计量反向的闭式完整链式
            # 公式（题面数学定义）逐式一致。
            mean_eff = mean.to(torch.float32).reshape(n, g) + mean_t - mean_t.detach()
            inv_eff = inv_var.to(torch.float32).reshape(n, g) + inv_t - inv_t.detach()
            mean_c = mean_eff.repeat_interleave(rep, dim=1).reshape(n, c, 1, 1)
            inv_c = inv_eff.repeat_interleave(rep, dim=1).reshape(n, c, 1, 1)
            y = (xf - mean_c) * inv_c * sf.reshape(1, c, 1, 1) + bf.reshape(1, c, 1, 1)

            # 上游梯度 dy 注入为 grad_outputs：L = sum(y * dy) 即 dL/dy = dy
            dx, dgamma, dbeta = torch.autograd.grad(
                (y * dy.to(torch.float32)).sum(), (xf, sf, bf))

        # 单张量输出：dx | dgamma | dbeta 三段一维拼接（全 float32，不回 cast）
        return torch.cat([dx.reshape(-1), dgamma.reshape(-1), dbeta.reshape(-1)])


def get_init_inputs():
    return [2, 1e-5]   # groups, eps


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG（顺序 x -> dy -> scale）——评测器在
    # set_seed 后调用本函数，多轮 correctness trial 因此获得输入多样性。
    # 保存统计量按训练快照口径由 x 确定性导出（不消费 RNG、恒 float32）：
    # 逐组有偏均值 + rsqrt(有偏方差 + 1e-5)，(N, G) 行主序展平。
    n, c, h, w, g = 2, 16, 512, 512, 2
    x = torch.randn(n, c, h, w)
    dy = torch.randn(n, c, h, w)
    scale = torch.randn(c)
    xg = x.to(torch.float32).reshape(n, g, -1)
    mean = xg.mean(dim=-1)
    var = (xg - mean.unsqueeze(-1)).pow(2).mean(dim=-1)
    inv_var = torch.rsqrt(var + 1e-5)
    return [x, dy, scale, mean.reshape(-1), inv_var.reshape(-1)]


def make_inputs(batch: int, channels: int, height: int, width: int, groups: int,
                dtype: str = "float32", eps: float = 1e-5, seed: int = 0):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    随机部分消费顺序固定：x -> dy -> scale（均 torch.randn 后 cast 到 dtype），
    全部来自同一 torch.Generator(seed)，同 seed 下逐位可复现。
    mean / inv_var（恒 float32，不 cast 到 dtype）按训练快照口径由量化后的 x
    确定性导出（与 get_inputs 同式：逐组有偏均值 + rsqrt(有偏方差 + eps)），
    不消费 RNG。
    groups / eps 同时是构造 Model 的 init 超参，此处仅做合法性校验；eps 仅
    参与统计量导出，反向数学本身不含 eps。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert min(batch, channels, height, width) >= 1, "batch/channels/height/width 均需 >= 1"
    assert 1 <= groups <= channels and channels % groups == 0, "groups 需整除 channels 且 1 <= groups <= channels"
    assert eps > 0, "eps 需为正"
    x = torch.randn(batch, channels, height, width, generator=gen).to(dt)
    dy = torch.randn(batch, channels, height, width, generator=gen).to(dt)
    scale = torch.randn(channels, generator=gen).to(dt)
    xg = x.to(torch.float32).reshape(batch, groups, -1)
    mean = xg.mean(dim=-1)
    var = (xg - mean.unsqueeze(-1)).pow(2).mean(dim=-1)
    inv_var = torch.rsqrt(var + float(eps))
    return x, dy, scale, mean.reshape(-1), inv_var.reshape(-1)
