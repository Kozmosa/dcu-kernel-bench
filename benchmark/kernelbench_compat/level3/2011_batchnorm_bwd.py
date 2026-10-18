# 2011_batchnorm_bwd —— BatchNorm 训练反向（数据梯度 dx / 参数梯度
# dgamma+dbeta）的 KernelBench 兼容题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem-id=2011, backend=cuda（HIP）。

import torch
import torch.nn as nn


class Model(nn.Module):
    """BatchNorm 训练反向：给定上游梯度 dy、训练前向的输入 x 与保存的统计量
    （saved_mean / saved_inv_variance）及缩放参数 scale，计算归一化前向
        y = scale * xhat + bias，  xhat = (x - saved_mean) * saved_inv_variance
    的梯度。__init__ 参数 variant 选择返回哪些梯度：
        "dx"           ：只返回数据梯度 dx，形状 (N, C, H, W)，同 x dtype；
        "dgamma_dbeta" ：只返回参数梯度，dgamma、dbeta 各为 (1, C, 1, 1)，
                         按"一维拼接打包协议"输出单个 1-D 张量 (2*C,)：
                         前 C 个元素为 dgamma（按行主序平铺），后 C 个为 dbeta。
    两种 variant 的 forward 签名相同；返回的一律是单张量（评测器按单输出
    比较）。数学定义（全程 float32 中间计算，m = N*H*W 为每通道元素数）：
        xhat         = (x - saved_mean) * saved_inv_variance
        dgamma[c]    = sum_{n,h,w} dy * xhat          （对每通道求和）
        dbeta[c]     = sum_{n,h,w} dy                 （对每通道求和）
        dx           = (scale * saved_inv_variance / m)
                       * (m*dy - dbeta - xhat * dgamma)
    saved_mean / saved_inv_variance 按"给定值原样消费"：它们是训练前向保存的
    统计量（inv_variance = 1/sqrt(有偏方差 + eps)，eps 已在前向时吸收，反向
    数学本身不含 eps），可以是与 x 的真实统计量无关的任意正数——必须直接使用
    传入值，禁止由 x 重新计算统计量。saved_inv_variance 恒为正。

    输入输出规格（NCHW，contiguous）：
        dy                 float32 / float16 / bfloat16 (N, C, H, W)
        x                  同 dy dtype (N, C, H, W)
        scale              同 dy dtype (1, C, 1, 1)
        saved_mean         float32 (1, C, 1, 1)（统计量恒为 float32，与输入
                           dtype 无关）
        saved_inv_variance float32 (1, C, 1, 1)，元素恒为正
        out                单张量：dx 时 (N, C, H, W)、同 dy dtype；
                           dgamma_dbeta 时 (2*C,)、同 dy dtype（前 C 个
                           dgamma、后 C 个 dbeta，均以 float32 计算后 cast）
        评测器会把全部输入 cast 成 fp32 传入（fp16/bf16 -> fp32 无损）。

    实现约束（违规判负）：
        - 核心计算（xhat 重构、两路通道归约、dx 的逐元素链式乘加）必须在提交
          文件内以 HIP kernel 完成；任意规模的通道归约（m = N*H*W 可大于
          单 block 容量）都需掩码 / 分块归约正确处理，通道内元素一个不漏。
        - 禁止借助 torch.autograd / autograd.grad / .backward() 反传求梯度，
          禁止调用 batch_norm 原语族与任何闭源库原语（MIOpen / hipDNN 等）。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 归约与中间计算在 float32 中进行；输出 cast 回输入 dtype；
          dgamma_dbeta 的打包顺序固定为 dgamma 段在前、dbeta 段在后。

    边界行为：N、C、H、W 均可取 1（m 可为 1，此时 dx = scale * inv_var *
    (dy - dy*xhat^2)），允许非 2 的幂与不对齐维度；统计量形状恒为
    (1, C, 1, 1)；输入值域为常规 randn 量级（无 inf/NaN）。
    禁用列表：aiter / torch.batch_norm / torch.native_batch_norm /
    torch.nn.functional.batch_norm / F.batch_norm / torch.nn.BatchNorm1d /
    torch.nn.BatchNorm2d / torch.nn.BatchNorm3d /
    torch.ops.aten.batch_norm / torch.ops.aten.native_batch_norm /
    autograd / .backward(。
    目标硬件：海光 DCU（gfx936/gfx938），实现语言 HIP（经 hipcc 编译；在 CUDA_CU_SRC / CUDA_CPP_SRC 锚点区编写，设备侧必须 include <hip/hip_runtime.h>，本机无 <cuda_runtime.h>）。
    """

    def __init__(self, variant: str = "dx"):
        super().__init__()
        assert variant in ("dx", "dgamma_dbeta"), "variant 必须是 dx / dgamma_dbeta 之一"
        self.variant = variant

    def forward(self, dy, x, scale, saved_mean, saved_inv_variance):
        # 全程 float32 中间计算；统计量按给定值原样消费，不由 x 重算
        dyf = dy.to(torch.float32)
        xf = x.to(torch.float32)
        mean = saved_mean.to(torch.float32)
        inv_var = saved_inv_variance.to(torch.float32)

        xhat = (xf - mean) * inv_var                    # (N, C, H, W)
        dgamma_c = (dyf * xhat).sum(dim=(0, 2, 3))      # (C,)
        dbeta_c = dyf.sum(dim=(0, 2, 3))                # (C,)

        if self.variant == "dgamma_dbeta":
            # 单张量打包：dgamma 段在前、dbeta 段在后，各按行主序平铺
            return torch.cat([dgamma_c.to(x.dtype).reshape(-1),
                              dbeta_c.to(x.dtype).reshape(-1)])

        m = x.shape[0] * x.shape[2] * x.shape[3]        # 每通道元素数
        scalef = scale.to(torch.float32)
        dx = (scalef * inv_var / m) * (m * dyf - dbeta_c.reshape_as(scalef)
                                       - xhat * dgamma_c.reshape_as(scalef))
        return dx.to(x.dtype)


def get_init_inputs():
    return ["dx"]   # variant


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG（顺序 dy -> x -> scale）——评测器在
    # set_seed 后调用本函数，多轮 correctness trial 因此获得输入多样性。
    # saved 统计量按训练快照口径由 x 确定性导出（不消费 RNG）：
    # mean = x 的通道均值，inv_variance = 1/sqrt(有偏方差 + 1e-5)。
    n, c, h, w = 4, 16, 56, 56
    dy = torch.randn(n, c, h, w).to(torch.float32)
    x = torch.randn(n, c, h, w).to(torch.float32)
    scale = torch.randn(1, c, 1, 1).to(torch.float32)
    xf = x.to(torch.float32)
    mean = xf.mean(dim=(0, 2, 3), keepdim=True)
    var = xf.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
    inv_var = torch.rsqrt(var + 1e-5)
    return [dy, x, scale, mean, inv_var]


def make_inputs(n: int, c: int, h: int, w: int, variant: str = "dx",
                dtype: str = "float32", seed: int = 0, eps: float = 1e-5,
                stats: str = "derived"):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    随机部分消费顺序固定：dy -> x -> scale（均 torch.randn 后 cast 到 dtype），
    全部来自同一 torch.Generator(seed)，同 seed 下逐位可复现。
    saved 统计量（float32，不 cast 到 dtype）由 stats 口径决定：
        "derived"：训练快照口径（缺省）——按 2010 前向同口径由 x 导出，
                   mean = x.float() 的通道均值，inv_variance =
                   rsqrt(x.float() 的通道有偏方差 + eps)；不消费 RNG；
        "random" ：样例口径——统计量为独立随机值，RNG 继续消费
                   saved_mean = rand(1,C,1,1)、saved_inv_variance =
                   rand(1,C,1,1) + 1e-3（保证为正），此时统计量与 x 无关，
                   用于钉死"统计量按给定值消费、不得由 x 重算"的语义。
    eps 仅在 derived 口径下参与统计量生成，反向数学本身不含 eps。
    variant 同时是构造 Model 的 init 超参，此处仅做合法性校验。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert variant in ("dx", "dgamma_dbeta"), "variant 必须是 dx / dgamma_dbeta 之一"
    assert stats in ("derived", "random"), "stats 必须是 derived / random 之一"
    assert n >= 1 and c >= 1 and h >= 1 and w >= 1, "需 N/C/H/W 均 >= 1"

    dy = torch.randn(n, c, h, w, generator=gen).to(dt)
    x = torch.randn(n, c, h, w, generator=gen).to(dt)
    scale = torch.randn(1, c, 1, 1, generator=gen).to(dt)

    if stats == "random":
        saved_mean = torch.rand(1, c, 1, 1, generator=gen)
        saved_inv_variance = torch.rand(1, c, 1, 1, generator=gen) + 1.0e-3
    else:
        xf = x.to(torch.float32)
        saved_mean = xf.mean(dim=(0, 2, 3), keepdim=True)
        var = xf.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        saved_inv_variance = torch.rsqrt(var + float(eps))

    return dy, x, scale, saved_mean, saved_inv_variance
