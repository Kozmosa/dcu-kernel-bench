# 6001_fused_adamw —— 融合 AdamW 优化器家族（标准 / Transformer 调度）的
# model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=2, problem_id=6001, backend=cuda（HIP）。

import torch
import torch.nn as nn


class Model(nn.Module):
    """融合 AdamW 优化器家族：标准 AdamW 与 Transformer 调度 AdamW 的单步
    functional（out-of-place）参数更新。

    ① 算子语义：__init__ 超参 lr、beta1、beta2、eps、weight_decay 为构造
    常量，variant 选择两条更新路径之一。设 step 为 0 维 float32 标量张量
    （当前迭代步数），四张状态 params / grads / exp_avg / exp_avg_sq 同形
    逐元素对齐，更新全程在 float32 中计算（输入先提升到 float32）：
        1) 解耦权重衰减（AdamW 式 decoupled weight decay，两种 variant 都
           做，等价于 p = params - lr * weight_decay * params，不是 L2 正则）：
              p = params * (1 - lr * weight_decay)
        2) 一阶矩：m = beta1 * exp_avg + (1 - beta1) * grads
        3) 二阶矩：v = beta2 * exp_avg_sq + (1 - beta2) * grads^2
        4) variant="adamw"（标准 AdamW，bias correction 开启）：
              bc1 = 1 - beta1^step；bc2 = 1 - beta2^step
              out = p - lr * (m / bc1) / (sqrt(v / bc2) + eps)
           variant="transformer_adamw"（Transformer 调度口径，bias
           correction 关闭，两 variant 唯一差异点）：
              out = p - lr * m / (sqrt(v) + eps)
    两种 variant 除 bias correction 外逐行同式；eps 加在 sqrt 之外
    （denom = sqrt(...) + eps）；幂 beta^step 按 float32 标量幂运算。
    边界行为：step=1 时 bc1 = 1 - beta1、bc2 = 1 - beta2 取最小值（标准
    AdamW 路径放大最明显）；step 很大时两种 variant 数值收敛一致；
    exp_avg_sq 恒非负（二阶矩约束，生成域 [0, 0.5)），denom >= eps 保证
    不除零；本算子无归约、无索引，逐元素映射。

    ② 输入输出规格：
      - params / grads / exp_avg / exp_avg_sq：float32 或 float16 张量，
        shape 全同（ndim 1–4，numel >= 1，含非 2 的幂与边界规模），contiguous；
      - step：0 维 float32 标量张量（numel == 1），数值为正整数语义步数
        （>= 1）；
      - 返回：更新后参数，与 params 同形、同 dtype 的新张量（单张量输出，
        out-of-place，不得原地改写任何输入，也不返回 exp_avg / exp_avg_sq
        等状态）。

    ③ 实现约束（违规判负）：
      - 核心计算（解耦权重衰减、一阶/二阶矩更新、bias correction 幂运算、
        sqrt、逐元素更新）必须在提交文件内以 HIP kernel 完成；
      - out-of-place：不得原地修改 params / grads / exp_avg / exp_avg_sq；
      - ModelNew 的 __init__ 与 forward 签名不可更改；
      - 中间计算用 float32（fp16 输入亦先提升到 float32），最终结果 cast
        回 params 的 dtype；
      - step 以 0 维张量传入，kernel 内作为标量参与幂运算，不得物化为与
        params 同形的临时大张量。

    ④ 禁用列表与目标硬件：
      - 禁止调用 torch.optim 全族（torch.optim.AdamW / Adam / SGD / ... 及
        其 _functional / _multi_tensor 内部路径）；
      - 禁止调用任何 fused 优化器库与闭源库入口：apex / deepspeed /
        bitsandbytes / fused_adam 系列入口 / hipdnn / miopen；
      - 禁止以 ATen 捷径直接完成本题计算：torch.addcdiv / torch.addcmul
        （含 .addcdiv / .addcdiv_ / .addcmul / .addcmul_ 张量方法与原地
        形式，一次调用即完成参数更新或动量更新）；torch._foreach_addcdiv /
        torch._foreach_addcmul 多张量原语同样禁止（含原地 _ 形式，单调用
        即可完成核心更新步）（终审静态审计对
        docstring/注释之外的代码做子串匹配）；
      - 目标硬件：海光 DCU（gfx936/gfx938），实现语言 HIP（经 hipcc 编译；在 CUDA_CU_SRC / CUDA_CPP_SRC 锚点区编写，设备侧必须 include <hip/hip_runtime.h>，本机无 <cuda_runtime.h>）。
    """

    def __init__(self, variant: str = "adamw", lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__()
        assert variant in ("adamw", "transformer_adamw"), \
            "variant 必须是 adamw / transformer_adamw 之一"
        self.variant = variant
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

    def forward(self, params, grads, exp_avg, exp_avg_sq, step):
        # 输入提升到 float32；step 为 0 维标量张量（评测器可能透传数值张量）
        p = params.to(torch.float32)
        g = grads.to(torch.float32)
        m = exp_avg.to(torch.float32)
        v = exp_avg_sq.to(torch.float32)
        s = step.to(torch.float32) if torch.is_tensor(step) \
            else torch.tensor(float(step))

        # 1) 解耦权重衰减（AdamW 式，非 L2）
        p = p * (1.0 - self.lr * self.weight_decay)
        # 2) 一阶矩
        m = self.beta1 * m + (1.0 - self.beta1) * g
        # 3) 二阶矩
        v = self.beta2 * v + (1.0 - self.beta2) * g * g

        if self.variant == "adamw":
            # 标准 AdamW：bias correction 开启
            bc1 = 1.0 - torch.pow(self.beta1, s)
            bc2 = 1.0 - torch.pow(self.beta2, s)
            denom = torch.sqrt(v / bc2) + self.eps
            update = (m / bc1) / denom
        else:
            # transformer_adamw：bias correction 关闭（唯一差异点）
            denom = torch.sqrt(v) + self.eps
            update = m / denom

        return (p - self.lr * update).to(params.dtype)


def get_init_inputs():
    # variant, lr, beta1, beta2, eps, weight_decay（torch.optim.AdamW 缺省族）
    return ["adamw", 1e-3, 0.9, 0.999, 1e-8, 0.01]


def get_inputs():
    # 固定 shape 族（与 public_cases.json 一致）；随机部分消费全局 RNG——
    # 评测器在 set_seed 后调用本函数，多轮 correctness trial 因此获得输入
    # 多样性。消费顺序固定：params -> grads -> exp_avg -> exp_avg_sq。
    shape = (8, 3, 224, 224)
    params = torch.rand(*shape)
    grads = torch.randn(*shape)
    exp_avg = torch.rand(*shape) * 0.5
    exp_avg_sq = torch.rand(*shape) * 0.5
    step = torch.tensor(10.0)
    return [params, grads, exp_avg, exp_avg_sq, step]


def make_inputs(shape, dtype="float32", step=10.0, variant="adamw",
                lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                weight_decay=0.01, seed=0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到
    设备）。

    消费顺序固定：params -> grads -> exp_avg -> exp_avg_sq（全部来自同一
    torch.Generator(seed)，同 seed 下逐位可复现）；step 为确定性标量张量，
    不消费 RNG。variant / lr / beta1 / beta2 / eps / weight_decay 是构造
    Model 的 init 超参，此处仅做合法性校验（生成域与超参无关）。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    shape = tuple(shape)
    assert variant in ("adamw", "transformer_adamw"), \
        "variant 必须是 adamw / transformer_adamw 之一"
    assert len(shape) >= 1 and min(shape) >= 1, "需 ndim 1-4 且每维 >= 1"
    params = torch.rand(*shape, generator=gen).to(dt)
    grads = torch.randn(*shape, generator=gen).to(dt)
    exp_avg = (torch.rand(*shape, generator=gen) * 0.5).to(dt)
    exp_avg_sq = (torch.rand(*shape, generator=gen) * 0.5).to(dt)
    step_t = torch.tensor(float(step), dtype=torch.float32)
    return params, grads, exp_avg, exp_avg_sq, step_t
