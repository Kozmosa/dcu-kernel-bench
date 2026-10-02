# 2005_moe_activation —— MoE FFN 激活变体族（门控 SwiGLU 变体 × chunked /
# interleaved 布局 × 非 gated 逐元素激活）的 model_class（KernelBench 兼容）
# 题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=2, problem_id=2005, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """MoE FFN 激活变体族：gated 门控激活（SwiGLU 家族变体）与非 gated
    逐元素激活，由 __init__ 的 activation / alpha / limit 三参数分派。

    全部计算在 float32 中进行，输出 cast 回输入 dtype。记
        SiLU(u)      = u * sigmoid(u)
        GELU(u)      = 0.5 * u * (1 + erf(u * 0.7071067811865476))
        GELU_tanh(u) = 0.5 * u * (1 + tanh(0.7978845608028654
                                           * (u + 0.044715 * u^3)))
        ReLU2(u)     = max(u, 0)^2
    sigmoid / erf / tanh 均按 float32 数学函数求值。

    【gated 变体】输入 x 形状 (M, 2N)，输出 (M, N)。两种取数布局：
        chunked（默认）   : gate = x[:, :N]，up = x[:, N:]
        interleaved       : gate = x[:, 0::2]（偶数列），up = x[:, 1::2]（奇数列）
    记钳位量 gate' = min(gate, limit)（仅上限钳位）、
    up' = clamp(up, -limit, limit)（对称钳位），各变体输出为：
        activation="silu"，alpha=None、limit=None：
            out = SiLU(gate) * up                        （标准 SwiGLU）
        activation="silu"，alpha=None、limit=L：
            out = SiLU(gate') * up'                       （钳位 SwiGLU）
        activation="silu"，alpha=A、limit=L（A 与 L 必须同时给出）：
            out = gate' * sigmoid(A * gate') * (up' + 1)  （缩放钳位门控）
        activation="gelu"（不接受 alpha / limit）：
            out = GELU(gate) * up
        activation="gelu_tanh"（不接受 alpha / limit）：
            out = GELU_tanh(gate) * up
        activation="swiglustep"（limit 缺省 7.0，不接受 alpha）：
            out = clamp(SiLU(gate), max=limit) * up'      （先 SiLU 后上限钳位）
        activation="swiglu_interleaved"（interleaved 布局；alpha 缺省 1.702、
        limit 缺省 7.0）：
            out = gate' * sigmoid(alpha * gate') * (up' + 1)

    【非 gated 变体】输入 x 形状 (M, N)，输出 (M, N)，逐元素激活，忽略
    alpha / limit：
        activation="silu_no_mul"      : out = SiLU(x)
        activation="gelu_no_mul"      : out = GELU(x)
        activation="gelu_tanh_no_mul" : out = GELU_tanh(x)
        activation="relu2"            : out = ReLU2(x)

    边界行为：gate 的钳位只设上限（负向不钳），up 的钳位对称；钳位在
    激活之前施加，唯独 swiglustep 是先算 SiLU 再对结果上限钳位。非法参数
    组合（未知变体名；gelu / gelu_tanh 带 alpha 或 limit；swiglustep 带
    alpha；silu 给 alpha 不给 limit）在构造期直接拒绝。

    输入输出规格：
        x    float16 / bfloat16 (M, n)，contiguous，M >= 1；gated 变体要求
             n 为偶数（N = n/2 无对齐约束，允许非 2 次幂、小于常见块大小）；
             非 gated 变体 n >= 1 任意。评测器会把输入 cast 成 fp32 传入，
             fp16/bf16 -> fp32 无损。
        out  与 x 同 dtype；gated 变体形状 (M, n/2)，非 gated 变体 (M, n)。
        输入值域为常规 randn 量级（无 inf/NaN）。

    实现约束（违规判负）：
        - 核心计算（chunked / interleaved 取数、钳位、激活、门控乘）必须在
          提交文件内以 Triton kernel 完成；中间计算一律 float32。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 中间累加用 float32；输出 cast 回输入 dtype。

    ④ 禁用列表与目标硬件：
    - 禁止调用 torch.nn.functional.silu / torch.nn.functional.gelu /
      torch.nn.SiLU / torch.nn.GELU / torch.sigmoid / torch.erf /
      torch.tanh / torch.relu / torch.clamp 等 ATen 捷径直接完成本题
      计算；禁用清单以 task.yaml forbidden 列表为准（含来源算子库与
      其它闭源加速库，终审静态审计对 docstring / 注释之外的代码做子串
      匹配）；
    - 目标硬件：海光 DCU（gfx936），实现语言 Triton。
    """

    def __init__(self, activation: str = "silu", alpha=None, limit=None):
        super().__init__()
        gated = ("silu", "gelu", "gelu_tanh", "swiglustep", "swiglu_interleaved")
        nomul = ("silu_no_mul", "gelu_no_mul", "gelu_tanh_no_mul", "relu2")
        name = str(activation).lower()
        assert name in gated or name in nomul, (
            "activation 必须是 " + "/".join(gated + nomul) + " 之一")
        # 参数组合合法性（与变体定义一致，非法组合构造期拒绝）
        if name in ("gelu", "gelu_tanh"):
            assert alpha is None and limit is None, "gelu / gelu_tanh 不接受 alpha / limit"
        if name == "swiglustep":
            assert alpha is None, "swiglustep 不接受 alpha"
        if name == "silu" and alpha is not None:
            assert limit is not None, "silu 设置 alpha 时必须同时设置 limit"
        # 缺省值填充（仅以下两个变体带缺省）
        if name == "swiglustep" and limit is None:
            limit = 7.0
        if name == "swiglu_interleaved":
            alpha = 1.702 if alpha is None else alpha
            limit = 7.0 if limit is None else limit
        self.activation = name
        self.is_gated = name in gated
        self.alpha = None if alpha is None else float(alpha)
        self.limit = None if limit is None else float(limit)

    def forward(self, x):
        # 评测器把全部输入 cast 成 fp32；统一在 float32 中计算，输出 cast 回输入 dtype
        xf = x.to(torch.float32)
        assert xf.dim() == 2, "输入必须是 (M, n) 的 2D 张量"
        if not self.is_gated:
            # 非 gated：逐元素激活，形状不变（忽略 alpha / limit）
            if self.activation == "silu_no_mul":
                out = xf * torch.sigmoid(xf)
            elif self.activation == "gelu_no_mul":
                out = 0.5 * xf * (1.0 + torch.erf(xf * 0.7071067811865476))
            elif self.activation == "gelu_tanh_no_mul":
                inner = 0.7978845608028654 * (xf + 0.044715 * xf * xf * xf)
                out = 0.5 * xf * (1.0 + torch.tanh(inner))
            else:   # relu2
                act = torch.clamp(xf, min=0.0)
                out = act * act
            return out.to(x.dtype)

        assert xf.shape[-1] % 2 == 0, "gated 变体要求最后一维 n 为偶数"
        half = xf.shape[-1] // 2
        interleaved = self.activation == "swiglu_interleaved"
        gate = xf[:, 0::2] if interleaved else xf[:, :half]   # interleaved / chunked 布局
        up = xf[:, 1::2] if interleaved else xf[:, half:]
        if self.activation == "swiglustep":
            # 先 SiLU 后上限钳位：clamp(SiLU(gate), max=limit) * up'
            silu_g = gate * torch.sigmoid(gate)
            out = silu_g.clamp(max=self.limit) * up.clamp(min=-self.limit, max=self.limit)
        if self.activation == "silu" and self.alpha is not None:
            # 缩放钳位门控：gate' * sigmoid(alpha*gate') * (up' + 1)
            gate_p = gate.clamp(max=self.limit)
            up_p = up.clamp(min=-self.limit, max=self.limit)
            out = gate_p * torch.sigmoid(self.alpha * gate_p) * (up_p + 1.0)
        if self.activation == "silu" and self.alpha is None and self.limit is not None:
            # 钳位 SwiGLU：SiLU(gate') * up'
            gate_p = gate.clamp(max=self.limit)
            up_p = up.clamp(min=-self.limit, max=self.limit)
            out = gate_p * torch.sigmoid(gate_p) * up_p
        if self.activation == "silu" and self.alpha is None and self.limit is None:
            out = gate * torch.sigmoid(gate) * up             # 标准 SwiGLU
        if self.activation == "gelu":
            out = 0.5 * gate * (1.0 + torch.erf(gate * 0.7071067811865476)) * up
        if self.activation == "gelu_tanh":
            inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
            out = 0.5 * gate * (1.0 + torch.tanh(inner)) * up
        if self.activation == "swiglu_interleaved":
            # gate' * sigmoid(alpha*gate') * (up' + 1)
            gate_p = gate.clamp(max=self.limit)
            up_p = up.clamp(min=-self.limit, max=self.limit)
            out = gate_p * torch.sigmoid(self.alpha * gate_p) * (up_p + 1.0)
        return out.to(x.dtype)


def get_init_inputs():
    return ["silu", 1.702, 7.0]   # activation / alpha / limit -> 缩放钳位门控变体


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性（M=128，2N=4096 即 N=2048）
    m, n = 128, 4096
    x = torch.randn(m, n).to(torch.bfloat16)
    return [x]


def make_inputs(m: int, n: int, activation: str, alpha=None, limit=None,
                dtype: str = "float16", seed: int = 0, scale: float = 1.0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：x = torch.randn(m, n) * scale 后 cast 到 dtype，全部来自同一
    torch.Generator(seed)，同 seed 下逐位可复现；activation / alpha / limit 与
    case 字段一一对应，原样传给 Model 构造。scale 仅控制输入幅度以覆盖钳位
    边界，不参与算子语义。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert m >= 1 and n >= 1, "需 M >= 1 且 n >= 1"
    x = (torch.randn(m, n, generator=gen) * float(scale)).to(dt)
    return (x,)
