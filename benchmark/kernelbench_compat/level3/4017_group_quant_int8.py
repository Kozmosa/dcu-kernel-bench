# 4017_group_quant_int8 — per-token-group int8 动态量化（W8A8 整数 GEMM 的
# 激活前处理）的 model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=4017, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """per-token-group int8 动态量化（W8A8 整数 GEMM 的激活前处理）。

    输入 x 是连续的 2D float 张量 (rows, cols)，要求 cols 能被 group_size 整除。
    按行主序（展平）把 x 切成 G = rows*cols/group_size 个连续分组，每组
    group_size 个元素。对每组 g（中间算术全程 float32）：
        absmax_g = max( max_{i∈g} |x_i|, eps )       # eps 为除零保护
        scale_g  = absmax_g / 127                    # fp32 除法
        q_i      = trunc_int8( clamp( x_i / scale_g, -128, 127 ) )
    trunc_int8 指向零截断后转 int8，量化码字最终落在 [-127, 127]。
    边界行为：absmax_g < eps（含整组全零）时 scale_g = eps/127、组内码字全 0，
    eps 保护必须生效（产出 0 scale 或 NaN 均为错误实现）。

    输入输出规格：
      x    float16/bfloat16，连续 (rows, cols)，cols % group_size == 0；
           评测器会把输入 cast 成 fp32 传入——fp16/bf16 值在 fp32 中精确表示，
           量化语义不变
      out  一维 float32 张量，长度 rows*cols + rows*cols/group_size
           （单张量输出协议，两段打包）：
           out[:rows*cols]  = int8 码字提升为 float32（[-127,127] 内整数）
           out[rows*cols:]  = 各组 scale，行主序（(rows, cols/group_size) 展平）
      __init__ 超参：group_size（int，默认 128）、eps（float，默认 1e-10）。

    实现约束（违规判负）：
      - 核心计算（组内 absmax、scale、量化码字）必须在提交文件内的 Triton
        kernel 中完成；forward 里只允许做输入/输出整理（dtype 转换、两段
        拼接打包等）。禁止调用 aiter / torch.quantize_per_tensor /
        torch.quantize_per_channel。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间算术用 float32；scale 必须按 absmax/127 的 fp32 除法计算（预计算
        倒数再乘会在截断边界把码字移位 1）；码字先 clamp(-128, 127) 再向零
        截断转 int8。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：2D 连续输入、torch.int8 码字、fp32 scale，不含 GEMM 后端计算。
    """

    def __init__(self, group_size: int = 128, eps: float = 1e-10):
        super().__init__()
        self.group_size = int(group_size)
        self.eps = float(eps)

    def forward(self, x):
        gs = self.group_size
        assert x.dim() == 2, "x 必须是 2D 张量"
        assert x.shape[-1] % gs == 0, "最后一维必须能被 group_size 整除"
        assert x.is_contiguous(), "x 必须连续"

        rows, cols = x.shape
        int8_min, int8_max = -128.0, 127.0

        # 展平分组，全程 float32（eps 保护在 float32 下生效）
        y = x.to(torch.float32).reshape(-1, gs)
        amax = y.abs().amax(dim=1, keepdim=True)
        amax = torch.maximum(amax, torch.full_like(amax, self.eps))
        y_s = amax / int8_max                                 # (G, 1) float32
        y_q = (y / y_s).clamp(min=int8_min, max=int8_max).to(torch.int8)

        # 单张量输出协议：[码字提升 float32 | scale float32]
        codes = y_q.reshape(rows, cols).reshape(-1).to(torch.float32)
        scales = y_s.reshape(rows, cols // gs).reshape(-1)
        return torch.cat([codes, scales])


def get_init_inputs():
    return [128]   # group_size；eps 缺省 1e-10


def get_inputs():
    # 固定 shape 族（官方测试最高频 case 128x7168x128）；随机部分消费全局
    # RNG——评测器在 set_seed 后调用本函数，多轮 correctness trial 因此获得
    # 输入多样性
    rows, cols = 128, 7168
    x = torch.randn(rows, cols).to(torch.float16)
    return [x]


def make_inputs(rows: int, cols: int, group_size: int, dtype: str = "float16",
                eps: float = 1e-10, value_scale: float = 1.0,
                zero_rows: int = 0, seed: int = 0):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器
    搬运到设备）。eps/group_size 供评测端构造 Model 用，这里只做契约校验。

    消费顺序固定：randn(rows*cols) 一次，同 seed 下逐位可复现。
    value_scale 缩放数值幅度、zero_rows 把前 k 行置为精确 0，均用于构造
    eps 保护 / 全零组边界 case。
    """
    assert cols % group_size == 0, "cols 必须能被 group_size 整除"
    assert 0 <= zero_rows <= rows, "zero_rows 越界"

    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    x = torch.randn(rows, cols, generator=gen) * float(value_scale)
    if zero_rows > 0:
        x[:zero_rows] = 0.0
    x = x.to(dt)
    return [x]
