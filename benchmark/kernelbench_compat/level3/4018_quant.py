# 4018_quant — fp8(e4m3)/int8 量化 kernel 族（静态 per-tensor / 动态
# per-tensor / 动态 per-token）的 model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=4018, backend=triton。

import torch
import torch.nn as nn

_QUANT_DTYPES = {"int8": torch.int8, "float8_e4m3": torch.float8_e4m3fn}
_MODES = ("static_per_tensor", "dynamic_per_tensor", "dynamic_per_token")


class Model(nn.Module):
    """fp8(e4m3)/int8 量化 kernel 族：静态 per-tensor / 动态 per-tensor / 动态 per-token。

    输入 x 是连续 2D float 张量 (rows, cols)，按 __init__ 超参 mode 选择量化
    模式、quant_dtype 选择目标 dtype（int8 或 fp8 e4m3）。中间算术全程
    float32，各模式定义（dtype_max：int8 取 127，fp8 e4m3 取 448）：

      static_per_tensor（scale 由输入给出）：
          q_i  = x_i / scale                    # fp32 除法，scale 为 (1,) fp32
          y_i  = int8: round_half_even(q_i)；fp8: rne_e4m3(q_i)
          scale_out = scale（原样透传，段长 1）
      dynamic_per_tensor（整个张量单一 scale，无除零保护）：
          amax = max_i |x_i|                    # fp32
          s    = amax / dtype_max
          y_i  = int8: round_half_even(x_i / s)；fp8: rne_e4m3(x_i / s)
          scale_out = s（段长 1）
      dynamic_per_token（每行一个 scale，带除零保护）：
          amax_r = max_j |x_rj|                 # fp32
          s_r    = max(amax_r, 1e-10) / dtype_max
          y_rj   = int8: round_half_even(x_rj * (1/s_r))；fp8: rne_e4m3(x_rj * (1/s_r))
          scale_out = s（段长 rows，行主序）

    round_half_even 即银行家舍入（0.5 -> 0、-0.5 -> 0、1.5 -> 2、126.5 -> 126），
    与 torch.round 一致；rne_e4m3 为向最近可表示 e4m3 值的舍入（等价
    .to(float8_e4m3fn)）。边界行为：per-token 全零行 amax_r = 0，经 1e-10 保护
    得 s_r = 1e-10/dtype_max、该行码字全 0——输出必须有限，产出 NaN/Inf 即错误
    实现；per-tensor 模式无除零保护（输入契约保证 amax > 0）。

    输入输出规格：
      x     float16/bfloat16，连续 (rows, cols)；评测器会把输入 cast 成 fp32
            传入——fp16/bf16 值在 fp32 中精确表示，量化语义不变
      scale 仅 static_per_tensor 模式使用：(1,) float32 正数
      out   一维 float32 张量，长度 rows*cols + S（单张量输出协议，两段打包）：
            out[:rows*cols] = 量化码字提升为 float32（int8 为 [-127,127] 内
                              整数；fp8 为 e4m3 精确值）
            out[rows*cols:] = scale 段：per-token 为 rows 个行 scale（行主序），
                              per-tensor/static 为 1 个张量 scale
      __init__ 超参：mode（str，默认 "dynamic_per_token"）、quant_dtype
            （str，"int8" 或 "float8_e4m3"，默认 "int8"）。
      输入契约：x 非全零（per-tensor 模式）；各模式量化前的商/积均落在目标
      dtype 值域内（cast 无溢出语义分歧）。

    实现约束（违规判负）：
      - 核心计算（amax 归约、scale、量化码字）必须在提交文件内的 Triton
        kernel 中完成；forward 里只允许做输入/输出整理（dtype 转换、两段
        拼接打包等）。禁止调用任何现成量化库 / 量化捷径（含
        torch.quantize_per_tensor / torch.quantize_per_channel）。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间算术用 float32；scale 与商的计算顺序必须与上述定义一致——
        per-token 先算倒数 1/s_r 再乘 x，static/per-tensor 用 x_i / s 的
        fp32 直接除法（倒数乘法与直接除法存在 1 ulp 差异，会把 tie 边界
        的码字移位）；int8 取整用 round_half_even。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：2D 连续输入、int8 与 fp8 e4m3 目标 dtype、三模式单算子；
    不含 GEMM 后端计算与块缩放浮点（mxfp4）量化。
    """

    def __init__(self, mode: str = "dynamic_per_token", quant_dtype: str = "int8"):
        super().__init__()
        # 注意：Model 方法体保持单行语句形态（评测脚手架按顶层语句切分锚点）
        assert mode in ("static_per_tensor", "dynamic_per_tensor", "dynamic_per_token"), "mode 非法"
        assert quant_dtype in ("int8", "float8_e4m3"), "quant_dtype 必须是 int8 或 float8_e4m3"
        self.mode = mode
        self.quant_dtype = torch.int8 if quant_dtype == "int8" else torch.float8_e4m3fn

    def forward(self, x, scale=None):
        # 三模式分派写成单行语句体（评测脚手架按顶层语句切分锚点，多行
        # if/else 块会被切坏）；计算路径与文档字符串定义逐条对应
        assert x.dim() == 2, "x 必须是 2D 张量"
        assert x.is_contiguous(), "x 必须连续"
        qdt = self.quant_dtype
        dtype_max = torch.iinfo(qdt).max if qdt == torch.int8 else torch.finfo(qdt).max
        x_f32 = x.to(torch.float32)
        absf = torch.abs(x_f32)

        if self.mode == "static_per_tensor": s = scale.to(torch.float32).reshape(1)
        if self.mode == "dynamic_per_tensor": s = (torch.max(absf) / dtype_max).reshape(1)
        if self.mode == "dynamic_per_token": s = torch.max(absf, dim=-1).values.clamp(min=1.0e-10) / dtype_max
        if self.mode == "dynamic_per_token": q = x_f32 * (1.0 / s)[:, None]
        if self.mode != "dynamic_per_token": q = x_f32 / s
        if qdt == torch.int8: q = q.round()
        qx = q.to(qdt)

        # 单张量输出协议：[码字段 float32 | scale 段 float32]
        codes = qx.reshape(-1).to(torch.float32)
        return torch.cat([codes, s.reshape(-1).to(torch.float32)])


def output_segments(init_kwargs, numel, inputs=None):
    """给评测器：输出里两段的边界（见 task.yaml 的 tolerance_segments）。

    scale 段长随 mode 变化：per-tensor 为 1；per-token 为输入行数（故需要 inputs）。
    """
    mode = str(init_kwargs.get("mode", "dynamic_per_token"))
    s_len = 1
    if mode == "dynamic_per_token" and inputs:
        s_len = int(inputs[0].shape[0])
    return [("codes", 0, int(numel) - s_len), ("scales", int(numel) - s_len, int(numel))]


def get_init_inputs():
    return ["dynamic_per_token", "int8"]   # mode；quant_dtype


def get_inputs():
    # 固定 shape 族（cols=128 触发行宽 128 的特化路径；随机部分消费全局
    # RNG——评测器在 set_seed 后调用本函数，多轮 correctness trial 因此获得
    # 输入多样性）
    rows, cols = 1024, 128
    x = torch.randn(rows, cols).to(torch.float16)
    return [x]


def make_inputs(mode: str, quant_dtype: str, rows: int, cols: int,
                dtype: str = "float16", seed: int = 0, zero_rows: int = 0):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器
    搬运到设备）。mode/quant_dtype 供评测端构造 Model 用，这里只做契约校验。

    消费顺序固定：randn(rows*cols) 一次、（static 模式）rand(1) 一次，同 seed
    下逐位可复现。zero_rows 把前 k 行置为精确 0，用于构造 per-token 的
    1e-10 保护边界；static 模式的 scale 取 [0.5, 1.5) 内正数，保证商落在
    目标 dtype 值域内。
    """
    assert mode in _MODES, f"mode 必须是 {_MODES} 之一"
    assert quant_dtype in _QUANT_DTYPES, "quant_dtype 必须是 int8 或 float8_e4m3"
    assert 0 <= zero_rows <= rows, "zero_rows 越界"
    # dynamic_per_tensor 无除零保护：不允许把全部行置零
    assert not (mode == "dynamic_per_tensor" and zero_rows == rows), \
        "per-tensor 模式要求 amax > 0"

    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    x = torch.randn(rows, cols, generator=gen).to(dt)
    if zero_rows > 0:
        x[:zero_rows] = 0.0
    if mode == "static_per_tensor":
        scale = 0.5 + torch.rand(1, generator=gen, dtype=torch.float32)
        return [x, scale]
    return [x]
