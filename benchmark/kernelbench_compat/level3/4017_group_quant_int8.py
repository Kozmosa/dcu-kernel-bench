# 4017_group_quant_int8 -- per-token-group int8 dynamic quantization
# (activation preprocessing for W8A8 integer GEMM) model_class (KernelBench
# compatible) task file.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=4017, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """per-token-group int8 dynamic quantization (activation preprocessing for
    W8A8 integer GEMM).

    The input x is a contiguous 2D float tensor (rows, cols) and cols must be
    divisible by group_size. In row-major (flattened) order x is cut into
    G = rows*cols/group_size contiguous groups of group_size elements each.
    For every group g (all intermediate arithmetic in float32):
        absmax_g = max( max_{i in g} |x_i|, eps )    # eps is the division-by-zero guard
        scale_g  = absmax_g / 127                    # fp32 division
        q_i      = trunc_int8( clamp( x_i / scale_g, -128, 127 ) )
    trunc_int8 means truncate toward zero and then convert to int8, so the
    quantized code words end up in [-127, 127].
    Boundary behaviour: when absmax_g < eps (including an all-zero group)
    scale_g = eps/127 and every code word in the group is 0; the eps guard must
    take effect (producing a 0 scale or NaN is a wrong implementation).

    Input/output specification:
      x    float16/bfloat16, contiguous (rows, cols), cols % group_size == 0;
           the evaluator casts the input to fp32 before passing it in --
           fp16/bf16 values are exact in fp32, so the quantization semantics are
           unchanged
      out  a 1-D float32 tensor of length rows*cols + rows*cols/group_size
           (single-tensor output protocol, two segments packed):
           out[:rows*cols]  = int8 code words promoted to float32 (integers in
                              [-127,127])
           out[rows*cols:]  = the per-group scales, in row-major order
                              ((rows, cols/group_size) flattened)
      __init__ hyperparameters: group_size (int, default 128), eps (float,
      default 1e-10).

    Implementation constraints (violations are scored as failures):
      - The core computation (in-group absmax, scale, quantized code words)
        must be completed in Triton kernels inside the submitted file; forward
        may only do input/output shaping (dtype conversion, the two-segment
        packing, etc.). Calling aiter / torch.quantize_per_tensor /
        torch.quantize_per_channel is forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Intermediate arithmetic uses float32; the scale must be computed as the
        fp32 division absmax/127 (precomputing a reciprocal and multiplying
        shifts the code words by 1 at the truncation boundary); the code words
        are first clamped to [-128, 127] and then truncated toward zero into
        int8.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: 2D contiguous input, torch.int8 code words, fp32 scale, without
    the GEMM back-end computation.
    """

    def __init__(self, group_size: int = 128, eps: float = 1e-10):
        super().__init__()
        self.group_size = int(group_size)
        self.eps = float(eps)

    def forward(self, x):
        gs = self.group_size
        assert x.dim() == 2, "x must be a 2D tensor"
        assert x.shape[-1] % gs == 0, "the last dimension must be divisible by group_size"
        assert x.is_contiguous(), "x must be contiguous"

        rows, cols = x.shape
        int8_min, int8_max = -128.0, 127.0

        # Flatten into groups, float32 throughout (the eps guard acts in float32)
        y = x.to(torch.float32).reshape(-1, gs)
        amax = y.abs().amax(dim=1, keepdim=True)
        amax = torch.maximum(amax, torch.full_like(amax, self.eps))
        y_s = amax / int8_max                                 # (G, 1) float32
        y_q = (y / y_s).clamp(min=int8_min, max=int8_max).to(torch.int8)

        # Single-tensor output protocol: [code words promoted to float32 | scale float32]
        codes = y_q.reshape(rows, cols).reshape(-1).to(torch.float32)
        scales = y_s.reshape(rows, cols // gs).reshape(-1)
        return torch.cat([codes, scales])


def output_segments(init_kwargs, numel, inputs=None):
    """For the evaluator: the boundaries of the two output segments (see
    tolerance_segments in task.yaml).

    The codes segment comes first and the scales segment after it; the scale
    segment length = number of groups = numel/(group_size+1).
    """
    gs = int(init_kwargs.get("group_size", 128))
    s_len = int(numel) // (gs + 1)
    return [("codes", 0, int(numel) - s_len), ("scales", int(numel) - s_len, int(numel))]


def get_init_inputs():
    return [128]   # group_size; eps defaults to 1e-10


def get_inputs():
    # Fixed shape family (128x7168x128, the most frequent case in the official
    # test); the random part consumes the global RNG -- the evaluator calls this
    # function after set_seed, so multiple correctness trials therefore see
    # diverse inputs
    rows, cols = 128, 7168
    x = torch.randn(rows, cols).to(torch.float16)
    return [x]


def make_inputs(rows: int, cols: int, group_size: int, dtype: str = "float16",
                eps: float = 1e-10, value_scale: float = 1.0,
                zero_rows: int = 0, seed: int = 0):
    """Generate inputs from a public/hidden/perf case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the
    device). eps/group_size are used by the evaluation side to construct Model;
    here they only check the contract.

    Consumption order is fixed: one randn(rows*cols) call, bit-reproducible for
    the same seed.
    value_scale scales the numerical magnitude and zero_rows sets the first k
    rows to exact 0; both are used to build the eps-guard / all-zero-group
    boundary cases.
    """
    assert cols % group_size == 0, "cols must be divisible by group_size"
    assert 0 <= zero_rows <= rows, "zero_rows out of range"

    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    x = torch.randn(rows, cols, generator=gen) * float(value_scale)
    if zero_rows > 0:
        x[:zero_rows] = 0.0
    x = x.to(dt)
    return [x]
