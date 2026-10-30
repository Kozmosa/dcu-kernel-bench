# 4018_quant -- fp8(e4m3)/int8 quantization kernel family (static per-tensor /
# dynamic per-tensor / dynamic per-token) model_class (KernelBench compatible)
# task file.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=4018, backend=triton.

import torch
import torch.nn as nn

_QUANT_DTYPES = {"int8": torch.int8, "float8_e4m3": torch.float8_e4m3fn}
_MODES = ("static_per_tensor", "dynamic_per_tensor", "dynamic_per_token")


class Model(nn.Module):
    """fp8(e4m3)/int8 quantization kernel family: static per-tensor / dynamic
    per-tensor / dynamic per-token.

    The input x is a contiguous 2D float tensor (rows, cols); the __init__
    hyperparameter mode selects the quantization mode and quant_dtype selects
    the target dtype (int8 or fp8 e4m3). Intermediate arithmetic is float32
    throughout. Definitions per mode (dtype_max: 127 for int8, 448 for fp8
    e4m3):

      static_per_tensor (the scale is given by the input):
          q_i  = x_i / scale                    # fp32 division, scale is (1,) fp32
          y_i  = int8: round_half_even(q_i); fp8: rne_e4m3(q_i)
          scale_out = scale (passed through as is, segment length 1)
      dynamic_per_tensor (single scale for the whole tensor, no division-by-zero guard):
          amax = max_i |x_i|                    # fp32
          s    = amax / dtype_max
          y_i  = int8: round_half_even(x_i / s); fp8: rne_e4m3(x_i / s)
          scale_out = s (segment length 1)
      dynamic_per_token (one scale per row, with a division-by-zero guard):
          amax_r = max_j |x_rj|                 # fp32
          s_r    = max(amax_r, 1e-10) / dtype_max
          y_rj   = int8: round_half_even(x_rj * (1/s_r)); fp8: rne_e4m3(x_rj * (1/s_r))
          scale_out = s (segment length rows, row-major)

    round_half_even is banker's rounding (0.5 -> 0, -0.5 -> 0, 1.5 -> 2,
    126.5 -> 126), consistent with torch.round; rne_e4m3 is rounding to the
    nearest representable e4m3 value (equivalent to .to(float8_e4m3fn)).
    Boundary behaviour: an all-zero row under per-token has amax_r = 0, the
    1e-10 guard gives s_r = 1e-10/dtype_max and every code word of that row is
    0 -- the output must be finite, producing NaN/Inf is a wrong
    implementation; per-tensor modes have no division-by-zero guard (the input
    contract guarantees amax > 0).

    Input/output specification:
      x     float16/bfloat16, contiguous (rows, cols); the evaluator casts the
            input to fp32 before passing it in -- fp16/bf16 values are exact in
            fp32, so the quantization semantics are unchanged
      scale used only in static_per_tensor mode: (1,) float32 positive number
      out   a 1-D float32 tensor of length rows*cols + S (single-tensor output
            protocol, two segments packed):
            out[:rows*cols] = quantized code words promoted to float32 (int8
                              are integers in [-127,127]; fp8 are exact e4m3
                              values)
            out[rows*cols:] = the scale segment: per-token gives rows row
                              scales (row-major), per-tensor/static gives 1
                              tensor scale
      __init__ hyperparameters: mode (str, default "dynamic_per_token"),
            quant_dtype (str, "int8" or "float8_e4m3", default "int8").
      Input contract: x is not all zero (per-tensor modes); for every mode the
      quotient/product before quantization falls inside the target dtype range
      (the cast has no overflow semantics divergence).

    Implementation constraints (violations are scored as failures):
      - The core computation (amax reduction, scale, quantized code words) must
        be completed in Triton kernels inside the submitted file; forward may
        only do input/output shaping (dtype conversion, the two-segment packing,
        etc.). Calling any ready-made quantization library / quantization
        shortcut is forbidden (including torch.quantize_per_tensor /
        torch.quantize_per_channel).
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Intermediate arithmetic uses float32; the computation order of the
        scale and the quotient must match the definitions above -- per-token
        computes the reciprocal 1/s_r first and then multiplies by x, whereas
        static/per-tensor use the fp32 direct division x_i / s (reciprocal
        multiplication and direct division differ by 1 ulp, which shifts code
        words on tie boundaries); int8 rounding uses round_half_even.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: 2D contiguous input, int8 and fp8 e4m3 target dtypes, three
    modes in one operator; without the GEMM back-end computation and without
    block-scaling floating-point (mxfp4) quantization.
    """

    def __init__(self, mode: str = "dynamic_per_token", quant_dtype: str = "int8"):
        super().__init__()
        # NOTE: the Model method bodies keep single-line statement form (the
        # evaluation scaffold splits anchors by top-level statement)
        assert mode in ("static_per_tensor", "dynamic_per_tensor", "dynamic_per_token"), "invalid mode"
        assert quant_dtype in ("int8", "float8_e4m3"), "quant_dtype must be int8 or float8_e4m3"
        self.mode = mode
        self.quant_dtype = torch.int8 if quant_dtype == "int8" else torch.float8_e4m3fn

    def forward(self, x, scale=None):
        # The three-mode dispatch is written as single-line statement bodies
        # (the evaluation scaffold splits anchors by top-level statement, so
        # multi-line if/else blocks would be cut apart); the compute path
        # corresponds to the docstring definitions one by one
        assert x.dim() == 2, "x must be a 2D tensor"
        assert x.is_contiguous(), "x must be contiguous"
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

        # Single-tensor output protocol: [code-word segment float32 | scale segment float32]
        codes = qx.reshape(-1).to(torch.float32)
        return torch.cat([codes, s.reshape(-1).to(torch.float32)])


def output_segments(init_kwargs, numel, inputs=None):
    """For the evaluator: the boundaries of the two output segments (see
    tolerance_segments in task.yaml).

    The scale segment length varies with mode: 1 for per-tensor; the number of
    input rows for per-token (hence inputs is needed).
    """
    mode = str(init_kwargs.get("mode", "dynamic_per_token"))
    s_len = 1
    if mode == "dynamic_per_token" and inputs:
        s_len = int(inputs[0].shape[0])
    return [("codes", 0, int(numel) - s_len), ("scales", int(numel) - s_len, int(numel))]


def get_init_inputs():
    return ["dynamic_per_token", "int8"]   # mode; quant_dtype


def get_inputs():
    # Fixed shape family (cols=128 triggers the row-width-128 specialized path;
    # the random part consumes the global RNG -- the evaluator calls this
    # function after set_seed, so multiple correctness trials therefore see
    # diverse inputs)
    rows, cols = 1024, 128
    x = torch.randn(rows, cols).to(torch.float16)
    return [x]


def make_inputs(mode: str, quant_dtype: str, rows: int, cols: int,
                dtype: str = "float16", seed: int = 0, zero_rows: int = 0):
    """Generate inputs from a public/hidden/perf case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the
    device). mode/quant_dtype are used by the evaluation side to construct
    Model; here they only check the contract.

    Consumption order is fixed: one randn(rows*cols) call and (static mode) one
    rand(1) call, bit-reproducible for the same seed. zero_rows sets the first k
    rows to exact 0, used to build the per-token 1e-10 guard boundary; the
    static mode scale takes a positive value in [0.5, 1.5), which keeps the
    quotient inside the target dtype range.
    """
    assert mode in _MODES, f"mode must be one of {_MODES}"
    assert quant_dtype in _QUANT_DTYPES, "quant_dtype must be int8 or float8_e4m3"
    assert 0 <= zero_rows <= rows, "zero_rows out of range"
    # dynamic_per_tensor has no division-by-zero guard: zeroing every row is
    # not allowed
    assert not (mode == "dynamic_per_tensor" and zero_rows == rows), \
        "per-tensor mode requires amax > 0"

    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    x = torch.randn(rows, cols, generator=gen).to(dt)
    if zero_rows > 0:
        x[:zero_rows] = 0.0
    if mode == "static_per_tensor":
        scale = 0.5 + torch.rand(1, generator=gen, dtype=torch.float32)
        return [x, scale]
    return [x]
