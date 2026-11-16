# 4009_gemm_a8w8 -- model_class (KernelBench compatible) task file for the 2D
# 8-bit quantized GEMM (int8 / fp8 activation x weight, per-token x per-channel
# scalar-scale dequantization, optional bias).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=4009, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """8-bit quantized GEMM: int8 / fp8 activations x same-dtype weights, with an
    outer-product scale applied for dequantization.

    Mathematical definition: given the quantized activation x (M, K), the quantized
    weight w (N, K) (row-major with output channels as rows, i.e. transposed-GEMM
    semantics), the per-token scale x_scale (M, 1) float32, the per-channel scale
    w_scale (1, N) float32, and an optional bias (1, N) float32:
      acc[m, n] = sum_k x[m, k] * w[n, k]                (float32 accumulation)
      out[m, n] = acc[m, n] * x_scale[m, 0] * w_scale[0, n]  (outer-product dequantization)
      out[m, n] = out[m, n] + bias[0, n]                 (when use_bias=True, the
                                                     dequantized result is added in float32)
    Finally the result is cast to out_dtype. The numeric value is determined jointly
    by the quantization code words and the scales: a dequantized element equals the
    code word times the corresponding position of the outer product of x_scale and
    w_scale.

    Input/output specification:
      x        in_dtype   (M, K)  row-major activation code words; in_dtype in
                          {int8, fp8e4m3, fp8e5m2} (e4m3 = 4-bit exponent, 3-bit
                          mantissa; e5m2 = 5-bit exponent, 2-bit mantissa), set by
                          __init__'s in_dtype
      w        in_dtype   (N, K)  row-major weight code words (output channels as rows)
      x_scale  float32    (M, 1)  per-token scale
      w_scale  float32    (1, N)  per-channel scale
      bias     float32    (1, N)  a placeholder tensor when use_bias=False, ignored
      out      out_dtype  (M, N)  out_dtype in {bfloat16, float16}; M/N/K >= 1,
                          non-powers-of-two allowed and K may not divide the tile
                          (the tail is zero-padded by masking)

    Implementation constraints (violations are scored as failures):
      - The core computation (the 8-bit GEMM accumulation and the dequantization)
        must be completed inside the submitted file; calling torch.matmul /
        torch.bmm / torch.mm / torch.addmm / torch.einsum / torch._int_mm /
        torch._scaled_mm / F.linear / functional.linear is forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Accumulate intermediates in float32; dequantization and the bias add are
        completed in the float32 domain before casting to out_dtype.
      - The evaluator casts all inputs to fp32 before calling forward: int8 / fp8
        code words and the scales are all exactly representable in fp32, so
        .to(in_dtype) at the entry point recovers the quantization code words
        losslessly.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: 2D GEMM (no batch dimension), x row-major (M,K) / w row-major
    (N,K), scalar (per-token x per-channel) scales; excluding 128x128 blocked
    scales, a batch dimension, and output-buffer preallocation variants.
    """

    def __init__(self, in_dtype="fp8e4m3", out_dtype="bfloat16", use_bias=True):
        super().__init__()
        quant = {"fp8e4m3": torch.float8_e4m3fn,
                 "fp8e5m2": torch.float8_e5m2,
                 "int8": torch.int8}
        out = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        self.in_dtype = quant[in_dtype]
        self.out_dtype = out[out_dtype]
        self.use_bias = bool(use_bias)

    def forward(self, x, w, x_scale, w_scale, bias):
        # The evaluator casts every input to fp32; the quantization code words are
        # recovered losslessly back to in_dtype and the scales back to float32.
        x = x.to(self.in_dtype)
        w = w.to(self.in_dtype)
        x_scale = x_scale.to(torch.float32)
        w_scale = w_scale.to(torch.float32)

        M, K = x.shape
        N = w.shape[0]
        assert K == w.shape[1], "x and w must agree on the K dimension"

        # Promote the code words to float32 and run the GEMM (x @ w^T); the whole
        # accumulation is float32
        acc = torch.matmul(x.to(torch.float32), w.to(torch.float32).t())

        # The (M,1) x (1,N) outer product forms the (M,N) dequantization matrix,
        # which is then multiplied in.
        out = acc * torch.matmul(x_scale, w_scale)

        if self.use_bias:
            out = out.to(torch.float32) + bias.to(torch.float32)
        return out.to(self.out_dtype)


def get_init_inputs():
    return ["fp8e4m3", "bfloat16", True]   # in_dtype; out_dtype; use_bias


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the evaluator
    # calls this function after set_seed, so multiple correctness trials therefore
    # see diverse inputs. The quantization scheme matches make_inputs: scale each
    # row by its absmax onto the code-word range limit, then cast.
    m, n, k = 256, 512, 1024
    qmax = torch.finfo(torch.float8_e4m3fn).max

    x = torch.randn((m, k), dtype=torch.float32)
    w = torch.randn((n, k), dtype=torch.float32)
    x_scale = x.abs().amax(dim=1, keepdim=True) / qmax
    x = (x / x_scale).to(torch.float8_e4m3fn)
    w_scale = w.abs().amax(dim=1, keepdim=True).T.contiguous() / qmax
    w = (w / w_scale.T).to(torch.float8_e4m3fn)
    bias = torch.rand([1, n], dtype=torch.float32) * 10
    return [x, w, x_scale, w_scale, bias]


def make_inputs(m: int, n: int, k: int, in_dtype: str = "fp8e4m3",
                out_dtype: str = "bfloat16", use_bias: bool = True,
                seed: int = 0):
    """Generate inputs from a public/hidden/performance case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    The quantization path matches the official test generator: scale each row by
    its absmax onto the code-word range limit and then cast to in_dtype. The
    consumption order is fixed: x -> w -> bias, all from the same
    torch.Generator(seed), so the same seed reproduces inputs bit-for-bit.
    use_bias only controls whether the Model's bias takes effect; the bias tensor
    is always generated (and ignored by forward when use_bias=False).
    """
    gen = torch.Generator().manual_seed(seed)
    quant = {"fp8e4m3": torch.float8_e4m3fn, "fp8e5m2": torch.float8_e5m2,
             "int8": torch.int8}
    assert out_dtype in ("bfloat16", "float16")
    dt = quant[in_dtype]
    qmax = (torch.finfo(dt) if dt.is_floating_point else torch.iinfo(dt)).max

    x = torch.randn((m, k), generator=gen, dtype=torch.float32)
    w = torch.randn((n, k), generator=gen, dtype=torch.float32)
    x_scale = x.abs().amax(dim=1, keepdim=True) / qmax
    x = (x / x_scale).to(dt)
    w_scale = w.abs().amax(dim=1, keepdim=True).T.contiguous() / qmax
    w = (w / w_scale.T).to(dt)
    bias = torch.rand([1, n], generator=gen, dtype=torch.float32) * 10
    return x, w, x_scale, w_scale, bias
