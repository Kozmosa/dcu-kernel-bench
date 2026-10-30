# 4001_batched_gemm_a8w8 -- model_class (KernelBench compatible) task file for
# the batch-dimension int8 quantized GEMM (per-token x per-channel scale
# dequantization, optional bias).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=4001, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """Batch-dimension int8 quantized GEMM: int8 activations x int8 weights, with an
    outer-product scale applied for dequantization.

    Computed independently for every matrix in the batch (mathematical definition):
      acc[b, m, n] = sum_k xq[b, m, k] * wq[b, n, k]          (float32 accumulation)
      out[b, m, n] = acc[b, m, n] * xs[b, m, 0] * ws[b, 0, n] (+ bias[b, 0, n])

    i.e. out[b] = xq[b] @ wq[b]^T followed by dequantization. wq is stored
    row-major as (B, N, K), so the computation is equivalent to a transposed GEMM
    (row n of the weight is one output-channel column). xs is the per-token scale:
    one value per row of the M dimension, (B, M, 1) float32; ws is the per-channel
    scale: one value per column of the N dimension, (B, 1, N) float32; their outer
    product is the (B, M, N) dequantization matrix. bias (B, 1, N) is optional and
    has the same dtype as the output; with bias the quantized-GEMM kernel fusion
    order applies: the float32 scaled result is first cast to the output dtype, then
    the bias is added.

    Input/output specification:
      xq      int8       (B, M, K)   activation code words
      wq      int8       (B, N, K)   weight code words (row-major, transposed-GEMM
                                     equivalent)
      xs      float32    (B, M, 1)   per-token scale
      ws      float32    (B, 1, N)   per-channel scale
      bias    output dtype (B, 1, N) optional; when use_bias=False a placeholder
                                     tensor is passed and ignored
      out     output dtype (B, M, N) the dtype is set by __init__'s out_dtype
                                     (bfloat16 / float16), B/M/N/K >= 1

    Implementation constraints (violations are scored as failures):
      - The core computation (the int8 GEMM accumulation) must be completed inside
        the submitted file; calling torch.matmul / torch.bmm / torch.mm /
        torch.addmm / torch.einsum / torch._int_mm / F.linear /
        functional.linear is forbidden, as is calling any ready-made operator
        library or closed-source kernel wrapper to complete this computation
        directly.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Accumulate intermediates in float32; cast the output to out_dtype.
      - The evaluator casts all inputs to fp32 before calling forward; the code
        words are small integers in [-20, 20) and the bias values come from
        low-precision quantization, all exactly representable in fp32, so
        .to(torch.int8) / .to(out_dtype) at the entry point recovers them
        losslessly.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: TN layout (x row-major (B,M,K), w row-major (B,N,K)), no split-K;
    B/M/N/K are any values >= 1 with no alignment requirement; the accumulation
    runs in float32 and must agree with the reference fp32 GEMM within tolerance.
    """

    def __init__(self, use_bias: bool = True, out_dtype: str = "bfloat16"):
        super().__init__()
        self.use_bias = bool(use_bias)
        self.out_dtype = getattr(torch, out_dtype)

    def forward(self, xq, wq, x_scale, w_scale, bias):
        # Note: keep the top level of forward made of single-line statements
        # (ternary expressions included). The framework loader extracts the
        # top-level statements one by one to build the ModelNew scaffold, and a
        # multi-line if/else block would break the scaffold indentation (the loader
        # adds a uniform 8 spaces to each statement block, so the relative
        # indentation inside a block would be misaligned).
        xq = xq.to(torch.int8)
        wq = wq.to(torch.int8)
        bias = bias.to(self.out_dtype) if self.use_bias else None

        B, M, K = xq.shape
        N = wq.shape[1]
        assert K == wq.shape[2], "xq and wq must agree on the K dimension"

        # Promote the int8 code words to float32 and run the GEMM; the whole
        # accumulation is float32
        a = xq.to(torch.float32)
        b = wq.to(torch.float32)
        acc = torch.matmul(a, b.transpose(1, 2))   # (B, M, N) float32

        # Dequantize by the outer product of the per-token and per-channel scales
        # (broadcasting)
        acc = acc * x_scale                        # (B, M, 1) broadcasts over the M dimension
        acc = acc * w_scale                        # (B, 1, N) broadcasts over the N dimension

        # Fusion order matches the quantized GEMM kernel: cast to the output dtype
        # first, then add the bias.
        acc = acc.to(self.out_dtype) if self.use_bias else acc
        acc = acc + bias if self.use_bias else acc
        return acc.to(self.out_dtype)


def get_init_inputs():
    return [True, "bfloat16"]   # use_bias; out_dtype


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the evaluator
    # calls this function after set_seed, so multiple correctness trials therefore
    # see diverse inputs. The code-word range and the scale/bias magnitudes match
    # the generator make_inputs.
    b, m, n, k = 4, 128, 256, 512

    xq = torch.randint(-20, 20, (b, m, k), dtype=torch.int8)
    wq = torch.randint(-20, 20, (b, n, k), dtype=torch.int8)
    x_scale = torch.rand(b, m, 1, dtype=torch.float32) + 1e-6
    w_scale = torch.rand(b, 1, n, dtype=torch.float32) + 1e-6
    bias = (torch.rand(b, 1, n, dtype=torch.float32) * 10).to(torch.bfloat16)
    return [xq, wq, x_scale, w_scale, bias]


def make_inputs(b: int, m: int, n: int, k: int, use_bias: bool = True,
                out_dtype: str = "bfloat16", seed: int = 0):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    The consumption order is fixed: xq -> wq -> x_scale -> w_scale -> bias, all
    from the same torch.Generator(seed), so the same seed reproduces inputs
    bit-for-bit.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, out_dtype)

    xq = torch.randint(-20, 20, (b, m, k), generator=gen, dtype=torch.int8)
    wq = torch.randint(-20, 20, (b, n, k), generator=gen, dtype=torch.int8)
    x_scale = torch.rand(b, m, 1, generator=gen, dtype=torch.float32) + 1e-6
    w_scale = torch.rand(b, 1, n, generator=gen, dtype=torch.float32) + 1e-6
    bias = (torch.rand(b, 1, n, generator=gen, dtype=torch.float32) * 10).to(dt)
    return xq, wq, x_scale, w_scale, bias
