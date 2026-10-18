# 4004_batched_gemm_bf16 -- model_class (KernelBench compatible) task file for
# the batch-dimension bf16 x bf16 GEMM (optional bias).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=4004, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """Batch-dimension bf16 GEMM: computes x[b] @ weight[b]^T independently for
    every matrix in the batch.

    Mathematical definition (independent per b in the batch, B/M/N/K >= 1):
      acc[b, m, n] = sum_k x[b, m, k] * weight[b, n, k]    (float32 accumulation)
      out[b, m, n] = acc[b, m, n] + bias[b, 0, n]          (optional bias)
    weight is stored row-major as (B, N, K), so the computation is equivalent to a
    transposed GEMM (each output channel corresponds to one row of weight). The
    bias fusion order follows GEMM kernel convention: the float32 accumulated
    result is first cast to the input dtype (bfloat16), bias is then added
    (bfloat16 arithmetic), and finally the result is cast to the output dtype. With
    no bias it is cast directly. Boundary behaviour: B/M/N/K may be any value >= 1
    (including the minimal 1x1x1 size) and the dimensions may be
    non-powers-of-two; when K does not divide the tile size, the kernel masks the
    tail with zeros, which does not change the mathematical result.

    Input/output specification:
      x      bfloat16  (B, M, K)  the activation matrix
      weight bfloat16  (B, N, K)  the weight matrix (row-major, transposed-GEMM semantics)
      bias   bfloat16  (B, 1, N)  optional; when use_bias=False a placeholder tensor is
                                  passed and ignored
      out    output dtype (B, M, N), where the dtype is set by __init__'s out_dtype
                                  (bfloat16 / float16)

    Implementation constraints (violations are scored as failures):
      - The core computation (the bf16 GEMM multiply-accumulate and the bias fusion)
        must be completed inside the submitted file; calling torch.matmul /
        torch.bmm / torch.mm / torch.addmm / torch.baddbmm / torch.einsum /
        F.linear / functional.linear / aiter is forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Accumulate intermediates in float32; the bias branch follows the fusion
        order above; cast the output to out_dtype.
      - The evaluator casts all inputs to fp32 before calling forward; the input
        values (small integer codes and values representable in bfloat16) are all
        exactly representable in fp32, so .to(torch.bfloat16) at the entry point
        recovers them losslessly.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: TN layout (x row-major (B,M,K), weight row-major (B,N,K)), no
    split-K, no quantization component at all (pure bf16 inputs, no scale
    dequantization).
    """

    def __init__(self, use_bias: bool = True, out_dtype: str = "bfloat16"):
        super().__init__()
        self.use_bias = bool(use_bias)
        self.out_dtype = getattr(torch, out_dtype)

    def forward(self, x, weight, bias):
        # The evaluator casts every input to fp32; the bfloat16 value range is
        # exactly representable in fp32, so it is recovered losslessly.
        x = x.to(torch.bfloat16)
        weight = weight.to(torch.bfloat16)
        # Single-statement form (forward statements must be extractable and
        # reorderable one by one, so avoid an if/else compound block).
        bias = bias.to(torch.bfloat16) if self.use_bias else None

        B, M, K = x.shape
        N = weight.shape[1]
        assert K == weight.shape[2], "x and weight must agree on the K dimension"

        # Promote bf16 to float32 and run the GEMM; the whole accumulation is float32
        acc = torch.matmul(x.to(torch.float32), weight.to(torch.float32).transpose(1, 2))

        if bias is not None:
            # Matches the GEMM kernel fusion order: cast to the input dtype first,
            # then add the bias.
            acc = acc.to(x.dtype) + bias
        return acc.to(self.out_dtype)


def get_init_inputs():
    return [True, "bfloat16"]   # use_bias; out_dtype


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness trials
    # therefore see diverse inputs. The value ranges match the generator
    # make_inputs (randint(-20,20), rand*10).
    b, m, n, k = 8, 128, 256, 512

    x = torch.randint(-20, 20, (b, m, k), dtype=torch.bfloat16)
    weight = torch.randint(-20, 20, (b, n, k), dtype=torch.bfloat16)
    bias = torch.rand(b, 1, n, dtype=torch.bfloat16) * 10
    return [x, weight, bias]


def make_inputs(b: int, m: int, n: int, k: int, use_bias: bool = True,
                out_dtype: str = "bfloat16", seed: int = 0):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    The consumption order is fixed: x -> weight -> bias, all from the same
    torch.Generator(seed), so the same seed reproduces inputs bit-for-bit.
    out_dtype only describes the output dtype (a Model construction parameter) and
    does not participate in input generation -- inputs are always bfloat16.
    """
    gen = torch.Generator().manual_seed(seed)

    x = torch.randint(-20, 20, (b, m, k), generator=gen, dtype=torch.bfloat16)
    weight = torch.randint(-20, 20, (b, n, k), generator=gen, dtype=torch.bfloat16)
    bias = torch.rand(b, 1, n, generator=gen, dtype=torch.bfloat16) * 10
    return x, weight, bias
