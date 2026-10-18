# 4006_gemm_a16w16 -- model_class (KernelBench compatible) task file for the
# bias-free 16-bit floating-point matrix multiplication (regular blocked-store
# variant).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=4006, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """Bias-free 16-bit floating-point matrix multiplication (regular blocked
    whole-reduction variant).

    Operator semantics:
      x has shape [M, K], weight has shape [N, K] (the linear-layer weight
      layout), and the output out has shape [M, N]:
        out[m, n] = sum_{k=0}^{K-1} x[m, k] * weight[n, k]
      equivalent to torch.nn.functional.linear(x, weight, bias=None). M, N, K are
      arbitrary positive integers (including 1 and non-powers-of-two; K need not
      align with any block size, and out-of-range lanes along K contribute 0 to the
      accumulation). Reduction strategy: the full K reduction is completed inside a
      single program -- each output tile is computed and written back by exactly
      one program (M/N boundaries are handled by masking, tiles never overlap),
      with no split-K, no atomics and no cross-writes to the same output; the
      float32 accumulated result is cast back to the input dtype once, so there is
      no intermediate half-precision rounding.

    Input/output specification:
      forward(x, weight) -> out
        x      [M, K], float16 or bfloat16, row-major contiguous
        weight [N, K], same dtype as x, row-major contiguous
        out    [M, N], same dtype as x
      __init__(in_features, out_features): in_features corresponds to K and
      out_features to N; they are instance metadata only. forward always follows
      the actual shapes of the runtime tensors and does not enforce a hard match
      against the init parameters.

    Implementation constraints (violations are scored as failures):
      - The core computation (the multiply-accumulate main loop) must be completed
        with a Triton kernel inside the submitted file; calling ATen GEMM shortcuts
        such as torch.matmul / torch.bmm / torch.mm / torch.addmm / torch.einsum /
        torch.nn.functional.linear / F.linear / aiter is forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Accumulate intermediates in float32; cast the output back to the input
        dtype.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: row-major TN layout, no bias, no batch dimension, single output
    tensor, float16 and bfloat16 input dtypes; the block sizes, tile-group
    scheduling and parallel strategy are left to the implementation and judged by
    the output tolerance.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, weight):
        # The KernelBench evaluator casts every input tensor to the evaluation
        # precision (fp32 for the triton backend); this task has no integer
        # tensors, so fp32 inputs simply take the same float32 accumulation path,
        # and the output dtype follows the runtime input.
        x32 = x.to(torch.float32)
        w32 = weight.to(torch.float32)
        assert x32.shape[1] == w32.shape[1], "x and weight must agree on the K dimension"
        # [M, K] @ [K, N] -> [M, N], accumulated once in float32
        out = torch.mm(x32, w32.t())
        return out.to(x.dtype)


def get_init_inputs():
    return [1024, 8192]  # in_features(K), out_features(N)


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs.
    m = 256
    n = 8192
    k = 1024
    x = torch.randn(m, k).to(torch.bfloat16)
    weight = torch.randn(n, k).to(torch.bfloat16)
    return [x, weight]


def make_inputs(m: int, n: int, k: int, dtype: str = "bfloat16", seed: int = 0):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    x = torch.randn(m, k, generator=gen).to(dt)
    weight = torch.randn(n, k, generator=gen).to(dt)
    return x, weight
