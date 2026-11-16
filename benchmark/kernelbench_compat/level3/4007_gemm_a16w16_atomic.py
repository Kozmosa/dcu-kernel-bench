# 4007_gemm_a16w16_atomic -- model_class (KernelBench compatible) task file for
# the bias-free 16-bit floating-point matrix multiplication (split-K atomic
# reduction variant).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=4007, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """Bias-free 16-bit floating-point matrix multiplication (the semantic core of
    the split-K atomic-reduction variant).

    Operator semantics:
      x has shape [M, K], weight has shape [N, K] (the linear-layer weight
      layout), and the output out has shape [M, N]:
        out[m, n] = sum_{k=0}^{K-1} x[m, k] * weight[n, k]
      equivalent to torch.nn.functional.linear(x, weight, bias=None). M, N, K are
      arbitrary positive integers (including 1 and non-powers-of-two; K need not
      align with any block size, and out-of-range lanes along K contribute 0 to the
      accumulation). The original operator splits along K: each split performs its
      own float32 multiply-accumulate, and the partial sums are merged into the
      same output tensor by atomic adds, so the output as a whole is narrowed only
      once -- the float32 accumulated result is cast straight back to the input
      dtype with no per-split half-precision intermediate rounding. This reference
      implementation performs the entire accumulation once in float32 under the
      same semantics.

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
        torch.nn.functional.linear / F.linear, or any ready-made operator library,
        is forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Accumulate intermediates in float32; cast the output back to the input
        dtype.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: row-major TN layout, no bias, single output tensor, float16 and
    bfloat16 input dtypes; the number of split-K splits and the parallel strategy
    are left to the implementation and judged by the output tolerance.
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
    return [4096, 1280]  # in_features(K), out_features(N)


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs.
    m = 64
    n = 1280
    k = 4096
    x = torch.randn(m, k).to(torch.float16)
    weight = torch.randn(n, k).to(torch.float16)
    return [x, weight]


def make_inputs(m: int, n: int, k: int, dtype: str = "float16", seed: int = 0):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    x = torch.randn(m, k, generator=gen).to(dt)
    weight = torch.randn(n, k, generator=gen).to(dt)
    return x, weight
