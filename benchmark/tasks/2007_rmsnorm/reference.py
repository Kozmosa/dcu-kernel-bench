# 2007_rmsnorm -- model_class (KernelBench compatible) task file for the RMSNorm
# forward family (standard / residual-fused).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=2, problem_id=2007, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """RMSNorm forward family: standard RMSNorm and residual-fused
    (fused residual-add) RMSNorm.

    For each row of an (M, N) input, the normalization is computed in float32 as
    defined below:
        mean_sq = sum_i x_i^2 / N
        rsigma  = rsqrt(mean_sq + eps)
        y_i     = x_i * rsigma * w_i
    where w is the per-channel scale vector. The __init__ parameter `variant`
    selects one of two paths:
        "fused_add": first compute residual_out = x + residual in the input dtype
        (the residual stream passes straight through to the output), then apply the
        RMSNorm above to residual_out;
        "rmsnorm"  : apply the RMSNorm directly to x; residual takes no part in the
        computation (its content is arbitrary and may be garbage data).
    forward returns a single tensor out (same dtype as x): with fused_add it is
    (M, 2N) with the first N columns equal to y and the last N columns equal to
    residual_out (the two logical outputs concatenated along the last dimension);
    with rmsnorm it is (M, N), i.e. y.
    Boundary behaviour: for an all-zero row mean_sq = 0, so rsigma = rsqrt(eps)
    (eps guarantees no division by zero); N is arbitrary (including 1,
    non-powers-of-two, sizes not multiple of 16, and large N beyond a single
    block's reduction capacity) and must all be handled correctly with masking / a
    blocked reduction, and out-of-range elements at the row tail must not
    contribute to the sum of squares.

    Input/output specification:
        x        float16 / bfloat16 (M, N), contiguous, M >= 1, N >= 1
        weight   same dtype as x, (N,), contiguous
        residual same dtype as x, (M, N), contiguous (ignored when variant="rmsnorm")
        out      a single tensor, same dtype as x: (M, 2N) with fused_add (first N
                 columns y, last N columns residual_out), (M, N) with rmsnorm (i.e. y)
        The evaluator casts all inputs to fp32 before calling forward
        (fp16/bf16 -> fp32 is lossless).

    Implementation constraints (violations are scored as failures):
        - The core computation (residual add, per-row sum-of-squares reduction,
          rsqrt, element-wise scaling) must be completed with a Triton kernel inside
          the submitted file.
        - The __init__ and forward signatures of ModelNew must not be changed.
        - The sum-of-squares accumulation and the normalization run in float32; y is
          cast back to the input dtype; residual_out is added in the input dtype and
          written out as-is.
        - forward returns the single tensor `out` (the evaluator compares a single
          output); with fused_add the residual_out column block must be bit-identical
          to x + residual in the input dtype.

    Forbidden list: aiter / torch.rms_norm / torch.nn.functional.rms_norm /
    torch.nn.RMSNorm / torch.layer_norm / torch.nn.functional.layer_norm /
    torch.nn.LayerNorm / torch.norm / torch.linalg.norm /
    torch.linalg.vector_norm.
    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    """

    def __init__(self, variant: str = "fused_add", eps: float = 1e-5):
        super().__init__()
        assert variant in ("rmsnorm", "fused_add"), "variant must be one of rmsnorm / fused_add"
        self.variant = variant
        self.eps = float(eps)

    def forward(self, x, weight, residual):
        # variant="fused_add": the residual stream is added in the input dtype and
        # passes straight through to the output; variant="rmsnorm": normalize x
        # directly and residual takes no part in the computation
        # (a conditional expression splits the paths, avoiding a multi-clause
        # compound statement).
        fused = self.variant == "fused_add"
        norm_input = x + residual if fused else x
        residual_out = norm_input

        # The RMSNorm body: float32 intermediates, output cast back to the input dtype
        xf = norm_input.to(torch.float32)
        wf = weight.to(torch.float32)
        mean_sq = (xf * xf).mean(dim=-1, keepdim=True)
        y = (xf * torch.rsqrt(mean_sq + self.eps) * wf).to(x.dtype)

        # Single-tensor output: fused_add concatenates y and residual_out along the
        # last dimension (first N columns y, last N columns residual_out).
        return torch.cat([y, residual_out], dim=-1) if fused else y


def get_init_inputs():
    return ["fused_add", 1e-5]   # variant, eps


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the evaluator
    # calls this function after set_seed, so multiple correctness trials therefore
    # see diverse inputs.
    m, n = 1024, 4096
    x = torch.randn(m, n).to(torch.float16)
    weight = torch.randn(n).to(torch.float16)
    residual = torch.randn(m, n).to(torch.float16)
    return [x, weight, residual]


def make_inputs(m: int, n: int, variant: str = "fused_add",
                dtype: str = "float16", seed: int = 0):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    The consumption order is fixed: x -> weight -> residual (all torch.randn, then
    cast to dtype), from the same torch.Generator(seed), so the same seed
    reproduces inputs bit-for-bit.
    variant is also the init hyper-parameter used to construct the Model and is
    only validated here.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert variant in ("rmsnorm", "fused_add"), "variant must be one of rmsnorm / fused_add"
    assert m >= 1 and n >= 1, "requires M >= 1 and N >= 1"
    x = torch.randn(m, n, generator=gen).to(dt)
    weight = torch.randn(n, generator=gen).to(dt)
    residual = torch.randn(m, n, generator=gen).to(dt)
    return x, weight, residual
