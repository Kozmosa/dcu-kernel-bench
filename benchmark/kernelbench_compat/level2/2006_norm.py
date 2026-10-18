# 2006_norm -- model_class (KernelBench compatible) task file for the forward
# pass of residual-fused LayerNorm (fused add LayerNorm).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=2, problem_id=2006, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """Forward pass of residual-fused LayerNorm (fused add LayerNorm).

    The inputs x and residual are added element-wise to form the residual sum s
    (the addition happens in float32, then s is rounded back to the input dtype
    for storage); LayerNorm is then applied to the last dimension of s together
    with an affine transform (everything evaluated in float32):
        mean_i  = (1/N) * sum_j s[i, j]
        var_i   = (1/N) * sum_j (s[i, j] - mean_i)^2      (biased variance)
        rstd_i  = 1 / sqrt(var_i + eps)                   (eps comes from __init__)
        y[i, j] = (s[i, j] - mean_i) * rstd_i * weight[j] + bias[j]
    The statistics are taken over s as rounded back to the input dtype; y is
    finally rounded back to the input dtype as well.
    Boundary behaviour: N is any positive integer (including 1, non-power-of-two,
    and very long rows), columns outside the mask do not participate in the
    statistics; when var = 0, eps keeps rstd finite; the inputs contain no
    inf/NaN.

    The output is a single tensor out (M, 2N) with the same dtype as the inputs:
        out[:, :N] = y           the LayerNorm output
        out[:, N:] = s           residual_out (the residual sum rounded to the input dtype)

    Input/output specification:
        x        (M, N)  float16 / bfloat16 / float32, contiguous, M >= 1, N >= 1
        residual (M, N)  same as x
        weight   (N,)    same as x (per-channel scale)
        bias     (N,)    same as x (per-channel shift)
        (the evaluator casts the inputs to fp32 before calling forward;
         fp16/bf16 -> fp32 is lossless)

    Implementation constraints (violations are scored as failures):
        - The core computation (residual add, mean/var statistics, normalization,
          affine transform) must be completed with a Triton kernel inside the
          submitted file.
        - The __init__ and forward signatures of ModelNew must not be changed.
        - Accumulate intermediates in float32; both y and residual_out are
          rounded back to the input dtype.

    Forbidden list: torch.nn.functional.layer_norm / torch.layer_norm /
    torch.nn.LayerNorm.
    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x, residual, weight, bias):
        # The evaluator casts every input to fp32; fp16/bf16 -> fp32 is lossless.
        # Compute uniformly in float32 and cast the outputs back to the input
        # dtype (fp32 online, the original dtype offline).
        dt = x.dtype
        x = x.to(torch.float32)
        residual = residual.to(torch.float32)
        w = weight.to(torch.float32)
        b = bias.to(torch.float32)

        # 1) residual add: sum in float32, round the stored value back to the
        #    input dtype
        residual_out = (x + residual).to(dt)

        # 2) LayerNorm statistics and normalization: taken over the rounded
        #    residual_out (matching the kernel semantics of writing back and then
        #    re-reading), all in float32
        z = residual_out.to(torch.float32)
        mean = z.mean(dim=-1, keepdim=True)
        var = (z - mean).pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + self.eps)
        y = ((z - mean) * rstd) * w + b
        y = y.to(dt)

        # Concatenate the two outputs into a single tensor: the first N columns
        # are y, the last N columns are residual_out.
        return torch.cat([y, residual_out], dim=-1)


def get_init_inputs():
    return [1e-5]  # eps


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs. Consumption order: x / residual /
    # weight / bias (randn / randn / rand / rand).
    m, n = 128, 1024
    x = torch.randn(m, n).to(torch.float16)
    residual = torch.randn(m, n).to(torch.float16)
    weight = torch.rand(n).to(torch.float16)
    bias = torch.rand(n).to(torch.float16)
    return [x, residual, weight, bias]


def make_inputs(m: int, n: int, dtype: str = "float16", seed: int = 0):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    The consumption order is fixed: x (randn), residual (randn), weight (rand),
    bias (rand); everything comes from the same torch.Generator(seed), so the
    same seed reproduces inputs bit-for-bit.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert m >= 1 and n >= 1, "requires M >= 1 and N >= 1"
    x = torch.randn(m, n, generator=gen).to(dt)
    residual = torch.randn(m, n, generator=gen).to(dt)
    weight = torch.rand(n, generator=gen).to(dt)
    bias = torch.rand(n, generator=gen).to(dt)
    return (x, residual, weight, bias)
