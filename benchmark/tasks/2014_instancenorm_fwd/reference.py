# 2014_instancenorm_fwd -- model_class (KernelBench compatible) task file for the
# InstanceNorm forward pass (per-(N,C) spatial statistics plus per-channel
# affine).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=2, problem_id=2014, backend=cuda (HIP).

import torch
import torch.nn as nn


class Model(nn.Module):
    """InstanceNorm forward pass: per-(N,C) spatial-statistics normalization plus
    per-channel affine (single variant).

    The input x (N, C, H, W) is NCHW contiguous; scale / bias are the (1, C, 1, 1)
    per-channel affine parameters, broadcast along the channel dimension; eps
    defaults to 1e-5. Statistics are taken over the spatial dimensions (H, W) for
    every (n, c) plane (HW = H*W elements per plane), in float32:
        mean_nc    = sum_{h,w} x[n,c,h,w] / HW                  (biased mean)
        var_nc     = sum_{h,w} (x[n,c,h,w] - mean_nc)^2 / HW    (biased variance)
        inv_std_nc = 1 / sqrt(var_nc + eps)
        y[n,c,h,w] = scale[c] * (x[n,c,h,w] - mean_nc) * inv_std_nc + bias[c]
    Instance norm has no train / inference semantic difference: the normalization
    statistics always come from the input itself. The convention is biased (divide
    by HW rather than HW-1), and eps sits inside the rsqrt (its only purpose is the
    division-by-zero guard; it takes part in no unbiased correction).

    Boundary behaviour: when HW == 1 (a single spatial element) var = 0 and
    x - mean = 0, so inv_std = 1/sqrt(eps) and y = bias (eps guarantees no division
    by zero); N/C/H/W are arbitrary (including 1, non-powers-of-two, and unaligned
    sizes); a large HW requires a blocked reduction, and out-of-range elements at
    the row tail must not participate in the statistics.

    Input/output specification:
        x     float32 / float16 / bfloat16 (N, C, H, W), contiguous,
              N/C/H/W >= 1
        scale same dtype as x, (1, C, 1, 1), contiguous
        bias  same dtype as x, (1, C, 1, 1), contiguous
        out   a single tensor, y itself: (N, C, H, W) with the same dtype as x
              (statistics and affine run in float32 intermediates, then cast back
              to the x dtype); no packing
        The inputs are ordinary randn/rand magnitude (no inf/NaN); the evaluator
        casts all inputs to fp32 before calling forward (fp16/bf16 -> fp32 is
        lossless).

    Implementation constraints (violations are scored as failures):
        - The core computation (the per-(N,C) spatial reduction statistics, rsqrt,
          and the per-channel affine) must be completed with a HIP kernel inside
          the submitted file.
        - The __init__ and forward signatures of ModelNew must not be changed.
        - The statistics, rsqrt and affine all run in float32 intermediates; y is
          cast back to the x dtype.
        - forward returns the single tensor `out` (the evaluator compares
          shape/dtype/allclose on a single output).

    Forbidden list: torch.nn.InstanceNorm1d / torch.nn.InstanceNorm2d /
    torch.nn.InstanceNorm3d / torch.nn.functional.instance_norm /
    F.instance_norm / torch.instance_norm / native_instance_norm /
    torch.nn.GroupNorm / nn.GroupNorm / torch.nn.functional.group_norm /
    F.group_norm / torch.group_norm / native_group_norm /
    torch.nn.BatchNorm1d / torch.nn.BatchNorm2d / torch.nn.BatchNorm3d /
    torch.nn.SyncBatchNorm / torch.nn.functional.batch_norm /
    F.batch_norm / torch.batch_norm / torch.native_batch_norm /
    miopen / hipdnn.
    (GroupNorm with num_groups=C and BatchNorm via a (1, N*C, H, W) reshape both
    complete this task's core computation in a single step, so they are forbidden
    alongside the instance_norm family.)
    Target hardware: Hygon DCU (gfx936/gfx938). Implementation language: HIP
    (compiled by hipcc; write inside the CUDA_CU_SRC / CUDA_CPP_SRC anchor
    regions; device code must include <hip/hip_runtime.h> -- <cuda_runtime.h>
    is not available on this machine).
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        assert eps > 0, "eps must be positive"
        self.eps = float(eps)

    def forward(self, x, scale, bias):
        # per-(N,C) spatial statistics: reduce over (H, W) with float32
        # intermediates (biased mean / biased variance)
        xf = x.to(torch.float32)
        mean = xf.mean(dim=(2, 3), keepdim=True)                   # (N,C,1,1)
        var = ((xf - mean) ** 2).mean(dim=(2, 3), keepdim=True)    # (N,C,1,1) biased variance
        inv_std = torch.rsqrt(var + self.eps)

        # Per-channel affine: (1,C,1,1) broadcasts along the channel dimension;
        # computed in float32 and then cast back to the x dtype.
        sf = scale.to(torch.float32)
        bf = bias.to(torch.float32)
        y = (xf - mean) * inv_std * sf + bf
        return y.to(x.dtype)


def get_init_inputs():
    return [1e-5]   # eps


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the evaluator
    # calls this function after set_seed, so multiple correctness trials therefore
    # see diverse inputs. The generation rules match make_inputs (consumption order
    # x -> scale -> bias; here the global RNG is consumed).
    n, c, h, w = 16, 32, 64, 64
    x = torch.randn(n, c, h, w)
    scale = torch.rand(1, c, 1, 1) + 0.5
    bias = torch.randn(1, c, 1, 1)
    return [x, scale, bias]


def make_inputs(n: int, c: int, h: int, w: int,
                dtype: str = "float32", seed: int = 0, eps: float = 1e-5):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    The consumption order is fixed: x -> scale -> bias. x/bias use torch.randn and
    scale uses torch.rand + 0.5 (a positive scaling range [0.5, 1.5)); everything
    comes from the same torch.Generator(seed), so the same seed reproduces inputs
    bit-for-bit.
    eps is also the init hyper-parameter used to construct the Model and is only
    validated here (it does not participate in input generation).
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert min(n, c, h, w) >= 1, "N/C/H/W must all be >= 1"
    assert eps > 0, "eps must be positive"
    assert dt in (torch.float32, torch.float16, torch.bfloat16), \
        "dtype must be float32 / float16 / bfloat16"
    x = torch.randn(n, c, h, w, generator=gen).to(dt)
    scale = (torch.rand(1, c, 1, 1, generator=gen) + 0.5).to(dt)
    bias = torch.randn(1, c, 1, 1, generator=gen).to(dt)
    return x, scale, bias
