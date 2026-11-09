# 2011_batchnorm_bwd -- KernelBench compatible task file for the BatchNorm training
# backward pass (data gradient dx / parameter gradients dgamma+dbeta).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem-id=2011, backend=cuda (HIP).

import torch
import torch.nn as nn


class Model(nn.Module):
    """BatchNorm training backward: given the upstream gradient dy, the input x of
    the training forward, the saved statistics (saved_mean / saved_inv_variance) and
    the scale parameter, compute the gradients of the normalized forward
        y = scale * xhat + bias,  xhat = (x - saved_mean) * saved_inv_variance
    The __init__ argument variant selects which gradients are returned:
        "dx"           : returns only the data gradient dx, shape (N, C, H, W), same
                         dtype as x;
        "dgamma_dbeta" : returns only the parameter gradients; dgamma and dbeta are
                         each (1, C, 1, 1) and are emitted as a single 1-D tensor
                         (2*C,) following the "one-dimensional concatenation packing
                         protocol": the first C elements are dgamma (flattened in
                         row-major order) and the last C are dbeta.
    Both variants share the same forward signature; the return value is always a
    single tensor (the evaluator compares a single output). Mathematical definition
    (float32 intermediate computation throughout, m = N*H*W elements per channel):
        xhat         = (x - saved_mean) * saved_inv_variance
        dgamma[c]    = sum_{n,h,w} dy * xhat          (summed per channel)
        dbeta[c]     = sum_{n,h,w} dy                 (summed per channel)
        dx           = (scale * saved_inv_variance / m)
                       * (m*dy - dbeta - xhat * dgamma)
    saved_mean / saved_inv_variance are consumed verbatim as the given values: they
    are the statistics saved by the training forward (inv_variance = 1/sqrt(biased
    variance + eps); eps was already absorbed in the forward, and the backward math
    itself contains no eps) and may be any positive numbers unrelated to the true
    statistics of x -- the given values must be used directly, and recomputing the
    statistics from x is forbidden. saved_inv_variance is always positive.

    Input/output specification (NCHW, contiguous):
        dy                 float32 / float16 / bfloat16 (N, C, H, W)
        x                  same dtype as dy (N, C, H, W)
        scale              same dtype as dy (1, C, 1, 1)
        saved_mean         float32 (1, C, 1, 1) (the statistics are always float32,
                           independent of the input dtype)
        saved_inv_variance float32 (1, C, 1, 1), its elements are always positive
        out                single tensor: (N, C, H, W) with the same dtype as dy for
                           dx; (2*C,) with the same dtype as dy for dgamma_dbeta (the
                           first C are dgamma and the last C are dbeta, both computed
                           in float32 and then cast)
        The evaluator casts all inputs to fp32 before passing them in (fp16/bf16 ->
        fp32 is lossless).

    Implementation constraints (violations are scored as failures):
        - The core computation (xhat reconstruction, the two channel reductions, and
          the element-wise chained multiply-add of dx) must be completed inside the
          submitted file with a HIP kernel; a channel reduction of any size (m =
          N*H*W may exceed the capacity of a single block) must be handled correctly
          with mask / tiling (block-wise) reductions, leaving out no element of the
          channel.
        - Computing the gradients through torch.autograd / autograd.grad /
          .backward() is forbidden, as is calling the batch_norm primitive family or
          any closed-source library primitive (MIOpen / hipDNN and the like).
        - The __init__ and forward signatures of ModelNew must not be changed.
        - The reductions and the intermediate computation run in float32; the output
          is cast back to the input dtype; the packing order of dgamma_dbeta is fixed
          as the dgamma segment first and the dbeta segment last.

    Boundary behaviour: N, C, H and W may all be 1 (m may be 1, in which case
    dx = scale * inv_var * (dy - dy*xhat^2)); non-powers of two and unaligned
    dimensions are allowed; the statistics shape is always (1, C, 1, 1); the input
    value range is the usual randn magnitude (no inf/NaN).
    Forbidden: aiter / torch.batch_norm / torch.native_batch_norm /
    torch.nn.functional.batch_norm / F.batch_norm / torch.nn.BatchNorm1d /
    torch.nn.BatchNorm2d / torch.nn.BatchNorm3d /
    torch.ops.aten.batch_norm / torch.ops.aten.native_batch_norm /
    autograd / .backward(.
    Target hardware: Hygon DCU (gfx936/gfx938). Implementation language: HIP
    (compiled by hipcc; write inside the CUDA_CU_SRC / CUDA_CPP_SRC anchor regions;
    device code must include <hip/hip_runtime.h> -- <cuda_runtime.h> is not available
    on this machine).
    """

    def __init__(self, variant: str = "dx"):
        super().__init__()
        assert variant in ("dx", "dgamma_dbeta"), "variant must be one of dx / dgamma_dbeta"
        self.variant = variant

    def forward(self, dy, x, scale, saved_mean, saved_inv_variance):
        # float32 intermediate computation throughout; the statistics are consumed
        # verbatim as given and are not recomputed from x
        dyf = dy.to(torch.float32)
        xf = x.to(torch.float32)
        mean = saved_mean.to(torch.float32)
        inv_var = saved_inv_variance.to(torch.float32)

        xhat = (xf - mean) * inv_var                    # (N, C, H, W)
        dgamma_c = (dyf * xhat).sum(dim=(0, 2, 3))      # (C,)
        dbeta_c = dyf.sum(dim=(0, 2, 3))                # (C,)

        if self.variant == "dgamma_dbeta":
            # single-tensor packing: the dgamma segment first and the dbeta segment
            # last, each flattened in row-major order
            return torch.cat([dgamma_c.to(x.dtype).reshape(-1),
                              dbeta_c.to(x.dtype).reshape(-1)])

        m = x.shape[0] * x.shape[2] * x.shape[3]        # elements per channel
        scalef = scale.to(torch.float32)
        dx = (scalef * inv_var / m) * (m * dyf - dbeta_c.reshape_as(scalef)
                                       - xhat * dgamma_c.reshape_as(scalef))
        return dx.to(x.dtype)


def get_init_inputs():
    return ["dx"]   # variant


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG (order dy -> x ->
    # scale) -- the evaluator calls this function after set_seed, so multiple
    # correctness trials therefore see diverse inputs.
    # The saved statistics are derived deterministically from x in the training
    # snapshot convention (no RNG consumed):
    # mean = the channel mean of x, inv_variance = 1/sqrt(biased variance + 1e-5).
    n, c, h, w = 4, 16, 56, 56
    dy = torch.randn(n, c, h, w).to(torch.float32)
    x = torch.randn(n, c, h, w).to(torch.float32)
    scale = torch.randn(1, c, 1, 1).to(torch.float32)
    xf = x.to(torch.float32)
    mean = xf.mean(dim=(0, 2, 3), keepdim=True)
    var = xf.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
    inv_var = torch.rsqrt(var + 1e-5)
    return [dy, x, scale, mean, inv_var]


def make_inputs(n: int, c: int, h: int, w: int, variant: str = "dx",
                dtype: str = "float32", seed: int = 0, eps: float = 1e-5,
                stats: str = "derived"):
    """Generate inputs from a public/hidden/perf case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the device).

    The consumption order of the random part is fixed: dy -> x -> scale (all
    torch.randn, then cast to dtype); everything comes from the same
    torch.Generator(seed), so the same seed reproduces inputs bit-for-bit.
    The saved statistics (float32, not cast to dtype) are decided by the stats
    convention:
        "derived": the training snapshot convention (default) -- derived from x in
                   the same convention as the 2010 forward, with mean = the channel
                   mean of x.float() and inv_variance = rsqrt(the biased channel
                   variance of x.float() + eps); no RNG is consumed;
        "random" : the sample convention -- the statistics are independent random
                   values and the RNG keeps being consumed:
                   saved_mean = rand(1,C,1,1), saved_inv_variance =
                   rand(1,C,1,1) + 1e-3 (guaranteed positive); the statistics are
                   then unrelated to x, which pins down the semantics that the given
                   statistics must be consumed verbatim and must not be recomputed
                   from x.
    eps takes part in the statistics generation only under the derived convention;
    the backward math itself contains no eps.
    variant is also the init hyper-parameter used to construct the Model and is only
    validated here.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert variant in ("dx", "dgamma_dbeta"), "variant must be one of dx / dgamma_dbeta"
    assert stats in ("derived", "random"), "stats must be one of derived / random"
    assert n >= 1 and c >= 1 and h >= 1 and w >= 1, "requires N/C/H/W to all be >= 1"

    dy = torch.randn(n, c, h, w, generator=gen).to(dt)
    x = torch.randn(n, c, h, w, generator=gen).to(dt)
    scale = torch.randn(1, c, 1, 1, generator=gen).to(dt)

    if stats == "random":
        saved_mean = torch.rand(1, c, 1, 1, generator=gen)
        saved_inv_variance = torch.rand(1, c, 1, 1, generator=gen) + 1.0e-3
    else:
        xf = x.to(torch.float32)
        saved_mean = xf.mean(dim=(0, 2, 3), keepdim=True)
        var = xf.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        saved_inv_variance = torch.rsqrt(var + float(eps))

    return dy, x, scale, saved_mean, saved_inv_variance
