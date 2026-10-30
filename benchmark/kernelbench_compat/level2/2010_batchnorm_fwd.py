# 2010_batchnorm_fwd -- model_class (KernelBench compatible) task file for the
# BatchNorm forward family (inference / training).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=2, problem_id=2010, backend=cuda (HIP).

import torch
import torch.nn as nn


class Model(nn.Module):
    """BatchNorm forward family: inference affine (inference) and training batch
    statistics (training).

    The input is x (N, C, H, W), NCHW contiguous; scale / bias / mean / variance are
    all (1, C, 1, 1) and broadcast along the channel dimension. Reduction dimension:
    per-channel, computing the statistics of every channel c over the three
    dimensions (N, H, W), with m = N*H*W elements per channel. The __init__ argument
    variant selects one of two paths (eps defaults to 1e-5 and momentum to 0.1; both
    are scalar hyper-parameters only):

        "inference": the mean and the variance are given as inputs, element-wise
                     affine (float32 intermediate computation):
            y = scale * (x - mean) / sqrt(variance + eps) + bias
        "training": batch statistics + affine + running update (float32
                    throughout):
            batch_mean_c = sum_{n,h,w} x[n,c,h,w] / m                    (biased batch mean)
            batch_var_c  = sum_{n,h,w} (x[n,c,h,w] - batch_mean_c)^2 / m (biased batch variance)
            saved_mean         = batch_mean                              (shape (1,C,1,1))
            saved_inv_variance = 1 / sqrt(batch_var + eps)               (shape (1,C,1,1))
            y = scale * (x - saved_mean) * saved_inv_variance + bias
            next_running_mean = (1 - momentum) * mean + momentum * batch_mean
            next_running_var  = (1 - momentum) * variance
                                + momentum * batch_var * m / (m - 1)
            In the training variant the 4th/5th forward arguments are the running
            statistics of the previous step (prev_running_mean / prev_running_var);
            the running variance update uses the unbiased batch variance (the
            m/(m-1) correction, hence the requirement m >= 2) -- this is the
            standard PyTorch / cuDNN / MIOpen training-BN convention.

    The forward signature is (x, scale, bias, mean, variance) for both variants: in
    inference the 4th/5th arguments are the given mean / variance, in training they
    are prev_running_mean / prev_running_var.

    Single-tensor output protocol (forward must return a single tensor; the
    evaluator compares a single output):
        "inference": returns y itself, shape (N, C, H, W), same dtype as x;
        "training" : returns a one-dimensional float32 packed tensor of length
                     N*C*H*W + 4*C; the five logical outputs are each cast to
                     float32, reshaped with reshape(-1) and torch.cat'ed in the
                     sample-output order, with the segment protocol (half-open
                     intervals):
            [0,            N*C*H*W)      y
            [N*C*H*W,      N*C*H*W + C)  saved_mean
            [N*C*H*W + C,  N*C*H*W + 2C) saved_inv_variance
            [N*C*H*W + 2C, N*C*H*W + 3C) next_running_mean
            [N*C*H*W + 3C, N*C*H*W + 4C) next_running_var

    Input/output specification:
        x        float32 (always float32 for training) / float16 / bfloat16
                 (inference only), (N, C, H, W), contiguous, N/C/H/W >= 1
        scale    same dtype as x, (1, C, 1, 1), contiguous
        bias     same dtype as x, (1, C, 1, 1), contiguous
        mean     same dtype as x, (1, C, 1, 1), contiguous (inference: the given
                 mean, any real value; training: prev_running_mean)
        variance same dtype as x, (1, C, 1, 1), contiguous, always positive
                 (inference: the given variance; training: prev_running_var)
        out      inference: (N, C, H, W), same dtype as x; training: one-dimensional
                 float32 of length N*C*H*W + 4*C (the five-segment packing protocol
                 above)
        The inputs are the usual randn/rand magnitude (no inf/NaN); the evaluator
        casts all inputs to fp32 before passing them in (fp16/bf16 -> fp32 is
        lossless).

    Implementation constraints (violations are scored as failures):
        - The core computation (the per-channel reduction statistics, rsqrt, the
          affine, and the running update) must be completed inside the submitted
          file with a HIP kernel.
        - The __init__ and forward signatures of ModelNew must not be changed.
        - The statistics, the normalization and the running update all use float32
          intermediates; the y of inference is cast back to the x dtype; all five
          training segments are float32.
        - forward returns the single tensor out (the evaluator compares
          shape/dtype/allclose on a single output).

    Forbidden: torch.nn.BatchNorm1d / torch.nn.BatchNorm2d /
    torch.nn.BatchNorm3d / torch.nn.SyncBatchNorm /
    torch.nn.functional.batch_norm / torch.batch_norm /
    torch.native_batch_norm / miopen / hipdnn.
    Target hardware: Hygon DCU (gfx936/gfx938). Implementation language: HIP
    (compiled by hipcc; write inside the CUDA_CU_SRC / CUDA_CPP_SRC anchor regions;
    device code must include <hip/hip_runtime.h> -- <cuda_runtime.h> is not available
    on this machine).
    """

    def __init__(self, variant: str = "training", eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        assert variant in ("inference", "training"), "variant must be one of inference / training"
        assert eps > 0, "eps must be positive"
        assert 0.0 <= momentum <= 1.0, "momentum must be in [0, 1]"
        self.variant = variant
        self.eps = float(eps)
        self.momentum = float(momentum)

    def forward(self, x, scale, bias, mean, variance):
        eps = self.eps
        xf = x.to(torch.float32)
        sf = scale.to(torch.float32)
        bf = bias.to(torch.float32)
        mf = mean.to(torch.float32)
        vf = variance.to(torch.float32)

        if self.variant == "inference":
            # element-wise affine with the given mean/variance; (1,C,1,1)
            # broadcasts along the channel dimension
            inv_std = torch.rsqrt(vf + eps)
            y = (xf - mf) * inv_std * sf + bf
            return y.to(x.dtype)

        # training: per-channel biased batch statistics (reduction over (N,H,W));
        # y is normalized with the batch statistics
        m = xf.shape[0] * xf.shape[2] * xf.shape[3]
        assert m >= 2, "the training variant requires N*H*W >= 2 (the m/(m-1) unbiased correction factor of the running variance)"
        batch_mean = xf.mean(dim=(0, 2, 3), keepdim=True)                        # (1,C,1,1) biased mean
        batch_var = ((xf - batch_mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)   # (1,C,1,1) biased variance
        saved_inv_variance = torch.rsqrt(batch_var + eps)
        y = (xf - batch_mean) * saved_inv_variance * sf + bf

        # running statistics update: the mean uses the batch mean, the variance uses
        # the unbiased batch variance (the m/(m-1) correction)
        batch_var_unbiased = batch_var * (m / (m - 1))
        next_running_mean = (1.0 - self.momentum) * mf + self.momentum * batch_mean
        next_running_var = (1.0 - self.momentum) * vf + self.momentum * batch_var_unbiased

        # single-tensor output: the five segments concatenated in one dimension in
        # the sample-output order (all float32)
        return torch.cat([
            y.to(torch.float32).reshape(-1),
            batch_mean.to(torch.float32).reshape(-1),
            saved_inv_variance.to(torch.float32).reshape(-1),
            next_running_mean.to(torch.float32).reshape(-1),
            next_running_var.to(torch.float32).reshape(-1),
        ])


def get_init_inputs():
    return ["training", 1e-5, 0.1]   # variant, eps, momentum


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the evaluator
    # calls this function after set_seed, so multiple correctness trials therefore
    # see diverse inputs. The generation rules match make_inputs (consumption order
    # x -> scale -> bias -> mean -> variance; here the global RNG is consumed)
    n, c, h, w = 4, 16, 56, 56
    x = torch.randn(n, c, h, w)
    scale = torch.rand(1, c, 1, 1) + 0.5
    bias = torch.randn(1, c, 1, 1)
    mean = torch.randn(1, c, 1, 1)              # training: prev_running_mean
    variance = torch.rand(1, c, 1, 1) + 0.25    # training: prev_running_var
    return [x, scale, bias, mean, variance]


def make_inputs(n: int, c: int, h: int, w: int, variant: str = "training",
                dtype: str = "float32", seed: int = 0,
                eps: float = 1e-5, momentum: float = 0.1):
    """Generate inputs from a public/hidden case description. Runs on the evaluation
    side only (generated on CPU; the evaluator moves them to the device).

    The consumption order is fixed: x -> scale -> bias -> mean -> variance. x/bias/
    mean use torch.randn, scale uses torch.rand + 0.5 (the positive scaling range
    [0.5, 1.5)), variance uses torch.rand + 0.25 (the always-positive range
    [0.25, 1.25)); everything comes from the same torch.Generator(seed), so the same
    seed reproduces inputs bit-for-bit.
    variant / eps / momentum are also the init hyper-parameters used to construct the
    Model and are only validated here; the dtype domain of the training variant
    declares float32 only (the packed-output protocol) and it requires N*H*W >= 2.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert variant in ("inference", "training"), "variant must be one of inference / training"
    assert min(n, c, h, w) >= 1, "N/C/H/W must all be >= 1"
    assert eps > 0 and 0.0 <= momentum <= 1.0, "eps must be positive and momentum must be in [0, 1]"
    if variant == "training":
        assert dt == torch.float32, "the dtype domain of the training variant declares float32 only"
        assert n * h * w >= 2, "the training variant requires N*H*W >= 2"
    x = torch.randn(n, c, h, w, generator=gen).to(dt)
    scale = (torch.rand(1, c, 1, 1, generator=gen) + 0.5).to(dt)
    bias = torch.randn(1, c, 1, 1, generator=gen).to(dt)
    mean = torch.randn(1, c, 1, 1, generator=gen).to(dt)
    variance = (torch.rand(1, c, 1, 1, generator=gen) + 0.25).to(dt)
    return x, scale, bias, mean, variance
