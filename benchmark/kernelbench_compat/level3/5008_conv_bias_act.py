# 5008_conv_bias_act -- model_class (KernelBench compatible) task file for the
# fused Conv+Bias+element-wise-activation forward pass (relu / swish / prelu).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=5008, backend=cuda (HIP).

import torch
import torch.nn as nn


class Model(nn.Module):
    """Conv + Bias + element-wise activation fused forward family: three
    variants, relu / swish / prelu.

    For the input x (N, C, H, W) and the kernel weight (K, C, R, S), run the 2D
    cross-correlation forward pass (the deep-learning-library conv_fprop
    convention, not the kernel-flipped mathematical convolution), add the
    per-channel bias bias (1, K, 1, 1) (broadcast along the batch and spatial
    dimensions), then apply the element-wise activation. The convolution
    hyper-parameters are shared by the three variants: stride=(1, 1),
    dilation=(1, 1), symmetric zero padding=(1, 1) (1 added on each of the
    top/bottom/left/right sides; the padding value is always zero). Element-wise
    definition:
        z[b, k, i, j] = ( sum_{c, u, v} x[b, c, i+u-1, j+v-1] *
                          weight[k, c, u, v] ) + bias[0, k, 0, 0]
        out[b, k, i, j] = act(z[b, k, i, j])
    where i in [0, H_out), j in [0, W_out), H_out = H - R + 3 and
    W_out = W - S + 3 (the standard convolution output formula once
    stride/padding/dilation are fixed). The activation is selected among three
    by the __init__ parameter variant (z is the convolution result after adding
    bias):
        "relu"  : out = max(z, 0)
        "swish" : out = z * sigmoid(z) = z / (1 + exp(-z))
        "prelu" : out = z (when z > 0), otherwise slope * z (a scalar slope,
                  equivalent to leaky-ReLU with negative_slope=slope; slope is a
                  forward input tensor rather than a module parameter)
    forward returns the single tensor out (N, K, H_out, W_out), with the same
    dtype as x (convolution is inherently single-output, so there are no
    multi-output packing segments).

    Input/output specification:
        x      float16 / bfloat16 / float32 (N, C, H, W), contiguous,
               N/C/H/W >= 1
        weight same dtype as x (K, C, R, S), contiguous, K/R/S >= 1, the
               input-channel dimension equals C
        bias   same dtype as x (1, K, 1, 1), contiguous (per-channel bias,
               broadcast over the whole batch and all spatial positions)
        slope  a 0-dim scalar tensor of the same dtype as x; it takes part in
               the computation only for variant="prelu" (default 0.01, may
               vary); for variant="relu"/"swish" its contents are arbitrary
               (garbage data) and must not affect the output
        out    a single tensor, same dtype as x (N, K, H_out, W_out)
        Inputs are ordinary randn magnitude (no inf/NaN); the evaluator casts
        all inputs to fp32 before the call (fp16/bf16 -> fp32 is lossless; the
        output dtype of this task always matches the dtype of the passed x).

    Implementation constraints (violations are scored as failures):
        - The core computation (the padding-aware index/mask expansion and the
          dot-product accumulation, the bias addition, the element-wise
          activation) must all be done and fused with a HIP kernel inside the
          submitted file; replacing it with any form of library convolution or
          ATen GEMM (matmul/mm/bmm/addmm/baddbmm/einsum/tensordot/unfold/linear
          etc., including one-step equivalent routes such as a conv3d
          depth-axis unsqueeze or a conv_transpose2d kernel-flip rewrite) is
          forbidden.
        - The __init__ and forward signatures of ModelNew must not be changed.
        - The convolution dot-product accumulation, the bias addition and the
          activation are computed in float32 and the output is cast back to the
          input dtype exactly once (do not accumulate in fp16/bf16 low
          precision, and do not make repeated round-trip casts); padding must be
          handled inside the kernel with boundary masks/indices -- do not
          materialize a padded copy on the host side in place of in-kernel
          handling.
        - forward returns the single tensor out (the evaluator compares
          shape/dtype/allclose on a single output).

    Forbidden (static audit strips docstrings and comments, then does literal
    substring matching):
    torch.nn.Conv2d / torch.nn.functional.conv2d / F.conv2d /
    torch.conv2d / aten.conv2d / aten.convolution /
    torch.nn.functional.conv3d / torch.conv3d / torch.nn.Conv3d /
    torch.nn.functional.conv_transpose2d / torch.conv_transpose2d /
    torch.nn.ConvTranspose2d /
    torch.nn.functional.unfold / torch.nn.Unfold / .unfold( /
    torch.matmul / .matmul( / torch.mm / torch.bmm / torch.addmm /
    torch.baddbmm / torch.einsum / torch.tensordot /
    torch.nn.functional.linear / torch.nn.Linear /
    miopen / hipdnn.
    Target hardware: Hygon DCU (gfx936/gfx938). Implementation language: HIP
    (compiled by hipcc; write inside the CUDA_CU_SRC / CUDA_CPP_SRC anchor
    regions; device code must include <hip/hip_runtime.h> -- <cuda_runtime.h>
    is not available on this machine).
    """

    # Convolution hyper-parameters shared by the three variants (sample
    # parameter set)
    _STRIDE = (1, 1)
    _PADDING = (1, 1)   # symmetric zero padding: 1 added on each side
    _DILATION = (1, 1)

    def __init__(self, variant: str = "relu"):
        super().__init__()
        assert variant in ("relu", "swish", "prelu"), \
            "variant must be one of relu / swish / prelu"
        self.variant = variant

    def forward(self, x, weight, bias, slope):
        # conv accumulates in float32 (inputs of the io dtype are first promoted
        # losslessly), and bias is broadcast-added as (1, K, 1, 1)
        z = torch.nn.functional.conv2d(
            x.to(torch.float32), weight.to(torch.float32), bias=None,
            stride=self._STRIDE, padding=self._PADDING, dilation=self._DILATION,
        )
        z = z + bias.to(torch.float32)

        # Activation in three branches (if/elif branching avoids multi-clause
        # compound statements)
        if self.variant == "relu":
            z = torch.relu(z)
        elif self.variant == "swish":
            z = z * torch.sigmoid(z)
        else:   # prelu: pass z through when z > 0, otherwise slope * z
            sf = slope.to(torch.float32)
            z = torch.where(z > 0, z, z * sf)

        # Cast the output back to the input dtype exactly once
        return z.to(x.dtype)


def get_init_inputs():
    return ["relu"]   # variant


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs
    n, c, h, w = 2, 16, 32, 32
    k, r, s = 16, 3, 3
    x = torch.randn(n, c, h, w).to(torch.float16)
    weight = torch.randn(k, c, r, s).to(torch.float16)
    bias = torch.randn(1, k, 1, 1).to(torch.float16)
    slope = torch.tensor(0.01).to(torch.float16)   # 0-dim scalar tensor, does not consume the RNG
    return [x, weight, bias, slope]


def make_inputs(n: int, c: int, h: int, w: int, k: int, r: int, s: int,
                variant: str = "relu", dtype: str = "float16",
                slope: float = 0.01, seed: int = 0):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    The consumption order is fixed: x -> weight -> bias (all torch.randn, then
    cast to dtype), all drawn from the same torch.Generator(seed), so the same
    seed reproduces inputs bit-for-bit; slope is materialized directly from the
    case field as a 0-dim scalar tensor (cast to dtype) and does not consume the
    RNG.
    variant is also the init hyper-parameter used to construct the Model; its
    validity, the declared dtype domain and the shape domain are checked here
    (every dimension >= 1 and the output dimension >= 1, i.e. R <= H+2 and
    S <= W+2).
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert dt in (torch.float16, torch.bfloat16, torch.float32), \
        "the dtype domain declared by this task is float16 / bfloat16 / float32 only"
    assert variant in ("relu", "swish", "prelu"), \
        "variant must be one of relu / swish / prelu"
    assert min(n, c, h, w, k, r, s) >= 1, "N/C/H/W/K/R/S must all be >= 1"
    assert r <= h + 2 and s <= w + 2, "the kernel exceeds the padded input, so the output dimension is negative"
    x = torch.randn(n, c, h, w, generator=gen).to(dt)
    weight = torch.randn(k, c, r, s, generator=gen).to(dt)
    bias = torch.randn(1, k, 1, 1, generator=gen).to(dt)
    slope_t = torch.tensor(float(slope)).to(dt)
    return x, weight, bias, slope_t
