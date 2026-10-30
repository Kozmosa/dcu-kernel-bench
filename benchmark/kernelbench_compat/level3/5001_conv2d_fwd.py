# 5001_conv2d_fwd -- model_class (KernelBench compatible) task file for the
# standard NCHW 2D convolution forward pass (symmetric / asymmetric padding).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=5001, backend=cuda (HIP).

import torch
import torch.nn as nn


class Model(nn.Module):
    """Standard NCHW 2D convolution forward family: symmetric padding
    (standard) and asymmetric padding (asymmetric).

    For the input x (N, C, H, W) and the kernel weight (K, C, R, S), run the 2D
    cross-correlation forward pass (the deep-learning-library conv_fprop
    convention, not the kernel-flipped mathematical convolution). Element-wise
    definition:
        out[b, k, i, j] = sum_{c, u, v}
            x[b, c, i*stride_h + u*dil_h - pad_top, j*stride_w + v*dil_w -
              pad_left] * weight[k, c, u, v]
    Out-of-range indices read zero (zero padding). No bias; stride=(1, 1) and
    dilation=(1, 1) are fixed. The __init__ parameter variant selects the
    padding scheme:
        "standard"   : symmetric zero padding (pad_h, pad_w) = (1, 1) (1 added
                       on each of the top/bottom/left/right sides);
        "asymmetric" : asymmetric zero padding, pre_padding = (2, 2) (added on
                       the top of the height axis and the left of the width
                       axis), post_padding = (1, 1) (added on the bottom of the
                       height axis and the right of the width axis).
    Output size: H_out = floor((H + pad_pre_h + pad_post_h - (R-1)*dil_h - 1)
    / stride_h) + 1 (likewise for W_out), which with this task's fixed
    parameters becomes:
        standard   : H_out = H - R + 3 (W_out = W - S + 3)
        asymmetric : H_out = H - R + 4 (W_out = W - S + 4)
    forward returns the single tensor out (N, K, H_out, W_out), float32
    (convolution is inherently single-output, so there are no multi-output
    packing segments).

    Input/output specification:
        x      float32 (N, C, H, W), contiguous, N/C/H/W >= 1
        weight float32 (K, C, R, S), contiguous, K/R/S >= 1, the input-channel
               dimension equals C
        out    a single float32 tensor (N, K, H_out, W_out)
        Inputs are ordinary randn magnitude (no inf/NaN); the evaluator casts
        all inputs to fp32 before the call (the inputs of this task are already
        fp32, so the cast is lossless).

    Implementation constraints (violations are scored as failures):
        - The core computation (the padding-aware index/mask expansion and the
          dot-product accumulation) must be done with a HIP kernel inside the
          submitted file; replacing it with any form of library convolution or
          ATen GEMM (matmul/mm/bmm/addmm/baddbmm/einsum/tensordot/unfold/linear
          etc., including one-step equivalent routes such as a conv3d
          depth-axis unsqueeze or a conv_transpose2d kernel-flip rewrite) is
          forbidden.
        - The __init__ and forward signatures of ModelNew must not be changed.
        - Compute in float32 throughout; the padding value is always zero and
          must be handled inside the kernel with boundary masks/indices -- do
          not materialize a padded copy on the host side in place of in-kernel
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

    # Fixed convolution hyper-parameters of this task (shared across variants,
    # sample parameter set):
    _STRIDE = (1, 1)
    _DILATION = (1, 1)
    _PAD_SYMMETRIC = (1, 1)                 # standard: symmetric padding
    _PRE_PADDING = (2, 2)                   # asymmetric: top/left side
    _POST_PADDING = (1, 1)                  # asymmetric: bottom/right side

    def __init__(self, variant: str = "standard"):
        super().__init__()
        assert variant in ("standard", "asymmetric"), "variant must be either standard or asymmetric"
        self.variant = variant

    def forward(self, x, weight):
        # variant="asymmetric": pre=(2,2)/post=(1,1) are in (height, width)
        # order, while F.pad takes (width left, width right, height top, height
        # bottom); both dimensions use the same pair of values here, i.e.
        # (2, 1, 2, 1).
        # variant="standard": symmetric padding goes straight to conv (a
        # conditional expression branches here to avoid multi-clause compound
        # statements)
        if self.variant == "asymmetric":
            xp = torch.nn.functional.pad(x, (2, 1, 2, 1))
            return torch.nn.functional.conv2d(
                xp, weight, bias=None, stride=self._STRIDE,
                padding=(0, 0), dilation=self._DILATION,
            )
        return torch.nn.functional.conv2d(
            x, weight, bias=None, stride=self._STRIDE,
            padding=self._PAD_SYMMETRIC, dilation=self._DILATION,
        )


def get_init_inputs():
    return ["standard"]   # variant


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs
    n, c, h, w = 4, 16, 56, 56
    k, r, s = 4, 3, 3
    x = torch.randn(n, c, h, w)
    weight = torch.randn(k, c, r, s)
    return [x, weight]


def make_inputs(n: int, c: int, h: int, w: int, k: int, r: int, s: int,
                variant: str = "standard", dtype: str = "float32",
                seed: int = 0):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    The consumption order is fixed: x -> weight (both torch.randn, float32), all
    drawn from the same torch.Generator(seed), so the same seed reproduces
    inputs bit-for-bit.
    variant is also the init hyper-parameter used to construct the Model; its
    validity and the shape domain are checked here (every dimension >= 1 and
    the output dimension >= 1); the dtype domain declared by this task is
    float32 only.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert dt == torch.float32, "the dtype domain declared by this task is float32 only"
    assert variant in ("standard", "asymmetric"), "variant must be either standard or asymmetric"
    assert min(n, c, h, w, k, r, s) >= 1, "N/C/H/W/K/R/S must all be >= 1"
    pad_pre_h, pad_post_h, pad_pre_w, pad_post_w = (
        (1, 1, 1, 1) if variant == "standard" else (2, 1, 2, 1)
    )
    h_out = (h + pad_pre_h + pad_post_h - (r - 1) - 1) // 1 + 1
    w_out = (w + pad_pre_w + pad_post_w - (s - 1) - 1) // 1 + 1
    assert h_out >= 1 and w_out >= 1, "the kernel exceeds the padded input, so the output dimension is negative"
    x = torch.randn(n, c, h, w, generator=gen)
    weight = torch.randn(k, c, r, s, generator=gen)
    return x, weight
