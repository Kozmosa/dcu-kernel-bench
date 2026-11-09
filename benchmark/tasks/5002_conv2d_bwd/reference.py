# 5002_conv2d_bwd -- model_class (KernelBench compatible) task file for the
# standard 2D convolution backward pass (dgrad / wgrad).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=5002, backend=cuda (HIP).

import torch
import torch.nn as nn


class Model(nn.Module):
    """Standard 2D convolution backward: gradient with respect to the input
    (dgrad) and gradient with respect to the weights (wgrad).

    Semantics (NCHW, groups=1, no bias, strictly consistent with the PyTorch
    forward conv2d):
        forward y = conv2d(x, weight, stride=(stride_h,stride_w),
                        padding=(pad_h,pad_w), dilation=(dil_h,dil_w)),
        output size
            out_h = floor((H + 2*pad_h - dil_h*(R-1) - 1) / stride_h) + 1
        (likewise for out_w); given the upstream gradient dy = dL/dy (shape
        (N, out_c, out_h, out_w)):
            dgrad: dx = dL/dx, shape (N, C, H, W)
            wgrad: dw = dL/dw, shape (out_c, C, R, S)
    The __init__ parameter variant selects which direction is returned:
        "dgrad": returns dx; the values of x do not take part in the output
                 here (shape convention only);
        "wgrad": returns dw; the values of weight do not take part in the
                 output here (shape convention only).
    forward(dy, weight, x) has a fixed three-input signature; the tensors mean:
        dy     (N, out_c, out_h, out_w)   upstream gradient, consumed by both
                                          directions
        weight (out_c, C, R, S)           convolution kernel (out_c output
                                          channels / C input channels)
        x      (N, C, H, W)               forward input
    forward returns the single tensor out -- both directions are inherently
    single-output, with no concatenation packing:
        dgrad -> (N, C, H, W); wgrad -> (out_c, C, R, S); dtype as the inputs
        (float32 for this task).

    Input/output specification:
        dy / weight / x are all float32, contiguous, NCHW layout; every
        dimension >= 1;
        R/S may be 1 (1x1 convolution); H/W may be 1, odd, non-power-of-two,
        not 16-aligned;
        stride/dilation >= 1, padding >= 0, padding follows the four-sided
        symmetric (pad_h, pad_w) semantics of the forward conv (asymmetric
        pre/post padding is outside the domain of this task); consistency
        constraint
            H + 2*pad_h >= dil_h*(R-1) + 1, W + 2*pad_w >= dil_w*(S-1) + 1,
        and the spatial size of dy must equal out_h/out_w from the forward
        formula.
        Input values are ordinary randn magnitude (no inf/NaN). The evaluator
        casts all inputs to fp32 before the call.

    Implementation constraints (violations are scored as failures):
        - The core computation must be done with a HIP kernel inside the
          submitted file: the scatter accumulation of dgrad (dy convolved back
          over the kernel window following stride/dilation) or the correlation
          accumulation of wgrad (summing the cross-correlation of x windows
          with dy); falling back to torch convolution or autograd is
          forbidden.
        - The __init__ and forward signatures of ModelNew must not be changed;
          forward returns the single tensor out (the evaluator compares a
          single output).
        - Numeric contract: float32 inputs are accumulated directly in float32,
          with no downcast to half precision.

    Forbidden (static audit strips docstrings and comments, then does literal
    substring matching):
    aiter / torch.nn.functional.conv2d / F.conv2d / torch.conv2d /
    torch.nn.Conv2d / torch.nn.functional.conv_transpose2d /
    F.conv_transpose2d / torch.nn.ConvTranspose2d /
    torch.nn.grad.conv2d_input / torch.nn.grad.conv2d_weight /
    aten.conv2d / aten.convolution / aten._convolution /
    aten.conv_transpose2d / aten.convolution_backward (including every
    spelling such as torch.ops.aten.*, at::.*) / torch.autograd / .backward( /
    unfold / F.fold / torch.nn.functional.fold / im2col / col2im / as_strided;
    the closed-source libraries MIOpen / hipDNN / cuDNN / rocBLAS / hipBLASLt
    are forbidden as well (global blacklist).
    Target hardware: Hygon DCU (gfx936/gfx938). Implementation language: HIP
    (compiled by hipcc; write inside the CUDA_CU_SRC / CUDA_CPP_SRC anchor
    regions; device code must include <hip/hip_runtime.h> -- <cuda_runtime.h>
    is not available on this machine).
    """

    def __init__(self, variant: str = "dgrad", stride_h: int = 1, stride_w: int = 1, pad_h: int = 1, pad_w: int = 1, dil_h: int = 1, dil_w: int = 1):
        # Note: the def signatures of __init__/forward must stay single-line
        # (the framework loader extracts the signature from the first line to
        # generate the ModelNew scaffold).
        super().__init__()
        assert variant in ("dgrad", "wgrad"), "variant must be either dgrad or wgrad"
        self.variant = variant
        self.stride_h, self.stride_w = int(stride_h), int(stride_w)
        self.pad_h, self.pad_w = int(pad_h), int(pad_w)
        self.dil_h, self.dil_w = int(dil_h), int(dil_w)

    def forward(self, dy, weight, x):
        # Build the forward conv2d graph and backpropagate with autograd: this
        # keeps dgrad / wgrad strictly consistent with the forward convolution
        # semantics (the reference is not bound by the forbidden list;
        # mathematically dx does not depend on the values of x and dw does not
        # depend on the values of weight, and autograd naturally yields exactly
        # that definition).
        xg = x.detach().requires_grad_(True)
        wg = weight.detach().requires_grad_(True)
        y = torch.nn.functional.conv2d(
            xg, wg, None,
            (self.stride_h, self.stride_w),
            (self.pad_h, self.pad_w),
            (self.dil_h, self.dil_w),
        )
        if self.variant == "dgrad":
            (dx,) = torch.autograd.grad(y, (xg,), dy)
            return dx
        (dw,) = torch.autograd.grad(y, (wg,), dy)
        return dw


def get_init_inputs():
    return ["dgrad", 1, 1, 1, 1, 1, 1]   # variant, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w


def get_inputs():
    # Fixed shape family (sample convention: stride=1, pad=1, dilation=1, so dy
    # and x are corresponding members of the same (4, 64/32, 16, 16) family);
    # the random part consumes the global RNG -- the evaluator calls this
    # function after set_seed, so multiple correctness trials therefore see
    # diverse inputs
    n, c, h, w = 4, 32, 16, 16
    out_c, r, s = 64, 3, 3
    dy = torch.randn(n, out_c, h, w).to(torch.float32)
    weight = torch.randn(out_c, c, r, s).to(torch.float32)
    x = torch.randn(n, c, h, w).to(torch.float32)
    return [dy, weight, x]


def make_inputs(n: int, c: int, h: int, w: int, out_c: int, r: int, s: int,
                stride_h: int = 1, stride_w: int = 1, pad_h: int = 1, pad_w: int = 1,
                dil_h: int = 1, dil_w: int = 1, variant: str = "dgrad",
                dtype: str = "float32", seed: int = 0):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    The consumption order is fixed: dy -> weight -> x (all torch.randn, then
    cast to dtype), all drawn from the same torch.Generator(seed), so the same
    seed reproduces inputs bit-for-bit. The spatial size of dy is derived by the
    forward formula from x/weight/the convolution attributes
    (out = floor((in + 2*pad - dil*(k-1) - 1) / stride) + 1), which keeps the
    three tensors self-consistent for any variant. variant and the convolution
    attributes are also the init hyper-parameters used to construct the Model;
    only validity checks are done here.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert variant in ("dgrad", "wgrad"), "variant must be either dgrad or wgrad"
    assert min(n, c, h, w, out_c, r, s) >= 1, "every dimension must be >= 1"
    assert min(stride_h, stride_w, dil_h, dil_w) >= 1 and min(pad_h, pad_w) >= 0, \
        "stride/dilation >= 1 and padding >= 0"
    eff_r, eff_s = dil_h * (r - 1) + 1, dil_w * (s - 1) + 1
    assert h + 2 * pad_h >= eff_r and w + 2 * pad_w >= eff_s, "the forward convolution sizes are inconsistent"
    out_h = (h + 2 * pad_h - eff_r) // stride_h + 1
    out_w = (w + 2 * pad_w - eff_s) // stride_w + 1
    dy = torch.randn(n, out_c, out_h, out_w, generator=gen).to(dt)
    weight = torch.randn(out_c, c, r, s, generator=gen).to(dt)
    x = torch.randn(n, c, h, w, generator=gen).to(dt)
    return dy, weight, x
