# 5004_deform_conv2d_bwd -- model_class (KernelBench compatible) task file for
# the deformable 2D convolution backward pass (DCNv2 dgrad / wrw).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=5004, backend=cuda (HIP).

import torch
import torch.nn as nn


def _bilinear_zero_pad(x, y_coord, x_coord):
    """Bilinear sampling (an integer coordinate is the pixel centre);
    contributions outside the boundary are 0.

    x: (n, c, h, w); y_coord / x_coord: (n, oh, ow) floating-point sampling
    coordinates (in the H/W coordinate system).
    Returns (n, c, oh, ow). Differentiable with respect to y_coord / x_coord:
    the coordinate gradient flows back only through the interpolation fractions
    (the frac terms) while the floor/index path is treated as a constant -- i.e.
    the subgradient of the standard zero-padding bilinear Theano/DCNv2
    convention.
    """
    n, c, h, w = x.shape
    oh, ow = y_coord.shape[1], y_coord.shape[2]
    y0 = torch.floor(y_coord)
    x0 = torch.floor(x_coord)
    y1 = y0 + 1
    x1 = x0 + 1
    ty = (y_coord - y0).unsqueeze(1)   # (n, 1, oh, ow)
    tx = (x_coord - x0).unsqueeze(1)

    def gather(yc, xc):
        valid = ((yc >= 0) & (yc <= h - 1) & (xc >= 0) & (xc <= w - 1)).unsqueeze(1)
        yi = yc.clamp(0, h - 1).long()
        xi = xc.clamp(0, w - 1).long()
        # The index tensors carry a batch dimension, so plain advanced indexing
        # would move the index dimensions to the front and produce 5-D;
        # flattening plus gather keeps (n, c, oh, ow) and stays differentiable
        # with respect to x
        idx = (yi * w + xi).reshape(n, 1, oh * ow).expand(n, c, oh * ow)
        got = x.reshape(n, c, h * w).gather(2, idx).reshape(n, c, oh, ow)
        return got * valid.to(x.dtype)

    return ((1 - ty) * (1 - tx) * gather(y0, x0)
            + (1 - ty) * tx * gather(y0, x1)
            + ty * (1 - tx) * gather(y1, x0)
            + ty * tx * gather(y1, x1))


class Model(nn.Module):
    """Deformable 2D convolution backward (DCNv2, with mask modulation):
    gradient with respect to the input (dgrad) and gradient with respect to the
    weights (wrw).

    Semantics (NCHW, offset group = 1, no bias, strictly consistent with the
    standard DCNv2 forward):
        forward y = deform_conv2d(x, offset, weight, mask, stride=(sh,sw),
                               padding=(ph,pw), dilation=(dh,dw))
        output size out_h = floor((H + 2*ph - dh*(R-1) - 1) / sh) + 1 (likewise
        for out_w)
        the m-th kernel sampling point in row-major order m = pm*S + pn
        (pm = m // S, pn = m % S), at output position (i, j):
            sampling coordinate  Y = i*sh - ph + dh*pm + offset[b, 2*m,   i, j]
                                 X = j*sw - pw + dw_*pn + offset[b, 2*m+1, i, j]
            sampled value  V[b, :, i, j, m] = bilinear(x, Y, X): an integer
                coordinate is the pixel centre, interpolated bilinearly; the part
                of a sampling point that falls outside [0,H-1]x[0,W-1]
                contributes 0.
            y[b, ko, i, j] = sum_m mask[b, m, i, j] *
                                 sum_ci weight[ko, ci, pm, pn] * V[b, ci, i, j, m]
        offset channels correspond one-to-one with the kernel sampling points in
        row-major order: channel 2*m is the H-direction offset and 2*m+1 the
        W-direction offset (the only widespread DCNv2 convention; offset
        group = 1 for this task).
    Given the upstream gradient dy = dL/dy (shape (N, out_c, out_h, out_w)):
        dgrad: dx = dL/dx (N,C,H,W), doffset = dL/doffset (N,2*R*S,out_h,out_w),
               dmask = dL/dmask (N,R*S,out_h,out_w)
        wrw:   dw = dL/dw (out_c,C,R,S)
    The __init__ parameter variant selects which direction is returned:
        "dgrad": returns the three gradients packed into a single 1-D tensor by
                 the segmentation protocol;
        "wrw":   returns dw; the values of weight do not take part in the
                 output here (shape convention only).
    forward(dy, weight, x, offset, mask) has a fixed five-input signature; the
    tensors mean:
        dy     (N, out_c, out_h, out_w)     upstream gradient, consumed by both
                                            directions
        weight (out_c, C, R, S)             convolution kernel (out_c output
                                            channels / C input channels)
        x      (N, C, H, W)                 forward input
        offset (N, 2*R*S, out_h, out_w)     sampling offsets
        mask   (N, R*S, out_h, out_w)       modulation mask
    forward returns the single tensor out (float32) -- dgrad is a 1-D
    concatenated packing (segmentation protocol):
        out = cat([dx.reshape(-1), doffset.reshape(-1), dmask.reshape(-1)])
        length = N*C*H*W + N*2*R*S*out_h*out_w + N*R*S*out_h*out_w, with the
        segment order fixed as dx | doffset | dmask, each segment being the
        C-order flattening of the corresponding 4-D tensor; the segment order
        must not be swapped;
        wrw -> (out_c, C, R, S), inherently single-output with no packing.

    Input/output specification:
        dy / weight / x / offset / mask are all float32, contiguous, NCHW
        layout; every dimension >= 1; R/S may be 1 (1x1 kernel); H/W may be 1,
        odd, non-power-of-two, not 16-aligned; stride/dilation >= 1, padding
        >= 0, padding follows the forward four-sided symmetric (ph, pw)
        semantics; consistency constraint
            H + 2*ph >= dh*(R-1) + 1, W + 2*pw >= dw_*(S-1) + 1,
        and the spatial sizes of dy/offset/mask must equal out_h/out_w from the
        forward formula, offset channels = 2*R*S, mask channels = R*S.
        Input values are ordinary randn magnitude (no inf/NaN). The evaluator
        casts all inputs to fp32 before the call.

    Implementation constraints (violations are scored as failures):
        - The core computation must be done with a HIP kernel inside the
          submitted file: the backward bilinear-sampling scatter of dgrad (the
          analytic partial derivatives with respect to Y/X chained back through
          the kernel weights and the mask into dx/doffset/dmask, with
          out-of-boundary contributions set to 0) or the sampled-value
          correlation accumulation of wrw (summing the cross-correlation of the
          x sampling windows with dy); falling back to torch autograd /
          convolution primitives / grid_sample for the core computation is
          forbidden.
        - The __init__ and forward signatures of ModelNew must not be changed;
          forward returns the single tensor out (the evaluator compares a single
          output); the three dgrad segments must be packed by the segmentation
          protocol above and the segment order must not be swapped.
        - Numeric contract: float32 inputs are accumulated directly in float32,
          with no downcast to half precision; the three dgrad gradient segments
          and the wrw output are all float32.

    Forbidden (static audit strips docstrings and comments, then does literal
    substring matching):
    aiter / torchvision / torch.nn.functional.conv2d / F.conv2d /
    torch.conv2d / torch.nn.Conv2d / torch.nn.functional.conv_transpose2d /
    F.conv_transpose2d / torch.nn.ConvTranspose2d /
    torch.nn.grad.conv2d_input / torch.nn.grad.conv2d_weight /
    aten.convolution_backward / deform_conv2d / deform_conv2d_backward /
    grid_sample / F.grid_sample / torch.nn.functional.grid_sample /
    torch.autograd / .backward( / unfold / F.fold /
    torch.nn.functional.fold / im2col / col2im / as_strided;
    the closed-source libraries MIOpen / hipDNN / cuDNN / rocBLAS / hipBLASLt
    are forbidden as well (global blacklist).
    Target hardware: Hygon DCU (gfx936/gfx938). Implementation language: HIP
    (compiled by hipcc; write inside the CUDA_CU_SRC / CUDA_CPP_SRC anchor
    regions; device code must include <hip/hip_runtime.h> -- <cuda_runtime.h>
    is not available on this machine).
    """

    def __init__(self, variant: str = "dgrad", stride_h: int = 1, stride_w: int = 1,
                 pad_h: int = 1, pad_w: int = 1, dil_h: int = 1, dil_w: int = 1):
        super().__init__()
        assert variant in ("dgrad", "wrw"), "variant must be either dgrad or wrw"
        self.variant = variant
        self.stride_h, self.stride_w = int(stride_h), int(stride_w)
        self.pad_h, self.pad_w = int(pad_h), int(pad_w)
        self.dil_h, self.dil_w = int(dil_h), int(dil_w)

    def _deform_conv2d_forward(self, x, offset, weight, mask):
        """Handwritten DCNv2 forward (a composition of standard differentiable
        torch operators), used for the autograd backward pass."""
        n, c, h, w = x.shape
        k, _, r, s = weight.shape
        sh, sw = self.stride_h, self.stride_w
        ph, pw = self.pad_h, self.pad_w
        dh, dwl = self.dil_h, self.dil_w
        oh, ow = offset.shape[2], offset.shape[3]
        out = torch.zeros(n, k, oh, ow, dtype=x.dtype, device=x.device)
        for m in range(r * s):
            pm, pn = m // s, m % s
            base_y = torch.arange(oh, dtype=x.dtype, device=x.device) * sh - ph + dh * pm
            base_x = torch.arange(ow, dtype=x.dtype, device=x.device) * sw - pw + dwl * pn
            yo = base_y.view(1, oh, 1) + offset[:, 2 * m]      # (n, oh, ow)
            xo = base_x.view(1, 1, ow) + offset[:, 2 * m + 1]  # (n, oh, ow)
            val = _bilinear_zero_pad(x, yo, xo)                # (n, c, oh, ow)
            wc = weight[:, :, pm, pn]                          # (k, c)
            contrib = torch.einsum("oc,ncij->noij", wc, val)
            out = out + mask[:, m].unsqueeze(1) * contrib
        return out

    def forward(self, dy, weight, x, offset, mask):
        # Build the handwritten DCNv2 forward graph and backpropagate with
        # autograd: this keeps dgrad / wrw strictly consistent with the forward
        # semantics (the reference is not bound by the forbidden list;
        # mathematically dw does not depend on the values of weight, and
        # autograd naturally yields exactly that definition).
        xg = x.detach().requires_grad_(True)
        og = offset.detach().requires_grad_(True)
        mg = mask.detach().requires_grad_(True)
        wg = weight.detach().requires_grad_(True)
        y = self._deform_conv2d_forward(xg, og, wg, mg)
        if self.variant == "dgrad":
            dx, doffset, dmask = torch.autograd.grad(y, (xg, og, mg), dy)
            # Single-tensor output: the three segments concatenated into 1-D,
            # segment order dx | doffset | dmask
            return torch.cat([dx.reshape(-1), doffset.reshape(-1), dmask.reshape(-1)])
        (dw,) = torch.autograd.grad(y, (wg,), dy)
        return dw


def get_init_inputs():
    return ["dgrad", 1, 1, 1, 1, 1, 1]   # variant, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w


def get_inputs():
    # Fixed shape family (sample convention: 3x3 kernel, stride=1, dilation=1;
    # pad=1 exercises the padding path and the sizes are a modest relaxation of
    # the sample); the random part consumes the global RNG -- the evaluator
    # calls this function after set_seed, so multiple correctness trials
    # therefore see diverse inputs
    n, c, h, w = 2, 16, 16, 16
    out_c, r, s = 32, 3, 3
    out_h = (h + 2 * 1 - 1 * (3 - 1) - 1) // 1 + 1
    out_w = (w + 2 * 1 - 1 * (3 - 1) - 1) // 1 + 1
    dy = torch.randn(n, out_c, out_h, out_w).to(torch.float32)
    weight = torch.randn(out_c, c, r, s).to(torch.float32)
    x = torch.randn(n, c, h, w).to(torch.float32)
    offset = torch.randn(n, 2 * r * s, out_h, out_w).to(torch.float32)
    mask = torch.randn(n, r * s, out_h, out_w).to(torch.float32)
    return [dy, weight, x, offset, mask]


def make_inputs(n: int, c: int, h: int, w: int, out_c: int, r: int, s: int,
                stride_h: int = 1, stride_w: int = 1, pad_h: int = 1, pad_w: int = 1,
                dil_h: int = 1, dil_w: int = 1, variant: str = "dgrad",
                dtype: str = "float32", seed: int = 0):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    The consumption order is fixed: dy -> weight -> x -> offset -> mask (all
    torch.randn, then cast to dtype), all drawn from the same
    torch.Generator(seed), so the same seed reproduces inputs bit-for-bit.
    The spatial sizes of dy/offset/mask are derived by the forward formula from
    x/weight/the convolution attributes
    (out = floor((in + 2*pad - dil*(k-1) - 1) / stride) + 1), which keeps the
    five tensors self-consistent for any variant. variant and the convolution
    attributes are also the init hyper-parameters used to construct the Model;
    only validity checks are done here.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert variant in ("dgrad", "wrw"), "variant must be either dgrad or wrw"
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
    offset = torch.randn(n, 2 * r * s, out_h, out_w, generator=gen).to(dt)
    mask = torch.randn(n, r * s, out_h, out_w, generator=gen).to(dt)
    return dy, weight, x, offset, mask
