# 5003_deform_conv2d_fwd -- model_class (KernelBench compatible) task file for
# the deformable 2D convolution forward pass (modulated / plain variants).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=5003, backend=cuda (HIP).

import torch
import torch.nn as nn


class Model(nn.Module):
    """Deformable 2D convolution forward (Deformable Convolution): sampling grid
    plus a handwritten bilinear interpolation.

    Semantics (NCHW, all intermediate computation in float32):
      For each output position (oi, oj) and kernel position p = i*S + j
      (i in [0,R), j in [0,S), row-major), the base sampling coordinate given by
      the regular convolutional receptive field is
          y0 = oi * stride_h + i * dilation_h - padding_h
          x0 = oj * stride_w + j * dilation_w - padding_w
      After adding the per-(output position x kernel position) offsets, the
      input map is sampled at real-valued coordinates with bilinear
      interpolation:
          yc = y0 + offset[n, 2*p,     oi, oj]     # even channel = y offset
          xc = x0 + offset[n, 2*p + 1, oi, oj]     # odd channel = x offset
          v(n, c, p, oi, oj) = bilinear(input map, yc, xc)
      The __init__ parameter variant selects one of the two paths:
          "modulated" (default, modulated form): the sampled value is
              additionally multiplied by the per-position modulation mask
              v *= mask[n, p, oi, oj]
          "plain" (mask-free form): mask does not take part in the computation
              (its contents are arbitrary, possibly garbage data).
      The output is a single tensor:
          out[n, k, oi, oj] = bias[k]
                              + sum_{c, p} v(n, c, p, oi, oj) * weight[k, c, i, j]
      After the computation it is cast back to the input dtype.

    Bilinear interpolation definition (point-wise identical to mainstream
    deformable convolution operators):
      Let iy = floor(yc), ix = floor(xc), wy = yc - iy, wx = xc - ix, and take
      the four corners (iy, ix), (iy, ix+1), (iy+1, ix), (iy+1, ix+1) with
      weights (1-wy)(1-wx), (1-wy)wx, wy(1-wx), wy*wx respectively; when either
      the row or the column of a corner is out of range (< 0 or > dim-1) that
      corner contributes 0 while the remaining corners are weighted as usual --
      i.e. zero-fill semantics: a partially out-of-range coordinate is still
      weighted by the remaining valid corners, and when it is entirely out of
      range (yc <= -1, outside the yc >= H-1 boundary band, etc.) the sampled
      value is 0. No clamp / nearest-neighbour pull-back is done anywhere.

    Input/output specification (all tensors contiguous, NCHW layout):
      x        float32 / float16 / bfloat16 (N, C, H, W)
      offset   same dtype as x (N, 2*R*S, H_out, W_out): 2 channels per kernel
               position (even channel = y offset, odd channel = x offset,
               p = i*S + j row-major)
      weight   same dtype as x (K, C, R, S)
      bias     same dtype as x (K,)
      mask     same dtype as x (N, R*S, H_out, W_out) (ignored when
               variant="plain")
      out      a single tensor, same dtype as x (N, K, H_out, W_out)
      H_out = (H + 2*padding_h - dilation_h*(R-1) - 1) // stride_h + 1, likewise
      for W_out. The evaluator casts all inputs to fp32 before the call
      (low-precision inputs are lossless).

    Implementation constraints (violations are scored as failures):
      - The core computation (sampling-grid construction, bilinear-interpolation
        sampling, mask modulation, the weighted reduction against weight) must
        be handwritten with a HIP kernel inside the submitted file.
      - forward returns the single tensor out (the evaluator compares a single
        output).
      - Interpolation and the weighted reduction are computed in float32, with
        the output cast back to the input dtype.
      - The __init__ and forward signatures of ModelNew must not be changed.

    Forbidden (static audit strips docstrings and comments, then does literal
    substring matching):
    torchvision (including torchvision.ops.deform_conv2d) /
    torch.nn.functional.conv2d (F.conv2d) / torch.nn.Conv2d /
    torch.nn.functional.grid_sample (F.grid_sample) /
    torch.nn.functional.unfold (F.unfold) / torch.matmul / torch.bmm /
    torch.mm / torch.einsum / torch.tensordot / miopen / hipdnn / aiter.
    Target hardware: Hygon DCU (gfx936/gfx938). Implementation language: HIP
    (compiled by hipcc; write inside the CUDA_CU_SRC / CUDA_CPP_SRC anchor
    regions; device code must include <hip/hip_runtime.h> -- <cuda_runtime.h>
    is not available on this machine).
    """

    def __init__(self, variant: str = "modulated", stride=(1, 1), padding=(1, 1),
                 dilation=(1, 1)):
        super().__init__()
        assert variant in ("modulated", "plain"), "variant must be either modulated or plain"
        self.variant = variant
        self.stride = (int(stride[0]), int(stride[1]))
        self.padding = (int(padding[0]), int(padding[1]))
        self.dilation = (int(dilation[0]), int(dilation[1]))

    def forward(self, x, offset, weight, bias, mask):
        n, c, h, w = x.shape
        k_out = weight.shape[0]
        r, s = weight.shape[2], weight.shape[3]
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation
        h_out = (h + 2 * ph - (dh * (r - 1) + 1)) // sh + 1
        w_out = (w + 2 * pw - (dw * (s - 1) + 1)) // sw + 1
        assert offset.shape[1] == 2 * r * s and mask.shape[1] == r * s, \
            "the channel counts of offset/mask must be 2*R*S / R*S"
        assert offset.shape[2] == h_out and offset.shape[3] == w_out \
            and mask.shape[2] == h_out and mask.shape[3] == w_out, \
            "the spatial sizes of offset/mask must be (H_out, W_out)"

        xf = x.to(torch.float32)
        off = offset.to(torch.float32)
        wf = weight.to(torch.float32)
        bf = bias.to(torch.float32)

        # Base receptive-field coordinate per kernel position (offsets not added
        # yet), row-major p = i*S + j
        # device=dev is required: these coordinates are added to the already
        # on-device off; a missing device raises "Expected all tensors to be on
        # the same device" on CUDA/DCU (harmless on CPU)
        dev = xf.device
        base_y = (torch.arange(h_out, dtype=torch.float32, device=dev) * sh - ph).view(1, 1, h_out, 1) \
            + torch.arange(r, dtype=torch.float32, device=dev).view(r, 1, 1, 1) * dh      # (r, 1, h_out, 1)
        base_x = (torch.arange(w_out, dtype=torch.float32, device=dev) * sw - pw).view(1, 1, 1, w_out) \
            + torch.arange(s, dtype=torch.float32, device=dev).view(1, s, 1, 1) * dw      # (1, s, 1, w_out)

        # Add the per-position offsets (even channel dy, odd channel dx) ->
        # sampling coordinates (n, R*S, H_out, W_out)
        y_coord = (base_y + off[:, 0::2].reshape(n, r, s, h_out, w_out))
        x_coord = (base_x + off[:, 1::2].reshape(n, r, s, h_out, w_out))
        y_coord = y_coord.reshape(n, r * s, h_out, w_out)
        x_coord = x_coord.reshape(n, r * s, h_out, w_out)

        # Handwritten bilinear interpolation: sample the four corners and let
        # out-of-range corners contribute 0 (zero-fill semantics)
        iy0 = torch.floor(y_coord).to(torch.long)
        ix0 = torch.floor(x_coord).to(torch.long)
        wy = y_coord - iy0.to(torch.float32)
        wx = x_coord - ix0.to(torch.float32)
        iy1 = iy0 + 1
        ix1 = ix0 + 1
        n_idx = torch.arange(n, device=xf.device).view(n, 1, 1, 1, 1)
        c_idx = torch.arange(c, device=xf.device).view(1, c, 1, 1, 1)

        def corner(iy, ix, wgt):
            # Single-corner sampling: the weight of an out-of-range corner is
            # set to 0; clamp only guards the index (cancelled by the times-0)
            valid = ((iy >= 0) & (iy <= h - 1) & (ix >= 0) & (ix <= w - 1)).to(torch.float32)
            vals = xf[n_idx, c_idx, iy.clamp(0, h - 1).unsqueeze(1),
                      ix.clamp(0, w - 1).unsqueeze(1)]
            return vals * valid.unsqueeze(1) * wgt.unsqueeze(1)

        sampled = (corner(iy0, ix0, (1.0 - wy) * (1.0 - wx))
                   + corner(iy0, ix1, (1.0 - wy) * wx)
                   + corner(iy1, ix0, wy * (1.0 - wx))
                   + corner(iy1, ix1, wy * wx))            # (n, c, R*S, H_out, W_out)

        # Modulation mask (under the plain variant mask does not take part)
        if self.variant == "modulated":
            sampled = sampled * mask.to(torch.float32).unsqueeze(1)

        # Weighted reduction against weight (float32), add bias, cast back to
        # the input dtype
        y = torch.matmul(wf.reshape(k_out, c * r * s),
                         sampled.reshape(n, c * r * s, h_out * w_out))
        y = y + bf.view(1, k_out, 1)
        return y.view(n, k_out, h_out, w_out).to(x.dtype)


def get_init_inputs():
    return ["modulated", [1, 1], [1, 1], [1, 1]]   # variant, stride, padding, dilation


def get_inputs():
    # Fixed shape family (N=4, C=16, H=W=32, K=32, R=S=3; stride, padding and
    # dilation all (1,1); variant=modulated, float32); the random part consumes
    # the global RNG -- the evaluator calls this function after set_seed, so
    # multiple correctness trials therefore see diverse inputs
    n, c, h, w, k, r, s = 4, 16, 32, 32, 32, 3, 3
    ph = pw = 1
    dh = dw = 1
    h_out = (h + 2 * ph - (dh * (r - 1) + 1)) // 1 + 1   # = 32
    w_out = (w + 2 * pw - (dw * (s - 1) + 1)) // 1 + 1   # = 32
    x = torch.randn(n, c, h, w).to(torch.float32)
    offset = torch.randn(n, 2 * r * s, h_out, w_out).to(torch.float32)
    weight = torch.randn(k, c, r, s).to(torch.float32)
    bias = torch.randn(k).to(torch.float32)
    mask = torch.randn(n, r * s, h_out, w_out).to(torch.float32)
    return [x, offset, weight, bias, mask]


def make_inputs(n: int, c: int, h: int, w: int, k: int, r: int = 3, s: int = 3,
                variant: str = "modulated", dtype: str = "float32", seed: int = 0,
                stride=(1, 1), padding=(1, 1), dilation=(1, 1)):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    The consumption order is fixed: x -> offset -> weight -> bias -> mask (all
    fp32 torch.randn, then cast to dtype), all drawn from the same
    torch.Generator(seed), so the same seed reproduces inputs bit-for-bit.
    variant/stride/padding/dilation are also the init hyper-parameters used to
    construct the Model; they are used here to derive H_out/W_out and for
    validity checks.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert variant in ("modulated", "plain"), "variant must be either modulated or plain"
    assert n >= 1 and c >= 1 and h >= 1 and w >= 1 and k >= 1 and r >= 1 and s >= 1, \
        "every dimension must be >= 1"
    sh, sw = int(stride[0]), int(stride[1])
    ph, pw = int(padding[0]), int(padding[1])
    dh, dw = int(dilation[0]), int(dilation[1])
    assert sh >= 1 and sw >= 1 and dh >= 1 and dw >= 1, "stride/dilation must be >= 1"
    assert ph >= 0 and pw >= 0, "padding must be >= 0"
    h_out = (h + 2 * ph - (dh * (r - 1) + 1)) // sh + 1
    w_out = (w + 2 * pw - (dw * (s - 1) + 1)) // sw + 1
    assert h_out >= 1 and w_out >= 1, "H_out/W_out must be >= 1 (padding must cover the dilated kernel)"
    x = torch.randn(n, c, h, w, generator=gen).to(dt)
    offset = torch.randn(n, 2 * r * s, h_out, w_out, generator=gen).to(dt)
    weight = torch.randn(k, c, r, s, generator=gen).to(dt)
    bias = torch.randn(k, generator=gen).to(dt)
    mask = torch.randn(n, r * s, h_out, w_out, generator=gen).to(dt)
    return x, offset, weight, bias, mask
