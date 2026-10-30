# 5009_conv_depthtospace -- model_class (KernelBench compatible) task file for
# the fused Conv+Bias+DepthToSpace (conv_bias_dts) and the pure DepthToSpace
# pixel rearrangement (dts).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=2, problem_id=5009, backend=cuda (HIP).

import torch
import torch.nn as nn


class Model(nn.Module):
    """Conv+Bias+DepthToSpace fusion (conv_bias_dts) and pure DepthToSpace
    pixel rearrangement (dts) forward family.

    DepthToSpace (D2S) rearranges the channel dimension of an (N, K, H, W)
    feature map into the spatial dimensions: K must be divisible by
    block_size^2; writing C = K // block_size^2, the output is
    (N, C, H*bs, W*bs) (bs = block_size). The rearrangement mode is selected
    between two by the __init__ parameter mode, defined in terms of a
    six-dimensional view + transpose + reshape of v = (N, K, H, W):
        "CRD": v.reshape(N, C, bs, bs, H, W).permute(0, 1, 4, 2, 5, 3)
                .reshape(N, C, H*bs, W*bs)
                (bit-for-bit equivalent to
                torch.nn.functional.pixel_shuffle(v, bs) -- that ATen primitive
                is on the forbidden list; the equivalence is only noted here and
                it must not be called directly)
        "DCR": v.reshape(N, bs, bs, C, H, W).permute(0, 3, 4, 1, 5, 2)
                .reshape(N, C, H*bs, W*bs)
    Both modes produce the same output shape; only the channel-to-spatial-block
    mapping differs. Element-wise formula:
        out[b, c, h*bs + r, w*bs + d] = v[b, idx(c, r, d), h, w]
        CRD: idx = c*bs^2 + r*bs + d; DCR: idx = (r*bs + d)*C + c
        (c in [0, C), r/d in [0, bs), h in [0, H), w in [0, W))

    The fused path is selected between two by the __init__ parameter variant:
        "conv_bias_dts": for x (N, C_in, H, W) and the kernel weight
            (K, C_in, 3, 3), run the 2D cross-correlation forward pass (the
            deep-learning-library conv_fprop convention, not the kernel-flipped
            mathematical convolution), add the per-channel bias bias
            (1, K, 1, 1) (broadcast along the batch and spatial dimensions),
            then apply the D2S above to the convolution result. The convolution
            hyper-parameters are fixed: stride=(1, 1), symmetric zero
            padding=(1, 1) (1 added on each of the top/bottom/left/right sides;
            the padding value is always zero), dilation=(1, 1), so the
            convolution output keeps the spatial size (H, W) and the final
            output is (N, K//bs^2, H*bs, W*bs). forward(x, weight, bias) uses all
            three parameters.
        "dts": pure data rearrangement without convolution (super-resolution /
            pixel-rearrangement scenario): x itself is the (N, K, H, W) feature
            map to rearrange, forward(x) is called with a single parameter, and
            weight/bias are not passed (the optional parameters default to None;
            even if they are passed they must not affect the output); the
            rearrangement is pure index movement and bit-for-bit lossless.
    The convolution dot-product accumulation and the bias addition are computed
    in float32 intermediate precision throughout (the dts branch likewise
    promotes to float32 losslessly before the rearrangement), and the output is
    cast back to the dtype of x exactly once.

    Input/output specification:
        x      float16 / bfloat16 / float32 4D contiguous;
               (N, C_in, H, W) with C_in >= 1 for
               variant="conv_bias_dts", (N, K, H, W) with K % bs^2 == 0 for
               variant="dts"
        weight same dtype as x (K, C_in, 3, 3), contiguous (conv_bias_dts only)
        bias   same dtype as x (1, K, 1, 1), contiguous (conv_bias_dts only)
        out    a single tensor, same dtype as x (N, K//bs^2, H*bs, W*bs) (both
               the convolution and D2S are inherently single-output, so there
               are no multi-output packing segments)
        N/K/H/W >= 1 (non-powers-of-two, primes and boundary values are
        allowed, including N=1 and H=W=1); K % block_size^2 == 0; block_size in
        {2, 3, 4}; mode in {"CRD", "DCR"};
        inputs are ordinary randn magnitude (no inf/NaN); the evaluator casts
        all inputs to fp32 before the call (fp16/bf16 -> fp32 is lossless; the
        output dtype of this task always matches the dtype of the passed x).

    Implementation constraints (violations are scored as failures):
        - The core computation (the padding-aware index/mask expansion and the
          dot-product accumulation, the bias addition, the six-dimensional
          rearrangement index mapping) must all be done and fused with a HIP
          kernel inside the submitted file; it is forbidden to replace it with
          any form of library convolution, ATen GEMM
          (matmul/mm/bmm/addmm/baddbmm/einsum/tensordot/unfold/fold/linear
          etc.), pixel_shuffle-family primitives (including a pixel_unshuffle
          round trip), or one-step equivalent routes such as a conv3d depth-axis
          unsqueeze or a conv_transpose2d kernel-flip rewrite.
        - The __init__ and forward signatures of ModelNew must not be changed.
        - The convolution dot-product accumulation and the bias addition are
          computed in float32 and the output is cast back to the input dtype
          exactly once (do not accumulate in fp16/bf16 low precision, and do not
          make repeated round-trip casts); padding must be handled inside the
          kernel with boundary masks/indices -- do not materialize a padded copy
          on the host side in place of in-kernel handling.
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
    torch.nn.functional.fold / torch.nn.Fold / .fold( /
    torch.nn.functional.pixel_shuffle / torch.pixel_shuffle / pixel_shuffle /
    torch.nn.PixelShuffle / PixelShuffle /
    torch.nn.functional.pixel_unshuffle / torch.pixel_unshuffle /
    pixel_unshuffle / torch.nn.PixelUnshuffle / PixelUnshuffle /
    torch.matmul / .matmul( / torch.mm / torch.bmm / torch.addmm /
    torch.baddbmm / torch.einsum / torch.tensordot /
    torch.nn.functional.linear / torch.nn.Linear /
    miopen / hipdnn.
    Target hardware: Hygon DCU (gfx936/gfx938). Implementation language: HIP
    (compiled by hipcc; write inside the CUDA_CU_SRC / CUDA_CPP_SRC anchor
    regions; device code must include <hip/hip_runtime.h> -- <cuda_runtime.h>
    is not available on this machine).
    """

    # The convolution hyper-parameters are fixed (both variants' D2S semantics
    # share the same parameter set)
    _STRIDE = (1, 1)
    _PADDING = (1, 1)   # symmetric zero padding: 1 added on each side
    _DILATION = (1, 1)

    def __init__(self, variant: str = "conv_bias_dts", mode: str = "CRD",
                 block_size: int = 2):
        super().__init__()
        assert variant in ("conv_bias_dts", "dts"), \
            "variant must be either conv_bias_dts or dts"
        assert mode in ("CRD", "DCR"), "mode must be either CRD or DCR"
        assert block_size in (2, 3, 4), "block_size must be 2 / 3 / 4"
        self.variant = variant
        self.mode = mode
        self.block_size = block_size

    def _depth_to_space(self, v):
        """Six-dimensional view + transpose + reshape: a pure index
        rearrangement (no numeric operations) matching mode/block_size one to
        one; the per-dimension permutation corresponds exactly to the fused
        graph's first reshape -> transpose -> second reshape."""
        bs = self.block_size
        n, k, h, w = v.shape
        c = k // (bs * bs)
        if self.mode == "CRD":
            t = v.reshape(n, c, bs, bs, h, w).permute(0, 1, 4, 2, 5, 3)
        else:   # DCR
            t = v.reshape(n, bs, bs, c, h, w).permute(0, 3, 4, 1, 5, 2)
        return t.reshape(n, c, h * bs, w * bs)

    def forward(self, x, weight=None, bias=None):
        # Inputs are promoted to float32 losslessly and the output is cast back
        # to the dtype of x exactly once
        xf = x.to(torch.float32)
        if self.variant == "conv_bias_dts":
            z = torch.nn.functional.conv2d(
                xf, weight.to(torch.float32), bias=None,
                stride=self._STRIDE, padding=self._PADDING, dilation=self._DILATION,
            )
            z = z + bias.to(torch.float32)
            return self._depth_to_space(z).to(x.dtype)
        else:   # dts: pure rearrangement, weight/bias do not take part
            return self._depth_to_space(xf).to(x.dtype)


def get_init_inputs():
    return ["conv_bias_dts", "CRD", 2]   # variant, mode, block_size


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs
    n, c_in, h, w = 2, 8, 64, 64
    k = 16
    x = torch.randn(n, c_in, h, w).to(torch.float16)
    weight = torch.randn(k, c_in, 3, 3).to(torch.float16)
    bias = torch.randn(1, k, 1, 1).to(torch.float16)
    return [x, weight, bias]


def make_inputs(n: int, c_in: int, k: int, h: int, w: int,
                variant: str = "conv_bias_dts", mode: str = "CRD",
                block_size: int = 2, dtype: str = "float16", seed: int = 0):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    The consumption order is fixed: x -> weight -> bias for conv_bias_dts (all
    torch.randn, then cast to dtype), and only x for dts; everything comes from
    the same torch.Generator(seed), so the same seed reproduces inputs
    bit-for-bit. variant/mode/block_size are also the init hyper-parameters used
    to construct the Model; their validity, the declared dtype domain and the
    shape domain are checked here (N/K/H/W >= 1, K % block_size^2 == 0,
    block_size in {2,3,4}; for variant=dts c_in does not take part and the case
    convention is to fill it with 0).
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert dt in (torch.float16, torch.bfloat16, torch.float32), \
        "the dtype domain declared by this task is float16 / bfloat16 / float32 only"
    assert variant in ("conv_bias_dts", "dts"), \
        "variant must be either conv_bias_dts or dts"
    assert mode in ("CRD", "DCR"), "mode must be either CRD or DCR"
    assert block_size in (2, 3, 4), "block_size must be 2 / 3 / 4"
    assert min(n, k, h, w) >= 1, "N/K/H/W must all be >= 1"
    assert k % (block_size * block_size) == 0, "K must be divisible by block_size^2"
    if variant == "conv_bias_dts":
        assert c_in >= 1, "conv_bias_dts requires C_in >= 1"
        x = torch.randn(n, c_in, h, w, generator=gen).to(dt)
        weight = torch.randn(k, c_in, 3, 3, generator=gen).to(dt)
        bias = torch.randn(1, k, 1, 1, generator=gen).to(dt)
        return x, weight, bias
    x = torch.randn(n, k, h, w, generator=gen).to(dt)
    return x,
