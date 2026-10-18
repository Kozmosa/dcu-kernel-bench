# 2012_groupnorm_fwd -- model_class (KernelBench compatible) task file for the
# GroupNorm forward family (standard / Swish fusion).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=2, problem_id=2012, backend=cuda (HIP).

import torch
import torch.nn as nn


class Model(nn.Module):
    """GroupNorm forward family: standard GroupNorm and the GroupNorm+Swish fusion.

    The input is x (N, C, H, W), NCHW contiguous; scale / bias are the (C,)
    per-channel affine parameters; the __init__ argument groups = G (dividing C,
    1 <= G <= C) splits the channels into G groups, with channel c belonging to
    group c // (C//G); eps defaults to 1e-5. The normalization statistics are
    computed per (n, g) group: x[n] is split along the channel dimension into G
    segments (group g occupies the channels [g*(C//G), (g+1)*(C//G)), and the number
    of elements in a group is D = (C//G)*H*W), in float32:
        mean_ng     = sum_D x / D                      (biased mean)
        var_ng      = sum_D (x - mean_ng)^2 / D        (biased variance)
        inv_var_ng  = 1 / sqrt(var_ng + eps)
        gn[n,c,h,w] = (x[n,c,h,w] - mean_ng) * inv_var_ng * scale[c] + bias[c]
                      (where the group index is g = c // (C//G))
    The __init__ argument variant selects one of two paths:
        "groupnorm"      : y = gn
        "groupnorm_swish": y = gn * sigmoid(gn) (Swish / SiLU)
    Both variants also emit the per-group statistics mean and inv_var (one entry per
    (n, g), (N, G) layout, float32).

    Boundary behaviour: when the in-group variance is 0 (e.g. D == 1, a single
    element in the group) inv_var = 1/sqrt(eps) (eps guarantees no division by zero)
    and gn = bias; G=1 is full-channel normalization and G=C is per-channel
    normalization, both of which are legal; N/C/H/W are arbitrary (including 1,
    non-powers of two and unaligned), a large in-group D needs a tiling (block-wise)
    reduction, and out-of-range elements at the row tail must not take part in the
    statistics.

    Single-tensor output protocol (forward must return a single tensor; the evaluator
    compares a single output):
    Returns a one-dimensional float32 packed tensor of length N*C*H*W + 2*N*G, with
    three segments packed back to back in order (half-open intervals; each segment is
    float32, reshaped with reshape(-1) and torch.cat'ed):
        [0,               N*C*H*W)           y
        [N*C*H*W,         N*C*H*W + N*G)     mean    ((N, G) flattened row-major)
        [N*C*H*W + N*G,   N*C*H*W + 2*N*G)   inv_var ((N, G) flattened row-major)
    The output is always float32: the y segment is not quantized to a lower
    precision; fp16/bf16 is only the input quantization level (the input is provided
    in that dtype and then cast to float32 to take part in the whole computation).

    Input/output specification:
        x      float32 / float16 / bfloat16 (N, C, H, W), contiguous, N/C/H/W >= 1
        scale  same dtype as x, (C,), contiguous
        bias   same dtype as x, (C,), contiguous
        out    one-dimensional float32 of length N*C*H*W + 2*N*G (the three-segment
               packing protocol above)
        The inputs are the usual randn magnitude (no inf/NaN); the evaluator casts
        all inputs to fp32 before passing them in (fp16/bf16 -> fp32 is lossless).

    Implementation constraints (violations are scored as failures):
        - The core computation (the in-group mean / biased var reduction, rsqrt, the
          per-channel affine, and Swish) must be completed inside the submitted file
          with a HIP kernel.
        - The __init__ and forward signatures of ModelNew must not be changed.
        - The statistics, rsqrt, the affine and Swish all use float32 intermediates;
          all three output segments are float32.
        - forward returns the single tensor out (the evaluator compares
          shape/dtype/allclose on a single output).

    Forbidden: torch.nn.GroupNorm / torch.nn.functional.group_norm /
    F.group_norm / torch.group_norm / native_group_norm / miopen / hipdnn.
    Target hardware: Hygon DCU (gfx936/gfx938). Implementation language: HIP
    (compiled by hipcc; write inside the CUDA_CU_SRC / CUDA_CPP_SRC anchor regions;
    device code must include <hip/hip_runtime.h> -- <cuda_runtime.h> is not available
    on this machine).
    """

    def __init__(self, variant: str = "groupnorm", groups: int = 2, eps: float = 1e-5):
        super().__init__()
        assert variant in ("groupnorm", "groupnorm_swish"), "variant must be one of groupnorm / groupnorm_swish"
        assert int(groups) >= 1, "groups must be a positive integer"
        assert eps > 0, "eps must be positive"
        self.variant = variant
        self.groups = int(groups)
        self.eps = float(eps)

    def forward(self, x, scale, bias):
        n, c, h, w = x.shape
        g = self.groups
        rep = c // g
        assert 1 <= g <= c and c % g == 0, "groups must divide the number of channels and 1 <= groups <= C"

        # in-group statistics: reduce (N, G, D) along D with float32 intermediates
        # (biased mean / biased variance)
        xf = x.to(torch.float32)
        xg = xf.reshape(n, g, -1)
        mean = xg.mean(dim=-1)                                   # (N, G)
        var = (xg - mean.unsqueeze(-1)).pow(2).mean(dim=-1)      # (N, G) biased variance
        inv_var = torch.rsqrt(var + self.eps)                    # (N, G)

        # broadcast the group statistics back to the channels: channel c belongs to
        # group c // rep
        mean_c = mean.repeat_interleave(rep, dim=1).reshape(n, c, 1, 1)
        inv_c = inv_var.repeat_interleave(rep, dim=1).reshape(n, c, 1, 1)
        sf = scale.to(torch.float32).reshape(1, c, 1, 1)
        bf = bias.to(torch.float32).reshape(1, c, 1, 1)
        gn = (xf - mean_c) * inv_c * sf + bf

        # variant dispatch: groupnorm_swish applies Swish to the normalized result
        # (gn * sigmoid(gn))
        y = gn * torch.sigmoid(gn) if self.variant == "groupnorm_swish" else gn

        # single-tensor output: the three segments y | mean | inv_var concatenated in
        # one dimension (all float32)
        return torch.cat([y.reshape(-1), mean.reshape(-1), inv_var.reshape(-1)])


def get_init_inputs():
    return ["groupnorm", 2, 1e-5]   # variant, groups, eps


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the evaluator
    # calls this function after set_seed, so multiple correctness trials therefore
    # see diverse inputs. The generation rules match make_inputs (consumption order
    # x -> scale -> bias; here the global RNG is consumed)
    n, c, h, w = 2, 16, 512, 512
    x = torch.randn(n, c, h, w)
    scale = torch.randn(c)
    bias = torch.randn(c)
    return [x, scale, bias]


def make_inputs(batch: int, channels: int, height: int, width: int,
                groups: int, variant: str = "groupnorm",
                dtype: str = "float32", eps: float = 1e-5, seed: int = 0):
    """Generate inputs from a public/hidden case description. Runs on the evaluation
    side only (generated on CPU; the evaluator moves them to the device).

    The consumption order is fixed: x -> scale -> bias (all torch.randn, then cast
    to dtype); everything comes from the same torch.Generator(seed), so the same
    seed reproduces inputs bit-for-bit.
    variant / groups / eps are also the init hyper-parameters used to construct the
    Model and are only validated here; the output packing protocol is always
    float32, so dtype only decides the input quantization level.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert variant in ("groupnorm", "groupnorm_swish"), "variant must be one of groupnorm / groupnorm_swish"
    assert min(batch, channels, height, width) >= 1, "batch/channels/height/width must all be >= 1"
    assert 1 <= groups <= channels and channels % groups == 0, "groups must divide channels and 1 <= groups <= channels"
    assert eps > 0, "eps must be positive"
    x = torch.randn(batch, channels, height, width, generator=gen).to(dt)
    scale = torch.randn(channels, generator=gen).to(dt)
    bias = torch.randn(channels, generator=gen).to(dt)
    return x, scale, bias
