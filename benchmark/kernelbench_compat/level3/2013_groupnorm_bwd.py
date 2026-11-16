# 2013_groupnorm_bwd -- model_class (KernelBench compatible) task file for the
# GroupNorm training backward pass (the full dx / dgamma / dbeta gradients).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem-id=2013, backend=cuda (HIP).

import torch
import torch.nn as nn


class Model(nn.Module):
    """GroupNorm training backward: given the upstream gradient dy, the forward input
    x, the scale parameter and the per-group statistics mean / inv_var saved by the
    training forward, compute the full gradients of the GroupNorm forward
        y[n,c,h,w] = xhat[n,c,h,w] * scale[c] + bias[c],
        xhat[n,c,h,w] = (x[n,c,h,w] - mean_ng) * inv_var_ng,
        group index g = c // (C//G)
    with respect to x / scale / bias. The __init__ argument groups = G (dividing C,
    1 <= G <= C); eps appears only in the forward-statistics convention (inv_var
    already carries the eps guard, and the backward math itself contains no eps).
    mean / inv_var are one-dimensional float32 of length N*G, the row-major
    flattening of (N, G), and are always the values saved by the forward: they are
    consumed verbatim as the given values, and recomputing them from x is forbidden.
    The dbeta gradient does not depend on the value of bias, so the forward signature
    does not take bias.
    Mathematical definition (float32 intermediate computation throughout; a group
    (n, g) holds rep = C//G channels and D = rep*H*W elements):
        xhat_nc   = (x_nc - mean_ng) * inv_var_ng
        dgamma[c] = sum_{n,h,w} dy * xhat                      (per channel)
        dbeta[c]  = sum_{n,h,w} dy                             (per channel)
        write S_ng    = sum_{c' in g,h,w} dy * scale_{c'}
              T_ng    = sum_{c' in g,h,w} dy * xhat * scale_{c'}
        dx_nc     = (inv_var_ng / D) * (D * dy_nc * scale_c - S_ng
                                         - xhat_nc * T_ng)
    (S_ng / T_ng are reductions over all channels and spatial positions inside the
    group (n,g), i.e. the complete chain terms of the data gradient; the per-channel
    scale_c is already folded into the two reductions.)

    Single-tensor output protocol (forward must return a single tensor; the evaluator
    compares a single output):
    Returns a one-dimensional float32 packed tensor of length N*C*H*W + 2*C, with
    three segments packed back to back in order (each segment float32, then
    reshape(-1) and torch.cat):
        [0, N*C*H*W)                   dx      ((N, C, H, W) flattened row-major)
        [N*C*H*W, N*C*H*W + C)         dgamma  ((C,))
        [N*C*H*W + C, N*C*H*W + 2*C)   dbeta   ((C,))
    The output is always float32: fp16/bf16 is only the input quantization level (the
    input is provided in that dtype and then cast to float32 to take part in the whole
    computation); the gradients are not quantized to a lower precision.

    Input/output specification:
        x       float32 / float16 / bfloat16 (N, C, H, W), NCHW contiguous,
                N/C/H/W >= 1
        dy      same dtype as x, (N, C, H, W), contiguous (the upstream gradient
                dL/dy)
        scale   same dtype as x, (C,), contiguous
        mean    float32 (N*G,), the row-major flattening of (N, G) (the per-group
                means saved by the forward)
        inv_var float32 (N*G,), the row-major flattening of (N, G), its elements are
                always positive
        out     one-dimensional float32 of length N*C*H*W + 2*C (the three-segment
                packing protocol above)
        The inputs are the usual randn magnitude (no inf/NaN); the evaluator casts
        all inputs to fp32 before passing them in (fp16/bf16 -> fp32 is lossless).

    Implementation constraints (violations are scored as failures):
        - The core computation (xhat reconstruction, the two in-group reductions
          S/T, the per-channel parameter reductions dgamma/dbeta, and the
          element-wise chained multiply-add of dx) must be completed inside the
          submitted file with a HIP kernel; an in-group reduction of any size (D may
          far exceed the capacity of a single block) must use mask / tiling
          (block-wise) reductions, leaving out no element.
        - Computing the gradients through torch.autograd / autograd.grad /
          .backward() is forbidden, as is calling the group_norm primitive family or
          any closed-source library primitive (MIOpen / hipDNN and the like).
        - The __init__ and forward signatures of ModelNew must not be changed.
        - The reductions and the intermediate computation are float32 throughout; all
          three output segments are float32, packed back to back per the segment
          protocol above.

    Boundary behaviour: N, C, H and W may all be 1 (D may be 1, in which case
    xhat = 0, dx = 0, dgamma = 0 and dbeta = dy); G=1 is the backward of full-channel
    normalization and G=C the backward of per-channel normalization, both of which are
    legal; non-powers of two and unaligned dimensions are allowed; the statistics
    shape is always (N*G,).
    Forbidden: aiter / torch.nn.GroupNorm / nn.GroupNorm /
    torch.nn.functional.group_norm / F.group_norm / torch.group_norm /
    native_group_norm / torch.ops.aten.native_group_norm /
    torch.ops.aten.group_norm / autograd / .backward( / miopen / hipdnn.
    Target hardware: Hygon DCU (gfx936/gfx938). Implementation language: HIP
    (compiled by hipcc; write inside the CUDA_CU_SRC / CUDA_CPP_SRC anchor regions;
    device code must include <hip/hip_runtime.h> -- <cuda_runtime.h> is not available
    on this machine).
    """

    def __init__(self, groups: int = 2, eps: float = 1e-5):
        super().__init__()
        assert int(groups) >= 1, "groups must be a positive integer"
        assert eps > 0, "eps must be positive"
        self.groups = int(groups)
        self.eps = float(eps)

    def forward(self, x, dy, scale, mean, inv_var):
        n, c, h, w = x.shape
        g = self.groups
        rep = c // g
        assert 1 <= g <= c and c % g == 0, "groups must divide the number of channels and 1 <= groups <= C"
        assert mean.numel() == n * g and inv_var.numel() == n * g, "mean/inv_var must be one-dimensional statistics of length N*G"

        # Semantic ground truth: backpropagate with autograd through "the 2012
        # same-convention GroupNorm forward". The online evaluator calls the
        # reference forward inside a no_grad context, so this block explicitly
        # enables grad.
        with torch.enable_grad():
            xf = x.to(torch.float32).detach().requires_grad_(True)
            sf = scale.to(torch.float32).detach().requires_grad_(True)
            bf = torch.zeros(c, dtype=torch.float32, device=xf.device,
                             requires_grad=True)  # dbeta does not depend on the value of bias, so take 0

            # the 2012 same-convention forward (in-group biased statistics ->
            # rsqrt(biased variance+eps) -> channel broadcast -> per-channel affine);
            # the derivative structure of the statistics comes from this forward graph
            xg = xf.reshape(n, g, -1)
            mean_t = xg.mean(dim=-1)                                  # (N, G)
            var_t = (xg - mean_t.unsqueeze(-1)).pow(2).mean(dim=-1)   # (N, G) biased variance
            inv_t = torch.rsqrt(var_t + self.eps)

            # identity splicing (separating value from derivative): the values of
            # mean_eff / inv_eff always equal the saved statistics given by the task
            # statement (t - t.detach() is always 0), while the gradients flow back
            # through the forward graph above -- i.e. "backpropagate through the 2012
            # forward at the given saved statistics". The statistics of the evaluation
            # cases are all derived deterministically from x in that forward
            # convention, in which case this matches the closed-form complete chain
            # formula (the task's mathematical definition) term by term.
            mean_eff = mean.to(torch.float32).reshape(n, g) + mean_t - mean_t.detach()
            inv_eff = inv_var.to(torch.float32).reshape(n, g) + inv_t - inv_t.detach()
            mean_c = mean_eff.repeat_interleave(rep, dim=1).reshape(n, c, 1, 1)
            inv_c = inv_eff.repeat_interleave(rep, dim=1).reshape(n, c, 1, 1)
            y = (xf - mean_c) * inv_c * sf.reshape(1, c, 1, 1) + bf.reshape(1, c, 1, 1)

            # the upstream gradient dy is injected as grad_outputs: L = sum(y * dy),
            # i.e. dL/dy = dy
            dx, dgamma, dbeta = torch.autograd.grad(
                (y * dy.to(torch.float32)).sum(), (xf, sf, bf))

        # single-tensor output: the three segments dx | dgamma | dbeta concatenated in
        # one dimension (all float32, no cast back)
        return torch.cat([dx.reshape(-1), dgamma.reshape(-1), dbeta.reshape(-1)])


def get_init_inputs():
    return [2, 1e-5]   # groups, eps


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG (order x -> dy ->
    # scale) -- the evaluator calls this function after set_seed, so multiple
    # correctness trials therefore see diverse inputs.
    # The saved statistics are derived deterministically from x in the training
    # snapshot convention (no RNG consumed, always float32): the per-group biased mean
    # + rsqrt(biased variance + 1e-5), flattened row-major as (N, G).
    n, c, h, w, g = 2, 16, 512, 512, 2
    x = torch.randn(n, c, h, w)
    dy = torch.randn(n, c, h, w)
    scale = torch.randn(c)
    xg = x.to(torch.float32).reshape(n, g, -1)
    mean = xg.mean(dim=-1)
    var = (xg - mean.unsqueeze(-1)).pow(2).mean(dim=-1)
    inv_var = torch.rsqrt(var + 1e-5)
    return [x, dy, scale, mean.reshape(-1), inv_var.reshape(-1)]


def make_inputs(batch: int, channels: int, height: int, width: int, groups: int,
                dtype: str = "float32", eps: float = 1e-5, seed: int = 0):
    """Generate inputs from a public/hidden/perf case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the device).

    The consumption order of the random part is fixed: x -> dy -> scale (all
    torch.randn, then cast to dtype); everything comes from the same
    torch.Generator(seed), so the same seed reproduces inputs bit-for-bit.
    mean / inv_var (always float32, not cast to dtype) are derived deterministically
    from the quantized x in the training snapshot convention (the same formula as
    get_inputs: the per-group biased mean + rsqrt(biased variance + eps)), consuming
    no RNG.
    groups / eps are also the init hyper-parameters used to construct the Model and
    are only validated here; eps takes part only in deriving the statistics, and the
    backward math itself contains no eps.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert min(batch, channels, height, width) >= 1, "batch/channels/height/width must all be >= 1"
    assert 1 <= groups <= channels and channels % groups == 0, "groups must divide channels and 1 <= groups <= channels"
    assert eps > 0, "eps must be positive"
    x = torch.randn(batch, channels, height, width, generator=gen).to(dt)
    dy = torch.randn(batch, channels, height, width, generator=gen).to(dt)
    scale = torch.randn(channels, generator=gen).to(dt)
    xg = x.to(torch.float32).reshape(batch, groups, -1)
    mean = xg.mean(dim=-1)
    var = (xg - mean.unsqueeze(-1)).pow(2).mean(dim=-1)
    inv_var = torch.rsqrt(var + float(eps))
    return x, dy, scale, mean.reshape(-1), inv_var.reshape(-1)
