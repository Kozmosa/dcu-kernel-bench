# 1032_deform_attention_bwd -- multi-scale deformable attention backward
# (deform_attn_dgrad), a model_class (KernelBench compatible) task file.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=1032, backend=cuda (HIP).

import torch
import torch.nn as nn


def _deform_attn_forward(value_f, spatial_shapes, level_start_index,
                         sampling_locations_f, attention_weights_f):
    """Multi-scale deformable attention forward (float32, hand-written the same way as task 1031).

    Semantically identical to the reference of task 1031: coordinate expansion,
    four-corner zero-padded bilinear interpolation, softmax normalization of the
    sampling weights over the flattened (levels x points) dimension, and the
    cross-sampling-point weighted reduction. The only difference is that the floor
    path is detached first (the subgradient convention that the floor gradient is
    0 is made explicit, so the gradient only flows back through the interpolation
    fractions fx/fy); numerically it is exactly the same as the forward.
    All inputs are float32 tensors (the caller is responsible for the cast and for
    requires_grad); returns the uncast float32 output (N, Q, heads*D) for the
    gradient chain to use.
    """
    n, n_keys, num_heads, d = value_f.shape
    n_levels = spatial_shapes.shape[0]
    n_queries = sampling_locations_f.shape[1]
    n_points = sampling_locations_f.shape[4]

    # Normalize the sampling weights: flatten (levels x points) into one
    # dimension and run a numerically stable softmax
    wf = attention_weights_f.reshape(n, n_queries, num_heads,
                                     n_levels * n_points)
    wf = torch.softmax(wf, dim=-1).view(n, n_queries, num_heads,
                                        n_levels, n_points)

    n_idx = torch.arange(n, device=value_f.device).view(n, 1, 1)
    h_idx = torch.arange(num_heads, device=value_f.device).view(1, 1, num_heads)

    out = torch.zeros(n, n_queries, num_heads, d,
                      dtype=torch.float32, device=value_f.device)
    for l in range(n_levels):
        h_l = int(spatial_shapes[l, 0])
        w_l = int(spatial_shapes[l, 1])
        base = int(level_start_index[l])
        v_l = value_f[:, base: base + h_l * w_l]   # (n, H_l*W_l, heads, d)

        # normalized coordinates x level size -> real-valued sampling
        # coordinates (channel 0 along W, channel 1 along H)
        sx = sampling_locations_f[:, :, :, l, :, 0] * float(w_l)
        sy = sampling_locations_f[:, :, :, l, :, 1] * float(h_l)
        # The floor / integer-index path is treated as a constant (detached
        # explicitly); the gradient only flows back through fx/fy
        ix0 = torch.floor(sx.detach()).to(torch.long)
        iy0 = torch.floor(sy.detach()).to(torch.long)
        fx = sx - ix0.to(torch.float32)
        fy = sy - iy0.to(torch.float32)

        def corner(iy, ix, wgt):
            # Single-corner sampling: an out-of-range corner contributes 0
            # (zero padding semantics); the clamp only guards the index
            # against going out of range and is cancelled by the 0 weight
            valid = ((iy >= 0) & (iy <= h_l - 1)
                     & (ix >= 0) & (ix <= w_l - 1)).to(torch.float32)
            flat = iy.clamp(0, h_l - 1) * w_l + ix.clamp(0, w_l - 1)
            vals = v_l[n_idx, flat, h_idx]        # (n, q, heads, P, d)
            return vals * (valid * wgt).unsqueeze(-1)

        for p in range(n_points):
            iy0p, ix0p = iy0[..., p], ix0[..., p]      # (n, q, heads)
            fyp, fxp = fy[..., p], fx[..., p]
            sampled = (corner(iy0p, ix0p, (1.0 - fyp) * (1.0 - fxp))
                       + corner(iy0p, ix0p + 1, (1.0 - fyp) * fxp)
                       + corner(iy0p + 1, ix0p, fyp * (1.0 - fxp))
                       + corner(iy0p + 1, ix0p + 1, fyp * fxp))
            out = out + wf[:, :, :, l, p].unsqueeze(-1) * sampled

    return out.view(n, n_queries, num_heads * d)


class Model(nn.Module):
    """Multi-Scale Deformable Attention Backward (deform_attn_dgrad): given the forward inputs and the upstream gradient grad_output, one call computes the gradients of value, sampling_locations and attention_weights at once.

    Forward semantics (all intermediate computation float32, the same convention
    as the companion forward task): value (N, n_keys, heads, D) is the
    concatenation of the multi-scale feature levels -- level l contributes
    spatial_shapes[l] = (H_l, W_l) keys, laid out inside the level in row-major
    order (H first, then W) over [level_start_index[l], level_start_index[l] +
    H_l*W_l), where level_start_index must be the prefix sum of the per-level
    H_l*W_l (first element 0) and n_keys = sum_l H_l*W_l. The last dimension of
    sampling_locations (N, Q, heads, L, P, 2) holds the normalized coordinates
    (x, y): channel 0 runs along W and channel 1 along H (generation domain
    [0,1)), so the actual sampling coordinates are loc_x = x * W_l and loc_y =
    y * H_l; the sampled value goes through four-corner bilinear interpolation
    (floor + the fractions fx/fy, with zero padding semantics where an
    out-of-range corner contributes 0, and no clamp / nearest-neighbour
    pull-back); attention_weights (N, Q, heads, L, P) are non-negative
    unnormalized scores, first flattened over (levels x points) into one
    dimension and normalized by a numerically stable softmax (max subtraction)
    into convex-combination weights, then used in the weighted sum against the
    interpolated value of each sampling point, producing the output o
    (N, Q, heads*D).

    Backward semantics (backpropagated through the float32 forward described
    above, at the point before the output is cast back to the value dtype;
    grad_output is first cast to float32): write dO for the float32 form of
    grad_output, w = softmax(attention_weights flattened over L*P), S_{l,p}[c] for
    the four-corner zero-padded bilinear sample of point (l,p), and
    t_{l,p} = <dO, S_{l,p}>; the three gradients are then the analytic gradients
    of o with respect to the three floating-point inputs:
      grad_value[n, key, h, c] = sum_{q, l, p} w_{l,p} * corner_weight *
                                 dO[n, q, h*D + c]
        (per-corner scatter accumulation; out-of-range corners do not scatter)
      grad_sampling_locations[n, q, h, l, p, 0] =
          sum_c dO[n, q, h*D + c] * w_{l,p} * W_l *
          ((1-fy) * (v01 - v00) + fy * (v11 - v10))
      grad_sampling_locations[n, q, h, l, p, 1] =
          sum_c dO[n, q, h*D + c] * w_{l,p} * H_l *
          ((1-fx) * (v10 - v00) + fx * (v11 - v01))
        (v00/v01/v10/v11 are the four corner pixel values of that point, with an
        out-of-range corner counted as 0; the gradient only flows back through
        the fractions fx/fy, and the floor / integer-index path has gradient 0 --
        the standard subgradient of bilinear sampling)
      grad_attention_weights[n, q, h, l, p] = w_{l,p} *
          (t_{l,p} - sum_{l', p'} w_{l',p'} * t_{l',p'})
        (softmax backprop; identically 0 when L*P = 1)
    The softmax, the interpolation, the weighted reduction and all gradient
    accumulation are done in float32, and the three gradients are each cast back
    to the value dtype and then packed into the output.

    Output contract (single tensor, one-dimensional concatenation packing, the
    segment order is fixed and must not be swapped):
      out = cat([grad_value.reshape(-1),
                 grad_sampling_locations.reshape(-1),
                 grad_attention_weights.reshape(-1)])
      a one-dimensional tensor with the same dtype as value, of length
          = n*n_keys*heads*D + n*n_queries*heads*L*P*2 + n*n_queries*heads*L*P
      segment 0 grad_value: n*n_keys*heads*D elements, (N, n_keys, heads, D)
          flattened row-major;
      segment 1 grad_sampling_locations: n*n_queries*heads*L*P*2 elements,
          (N, Q, heads, L, P, 2) flattened row-major (the (x, y) channel order of
          the last dimension is preserved);
      segment 2 grad_attention_weights: n*n_queries*heads*L*P elements,
          (N, Q, heads, L, P) flattened row-major.

    Input/output specification (all tensors contiguous):
      value              float32 / float16 (N, n_keys, heads, D)
      spatial_shapes     int64 (L, 2), each row (H_l, W_l), H_l / W_l >= 1
      level_start_index  int64 (L,), prefix sum of the per-level key counts
      sampling_locations same dtype as value (N, Q, heads, L, P, 2), generation domain [0,1)
      (with an fp16 input individual elements may round to exactly 1.0 through
       the cast; then loc_x = W_l is fully out of range, the sampled value is 0,
       and the zero padding semantics apply)
      attention_weights  same dtype as value (N, Q, heads, L, P), non-negative
      grad_output        same dtype as value (N, Q, heads*D), the upstream
                         gradient in the forward output layout, generation domain [0,1)
      out                single tensor, same dtype as value, 1-D three-segment
                         concatenation (see above)
      every dimension is an arbitrary positive integer (including 1, non-powers
      of 2, and non-16-aligned sizes; a single-level single-pixel (H_l, W_l) =
      (1, 1), 1xN / Nx1 thin-strip levels, Q=1, P=1 and L=1 are all legal
      boundaries, and when L*P=1 grad_attention_weights is identically 0). The
      evaluator casts every input to fp32 before passing it in (lossless for both
      the low precision and the int64 indices -- an index value <= 2^24 is exactly
      representable in fp32), so the implementation must restore
      spatial_shapes / level_start_index to integer indices before using them.

    Implementation constraints (violations are scored as failures):
      - The core computation (softmax backprop, the four-corner gradient scatter
        with the partial derivatives of the interpolation fractions, and the
        cross-query grad_value scatter accumulation) must be hand-written as a HIP
        kernel inside the submitted file.
      - Using torch.autograd / autograd.grad / .backward() to backpropagate for the
        gradients is forbidden; the gradients must be hand-written from the
        analytic formulas above.
      - All gradient accumulation is done in float32; the three gradient segments
        are cast back to the value dtype and concatenated in the segment order
        above; forward returns the single tensor out (the evaluator compares a
        single output).
      - The __init__ and forward signatures of ModelNew must not be changed.

    Forbidden list: torchvision (including ready-made deformable attention /
    deformable convolution primitives such as torchvision.ops.ms_deform_attn),
    mmcv, miopen, hipdnn, aiter,
    torch.nn.functional.grid_sample (F.grid_sample),
    torch.nn.functional.interpolate (F.interpolate), torch.matmul /
    torch.bmm / torch.mm / torch.einsum / torch.tensordot,
    torch.softmax / torch.nn.functional.softmax (F.softmax),
    torch.autograd (including autograd.grad), .backward().
    Target hardware: Hygon DCU (gfx936/gfx938). Implementation language: HIP (compiled by hipcc; write inside the CUDA_CU_SRC / CUDA_CPP_SRC anchor regions; device code must include <hip/hip_runtime.h> -- <cuda_runtime.h> is not available on this machine).
    """

    def __init__(self):
        super().__init__()

    def forward(self, value, spatial_shapes, level_start_index, sampling_locations, attention_weights, grad_output):
        # The evaluator casts every input to fp32 (cuda backend); spatial_shapes /
        # level_start_index are small integer indices that fp32 represents
        # losslessly, so they are restored to long here
        spatial_shapes = spatial_shapes.to(torch.long)
        level_start_index = level_start_index.to(torch.long)

        n, n_keys, num_heads, d = value.shape
        n_levels = spatial_shapes.shape[0]
        n_queries = sampling_locations.shape[1]
        n_points = sampling_locations.shape[4]
        assert sampling_locations.dim() == 6 \
            and sampling_locations.shape[2] == num_heads \
            and sampling_locations.shape[3] == n_levels \
            and sampling_locations.shape[5] == 2, \
            "sampling_locations must be (N, Q, heads, L, P, 2)"
        assert attention_weights.shape == (n, n_queries, num_heads, n_levels, n_points), \
            "attention_weights must be (N, Q, heads, L, P)"
        assert value.shape[1] == int(spatial_shapes.prod(dim=1).sum()), \
            "dimension 1 of value must equal the sum of the per-level H_l*W_l"
        assert grad_output.shape == (n, n_queries, num_heads * d), \
            "grad_output must be (N, Q, heads*D)"

        # Build the graph over the float32 intermediate forward and then
        # backpropagate with autograd (the reference is not bound by the forbidden
        # list; the gradient definition = the analytic gradient of the
        # float32-semantics forward)
        vg = value.detach().to(torch.float32).requires_grad_(True)
        lg = sampling_locations.detach().to(torch.float32).requires_grad_(True)
        ag = attention_weights.detach().to(torch.float32).requires_grad_(True)
        out = _deform_attn_forward(vg, spatial_shapes, level_start_index, lg, ag)
        grad_value, grad_sampling_loc, grad_attn_weight = torch.autograd.grad(
            out, (vg, lg, ag), grad_output.to(torch.float32))

        # Single-tensor output: the three segments are concatenated
        # one-dimensionally in the order grad_value | grad_sampling_locations |
        # grad_attention_weights, each cast back to the value dtype
        return torch.cat([grad_value.to(value.dtype).reshape(-1),
                          grad_sampling_loc.to(value.dtype).reshape(-1),
                          grad_attn_weight.to(value.dtype).reshape(-1)])


def get_init_inputs():
    return []   # this task has no constructor hyper-parameters (Model is built without arguments)


def get_inputs():
    # Fixed shape family (n=2, heads=2, D=32, L=2, P=2, Q=32; spatial_shapes gives
    # (H_l, W_l) ~ randint[1,16) per level, from which n_keys and the value shape
    # are derived); the random part consumes the global RNG -- the evaluator calls
    # this function after set_seed, so multiple correctness trials therefore see
    # diverse inputs. The consumption order is fixed: spatial_shapes -> value ->
    # sampling_locations -> attention_weights -> grad_output
    n, n_heads, d_head, n_levels, n_points, n_queries = 2, 2, 32, 2, 2, 32
    spatial_shapes = torch.randint(1, 16, (n_levels, 2), dtype=torch.int64)
    n_keys = int(spatial_shapes.prod(dim=1).sum())
    count_per_level = spatial_shapes.prod(dim=1)
    level_start_index = torch.zeros_like(count_per_level)
    level_start_index[1:] = torch.cumsum(count_per_level[:-1], dim=0)
    value = torch.rand(n, n_keys, n_heads, d_head, dtype=torch.float32)
    sampling_locations = torch.rand(
        n, n_queries, n_heads, n_levels, n_points, 2, dtype=torch.float32)
    attention_weights = torch.rand(
        n, n_queries, n_heads, n_levels, n_points, dtype=torch.float32)
    grad_output = torch.rand(n, n_queries, n_heads * d_head, dtype=torch.float32)
    return [value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights, grad_output]


def make_inputs(n: int, n_heads: int, embed_dims_per_head: int, n_queries: int,
                n_levels: int, n_points: int, spatial_shapes=None,
                max_size: int = 15, dtype: str = "float32", seed: int = 0):
    """Generate inputs from a public/hidden/perf case description. Runs on the evaluation side only (generated on CPU; the evaluator moves them to the device). This task has no constructor hyper-parameters, so the case fields are all parameters here.

    The consumption order is fixed: spatial_shapes (when not given explicitly,
    element by element ~ randint[1, max_size], default max_size=15) -> value ->
    sampling_locations -> attention_weights -> grad_output (rand tensors, then cast
    to dtype), all from the same torch.Generator(seed), so the same seed
    reproduces the result bit for bit; level_start_index is derived from
    spatial_shapes (prefix sum) and consumes no RNG.
    Returns, in the same order as the forward parameters,
    (value, spatial_shapes, level_start_index, sampling_locations,
    attention_weights, grad_output).
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert min(n, n_heads, embed_dims_per_head, n_queries, n_levels, n_points) >= 1, \
        "every dimension must be >= 1"
    max_size = int(max_size)
    assert max_size >= 1, "max_size must be >= 1"
    if spatial_shapes is None:
        ss = torch.randint(1, max_size + 1, (n_levels, 2),
                           generator=gen, dtype=torch.int64)
    else:
        ss = torch.as_tensor(spatial_shapes, dtype=torch.int64).reshape(n_levels, 2)
        assert int(ss.min()) >= 1, "H_l / W_l of every level must be >= 1"
    n_keys = int(ss.prod(dim=1).sum())
    assert n_keys <= 2 ** 24, "n_keys must be <= 2^24 (the upper bound for an fp32-exact index)"
    count_per_level = ss.prod(dim=1)
    level_start_index = torch.zeros_like(count_per_level)
    level_start_index[1:] = torch.cumsum(count_per_level[:-1], dim=0)
    value = torch.rand(n, n_keys, n_heads, embed_dims_per_head,
                       generator=gen).to(dt)
    sampling_locations = torch.rand(n, n_queries, n_heads, n_levels, n_points, 2,
                                    generator=gen).to(dt)
    attention_weights = torch.rand(n, n_queries, n_heads, n_levels, n_points,
                                   generator=gen).to(dt)
    grad_output = torch.rand(n, n_queries, n_heads * embed_dims_per_head,
                             generator=gen).to(dt)
    return (value, ss, level_start_index, sampling_locations, attention_weights,
            grad_output)
