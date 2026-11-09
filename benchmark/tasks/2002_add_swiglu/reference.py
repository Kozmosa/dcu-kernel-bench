# 2002_add_swiglu -- KernelBench compatible model_class task file for the
# training-oriented Add + SwiGLU fusion (forward activation plus first-order
# analytic backward).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here. The evaluation ground
# truth is Model's float32 intermediate computation with the output cast back to
# the input dtype; see task.yaml for the tolerances.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=2, problem_id=2002, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """Training-oriented Add + SwiGLU fused operator: forward activation plus
    first-order analytic backward in a single call.

    Operator semantics:
      base and delta are added element-wise, then split in half along the last
      dimension (let D = width // 2):
        gate = (base + delta)[..., :D], up = (base + delta)[..., D:]
      The forward is a plain (non-clipped, non-interleaved) SwiGLU:
        out = SiLU(gate) * up, where SiLU(g) = g * sigmoid(g)
      The backward receives the upstream gradient grad_out (same shape as out);
      the derivative of SiLU is
        SiLU'(g) = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
      base and delta enter the sum with the same coefficient, so their input
      gradients are element-wise identical:
        grad[..., :D] = (grad_out * up) * SiLU'(gate)
        grad[..., D:] = grad_out * SiLU(gate)
      This task merges forward and backward into a single forward call returning
      one fused tensor (3D columns):
        fused[..., 0:D)   = out (the activation output)
        fused[..., D:2D)  = the gate half of grad (the same for base and delta)
        fused[..., 2D:3D) = the up half of grad
      Boundary behaviour: width is a non-zero even number; num_rows may be 0 (an
      empty batch, output (0, 3D)); the input tensors are not modified; there is
      no broadcasting and no implicit dtype promotion; the evaluation ground truth
      is the float32 intermediate computation, and the tolerances cover the
      differences between half-precision rounding paths, so no specific eager
      rounding chain needs to be reproduced bit-for-bit.

    Input/output specification:
      base     (num_rows, width)      contiguous float16/bfloat16, width a non-zero even number
      delta    (num_rows, width)      same shape, dtype and device as base
      grad_out (num_rows, width // 2) same dtype as base, the upstream gradient of out
      returns  (num_rows, 3 * (width // 2)) same dtype as base, layout as above

    Implementation constraints (violations are scored as failures):
      - The core computation (the addition, SiLU/sigmoid, and the two gradient
        formulas) must be completed with a Triton kernel inside the submitted file;
        torch tensor operations are limited to output allocation and layout
        handling.
      - Calling ready-made implementations such as aiter /
        torch.nn.functional.silu / torch.sigmoid / torch.matmul / torch.bmm /
        torch.einsum is forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Accumulate intermediates in float32; cast the output back to the input
        dtype.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    """

    def __init__(self):
        super().__init__()

    def forward(self, base, delta, grad_out):
        # All inputs are floating point (upstream may cast everything to float32);
        # compute uniformly in float32 internally.
        x = base.to(torch.float32)
        y = delta.to(torch.float32)
        g = grad_out.to(torch.float32)

        gate, up = (x + y).chunk(2, dim=-1)
        sig = torch.sigmoid(gate)
        silu = gate * sig
        out = silu * up

        # First-order analytic backward (identical for base and for delta)
        dgate = (g * up) * (sig + gate * sig * (1.0 - sig))
        dup = g * silu

        fused = torch.cat([out, dgate, dup], dim=-1)
        return fused.to(base.dtype)


def get_init_inputs():
    return []   # the operator has no hyper-parameters; D is inferred from the input width


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs.
    num_rows = 2048
    width = 8192      # D = 4096

    base = torch.randn(num_rows, width).to(torch.float16)
    delta = torch.randn(num_rows, width).to(torch.float16)
    grad_out = torch.randn(num_rows, width // 2).to(torch.float16)
    return [base, delta, grad_out]


def make_inputs(num_rows: int, width: int, dtype: str = "float16", seed: int = 0):
    """Generate inputs from a public/hidden/performance case description
    (deterministic; generated on CPU and then moved to the device by the
    evaluator).

    base/delta have shape (num_rows, width) and grad_out has shape
    (num_rows, width // 2).
    """
    assert isinstance(num_rows, int) and num_rows >= 0
    assert isinstance(width, int) and width > 0 and width % 2 == 0, "width must be a non-zero even number"
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    base = torch.randn(num_rows, width, generator=gen).to(dt)
    delta = torch.randn(num_rows, width, generator=gen).to(dt)
    grad_out = torch.randn(num_rows, width // 2, generator=gen).to(dt)
    return base, delta, grad_out
