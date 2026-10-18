# 2003_fused_mul_add -- element-wise fused multiply-add out = a*x + b, as a
# KernelBench compatible model_class task file.
#
# This file must be self-contained: the evaluator loads the task via exec(source),
# so local module imports are forbidden -- the reference implementation
# (Model.forward) and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=2, problem_id=2003, backend=triton.

import torch
import torch.nn as nn

# The four value forms of a / b (matching the operator contract):
#   scalar_float -- a Python float scalar; scalar_int -- a Python int scalar;
#   tensor1      -- a single-element tensor (numel == 1); tensor_full -- a tensor
#                   with the same shape as x.
_OPERAND_KINDS = ("scalar_float", "scalar_int", "tensor1", "tensor_full")


def _draw_operand(kind, shape, dt, gen):
    """Draw a / b according to the form: scalars come from randn*100 (matching the
    magnitude convention of the official operator), tensors come from randn.

    When gen is None the global RNG is consumed (the get_inputs path); otherwise the
    given Generator is consumed (the make_inputs deterministic path).
    """
    if kind == "tensor_full":
        return torch.randn(*shape, dtype=dt, generator=gen)
    base = torch.randn(1, dtype=dt, generator=gen)
    if kind == "scalar_float":
        return float(base.item() * 100)
    if kind == "scalar_int":
        return int(base.item() * 100)
    if kind == "tensor1":
        return base.reshape(1)
    raise ValueError(f"unknown form: {kind}")


class Model(nn.Module):
    """Element-wise fused multiply-add: out = a * x + b.

    (1) Operator semantics: for every element i of x,
    out[i] = a_i * x[i] + b_i. a and b each independently take one of the
    following three forms (any combination is allowed, e.g. a a scalar while b is a
    same-shape tensor):
      1. a Python float / int scalar: a_i = a (the value is broadcast to every
         element of x);
      2. a single-element tensor (numel == 1, any shape, e.g. (1,)): a_i = its only
         element (broadcast);
      3. a tensor with the same shape as x (numel == x.numel()): a_i = the
         corresponding element of a.
    x is a floating-point tensor of arbitrary rank and shape (including
    non-powers-of-two and prime sizes). Intermediate computation runs in float32
    (x / a / b are first promoted to float32, then multiplied and added), and the
    final result is cast back to the dtype of x. Boundary behaviour: this operator
    has no integer tensor inputs and no indexing/length-like semantics; when a and
    b are both scalars the output still has the same shape as x.

    (2) Input/output specification:
      - x: float16 / bfloat16 / float32 tensor, arbitrary shape, contiguous;
      - a, b: one of the three forms above; when a tensor, contiguous with the
        same dtype as x;
      - returns: a new tensor with the same shape and dtype as x (out-of-place; no
        input may be overwritten in place).

    (3) Implementation constraints (violations are scored as failures):
      - The core computation (the multiply-add) must be completed inside the
        submitted file with a self-written Triton kernel: a single kernel and a
        single read-compute-write pass; splitting it into a multiply kernel followed
        by an add kernel is forbidden;
      - When a / b is a scalar or a single-element tensor it must take part in the
        kernel as a broadcast argument and must not be materialized into a
        temporary big tensor with the same shape as x;
      - The __init__ and forward signatures of ModelNew must not be changed;
      - Use float32 for intermediate computation; cast the output back to the dtype
        of x.

    (4) Forbidden list and target hardware:
      - Directly completing this task's computation with ATen shortcuts is
        forbidden: torch.addcmul (including the .addcmul / .addcmul_ methods),
        torch.add, torch.mul, and tensor methods with their in-place forms such as
        .add( / .mul( / .add_( / .mul_; calling a same-functionality entry point
        provided by any precompiled operator library is also forbidden (the final
        static audit does substring matching on the code outside
        docstrings/comments);
      - Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    """

    def __init__(self):
        super().__init__()
        # An element-wise operator has no hyper-parameters: the block size and the
        # broadcast strategy are implementation details chosen by the submitter.

    def forward(self, x, a, b):
        # The evaluator casts the input tensors to the evaluation precision (fp32
        # for the triton backend) before passing them in; Python scalars for a / b
        # are passed through unchanged. This operator has no integer tensor inputs,
        # so nothing needs restoring.
        return (a * x.to(torch.float32) + b).to(x.dtype)


def get_init_inputs():
    return []


def get_inputs():
    # Fixed shape family (matching public_cases.json); the forms of a / b are drawn
    # independently and with equal probability from the four forms using the global
    # RNG -- the evaluator calls this function after set_seed, so multiple
    # correctness trials therefore cover different combinations of scalar /
    # single-element / same-shape tensors.
    shape = (2048, 4096)
    dt = torch.float16
    a_kind = _OPERAND_KINDS[torch.randint(0, len(_OPERAND_KINDS), (1,)).item()]
    b_kind = _OPERAND_KINDS[torch.randint(0, len(_OPERAND_KINDS), (1,)).item()]
    x = torch.randn(*shape, dtype=dt)
    a = _draw_operand(a_kind, shape, dt, None)
    b = _draw_operand(b_kind, shape, dt, None)
    return [x, a, b]


def make_inputs(shape, dtype="float16", a_kind="tensor_full", b_kind="tensor1", seed=0):
    """Deterministic case generator: torch.Generator().manual_seed(seed), generated
    on CPU.

    The fields map one-to-one onto the hidden/perf cases (a shape list, a dtype
    string, the a_kind/b_kind form names, and seed); returns (x, a, b). Runs on the
    evaluation side only.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    shape = tuple(shape)
    x = torch.randn(*shape, dtype=dt, generator=gen)
    a = _draw_operand(a_kind, shape, dt, gen)
    b = _draw_operand(b_kind, shape, dt, gen)
    return x, a, b
