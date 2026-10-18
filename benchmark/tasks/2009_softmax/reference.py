# 2009_softmax -- model_class (KernelBench compatible) task file for the 2D
# row-wise online softmax (two-pass scan inside a single kernel).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=2, problem_id=2009, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """2D row-wise softmax (the numerically stable online two-pass form).

    For each row (the last dimension) of an (M, N) input, independently:
        m[i]     = max_j x[i, j]
        z[i, j]  = exp(x[i, j] - m[i])
        y[i, j]  = z[i, j] / sum_j z[i, j]
    Rows are completely independent, and after per-row normalization the elements
    of every row sum to 1. The reference implementation does max / exp / sum
    entirely in float32 and casts the output back to the input dtype.
    Target implementation shape (the computation organization with the same
    semantics as the task statement): inside a single Triton kernel, scan each row
    in blocks twice -- the first pass (online) maintains the running maximum m and
    the rescaled partial sum
    s <- s * exp(m_old - m_new) + sum_block exp(x - m_new), and the second pass
    writes out exp(x - m) / s; rows longer than a single block are handled with a
    blocked loop.

    Boundary behaviour: N is any positive integer -- when N=1 the row's output is
    always 1; non-power-of-two N and very long rows (N beyond a single block's
    limit -- counting a 64KB shared buffer, 65536/element_size = 32768 elements for
    fp16/bf16) must all be handled with masking / a blocked loop, and out-of-range
    elements at the row tail must not participate in the max and sum reductions;
    the inputs are ordinary randn magnitude (no inf/NaN).

    Input/output specification:
        x   float16 / bfloat16 (M, N), contiguous, M >= 1, N >= 1
        y   same dtype as x, (M, N), the row-wise softmax result
        The evaluator casts all inputs to fp32 before calling forward
        (fp16/bf16 -> fp32 is lossless); this operator has no integer tensor
        input, so nothing needs restoring.

    Implementation constraints (violations are scored as failures):
        - The core computation (the in-row max reduction, the exp sum reduction,
          and the element-wise normalization) must be completed with a Triton
          kernel inside the submitted file.
        - The __init__ and forward signatures of ModelNew must not be changed.
        - Intermediates (max / sum / exp) are computed in float32; the output is
          cast back to the input dtype.

    Forbidden list: aiter / torch.softmax / torch.nn.functional.softmax /
    torch.nn.Softmax / torch.log_softmax / torch.nn.functional.log_softmax /
    torch.nn.LogSoftmax / torch.special.softmax / torch.special.log_softmax.
    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    """

    def __init__(self):
        super().__init__()
        # The softmax forward has no hyper-parameters: the block size and the
        # in-row blocking strategy are implementation details chosen by the
        # submitter.

    def forward(self, x):
        # The evaluator casts the input to the evaluation precision (fp32 for the
        # triton backend) before calling forward; compute uniformly in float32
        # (numerically stable: subtract the row max before exp), then cast the
        # output back to the input dtype (fp32 online, the original dtype
        # offline).
        dt = x.dtype
        xf = x.to(torch.float32)
        m = xf.max(dim=-1, keepdim=True).values
        e = torch.exp(xf - m)
        y = e / e.sum(dim=-1, keepdim=True)
        return y.to(dt)


def get_init_inputs():
    return []   # no hyper-parameters


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs.
    m, n = 1024, 8192
    x = torch.randn(m, n).to(torch.float16)
    return [x]


def make_inputs(m: int, n: int, dtype: str = "float16", seed: int = 0):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    The consumption order is fixed: x = torch.randn(m, n, generator=gen).to(dtype),
    from the same torch.Generator(seed), so the same seed reproduces inputs
    bit-for-bit.
    Returns the 1-tuple (x,): the offline final audit consumes
    inputs = [t.to(device) for t in inputs] element by element and then expands
    Model(*inputs) positionally, so a bare tensor would be iterated into row vectors.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert m >= 1 and n >= 1, "requires M >= 1 and N >= 1"
    x = torch.randn(m, n, generator=gen).to(dt)
    return (x,)
