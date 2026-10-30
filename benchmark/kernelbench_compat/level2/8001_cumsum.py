# 8001_cumsum -- model_class (KernelBench compatible) task file for
# Cumsum (inclusive prefix sum).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=2, problem_id=8001, backend=cuda (HIP).

import torch
import torch.nn as nn


class Model(nn.Module):
    """Cumsum (inclusive prefix sum): a sequential accumulation along a given
    dimension.

    Operator definition: for an n-dimensional input x and the scan dimension
    `dim`, the output out has the same shape as x:
        out[i_0, ..., i_k, ...] = sum_{j <= i_dim} x[i_0, ..., j, ...]
    i.e. an inclusive prefix sum along dimension `dim` (including the current
    element), with every other dimension accumulating independently per chain.
    The __init__ parameter `dim` selects the scan dimension (torch convention:
    non-negative counts from 0, negative counts from the end, e.g. -1 means the
    last dimension), defaulting to 2.

    Numeric contract: accumulation always happens in float32 -- float16/bfloat16
    inputs are first cast to float32, accumulated, then cast back to the input
    dtype (matching the compute-in-fp32 convention of production scan
    libraries).

    Input/output specification:
        x     float32 / float16 / bfloat16, contiguous, ndim in [1, 4],
              every dimension >= 1, scan dimension length L in [1, 65536],
              total element count <= 2^25; values are ordinary randn magnitude
              (no inf/NaN)
        out   a single tensor, same shape / dtype as x, contiguous; forward must
              return a single tensor (the evaluator compares .shape / allclose
              on a single output).
        The online evaluator casts all inputs to fp32 before calling forward
        (fp16/bf16 -> fp32 is lossless); the offline final audit passes inputs
        natively in their declared dtype -- the implementation must be correct
        for both.

    Boundary behaviour: when L=1, out equals x (cast back to the input dtype);
    non-power-of-two L, prime lengths, and axes longer than a single block's
    capacity must all be handled correctly with masking plus a blocked scan, and
    out-of-range lanes must not contribute to the prefix sum; a negative `dim`
    follows the torch convention.

    Implementation constraints (violations are scored as failures):
        - The core scan (prefix sum) must be handwritten with a HIP kernel
          inside the submitted file; calling any library-level scan primitive is
          forbidden, as is bypassing via mathematically equivalent paths such as
          FFT / matrix multiplication / convolution.
        - The __init__ and forward signatures of ModelNew must not be changed.
        - Accumulate in float32 and cast the output back to the input dtype.
        - forward returns the single tensor `out` (the evaluator compares a
          single output).

    Forbidden (static audit strips docstrings and comments, then does literal
    substring matching):
    torch.cumsum / .cumsum( / torch.cumulative_sum / .cumulative_sum( /
    aten.cumsum / aten.cumulative_sum /
    torch.fft / torch.matmul / torch.mm / torch.bmm / torch.einsum /
    conv1d( / numpy / hipdnn.
    Target hardware: Hygon DCU (gfx936/gfx938). Implementation language: HIP
    (compiled by hipcc; write inside the CUDA_CU_SRC / CUDA_CPP_SRC anchor
    regions; device code must include <hip/hip_runtime.h> -- <cuda_runtime.h>
    is not available on this machine).
    """

    def __init__(self, dim: int = 2):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x):
        # `dim` follows the torch convention (negative counts from the end) and
        # must point at an existing dimension.
        ndim = x.dim()
        assert -ndim <= self.dim < ndim, "dim must point at an existing dimension of the input"
        # Numeric contract: accumulate in float32, cast the output back to the
        # input dtype.
        y = torch.cumsum(x.to(torch.float32), dim=self.dim)
        return y.to(x.dtype)


def get_init_inputs():
    return [2]   # dim


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs.
    b, c, h, w = 8, 16, 512, 128
    x = torch.randn(b, c, h, w)   # float32 as the primary dtype
    return [x]


def make_inputs(shape, dim, dtype="float32", seed=0):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    `shape` is the input shape list (ndim 1-4); `dim` is the scan dimension (also
    the init hyper-parameter used to construct the Model -- it is only validated
    here). Everything comes from the same torch.Generator(seed), so the same seed
    reproduces inputs bit-for-bit; the only tensor consumed is x (torch.randn,
    then cast to dtype).
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    ndim = len(shape)
    assert 1 <= ndim <= 4, "requires 1 <= ndim <= 4"
    assert all(s >= 1 for s in shape), "every dimension must be >= 1"
    d = int(dim)
    assert -ndim <= d < ndim, "dim must point at an existing dimension"
    L = shape[d % ndim]
    assert 1 <= L <= 65536, "scan dimension length must be in [1, 65536]"
    numel = 1
    for s in shape:
        numel *= s
    assert numel <= 2 ** 25, "total element count must be <= 2^25"
    x = torch.randn(*shape, generator=gen).to(dt)
    return (x,)
