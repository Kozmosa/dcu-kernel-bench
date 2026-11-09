# 8003_scatter_nd -- model_class (KernelBench compatible) task file for
# ScatterND (the index-write family).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=2, problem_id=8003, backend=cuda (HIP).

import torch
import torch.nn as nn


class Model(nn.Module):
    """ScatterND: the index-write family that overwrites positions of `data`
    with `updates` according to integer index tuples.

    Semantics: data has shape (S_0, ..., S_{r-1}) where r is its rank; indices is
    an int64 tensor of shape (K, r) -- each row is a full coordinate tuple that
    picks exactly one position in data; updates has shape (K,) and the same dtype
    as data. The output out has the same shape and dtype as data:
        out[indices[k,0], ..., indices[k,r-1]] = updates[k]
            (hit positions: overwritten, the original value is discarded and
             replaced by the corresponding element of updates)
        all other positions out[i_0, ..., i_{r-1}] = data[i_0, ..., i_{r-1}]
            (misses: the original value is preserved bit-for-bit)
    The write semantics is overwrite, not accumulate; the K coordinate tuples are
    pairwise distinct (the generator samples without replacement), so every data
    element is written at most once. Which write wins for duplicate coordinate
    tuples is undefined behaviour (mainstream frameworks leave the write order
    for duplicate indices undefined either way), and duplicate tuples are outside
    this task's input domain -- the implementation need not define behaviour for
    them, and must not let deduplication / sorting or any other preprocessing
    change the overwrite result for distinct tuples.

    Input/output specification:
        data    float32 / float16 / bfloat16, contiguous, ndim in [1, 6],
                every dimension >= 1, total element count <= 2^23; values are
                ordinary randn magnitude (no inf/NaN)
        indices int64 (K, r) with r = data rank, 1 <= K <= the total element
                count of data; every index value lies in [0, S_d) and the K
                coordinate tuples are pairwise distinct
        updates same dtype as data, shape (K,), contiguous
        out     a single tensor, same shape / dtype as data, contiguous -- this
                operator is naturally single-output, so no multi-output packing is
                needed. forward must return a single tensor (the evaluator
                compares .shape / allclose on a single output).
        The online evaluator casts all inputs to fp32 before calling forward
        (indices are small non-negative integers < 2^24 and exactly
        representable in fp32, so they are recovered losslessly; fp16/bf16 ->
        fp32 is also lossless). The offline final audit passes inputs natively in
        their declared dtype -- the implementation must be correct for both.

    Boundary behaviour: K=1, K equal to the total element count of data (a full
    permutation, every element written), ndim=1 (indices shaped (K, 1)), and
    shapes with non-power-of-two / prime lengths or axes of length 1 must all be
    handled correctly; the output must be a fresh tensor and must not modify
    data / indices / updates in place.

    Implementation constraints (violations are scored as failures):
        - The core computation (r-dimensional coordinate -> linear offset address
          computation, reading updates, and moving the original values out for
          miss positions) must be completed with a HIP kernel inside the
          submitted file.
        - The __init__ and forward signatures of ModelNew must not be changed.
        - Pure-movement contract: miss positions match data bit-for-bit and hit
          positions match updates bit-for-bit (bit patterns preserved); no
          numeric transformation may be introduced.
        - forward returns the single tensor `out` (the evaluator compares a
          single output).

    Forbidden (static audit strips docstrings and comments, then does literal
    substring matching):
    torch.scatter / .scatter( / .scatter_( / scatter_add / scatter_reduce /
    index_put / torch.put( / .put_( / index_copy / torch_scatter / hipdnn.
    Target hardware: Hygon DCU (gfx936/gfx938). Implementation language: HIP
    (compiled by hipcc; write inside the CUDA_CU_SRC / CUDA_CPP_SRC anchor
    regions; device code must include <hip/hip_runtime.h> -- <cuda_runtime.h>
    is not available on this machine).
    """

    def __init__(self):
        super().__init__()

    def forward(self, data, indices, updates):
        # The online evaluator casts every input (integer indices included) to
        # fp32; the indices are small non-negative integers exactly representable
        # in fp32, restored losslessly to int64 here.
        idx = indices.to(torch.long)
        out = data.clone()
        out[tuple(idx.unbind(dim=-1))] = updates.to(out.dtype)
        return out


def get_init_inputs():
    return []   # no construction hyper-parameters: the write semantics is fixed to overwrite


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs.
    data = torch.randn(4, 4)                          # float32 as the primary dtype
    flat = torch.randperm(16)[:2]                     # draw 2 linear positions without replacement
    indices = torch.stack((flat // 4, flat % 4), dim=1)   # int64 (2, 2)
    updates = torch.randn(2)
    return [data, indices, updates]


def make_inputs(data_shape, num_updates, dtype="float32", seed=0):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    The consumption order is fixed: data (torch.randn, then cast to dtype) ->
    indices (torch.randperm over the total element count of data, take the first
    K linear indices, decode them to coordinates per dimension in row-major
    order -- sampling without replacement guarantees the K tuples are pairwise
    distinct) -> updates (torch.randn, then cast to dtype). Everything comes from
    the same torch.Generator(seed), so the same seed reproduces inputs
    bit-for-bit.
    """
    gen = torch.Generator().manual_seed(seed)
    assert dtype in ("float32", "float16", "bfloat16"), "dtype must be float32/float16/bfloat16"
    dt = getattr(torch, dtype)
    shape = tuple(int(s) for s in data_shape)
    ndim = len(shape)
    assert 1 <= ndim <= 6, "requires 1 <= data ndim <= 6"
    assert all(s >= 1 for s in shape), "every data dimension must be >= 1"
    numel = 1
    for s in shape:
        numel *= s
    assert numel <= 2 ** 23, "total data elements must be <= 2^23"
    k = int(num_updates)
    assert 1 <= k <= numel, "requires 1 <= K <= total data elements (upper bound for distinct coordinates)"
    data = torch.randn(*shape, generator=gen).to(dt)
    flat = torch.randperm(numel, generator=gen)[:k]
    coords = []
    for d in range(ndim):
        stride_d = 1
        for j in range(d + 1, ndim):
            stride_d *= shape[j]
        coords.append(flat // stride_d % shape[d])
    indices = torch.stack(coords, dim=1).to(torch.long)   # int64 (K, r)
    updates = torch.randn(k, generator=gen).to(dt)
    return data, indices, updates
