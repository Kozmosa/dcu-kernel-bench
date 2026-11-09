# 8002_gather -- model_class task file for Gather / GatherElements
# (the index-selection family), KernelBench compatible.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=1, problem_id=8002, backend=cuda (HIP).

import torch
import torch.nn as nn


class Model(nn.Module):
    """Gather / GatherElements: the index-movement family that selects elements
    along a given axis by integer index.

    The __init__ parameter `variant` selects one of two paths; their semantics
    are defined as follows:
        "gather" (select along an axis): `indices` is a 1-D integer vector of
            arbitrary length K;
            out[i_0, ..., i_{axis-1}, j, i_{axis+1}, ...] =
                data[i_0, ..., i_{axis-1}, indices[j], i_{axis+1}, ...]
            i.e. indices[j] picks one whole slice along dimension `axis` at a
            time; the output shape equals the data shape with dimension `axis`
            replaced by K.
        "gather_elements" (element-wise selection with a same-shaped index):
            `indices` has the same rank as data, with every dimension equal to
            data except dimension `axis` (whose length K may differ from data);
            out[i_0, ..., i_{n-1}] = data[i_0, ..., m, ..., i_{n-1}] where
            m = indices[i_0, ..., i_{n-1}] -- each output element takes its own
            subscript along `axis`; the output shape equals the indices shape.
    The difference: gather's index is a 1-D vector that decides how a whole
    slice along `axis` is selected at once; gather_elements' index has the same
    shape as the output and specifies the subscript along `axis` per element.
    The __init__ parameter `axis` sets the selection dimension (torch
    convention: non-negative counts from 0, negative counts from the end),
    defaulting to 1.

    Numeric contract: this operator is pure index movement with no arithmetic.
    A correct implementation produces an output bit-identical to the selected
    elements of the input (bit patterns preserved; no dtype conversion, no
    numeric operation).

    Input/output specification:
        data    float32 / float16 / bfloat16, contiguous, ndim in [1, 4],
                every dimension >= 1, total element count <= 2^23; values are
                ordinary randn magnitude (no inf/NaN)
        indices an integer tensor: for variant="gather" an int64 1-D tensor
                (length K >= 1); for variant="gather_elements" an int32 tensor
                with the same rank as data (shape constraints above); every
                index value lies in [0, data.shape[axis]) -- the input contains
                no out-of-range index
        out     a single tensor, same dtype as data, contiguous: for gather its
                shape is the data shape with dimension `axis` replaced by K;
                for gather_elements its shape is the indices shape. forward must
                return a single tensor (the evaluator compares .shape /
                allclose on a single output).
        The online evaluator casts all inputs to fp32 before calling forward
        (indices are small non-negative integers < 2^24 and are exactly
        representable in fp32, so they are recovered losslessly; fp16/bf16 ->
        fp32 is also lossless). The offline final audit passes inputs natively
        in their declared dtype -- the implementation must be correct for both.

    Boundary behaviour: K=1, K different from data.shape[axis], an axis of
    length 1, non-power-of-two / prime lengths, and repeated indices hitting the
    same element must all be handled correctly; a negative `axis` follows the
    torch convention.

    Implementation constraints (violations are scored as failures):
        - The core computation (index expansion / address computation plus
          load/store at those addresses) must be completed with a HIP kernel
          inside the submitted file.
        - The __init__ and forward signatures of ModelNew must not be changed.
        - Pure-movement contract: the output is bit-identical to the selected
          elements of data (including float16 / bfloat16 bit patterns); no
          numeric transformation may be introduced.
        - forward returns the single tensor `out` (the evaluator compares a
          single output).

    Forbidden (static audit strips docstrings and comments, then does literal
    substring matching):
    torch.gather / .gather( / torch.index_select /
    .index_select( / torch.take_along_dim / .take_along_dim( /
    torch.take( / .take( (including the Tensor.take method form) /
    torch.embedding / nn.Embedding / .embedding( / hipdnn.
    Target hardware: Hygon DCU (gfx936/gfx938). Implementation language: HIP
    (compiled by hipcc; write inside the CUDA_CU_SRC / CUDA_CPP_SRC anchor
    regions; device code must include <hip/hip_runtime.h> -- <cuda_runtime.h>
    is not available on this machine).
    """

    def __init__(self, variant: str = "gather", axis: int = 1):
        super().__init__()
        assert variant in ("gather", "gather_elements"), "variant must be one of gather / gather_elements"
        self.variant = variant
        self.axis = int(axis)

    def forward(self, data, indices):
        # The online evaluator casts every input (integer indices included) to
        # fp32; the indices are small non-negative integers exactly representable
        # in fp32, restored losslessly to int64 here.
        ndim = data.dim()
        axis = self.axis % ndim
        idx = indices.to(torch.long)

        if self.variant == "gather_elements":
            # Same-shaped index: output shape = indices shape, each element
            # takes its own subscript along `axis`.
            return torch.gather(data, axis, idx)

        # 1-D index vector: expand it to an index tensor with the same rank as
        # data, then select along `axis` (broadcasting the existing subscripts
        # of data over all dimensions except `axis`); the output shape is the
        # data shape with dimension `axis` replaced by K.
        k = idx.shape[0]
        view = [1] * ndim
        view[axis] = k
        full = idx.view(view).expand(data.shape[:axis] + (k,) + data.shape[axis + 1:])
        return torch.gather(data, axis, full)


def get_init_inputs():
    return ["gather", 1]   # variant, axis


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs.
    b, c, h = 2, 3, 4
    k = 2
    data = torch.randn(b, c, h)          # float32 as the primary dtype
    indices = torch.randint(0, c, (k,))  # int64, values in [0, c)
    return [data, indices]


def make_inputs(variant, data_shape, index_shape, axis, dtype="float32", seed=0):
    """Generate inputs from a public/hidden case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    The consumption order is fixed: data (torch.randn, then cast to dtype) ->
    indices (torch.randint with the upper bound equal to the length of data
    along `axis`; dtype by convention: int64 for gather, int32 for
    gather_elements). Everything comes from the same torch.Generator(seed), so
    the same seed reproduces inputs bit-for-bit.
    `variant` and `axis` are also the init hyper-parameters used to construct
    the Model; they are validated here.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert variant in ("gather", "gather_elements"), "variant must be one of gather / gather_elements"
    ndim = len(data_shape)
    assert 1 <= ndim <= 4, "requires 1 <= data ndim <= 4"
    assert all(s >= 1 for s in data_shape), "every data dimension must be >= 1"
    d = int(axis) % ndim
    numel = 1
    for s in data_shape:
        numel *= s
    assert numel <= 2 ** 23, "total data elements must be <= 2^23"
    n = data_shape[d]
    if variant == "gather":
        assert len(index_shape) == 1, "gather indices must be 1-D"
        k = int(index_shape[0])
    else:
        assert len(index_shape) == ndim, "gather_elements indices must have the same rank as data"
        for i, (sd, si) in enumerate(zip(data_shape, index_shape)):
            if i != d:
                assert si == sd, "gather_elements indices must match data on every dimension except axis"
        k = int(index_shape[d])
    assert k >= 1, "index length K must be >= 1"
    assert (numel // n) * k <= 2 ** 23, "total output elements must be <= 2^23"
    data = torch.randn(*data_shape, generator=gen).to(dt)
    idx_dtype = torch.int64 if variant == "gather" else torch.int32
    indices = torch.randint(0, n, list(index_shape), generator=gen, dtype=idx_dtype)
    return data, indices
