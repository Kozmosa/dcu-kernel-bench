# 3010_topk -- in-row Top-K (values plus the original column indices), as a
# model_class (KernelBench-compatible) task file.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=3010, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """In-row Top-K selection: for every row of a 2-D score matrix take the k
    largest values (strictly descending) together with their original column
    indices. This operator is the selection stage of MoE gating, where top-k
    routing is applied to the expert logits.

    Operator semantics: the input x has shape (B, M) and, independently for every
    row b:
        (values[b, :], indices[b, :]) = TopK(x[b, :], k)
    values[b, :] are the k largest elements of x[b, :] in strictly descending
    order; indices[b, j] is the column index of values[b, j] inside that row
    (0 <= indices[b, j] < M, distinct within a row), that is out[b, k + j]
    satisfies x[b, out[b, k + j]] == out[b, j]. Mathematically this equals a full
    descending sort of each row followed by taking the first k values and their
    original column positions. Boundary behaviour: k = 1 degenerates to the row
    maximum and its argmax; k = M degenerates to a descending argsort of the whole
    row; with M = 1 the output is that row's only element and index 0. The
    evaluation inputs guarantee that the elements of each row are mutually
    unequal (built as a random permutation of a set of distinct values), so the
    Top-K result is uniquely determined -- the tie order of equal values is not
    part of this task's semantics.

    Implementation path hints (free choice):
      - short rows (M <= 1024): a single kernel selecting the maximum round by
        round -- each round takes the current maximum and the column index of its
        first occurrence, writes them out and then masks that position with the
        smallest float32 value, for exactly k rounds;
      - long rows (M > 1024): two stages -- cut the row into fixed-length chunks,
        run a local Top-K per chunk to obtain chunk_num*k candidates (values plus
        global column indices), then merge-sort the candidates and take the first
        k.

    Input/output specification:
        x   (B, M) float32 (the evaluator casts all inputs to fp32 before passing
            them in), contiguous, row elements mutually unequal; B >= 1,
            M >= 1 and < 2^24 (column indices are exactly representable in the
            fp32 packing), non powers of two allowed; the task scale domain is
            M in [16, 131072].
        k   constructor hyperparameter (passed via __init__), 1 <= k <= M.
        out (B, 2*k) a single float32 tensor (evaluation protocol: single-tensor
            output):
            out[:, :k]   = values (strictly descending, a bit-exact copy of the
                           input values)
            out[:, k:]   = indices (integer column indices, < 2^24, exactly
                           representable in fp32)
        The output is a pure selection result (no arithmetic), so any single bit
        of deviation is a semantic error.

    Implementation constraints (violations are scored as failures):
      - The core Top-K selection (scan / tiling / merge) must be done with a
        Triton kernel inside the submitted file, with both values and indices
        written by the kernel; numeric comparison happens in float32.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - The first k output columns must be strictly descending; the last k
        columns must be the original column indices (not chunk-local indices and
        not sort ranks).

    Disallowed list: aiter / torch.topk / torch.sort / torch.argsort /
    torch.kthvalue / torch.mode, plus the Tensor method forms of the shortcuts
    above .topk( / .sort( / .argsort( / .kthvalue( -- calling a selection
    operator directly and doing a full sort to take the first k are the same
    shortcut, so any appearance in any form fails (Triton primitives are
    unrestricted).
    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    """

    def __init__(self, k: int = 8):
        super().__init__()
        assert k >= 1, "k must be >= 1"
        self.k = int(k)

    def forward(self, x):
        # The evaluator casts all inputs to fp32; x is a (B, M) score matrix with
        # distinct row elements
        x = x.to(torch.float32).contiguous()
        values, indices = torch.topk(x, self.k, dim=-1, largest=True, sorted=True)
        # Single-tensor packing: the first k columns are the descending values,
        # the last k columns the original column indices (integers, exact in fp32)
        return torch.cat([values, indices.to(torch.float32)], dim=-1)


def get_init_inputs():
    return [8]   # k


def get_inputs():
    # Fixed shape family: B=8, M=128256 (long rows -> the two-stage path), k=8;
    # the random part consumes the global RNG -- the evaluator calls this
    # function after set_seed, so multiple correctness trials see diverse inputs
    return list(_gen_case(batch_size=8, row_len=128256, dtype=torch.float32, rng=None))


def make_inputs(batch_size: int, row_len: int, k: int = 8,
                dtype: str = "float32", seed: int = 0):
    """Generate inputs from a public/hidden case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the
    device). The fields map one-to-one onto the hidden/perf cases; k is also the
    Model constructor hyperparameter and is passed to Model as it is.

    The consumption order is fixed (the same torch.Generator(seed)): one randperm
    per row; bit-reproducible under the same seed.
    """
    assert 1 <= k <= row_len, "k must satisfy 1 <= k <= row_len"
    gen = torch.Generator().manual_seed(seed)
    return _gen_case(batch_size=batch_size, row_len=row_len,
                     dtype=getattr(torch, dtype), rng=gen)


def _gen_case(batch_size, row_len, dtype, rng):
    """Deterministic generator shared by get_inputs (global RNG) and make_inputs
    (Generator).

    The distribution follows the family of the official test: every row is an
    independent random permutation of the same set of distinct values (the
    official test shuffles an arange in place; for low-precision dtypes this uses
    a value set that is still bit-exact and mutually distinct in that dtype, see
    _distinct_values), so Top-K has no tie ambiguity.
    """
    vals = _distinct_values(row_len, dtype)
    x = torch.empty(batch_size, row_len, dtype=dtype)
    for b in range(batch_size):
        x[b] = vals[torch.randperm(row_len, generator=rng)].to(dtype)
    return [x]


def _distinct_values(row_len, dtype):
    """Return row_len ascending, mutually unequal values that are bit-exact in the
    given dtype.

    A dtype has a finite number of significant mantissa bits (fp32 24 / fp16 11 /
    bf16 8): while row_len stays inside the exact-integer range this uses arange
    directly (the value set of the official test); beyond that the values are laid
    out per binary binade -- the i-th value is 2^(i//P) * (1 + (i%P)/P), with P
    the number of mantissa slots inside one binade (2^(significant bits - 1));
    mantissas differ within a binade and exponents differ across binades, so any
    two values remain distinct in the dtype (the distinct-value ceiling of both
    fp16 and bf16 is 16384, which covers the task scale domain).
    """
    sig_bits = {torch.float32: 24, torch.float16: 11, torch.bfloat16: 8}[dtype]
    if row_len <= 2 ** sig_bits:
        return torch.arange(row_len, dtype=torch.float64)
    per_binade = 2 ** (sig_bits - 1)
    idx = torch.arange(row_len)
    e, r = idx // per_binade, idx % per_binade
    return torch.pow(2.0, e.double()) * (1.0 + r.double() / per_binade)
