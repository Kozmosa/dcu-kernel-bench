# 5005_spconv_fprop -- model_class (KernelBench compatible) task file for the
# sparse convolution forward pass (the gathered form spconv_fprop).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=5005, backend=cuda (HIP).

import torch
import torch.nn as nn


class Model(nn.Module):
    """Sparse convolution forward (the gathered form spconv_fprop): a per
    kernel-offset gather-GEMM-scatter forward over the sparse active points,
    driven by an index-pair table (spconv style).

    Semantic definition (out is first initialized to all zeros, then
    accumulated by the formula below):
        for each kernel offset k in [0, kv):
            n_k = indice_num_per_loc[k]
            for each j in [0, n_k):
                in_idx  = indice_pairs[0, k, j]      # input row index in [0, n)
                out_idx = indice_pairs[1, k, j]      # output row index in [0, num_act_out)
                out[out_idx, :] += features[in_idx, :] @ W[:, :, k].T
    i.e. for each index pair, gather the input row and matrix-multiply it with
    the weight matrix of offset k ((n_k, c_in) @ (c_in, c_out)), then
    scatter-accumulate by the output row index into out. The same output row may
    be hit by any number of index pairs (duplicates are allowed within the same
    k), and all of them accumulate with +=.

    The filters layout mapping (must be respected; it comes from the sample's
    as_strided rearrangement): let F be the contiguous (c_out, c_in, kv) tensor
    of the input; the library interprets its storage with strides
    (c_in*kv, 1, c_in), and the mathematical weight W and F satisfy:
        W[o, i, k] = F.reshape(c_out, kv, c_in)[o, k, i]
    which is equivalent to W = F.reshape(c_out, kv, c_in).permute(0, 2, 1)
    (still shaped (c_out, c_in, kv), a pure reinterpretation view for
    contiguous input); the weight matrix of kernel offset k is W[:, :, k],
    shaped (c_out, c_in).

    Boundary behaviour: kernel offsets with n_k = 0 are skipped entirely; when
    all n_k = 0, out stays all-zero; kv = 1 degenerates to a single-offset
    gather-scatter; num_act_out and n are independent of each other (either may
    be the larger one). Index-pair table entries with j >= n_k are padding
    garbage (they may hold any value, including duplicated valid row indices)
    and must be masked out according to indice_num_per_loc -- they must never be
    read into the computation.

    Input/output specification:
        features           float32 / float16 / bfloat16 (n, c_in), contiguous
        filters            same dtype as features (c_out, c_in, kv), contiguous
        indice_pairs       int32 (2, kv, n), contiguous
        indice_num_per_loc int32 (kv,), contiguous
        num_act_out        0-dim integer tensor (the online evaluator casts all
                           input tensors to fp32: integer indices are exact up
                           to 2^24 and are recovered with int(); the shape
                           domain of this task guarantees indices <= 2^24)
        out                a single tensor, same dtype as features
                           (num_act_out, c_out). This operator is inherently
                           single-output, with no multi-output packing segments.
    Dimension domain: n, c_in, c_out, kv, num_act_out >= 1, non-powers-of-two
    allowed; indice_num_per_loc[k] in [0, n]. Input values are ordinary randn
    magnitude (no inf/NaN).

    Implementation constraints (violations are scored as failures):
        - The core computation (gathering by index pair, the matrix
          multiplication with W[:, :, k], the scatter accumulation by output row
          index) must be done with a HIP kernel inside the submitted file; ATen
          is only allowed for non-core operations such as dtype recovery / shape
          cleanup.
        - The __init__ and forward signatures of ModelNew must not be changed.
        - Accumulate in float32 throughout (fp16/bf16 inputs are first cast
          losslessly to fp32), then cast the result back to the dtype of
          features; padding table entries must not take part in the computation
          (including out-of-bounds reads).
        - forward returns the single tensor out (the evaluator compares
          shape/dtype/allclose on a single output).

    Forbidden (static audit strips docstrings and comments, then does literal
    substring matching):
    aiter / torch.sparse / torch.sparse_coo_tensor /
    torch.sparse_csr_tensor / torch.sparse_csc_tensor / torch.sparse_mm /
    torch.matmul / .matmul( / torch.mm / torch.bmm / torch.addmm /
    torch.baddbmm / torch.einsum / torch.tensordot /
    torch.nn.functional.linear / torch.nn.Linear / index_add_ /
    torch.index_add / index_put / torch.index_put / scatter_add_ /
    .scatter_( / torch.scatter / segment_reduce / miopen / hipdnn.
    Target hardware: Hygon DCU (gfx936/gfx938). Implementation language: HIP
    (compiled by hipcc; write inside the CUDA_CU_SRC / CUDA_CPP_SRC anchor
    regions; device code must include <hip/hip_runtime.h> -- <cuda_runtime.h>
    is not available on this machine).
    """

    def __init__(self):
        # This task has no constructor hyper-parameters (get_init_inputs returns
        # an empty list); the PyramidKernel loader requires Model to define
        # __init__ explicitly, so an empty constructor is declared here
        super().__init__()

    def forward(self, features, filters, indice_pairs, indice_num_per_loc,
                num_act_out):
        # The online evaluator casts all input tensors to fp32: integer index
        # tensors are exact up to 2^24, so they are restored uniformly here;
        # num_act_out accepts a 0-dim tensor or a python int
        num_act_out = int(num_act_out)
        c_out, c_in, kv = filters.shape
        dt = features.dtype

        xf = features.to(torch.float32)
        # filters layout mapping (the semantics of the sample's
        # as_strided(shape, [c_in*kv, 1, c_in])):
        # W[o, i, k] = F.reshape(c_out, kv, c_in)[o, k, i]
        wf = filters.to(torch.float32).reshape(c_out, kv, c_in).permute(0, 2, 1)
        pairs = indice_pairs.to(torch.long)
        counts = indice_num_per_loc.to(torch.long)

        out = torch.zeros(num_act_out, c_out, dtype=torch.float32,
                          device=features.device)
        for k in range(kv):
            nvalid = int(counts[k])
            if nvalid == 0:
                continue
            in_idx = pairs[0, k, :nvalid]
            out_idx = pairs[1, k, :nvalid]
            contrib = xf[in_idx] @ wf[:, :, k].t()   # (nvalid, c_out)
            out.index_add_(0, out_idx, contrib)
        return out.to(dt)


def get_init_inputs():
    return []   # this task has no constructor hyper-parameters; all semantic inputs go through forward


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs.
    # The consumption order is fixed:
    # features -> filters -> counts -> in_tab -> out_tab.
    n, c_in, c_out, kv, num_act_out = 1024, 16, 32, 9, 768
    features = torch.randn(n, c_in)
    filters = torch.randn(c_out, c_in, kv)
    counts = torch.randint(0, n + 1, (kv,), dtype=torch.int32)
    in_tab = torch.randint(0, n, (kv, n), dtype=torch.int32)
    out_tab = torch.randint(0, num_act_out, (kv, n), dtype=torch.int32)
    indice_pairs = torch.stack([in_tab, out_tab], dim=0)
    num_act_out_t = torch.tensor(num_act_out, dtype=torch.int64)
    return [features, filters, indice_pairs, counts, num_act_out_t]


def make_inputs(n: int, c_in: int, c_out: int, kv: int, num_act_out: int,
                dtype: str = "float32", counts_max=None, seed: int = 0):
    """Generate inputs from a public/hidden/perf case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device). This task has no constructor hyper-parameters, so the case
    fields are all formal parameters here.

    The consumption order is fixed: features -> filters -> counts -> in_tab ->
    out_tab (all drawn from the same torch.Generator(seed), so the same seed
    reproduces inputs bit-for-bit).
        counts[k] = indice_num_per_loc[k] ~ randint[0, counts_max]
        (counts_max defaults to n; when it is 0 all n_k = 0 and out stays
        all-zero, the boundary case)
        in_tab / out_tab cover every j in [0, n) (the valid prefix and the
        padding garbage share one distribution; padding entries are random
        values within range, so a correct implementation must mask by counts).
    Returns, in the same order as the forward parameters,
    (features, filters, indice_pairs, indice_num_per_loc, num_act_out).
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert min(n, c_in, c_out, kv, num_act_out) >= 1, "every dimension must be >= 1"
    cm = n if counts_max is None else int(counts_max)
    assert 0 <= cm <= n, "counts_max must lie in [0, n]"
    assert n <= 2 ** 24 and num_act_out <= 2 ** 24, "indices must be <= 2^24 (exactly representable in fp32)"

    features = torch.randn(n, c_in, generator=gen).to(dt)
    filters = torch.randn(c_out, c_in, kv, generator=gen).to(dt)
    counts = torch.randint(0, cm + 1, (kv,), generator=gen, dtype=torch.int32)
    in_tab = torch.randint(0, n, (kv, n), generator=gen, dtype=torch.int32)
    out_tab = torch.randint(0, num_act_out, (kv, n), generator=gen,
                            dtype=torch.int32)
    indice_pairs = torch.stack([in_tab, out_tab], dim=0)
    return features, filters, indice_pairs, counts, \
        torch.tensor(num_act_out, dtype=torch.int64)
