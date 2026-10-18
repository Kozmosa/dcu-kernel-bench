# 5006_spconv_bwd -- model_class (KernelBench compatible) task file for the
# sparse convolution backward pass (dgrad / wgrad, gathered form spconv).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=5006, backend=cuda (HIP).

import torch
import torch.nn as nn


class Model(nn.Module):
    """Sparse convolution backward (gathered form spconv): dgrad (gradient with
    respect to the input features) and wgrad (gradient with respect to the
    weight buffer); it is the exact adjoint of the sparse convolution forward
    (gather-GEMM-scatter).

    Forward semantics (the sole authority for this task's convention; out is
    first initialized to all zeros, then accumulated by the formula below):
        for each kernel offset k in [0, kv):
            n_k = indice_num_per_loc[k]
            for each j in [0, n_k):
                in_idx  = indice_pairs[0, k, j]      # input row index in [0, n)
                out_idx = indice_pairs[1, k, j]      # output row index in [0, num_act_out)
                out[out_idx, :] += features[in_idx, :] @ W[:, :, k].T
    i.e. for each index pair, gather the input row and matrix-multiply it with
    the weight matrix of offset k ((n_k, c_in) @ (c_in, c_out)), then
    scatter-accumulate by the output row index into out.
    The mathematical weight W and the contiguous filters tensor F satisfy the
    layout mapping:
        W[o, i, k] = F.reshape(c_out, kv, c_in)[o, k, i]
    (equivalent to W = F.reshape(c_out, kv, c_in).permute(0, 2, 1), still shaped
    (c_out, c_in, kv)).

    Given the upstream gradient out_bp = dL/dy (shape (num_act_out, c_out)), the
    two directions are defined as the adjoint of that forward with respect to
    features / filters:
        dgrad: dx[in_idx, :] += out_bp[out_idx, :] @ W[:, :, k], output
               (n, c_in), i.e. dL/dfeatures;
        wgrad: dW_math[:, :, k] += out_bp[out_idx, :, None] *
               features[in_idx, None, :] (accumulated per index pair), output as
               the weight gradient dF (c_out, c_in, kv) interpreted with the same
               layout mapping as filters:
               dF.reshape(c_out, kv, c_in)[o, k, i] = dW_math[o, i, k],
               stored contiguously.
    The __init__ parameter variant selects which direction is returned:
        "dgrad": returns dx; here the values of features do not take part in the
                 output (shape convention only);
        "wgrad": returns dF; here the values of filters do not take part in the
                 output (shape convention only).
    forward(features, filters, out_bp, indice_pairs, indice_num_per_loc) has a
    fixed five-input signature; both directions are inherently single-tensor
    outputs (no concatenation packing).

    Boundary behaviour: kernel offsets with n_k = 0 are skipped entirely; when
    all n_k = 0 the output is all-zero; kv = 1 degenerates to a single offset;
    num_act_out and n are independent of each other (either may be the larger
    one). Index-pair table entries with j >= indice_num_per_loc[k] are padding
    garbage (they may hold any value, including duplicated valid row indices)
    and must be masked out according to indice_num_per_loc -- they must never be
    read into the computation. The same input/output row may be hit by any number
    of index pairs (duplicates are allowed within the same k), and all of them
    accumulate with +=.

    Input/output specification:
        features           float32 / float16 / bfloat16 (n, c_in), contiguous
        filters            same dtype as features (c_out, c_in, kv), contiguous
        out_bp             same dtype as features (num_act_out, c_out),
                           contiguous
        indice_pairs       int32 (2, kv, n), contiguous
        indice_num_per_loc int32 (kv,), contiguous
        out                a single tensor, same dtype as features: (n, c_in) for
                           dgrad, (c_out, c_in, kv) for wgrad
        Dimension domain: n, c_in, c_out, kv, num_act_out >= 1, non-powers-of-two
        allowed; indice_num_per_loc[k] in [0, n]; row indices <= 2^24. Input
        values are ordinary randn magnitude (no inf/NaN). The evaluator casts all
        input tensors to fp32 before the call (integer indices are exact up to
        2^24 and are recovered with int()/long()).

    Implementation constraints (violations are scored as failures):
        - The core computation must be done with a HIP kernel inside the
          submitted file: the scatter accumulation of dgrad (the out_bp rows
          transformed by W[:, :, k] and then scatter-accumulated by in_idx) or
          the per-index-pair outer-product correlation accumulation of wgrad;
          falling back to combinations of torch gather / matmul / scatter or to
          autograd is forbidden.
        - The __init__ and forward signatures of ModelNew must not be changed;
          forward returns the single tensor out (the evaluator compares
          shape/dtype/allclose on a single output).
        - Numeric contract: accumulate in float32 throughout (fp16/bf16 inputs
          are first cast losslessly to fp32), then cast the result back to the
          dtype of features; padding table entries must not take part in the
          computation (including out-of-bounds reads).

    Forbidden (static audit strips docstrings and comments, then does literal
    substring matching):
    aiter / torch.sparse / torch.sparse_coo_tensor /
    torch.sparse_csr_tensor / torch.sparse_csc_tensor / torch.sparse_mm /
    torch.matmul / .matmul( / torch.mm / torch.bmm / torch.addmm /
    torch.baddbmm / torch.einsum / torch.tensordot /
    torch.nn.functional.linear / torch.nn.Linear /
    index_add (including every spelling .index_add( / index_add_ /
    torch.index_add) / index_put / torch.index_put / scatter_add (including
    scatter_add_) / .scatter_( / torch.scatter / scatter_reduce /
    segment_reduce / torch.gather / index_select / torch.take / torch.sum /
    torch.ops.aten / torch.autograd / .backward( / as_strided;
    the closed-source libraries MIOpen / hipDNN / cuDNN / rocBLAS / hipBLASLt
    are forbidden as well (global blacklist).
    Target hardware: Hygon DCU (gfx936/gfx938). Implementation language: HIP
    (compiled by hipcc; write inside the CUDA_CU_SRC / CUDA_CPP_SRC anchor
    regions; device code must include <hip/hip_runtime.h> -- <cuda_runtime.h>
    is not available on this machine).
    """

    def __init__(self, variant: str = "dgrad"):
        # Note: the def signatures of __init__/forward must stay single-line
        # (the framework loader extracts the signature from the first line to
        # generate the ModelNew scaffold).
        super().__init__()
        assert variant in ("dgrad", "wgrad"), "variant must be either dgrad or wgrad"
        self.variant = variant

    def forward(self, features, filters, out_bp, indice_pairs, indice_num_per_loc):
        # Build the forward graph of the same family (gather-GEMM-scatter) and
        # backpropagate with autograd: this keeps dgrad / wgrad strictly
        # consistent with the forward semantics (including the filters layout
        # mapping) (the reference is not bound by the forbidden list;
        # mathematically dx does not depend on the values of features and dF
        # does not depend on the values of filters, and autograd naturally
        # yields exactly that definition).
        # The online evaluator casts all input tensors to fp32: integer index
        # tensors are exact up to 2^24 and are restored uniformly here;
        # floating-point inputs are cast losslessly to fp32 for the float32
        # accumulation throughout, and the result is finally cast back to the
        # dtype of features.
        dt = features.dtype
        c_out, c_in, kv = filters.shape
        f32 = torch.float32

        xf = features.detach().to(f32).requires_grad_(True)
        wf0 = filters.detach().to(f32).requires_grad_(True)
        yf = out_bp.detach().to(f32)
        pairs = indice_pairs.to(torch.long)
        counts = indice_num_per_loc.to(torch.long)

        # Forward graph construction: out starts all-zero, then accumulates per
        # index pair via gather-GEMM-scatter
        # (layout mapping: W[o, i, k] = F.reshape(c_out, kv, c_in)[o, k, i]).
        # When all n_k = 0 the forward has no dependency on either leaf, so the
        # 0.0-multiply hook keeps the graph connected while the values stay
        # exactly zero (the input domain guarantees no inf/NaN, 0.0*finite = 0.0
        # and x+0.0 = x are both exact), and the gradients are naturally
        # all-zero.
        wf = wf0.reshape(c_out, kv, c_in).permute(0, 2, 1)
        hook = xf.sum() * 0.0 + wf0.sum() * 0.0
        out = torch.zeros(yf.shape[0], c_out, dtype=f32,
                          device=features.device) + hook
        for k in range(kv):
            nvalid = int(counts[k])
            if nvalid == 0:
                continue
            in_idx = pairs[0, k, :nvalid]
            out_idx = pairs[1, k, :nvalid]
            contrib = xf[in_idx] @ wf[:, :, k].t()   # (nvalid, c_out)
            out = out.index_add(0, out_idx, contrib)

        if self.variant == "dgrad":
            (dx,) = torch.autograd.grad(out, (xf,), yf)
            return dx.to(dt)
        (dwf,) = torch.autograd.grad(out, (wf0,), yf)
        return dwf.to(dt)


def get_init_inputs():
    return ["dgrad"]   # variant


def get_inputs():
    # Fixed shape family (consistent with the forward task of the same family);
    # the random part consumes the global RNG -- the evaluator calls this
    # function after set_seed, so multiple correctness trials therefore see
    # diverse inputs.
    # The consumption order is fixed:
    # features -> filters -> out_bp -> counts -> in_tab -> out_tab.
    n, c_in, c_out, kv, num_act_out = 1024, 16, 32, 9, 768
    features = torch.randn(n, c_in)
    filters = torch.randn(c_out, c_in, kv)
    out_bp = torch.randn(num_act_out, c_out)
    counts = torch.randint(0, n + 1, (kv,), dtype=torch.int32)
    in_tab = torch.randint(0, n, (kv, n), dtype=torch.int32)
    out_tab = torch.randint(0, num_act_out, (kv, n), dtype=torch.int32)
    indice_pairs = torch.stack([in_tab, out_tab], dim=0)
    return [features, filters, out_bp, indice_pairs, counts]


def make_inputs(n: int, c_in: int, c_out: int, kv: int, num_act_out: int,
                variant: str = "dgrad", dtype: str = "float32",
                counts_max=None, seed: int = 0):
    """Generate inputs from a public/hidden/perf case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device). variant is also the init hyper-parameter used to construct
    the Model (the evaluation side takes it from the case per task.yaml
    io.init_inputs); only validity checks are done here.

    The consumption order is fixed: features -> filters -> out_bp -> counts ->
    in_tab -> out_tab (all drawn from the same torch.Generator(seed), so the same
    seed reproduces inputs bit-for-bit).
        counts[k] = indice_num_per_loc[k] ~ randint[0, counts_max]
        (counts_max defaults to n; when it is 0 all n_k = 0 and the output is
        the all-zero boundary case)
        in_tab / out_tab cover every j in [0, n) (the valid prefix and the
        padding garbage share one distribution; padding entries are random
        values within range, so a correct implementation must mask by counts).
    Returns, in the same order as the forward parameters,
    (features, filters, out_bp, indice_pairs, indice_num_per_loc).
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert variant in ("dgrad", "wgrad"), "variant must be either dgrad or wgrad"
    assert min(n, c_in, c_out, kv, num_act_out) >= 1, "every dimension must be >= 1"
    cm = n if counts_max is None else int(counts_max)
    assert 0 <= cm <= n, "counts_max must lie in [0, n]"
    assert n <= 2 ** 24 and num_act_out <= 2 ** 24, "row indices must be <= 2^24 (exactly representable in fp32)"

    features = torch.randn(n, c_in, generator=gen).to(dt)
    filters = torch.randn(c_out, c_in, kv, generator=gen).to(dt)
    out_bp = torch.randn(num_act_out, c_out, generator=gen).to(dt)
    counts = torch.randint(0, cm + 1, (kv,), generator=gen, dtype=torch.int32)
    in_tab = torch.randint(0, n, (kv, n), generator=gen, dtype=torch.int32)
    out_tab = torch.randint(0, num_act_out, (kv, n), generator=gen,
                            dtype=torch.int32)
    indice_pairs = torch.stack([in_tab, out_tab], dim=0)
    return features, filters, out_bp, indice_pairs, counts
