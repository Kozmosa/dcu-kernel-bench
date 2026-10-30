# 5007_spconv_indice_pairs -- model_class (KernelBench compatible) task file for
# sparse convolution indice pairs generation (spconv-style index-pair table
# construction).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=2, problem_id=5007, backend=cuda (HIP).

import torch
import torch.nn as nn


class Model(nn.Module):
    """Sparse convolution indice pairs generation: from the active input point
    table and the convolution geometry, build the index-pair table required by
    the sparse convolution forward pass (spconv-style kernel map, Native
    algorithm, not subm).

    The input indices is an (n, ndim+1) active point table, each row
    [batch_idx, x_0, ..., x_{ndim-1}] (column 0 is the batch index, the rest are
    the spatial coordinates); rows are distinct and in arbitrary order.

    The convolution geometry is given by 6 constructor hyper-parameters:
    batch_size (B), input_dims (the size of each grid dimension, D), ksize (K),
    stride (S), padding (P), dilation (L), all lists with one entry per dimension
    (1D/2D/3D). Derived quantities:
        kv = prod(K)
        output_dims[d] = (D[d] + 2*P[d] - L[d]*(K[d]-1) - 1) // S[d] + 1
    The kernel offset k in [0, kv) is decomposed into the per-dimension offsets
    o_d in mixed radix (C-order): dimension 0 is the slowest and dimension
    ndim-1 the fastest, i.e. k = o_0*K_1*...*K_{n-1} + ... + o_{ndim-1}.

    Generation rule (enumerate in this order: the outer loop k runs from 0 to
    kv-1, the inner loop over input rows i from 0 to n-1; each valid pair (k, i)
    is processed in turn):
        for each dimension d let val = x_d + P[d] - o_d*L[d];
        (k, i) is valid if and only if 0 <= batch_idx < B and, for every
        dimension,
            val % S[d] == 0 (divisible) and 0 <= val // S[d] < output_dims[d],
        in which case the output coordinate is q_d = val // S[d] (the batch is
        unchanged).
        (1) Output point deduplication and numbering: the first time a valid
            output coordinate (batch_idx, q_0..q_{ndim-1}) appears, it is given
            the next increasing row number r (starting at 0) in enumeration
            order; row r of out_indices is written with that coordinate. The
            number of valid outputs m is bounded by
            cap = min(kv * n, B * prod(output_dims)) (the smaller of the number
            of valid pairs and the output-grid capacity), and out_indices has the
            fixed shape (cap, ndim+1) (capacity allocation, with m <= cap valid
            rows); every row with row number >= m is filled with -1.
        (2) indice_pairs has shape (2, kv, n) and is first filled entirely with
            -1; for offset k, its c-th valid pair (c counting up from 0, a dense
            prefix of slots) writes:
                indice_pairs[0, k, c] = i      # input row index (gather source)
                indice_pairs[1, k, c] = r      # output row index (scatter target)
            Slots with c >= the number of valid pairs of that offset stay -1.
        (3) indice_num_per_loc has shape (kv,): indice_num_per_loc[k] is the
            number of valid pairs of offset k (i.e. the final value of the c
            counter in (2)).
    A 1D example (B=1, input_dims=[4], ksize=[2], stride=[2], padding=[0],
    dilation=[1], output_dims=[2], cap = min(kv*n, B*prod(output_dims)) =
    min(6, 2) = 2), indices = [[0,0],[0,1],[0,3]]:
        k=0 (o=0): i=0 hits output (0,0) -> r=0; i=1 (val=1, not divisible) and
        i=2 (val=3, not divisible) are invalid. k=1 (o=1): i=0 (val=-1) is
        invalid; i=1 (val=0) hits (0,0) -> r=0; i=2 (val=2) hits (0,1) -> r=1.
        out_indices = [[0,0],[0,1]] (m=2 exactly fills the capacity);
        indice_num_per_loc = [1, 2]; indice_pairs[0] = [[0,-1,-1],[1,2,-1]],
        indice_pairs[1] = [[0,-1,-1],[0,1,-1]].

    Single-tensor output protocol (forward returns a 1-D int32 tensor, three
    concatenated segments; cap = min(kv*n, B*prod(output_dims))):
        segment 1 [0, cap*(ndim+1))                       = out_indices flattened
        segment 2 [cap*(ndim+1), cap*(ndim+1) + 2*kv*n)   = indice_pairs flattened
        segment 3 the trailing kv elements                = indice_num_per_loc
    The total output length is always cap*(ndim+1) + 2*kv*n + kv, uniquely
    determined by the constructor hyper-parameters.

    Input/output specification:
        indices  int32 (n, ndim+1), contiguous; n >= 1, ndim in [1, 3];
                 rows distinct, 0 <= batch_idx < B, 0 <= x_d < D[d].
        hyper-parameter domain  D[d] >= 1, K[d] >= 1, S[d] >= 1, P[d] >= 0,
                 L[d] >= 1, output_dims[d] >= 1; n <= B * prod(D).
        out      a 1-D int32 tensor (see the packing protocol above). The index
                 semantics are exact matching (bit-for-bit), with no tolerance.
        The evaluator casts all input tensors to fp32 before the call (index
        values <= 2^24 are exactly representable in fp32), so the implementation
        must first restore int32 before using them in index arithmetic.

    Implementation constraints (violations are scored as failures):
        - The core computation (coordinate mapping and the divisibility / range
          tests, output point deduplication and numbering, filling of the three
          output tables) must be done with a HIP kernel inside the submitted
          file; ATen is only allowed for non-core operations such as dtype
          recovery, torch.full/zeros/arange, shape cleanup and torch.cat.
        - The __init__ and forward signatures of ModelNew must not be changed.
        - The output is indices: it must match bit-for-bit (row-number order,
          slot order and -1 padding all follow the rules above); any difference
          in order is scored as a failure.
        - forward returns a single tensor (the evaluator compares
          shape/dtype/allclose on a single output).

    Forbidden (static audit strips docstrings and comments, then does literal
    substring matching):
    aiter / miopen / hipdnn / torch.unique /
    torch.unique_consecutive / torch.sort / torch.argsort / torch.msort /
    .sort( / .argsort( / torch.mode / torch.histc / torch.bincount /
    torch.topk / scatter (including every form scatter_ / scatter_add_ /
    scatter_reduce / torch.scatter) / index_put (including index_put_ and
    torch.index_put) / segment_reduce.
    Target hardware: Hygon DCU (gfx936/gfx938). Implementation language: HIP
    (compiled by hipcc; write inside the CUDA_CU_SRC / CUDA_CPP_SRC anchor
    regions; device code must include <hip/hip_runtime.h> -- <cuda_runtime.h>
    is not available on this machine).
    """

    def __init__(self, batch_size: int = 2, input_dims=None, ksize=None,
                 stride=None, padding=None, dilation=None):
        super().__init__()
        # Constructor hyper-parameters: grid and convolution geometry (defaults
        # = the fixed family of get_inputs)
        self.batch_size = int(batch_size)
        self.input_dims = list(input_dims if input_dims is not None else [4, 32, 32])
        self.ksize = list(ksize if ksize is not None else [3, 3, 3])
        self.stride = list(stride if stride is not None else [1, 1, 1])
        self.padding = list(padding if padding is not None else [1, 1, 1])
        self.dilation = list(dilation if dilation is not None else [1, 1, 1])
        assert self.batch_size >= 1, "batch_size must be >= 1"
        ndim = len(self.input_dims)
        assert 1 <= ndim <= 3, "ndim must be in [1, 3]"
        assert all(len(p) == ndim for p in
                   (self.ksize, self.stride, self.padding, self.dilation)), \
            "ksize/stride/padding/dilation must have the same rank as input_dims"
        for d in range(ndim):
            assert self.input_dims[d] >= 1 and self.ksize[d] >= 1 \
                and self.stride[d] >= 1 and self.padding[d] >= 0 \
                and self.dilation[d] >= 1, "a per-dimension hyper-parameter is invalid"
            out_d = (self.input_dims[d] + 2 * self.padding[d]
                     - self.dilation[d] * (self.ksize[d] - 1) - 1) \
                // self.stride[d] + 1
            assert out_d >= 1, "every dimension of output_dims must be >= 1"

    def forward(self, indices):
        # The online evaluator casts all input tensors to fp32: index values
        # <= 2^24 are exact, so int32 is restored here
        rows = indices.to(torch.int32).tolist()
        n = len(rows)
        B = self.batch_size
        D, K = self.input_dims, self.ksize
        S, P, L = self.stride, self.padding, self.dilation
        ndim = len(D)

        kv = 1
        for k in K:
            kv *= k
        out_dims = [(D[d] + 2 * P[d] - L[d] * (K[d] - 1) - 1) // S[d] + 1
                    for d in range(ndim)]
        # out_indices capacity = upper bound on the number of valid outputs:
        # min(total valid pairs kv*n, output grid capacity)
        out_vol = B
        for od in out_dims:
            out_vol *= od
        cap = min(kv * n, out_vol)

        # C-order decomposition of the kernel offset (dimension 0 slowest,
        # dimension ndim-1 fastest)
        offs = []
        for k in range(kv):
            o = [0] * ndim
            t = k
            for d in range(ndim - 1, -1, -1):
                o[d] = t % K[d]
                t //= K[d]
            offs.append(o)

        out_flat = [-1] * (cap * (ndim + 1))
        pair_flat = [-1] * (2 * kv * n)
        num = [0] * kv
        table = {}   # output coordinate tuple -> output row number (first-appearance order)
        m = 0
        for k in range(kv):
            o = offs[k]
            base = k * n
            cnt = 0
            for i in range(n):
                row = rows[i]
                b = int(row[0])
                if b < 0 or b >= B:
                    continue
                q = [0] * ndim
                valid = True
                for d in range(ndim):
                    val = int(row[d + 1]) + P[d] - o[d] * L[d]
                    if val % S[d] != 0:
                        valid = False
                        break
                    qd = val // S[d]
                    if qd < 0 or qd >= out_dims[d]:
                        valid = False
                        break
                    q[d] = qd
                if not valid:
                    continue
                key = (b, *q)
                r = table.get(key, -1)
                if r < 0:
                    r = m
                    table[key] = r
                    start = r * (ndim + 1)
                    out_flat[start] = b
                    for d in range(ndim):
                        out_flat[start + 1 + d] = q[d]
                    m += 1
                pair_flat[base + cnt] = i
                pair_flat[kv * n + base + cnt] = r
                cnt += 1
            num[k] = cnt

        # Single-tensor packing: out_indices segment -> indice_pairs segment ->
        # indice_num_per_loc segment
        seg1 = torch.tensor(out_flat, dtype=torch.int32)
        seg2 = torch.tensor(pair_flat, dtype=torch.int32).reshape(2, kv, n)
        seg3 = torch.tensor(num, dtype=torch.int32)
        return torch.cat([seg1.reshape(-1), seg2.reshape(-1), seg3])


def get_init_inputs():
    # batch_size, input_dims, ksize, stride, padding, dilation (fixed family)
    return [2, [4, 32, 32], [3, 3, 3], [1, 1, 1], [1, 1, 1], [1, 1, 1]]


def _sample_indices(n, batch_size, input_dims, gen):
    # Uniformly sample n active points without replacement on the B x D grid
    # (rows distinct), keeping the sampling order (no sorting, to mimic the
    # arbitrary order of a real point cloud); when gen is None the global RNG is
    # consumed
    sv = 1
    for d in input_dims:
        sv *= d
    vol = batch_size * sv
    assert 1 <= n <= vol, "n must lie in [1, B*prod(D)] (active point rows are distinct)"
    if gen is not None:
        perm = torch.randperm(vol, generator=gen)[:n]
    else:
        perm = torch.randperm(vol)[:n]
    ndim = len(input_dims)
    rows = []
    for v in perm.tolist():
        b, t = v // sv, v % sv
        row = [b]
        for d in range(ndim):
            sd = 1
            for dd in range(d + 1, ndim):
                sd *= input_dims[dd]
            row.append(t // sd % input_dims[d])
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int32)


def get_inputs():
    # Fixed shape family (3D, B=2, D=[4,32,32], 3x3x3 convolution, with
    # stride/padding/dilation all 1); the random part consumes the global RNG --
    # the evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs. The only random consumption:
    # torch.randperm.
    return [_sample_indices(1024, 2, [4, 32, 32], None)]


def make_inputs(n: int, batch_size: int, input_dims, ksize, stride, padding,
                dilation, seed: int = 0):
    """Generate inputs from a public/hidden/perf case description.

    Runs on the evaluation side only (generated on CPU; the evaluator moves them
    to the device).

    The consumption order is fixed: the only random consumption is the first n
    positions of torch.randperm(B*prod(D), torch.Generator(seed)), decoded into
    (batch_idx, x_0..x_{ndim-1}) rows (C-order; the sampling order is the row
    order, no sorting). The same seed reproduces inputs bit-for-bit. The 6
    geometry hyper-parameters are also the init hyper-parameters used to
    construct the Model; only validity checks are done here (the same assertions
    as __init__).
    Returns, in the same order as the forward parameters, (indices,).
    """
    gen = torch.Generator().manual_seed(int(seed))
    # Validity checks (same domain as Model.__init__, so the evaluation side can
    # locate a bad case early)
    ndim = len(input_dims)
    assert n >= 1 and batch_size >= 1 and 1 <= ndim <= 3, "n/B/ndim are invalid"
    assert all(len(p) == ndim for p in (ksize, stride, padding, dilation)), \
        "ksize/stride/padding/dilation must have the same rank as input_dims"
    vol = 1
    for d in range(ndim):
        assert input_dims[d] >= 1 and ksize[d] >= 1 and stride[d] >= 1 \
            and padding[d] >= 0 and dilation[d] >= 1, "a per-dimension hyper-parameter is invalid"
        out_d = (input_dims[d] + 2 * padding[d] - dilation[d] * (ksize[d] - 1)
                 - 1) // stride[d] + 1
        assert out_d >= 1, "every dimension of output_dims must be >= 1"
        vol *= input_dims[d]
    assert n <= batch_size * vol, "n must be <= B*prod(D) (active point rows are distinct)"
    return (_sample_indices(n, batch_size, input_dims, gen),)
