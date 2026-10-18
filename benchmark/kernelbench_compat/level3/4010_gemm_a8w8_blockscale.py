# 4010_gemm_a8w8_blockscale -- 8bit GEMM with 2D block-scale dequantization
# (fp8 activations x fp8 weights, block scales tiled along both K and N)
# model_class (KernelBench compatible) task file.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=4010, backend=triton.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """8bit GEMM with 2D block-scale dequantization (fp8 activations x fp8
    weights).

    Mathematical definition: given the fp8(e4m3) code-word matrices xq (M, K),
    wq (N, K) and the float32 block scales x_scale (M, ceil(K/gk)),
    w_scale (ceil(N/gn), ceil(K/gk)), the output is
      dq_x[m, k] = xq[m, k] * x_scale[m, k // gk]
      dq_w[n, k] = wq[n, k] * w_scale[n // gn, k // gk]
      acc[m, n]  = sum_k dq_x[m, k] * dq_w[n, k]     (float32 accumulation)
      out[m, n]  = cast(acc[m, n], out_dtype)
    That is, every element is first dequantized with the block scale it belongs
    to ((m, k//gk) / (n//gn, k//gk)), then a float32 GEMM is performed
    (equivalent to the transposed GEMM xq @ wq^T, with wq row-major (N, K)).
    Scale ownership is decided element by element and is independent of the
    K-direction tile partition; when K is not divisible by gk / N is not
    divisible by gn, elements of the tail block still use the scale of the
    block they belong to (the scale count is the ceil division result); M/N/K
    >= 1, non-powers of two allowed. The dequantization multiply and the GEMM
    accumulation run in float32 throughout, and the output is finally cast to
    out_dtype.

    Input/output specification:
      x       float8_e4m3fn (M, K)             activation code words (row-major, TN layout)
      w       float8_e4m3fn (N, K)             weight code words (row-major, transposed GEMM)
      x_scale float32 (M, ceil(K/gk))          activation scale: per-M-row, K in gk blocks
      w_scale float32 (ceil(N/gn), ceil(K/gk)) weight 2D block scale (rows = N blocks, cols = K blocks)
      out     out_dtype (M, N)                 bfloat16 / float16, set by __init__
    The online evaluator casts every input to fp32 before calling forward; the
    finite values of fp8(e4m3) are exact in fp32, so the
    .to(torch.float8_e4m3fn) at the entry point recovers the code words
    losslessly.

    Final-audit compatible form: the offline final-audit harness constructs the
    model with a single argument Model(case["head_size"])
    (case["head_size"] = [group_k, group_n, out_dtype]), so the first __init__
    argument also accepts a [group_k, group_n, out_dtype] sequence and unpacks
    it in order; ModelNew must keep the same dual-form compatibility
    (constructible with either three arguments or a single sequence).

    Implementation constraints (violations are scored as failures):
      - The core computation (8bit GEMM accumulation + block-scale
        dequantization) must be completed inside the submitted file, and
        calling any ATen matmul shortcut is forbidden: torch.matmul /
        torch.bmm / torch.mm / torch.addmm / torch.einsum / torch._int_mm /
        torch._scaled_mm / F.linear / functional.linear, together with their
        torch.ops.aten.* fully qualified names and tensor-method forms
        (torch.ops.aten.mm, .mm( / .bmm( / .matmul( / .addmm( / .einsum( /
        ._int_mm( / ._scaled_mm( etc.), the matmul operator @ and the tensor
        transpose method .t() (x @ w.t() is the typical disguised form; the
        transpose should be handled with strides inside the kernel). The
        final-audit static audit scans the submitted file for those literal
        strings (docstrings excluded).
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation; the output is cast to
        out_dtype.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: a single matrix (no batch dimension), no bias, no splitK; the
    code-word and scale value ranges follow the task generator (code words are
    fp8 values in [0, 0.1), scales are float32 in [0, 1)).
    """

    def __init__(self, group_k=128, group_n=128, out_dtype="bfloat16"):
        super().__init__()
        # The offline final-audit harness constructs with a single argument
        # (Model(case["head_size"])); when the first argument is a list/tuple
        # it is unpacked as [group_k, group_n, out_dtype]
        if isinstance(group_k, (list, tuple)):
            group_k, group_n, out_dtype = group_k
        self.group_k = int(group_k)
        self.group_n = int(group_n)
        self.out_dtype = getattr(torch, out_dtype)

    def forward(self, x, w, x_scale, w_scale):
        # The evaluator casts every input to fp32; the finite values of
        # fp8(e4m3) are exact in fp32, so the round-trip at the entry point
        # recovers the code words losslessly (a no-op when the offline path
        # passes fp8 tensors directly)
        x = x.to(torch.float8_e4m3fn)
        w = w.to(torch.float8_e4m3fn)

        gk, gn = self.group_k, self.group_n
        M, K = x.shape
        N = w.shape[0]
        assert K == w.shape[1], "the K dimension of x and w must match"
        scale_k = (K + gk - 1) // gk
        scale_n = (N + gn - 1) // gn
        assert x_scale.shape == (M, scale_k), \
            f"x_scale shape should be ({M}, {scale_k}), got {tuple(x_scale.shape)}"
        assert w_scale.shape == (scale_n, scale_k), \
            f"w_scale shape should be ({scale_n}, {scale_k}), got {tuple(w_scale.shape)}"

        # The scale block each element belongs to: k -> k // gk, n -> n // gn
        # (independent of the tile partition; elements of a partial tail block
        # use the scale of the block they belong to as usual)
        k_idx = torch.arange(K, device=x.device) // gk      # (K,)
        n_idx = torch.arange(N, device=x.device) // gn      # (N,)
        xs = x_scale[:, k_idx]                              # (M, K) float32
        ws = w_scale[n_idx[:, None], k_idx[None, :]]        # (N, K) float32

        # Dequantization and GEMM accumulation run in float32 throughout
        dq_x = x.to(torch.float32) * xs
        dq_w = w.to(torch.float32) * ws
        out = F.linear(dq_x, dq_w)                          # (M, N) float32
        return out.to(self.out_dtype)


def get_init_inputs():
    return [128, 128, "bfloat16"]   # group_k; group_n; out_dtype


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs. The code-word and scale value ranges
    # match make_inputs.
    m, n, k = 128, 512, 1024
    group_k, group_n = 128, 128
    scale_k = (k + group_k - 1) // group_k
    scale_n = (n + group_n - 1) // group_n

    x = (torch.rand((m, k), dtype=torch.float16) / 10).to(torch.float8_e4m3fn)
    w = (torch.rand((n, k), dtype=torch.float16) / 10).to(torch.float8_e4m3fn)
    x_scale = torch.rand((m, scale_k), dtype=torch.float32)
    w_scale = torch.rand((scale_n, scale_k), dtype=torch.float32)
    return [x, w, x_scale, w_scale]


def make_inputs(m: int, n: int, k: int, group_k: int = 128, group_n: int = 128,
                out_dtype: str = "bfloat16", seed: int = 0,
                seq_lens=None, head_size=None, dtype=None):
    """Generate inputs from a public/hidden case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the
    device).

    out_dtype selects the output dtype of this case (a Model constructor
    argument) and does not change the input tensors themselves.
    Consumption order is fixed: x -> w -> x_scale -> w_scale, all from the same
    torch.Generator(seed); bit-reproducible for the same seed.

    Offline final-audit harness compatible parameters (audit_model_class calls
    make_inputs(seq_lens=None, **case_fields) uniformly, forwarding the case's
    extra fields): seq_lens does not exist for this task and must be None;
    head_size/dtype are forwarded from the case and must agree with the
    explicit parameters (consistency assertions).
    """
    assert seq_lens is None, "this task has no seq_lens dimension"
    if head_size is not None:
        assert list(head_size) == [group_k, group_n, out_dtype], \
            f"head_size {head_size} is inconsistent with (group_k, group_n, out_dtype)"
    if dtype is not None:
        assert dtype == out_dtype, f"dtype {dtype} is inconsistent with out_dtype {out_dtype}"

    gen = torch.Generator().manual_seed(seed)

    scale_k = (k + group_k - 1) // group_k
    scale_n = (n + group_n - 1) // group_n

    x = (torch.rand((m, k), generator=gen, dtype=torch.float16) / 10).to(torch.float8_e4m3fn)
    w = (torch.rand((n, k), generator=gen, dtype=torch.float16) / 10).to(torch.float8_e4m3fn)
    x_scale = torch.rand((m, scale_k), generator=gen, dtype=torch.float32)
    w_scale = torch.rand((scale_n, scale_k), generator=gen, dtype=torch.float32)
    return x, w, x_scale, w_scale
