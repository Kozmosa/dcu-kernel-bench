# 4016_gemm_w8a8 -- int8 block-quantized GEMM (activations with a
# per-token-group x128 scale, weights with a 128x128 block scale) model_class
# (KernelBench compatible) task file.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=4016, backend=triton.

import torch
import torch.nn as nn


def _per_token_group_quant_int8(x: torch.Tensor, group_k: int = 128):
    """int8 quantization grouped along the row (group_k elements per group):
    scaled by the in-group absmax.

    Returns (int8 code words shaped like x, float32 scale
    (m, ceil(k/group_k))). A tail group shorter than group_k takes its absmax
    over the valid elements only; the zero-padded elements are cropped after
    quantization.
    """
    m, k = x.shape
    g = (k + group_k - 1) // group_k
    xp = torch.zeros(m, g * group_k, dtype=x.dtype)
    xp[:, :k] = x
    xg = xp.view(m, g, group_k).to(torch.float32)
    absmax = xg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = (absmax / 127.0).squeeze(-1)                       # (m, g) float32
    q = torch.clamp(torch.round(xg / scale.unsqueeze(-1)), -128, 127)
    q = q.to(torch.int8).view(m, g * group_k)[:, :k].contiguous()
    return q, scale


def _per_block_quant_int8(w: torch.Tensor, block_n: int = 128, block_k: int = 128):
    """int8 quantization in (block_n x block_k) blocks: scaled by the in-block
    absmax.

    Returns (int8 code words shaped like w, float32 scale
    (ceil(n/block_n), ceil(k/block_k))). An incomplete block on the
    right/bottom edge takes its absmax over the valid elements only (zero
    padding does not affect the scaling).
    """
    n, k = w.shape
    gn = (n + block_n - 1) // block_n
    gk = (k + block_k - 1) // block_k
    wp = torch.zeros(gn * block_n, gk * block_k, dtype=w.dtype)
    wp[:n, :k] = w
    blocks = wp.view(gn, block_n, gk, block_k).to(torch.float32)   # (gn, bn, gk, bk)
    scale4d = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-10) / 127.0
    q = torch.clamp(torch.round(blocks / scale4d), -128, 127)
    scale = scale4d.squeeze(1).squeeze(-1)                      # (gn, gk) float32
    q = q.to(torch.int8).view(gn * block_n, gk * block_k)[:n, :k].contiguous()
    return q, scale


class Model(nn.Module):
    """int8 block-quantized GEMM (quantized activations x quantized weights,
    dequantized with block scales).

    Mathematical definition: given the int8 activation code words A (M,K), the
    int8 weight code words B (N,K), the activation per-token-group scale
    As (M, ceil(K/128)) and the weight 128x128 block scale
    Bs (ceil(N/128), ceil(K/128)) (all float32), dequantize first and then do
    the matrix multiply:
      Af[m,k] = A[m,k] * As[m, floor(k/128)]
      Bf[n,k] = B[n,k] * Bs[floor(n/128), floor(k/128)]
      out[m,n] = sum_k Af[m,k] * Bf[n,k]    (float32 accumulation, i.e. Af @ Bf^T)
    The equivalent compute order for block quantization: take an int8 dot
    product per 128-element group along K first (exactly accumulable in int32),
    then multiply by that group's As[m,g] * Bs[n_blk,g] and sum in float32 --
    equal to the expression above within float32 precision. The output is cast
    to the out_dtype passed to __init__.
    Boundary behaviour: when K or N is not divisible by 128 the floor index
    simply extends naturally -- a tail group/block covers only the valid
    elements (out-of-range elements do not take part in the accumulation), and
    the group counts of As/Bs are given by ceil(.../128).

    Input/output specification:
      aq      int8    (M, K)                 activation code words, row-major
      bq      int8    (N, K)                 weight code words, row-major (output channels on rows)
      a_scale float32 (M, ceil(K/128))       per-token scale, one group per 128 elements
      b_scale float32 (ceil(N/128), ceil(K/128))   weight 128x128 block scale
      out     out_dtype (M, N)              bfloat16 / float16
      M/N/K >= 1, all allow non-powers of two and a tail not aligned to 128.

    Implementation constraints (violations are scored as failures):
      - The core computation (int8 GEMM accumulation and the scale multiply)
        must be completed inside the submitted file, and calling library-level
        matrix-multiply shortcuts such as torch.matmul / torch.bmm / torch.mm /
        torch.addmm / torch.einsum / torch._int_mm / F.linear /
        functional.linear, or the other entries on the task set's static-audit
        forbidden list, is forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation (int8 products can be
        accumulated exactly in int32 first, then converted to float32); the
        output is cast to out_dtype.
      - The evaluator casts every input to fp32 before calling forward; the
        int8 code-word range [-128,127] and the fp32 scales are both exactly
        representable, so .to(torch.int8) / .to(torch.float32) at the entry
        point recover them losslessly.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: block shape fixed at [128,128], A is 2D (M,K), no bias, no
    splitK; the output is bfloat16 / float16.
    """

    def __init__(self, out_dtype: str = "bfloat16"):
        super().__init__()
        self.out_dtype = getattr(torch, out_dtype)
        self.block_n = 128   # weight block-quantization block shape fixed [128, 128]
        self.block_k = 128

    def forward(self, aq, bq, a_scale, b_scale):
        # The evaluator casts every input to fp32; int8 code words and fp32
        # scales are recovered losslessly
        aq = aq.to(torch.int8)
        bq = bq.to(torch.int8)
        a_scale = a_scale.to(torch.float32)
        b_scale = b_scale.to(torch.float32)

        M, K = aq.shape
        N = bq.shape[0]
        assert K == bq.shape[1], "the K dimension of aq and bq must match"

        # Dequantize A: A[m,k] * As[m, k//block_k] (one group of 128 per token)
        a = aq.to(torch.float32) * a_scale.repeat_interleave(self.block_k, dim=1)[:, :K]
        # Dequantize B: B[n,k] * Bs[n//block_n, k//block_k] (128x128 block scale)
        b_exp = (
            b_scale.repeat_interleave(self.block_n, dim=0)[:N]
            .repeat_interleave(self.block_k, dim=1)[:, :K]
        )
        b = bq.to(torch.float32) * b_exp

        out = torch.matmul(a, b.t())          # (M, N) float32 accumulation
        return out.to(self.out_dtype)


def get_init_inputs():
    return ["bfloat16"]   # out_dtype


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs. The generation scheme matches
    # make_inputs: activation fp16 randn -> per-token-group int8; weight randn
    # -> 128x128 block int8.
    m, n, k = 96, 1024, 768

    x = torch.randn(m, k).to(torch.float16)
    aq, a_scale = _per_token_group_quant_int8(x)
    w = torch.randn(n, k)
    bq, b_scale = _per_block_quant_int8(w)
    return [aq, bq, a_scale, b_scale]


def make_inputs(m: int, n: int, k: int, dtype: str = "bfloat16", seed: int = 0,
                seq_lens=None, head_size=None):
    """Generate inputs from a public/hidden case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the
    device).

    Consumption order is fixed: activation randn(m,k) -> weight randn(n,k), from
    the same torch.Generator(seed); bit-reproducible for the same seed. dtype is
    the output dtype (the case-side key name of the Model constructor argument
    out_dtype) and does not take part in input generation.
    The offline final-audit pipeline (audit_model_class.py) uniformly generates
    inputs for all model_class tasks with
    make_inputs(seq_lens=..., **case_fields), constructs Model from
    case["head_size"] positionally, and takes the tolerance key from
    case["dtype"]; for this task head_size is the compatibility alias of dtype
    (the pipeline passes it as the first positional argument to Model, landing
    on this model's out_dtype), and make_inputs only asserts that it equals
    dtype -- it does not take part in input generation.
    seq_lens is the pipeline's uniform-call compatibility parameter (a keyword
    reserved for the paged-attention task family); this operator has no such
    semantics, so any value passed in is ignored.
    """
    if head_size is not None and head_size != dtype:
        raise ValueError(f"head_size({head_size!r}) should equal dtype({dtype!r}) (compatibility alias)")
    gen = torch.Generator().manual_seed(seed)

    x = torch.randn(m, k, generator=gen).to(torch.float16)
    aq, a_scale = _per_token_group_quant_int8(x)
    w = torch.randn(n, k, generator=gen)
    bq, b_scale = _per_block_quant_int8(w)
    return aq, bq, a_scale, b_scale
