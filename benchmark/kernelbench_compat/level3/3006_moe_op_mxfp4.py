# 3006_moe_op_mxfp4 -- MoE grouped GEMM (mxfp4 e2m1 packed activations x mxfp4
# e2m1 packed expert weights, e8m0 block scales, with the routing weight
# multiply), as a model_class (KernelBench-compatible) task file.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=3006, backend=triton.

import torch
import torch.nn as nn

# e2m1 nibble (4 bit) decode table: bit3 is the sign, the low 3 bits index this table
_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _dequant_mxfp4(codes, scales):
    """mxfp4 dequantization: (..., K//2) uint8 code words plus (..., K//32) uint8
    e8m0 block scales -> (..., K) float32. In every byte the low 4 bits are the
    even-index element and the high 4 bits the odd-index element;
    x[..., k] = e2m1(nibble) * 2^(scales[..., k // 32] - 127)."""
    lo = codes & 0xF
    hi = codes >> 4
    nib = torch.stack((lo, hi), dim=-1).flatten(-2)      # (..., K)
    x = _E2M1_TABLE.to(codes.device)[nib.long()]
    scale = torch.exp2(scales.to(torch.float32) - 127.0)
    return x * scale.repeat_interleave(32, dim=-1)


def _mxfp4_quant(x):
    """MXFP4 dynamic quantization (block size 32, round to even), bit-identical
    to the input generator of the official test. Used on the input generation
    side only; it is not part of the computation the Agent has to implement."""
    MXFP4_QUANT_BLOCK_SIZE = 32
    x_shape = x.shape
    if x.shape[-1] % MXFP4_QUANT_BLOCK_SIZE != 0:
        shape = list(x_shape)
        shape = shape[:-1] + [
            ((shape[-1] - 1 + MXFP4_QUANT_BLOCK_SIZE) // MXFP4_QUANT_BLOCK_SIZE)
            * MXFP4_QUANT_BLOCK_SIZE
        ]
        x_padded = torch.zeros(tuple(shape), dtype=x.dtype)
        x_padded[..., : x.shape[-1]] = x
    else:
        x_padded = x

    # Block scale: amax rounded up to a power of two, e = clamp(floor(log2(amax)) - 2, -127, 127)
    x_padded = x_padded.reshape(
        -1, x_padded.shape[-1] // MXFP4_QUANT_BLOCK_SIZE, MXFP4_QUANT_BLOCK_SIZE
    ).to(torch.float32)
    amax, _ = torch.max(torch.abs(x_padded), dim=-1)
    amax = amax.view(torch.int32)
    amax = (amax + 0x200000) & 0xFF800000
    amax = amax.view(torch.float32)
    scale_e8m0_unbiased = torch.log2(amax).floor() - 2
    scale_e8m0_unbiased = torch.clamp(scale_e8m0_unbiased, min=-127, max=127)
    quant_scale = torch.exp2(-scale_e8m0_unbiased)

    qx = x_padded * quant_scale.unsqueeze(-1)
    bs_e8m0 = scale_e8m0_unbiased.to(torch.uint8) + 127

    # e2m1 rounding: round-half-up, saturating on overflow (min(.,7) of (E<<2 | M>>21) + 1 >> 1)
    qx = qx.view(torch.int32)
    s = qx & 0x80000000
    e = (qx >> 23) & 0xFF
    m = qx & 0x7FFFFF

    E8_BIAS = 127
    E2_BIAS = 1
    adjusted_exponents = E8_BIAS - e - 1
    m = torch.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)
    e = torch.where(e > E8_BIAS - E2_BIAS, e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

    combined_val = (((e << 2) | (m >> 21)) + 1) >> 1
    e2m1_tmp = torch.where(combined_val < 0x7, combined_val, 0x7)
    e2m1_value = (((s >> 28) & 0xF) | e2m1_tmp).to(torch.uint8)

    # Pack 2 four-bit code words per byte: the even index goes in the low 4 bits
    x_mxfp4 = e2m1_value[..., ::2] | (e2m1_value[..., 1::2] << 4)
    x_mxfp4 = torch.flatten(x_mxfp4, -2, -1)

    if x.shape[-1] % MXFP4_QUANT_BLOCK_SIZE != 0:
        x_mxfp4 = x_mxfp4[..., : x.shape[-1] // 2]

    mxfp4_shape = tuple(list(x_shape)[:-1] + [x_shape[-1] // 2])
    x_mxfp4 = x_mxfp4.reshape(mxfp4_shape)
    bs_shape = tuple(list(x_shape)[:-1] + [x_shape[-1] // MXFP4_QUANT_BLOCK_SIZE])
    bs_e8m0 = bs_e8m0.reshape(bs_shape)
    return x_mxfp4, bs_e8m0


def _gen_case(M, N, K, E, top_k, dtype, generator):
    # Official test generator: randn activations/weights -> dynamic mxfp4
    # quantization; softmax+topk routing. The consumption order is fixed:
    # a_raw -> b_raw -> values.
    a_raw = torch.randn(M, K, dtype=dtype, generator=generator)
    b_raw = torch.randn(E, N, K, dtype=dtype, generator=generator)
    values = torch.randn(M, E, dtype=dtype, generator=generator)
    softmax_vals = torch.softmax(values, dim=1)
    topk_weights, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)
    a_q, a_scales = _mxfp4_quant(a_raw)
    b_q, b_scales = _mxfp4_quant(b_raw)
    return [a_q, a_scales, b_q, b_scales, topk_weights, topk_ids]


class Model(nn.Module):
    """MoE (Mixture of Experts) grouped GEMM: mxfp4 quantized activations x
    mxfp4 quantized expert weights.

    Routing is already done (topk_ids gives the experts selected for each
    token); this operator runs one GEMM per (token, selected expert) slot and
    multiplies by the routing weight (mul_routed_weight is always True):
      c[m, j, n] = topk_weights[m, j] * sum_k adeq[m, k] * bdeq[topk_ids[m, j], n, k]
    The sum runs along the K dimension, intermediate accumulation is float32 and
    the result is cast to bfloat16 for output.

    mxfp4 dequantization definition (the same for activation a and weight b):
      - every byte of the last dimension of the code word tensor holds 2 e2m1
        codes: the low 4 bits are the even-index element and the high 4 bits the
        odd-index element, i.e. the low/high nibble of code[..., i] correspond
        to elements 2i / 2i+1.
      - e2m1 nibble (4 bit) decoding: bit3 is the sign, the low 3 bits index the
        table {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 3.0, 6: 4.0, 7: 6.0};
        a set sign bit negates the value (-0.0 takes part in the multiply-add as
        0).
      - e8m0 block scale: every 32 elements of the last dimension share 1 uint8
        scale byte s = scales[..., k // 32], decoded as 2^(s - 127).
      - dequantization:
        x[..., k] = e2m1(nibble(..., k)) * 2^(scales[..., k // 32] - 127).

    Input/output specification:
      a_q          (M, K//2)     uint8   activation code words (K is the logical
                                         feature dimension)
      a_scales     (M, K//32)    uint8   activation e8m0 block scales
      b_q          (E, N, K//2)  uint8   expert weight code words (E experts)
      b_scales     (E, N, K//32) uint8   expert weight e8m0 block scales
      topk_weights (M, top_k)    float16/bfloat16/float32  routing weights (>0)
      topk_ids     (M, top_k)    int64   selected expert ids, values in [0, E)
      out          (M, top_k, N) bfloat16
      Constraints: M >= 1, E >= 1, N >= 1, K % 32 == 0, 1 <= top_k <= E.
      Boundary behaviour: one expert may be selected by several tokens, and one
      token may select the same expert in several of its slots; every (m, j)
      slot is computed independently, with no reduction or weighted merge across
      experts and no dispatch-merge semantics (that belongs to a downstream
      operator).

    Implementation constraints (violations are scored as failures):
      - Core computation (nibble unpacking, e2m1/e8m0 decoding, the GEMM grouped
        by topk_ids, the routing weight multiply) must be done inside the
        submitted file; calling torch.matmul / torch.bmm / torch.mm /
        torch.addmm / torch.einsum / F.linear / functional.linear /
        torch._scaled_mm is forbidden, as is using the built-in torch
        float4/mxfp4 representations and ready-made quantization conversion APIs
        (torch.float4_e2m1fn_x2, torch.quantize_per_tensor, etc.).
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation; the output is cast to
        bfloat16.
      - The evaluator casts all inputs to fp32 before calling forward (uint8
        code words and scale bytes (0..255), int64 expert ids (small integers)
        are all exactly representable in fp32), so they must be restored
        losslessly at the entry point (.to(torch.uint8) / .to(torch.long)).

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: MoE grouped GEMM of mxfp4 activations x mxfp4 weights plus the
    routing weight multiply; the token-expert alignment scheduling metadata
    (sorted_token_ids / expert_ids / num_tokens_post_padded) is an
    implementation detail and not part of the input contract.
    """

    def __init__(self):
        super().__init__()

    def forward(self, a_q, a_scales, b_q, b_scales, topk_weights, topk_ids):
        # The evaluator casts all inputs to fp32: code words, scale bytes and
        # expert ids are exactly representable in fp32, so restore them here
        # losslessly
        a_q = a_q.to(torch.uint8)
        a_scales = a_scales.to(torch.uint8)
        b_q = b_q.to(torch.uint8)
        b_scales = b_scales.to(torch.uint8)
        topk_ids = topk_ids.to(torch.long)
        topk_weights = topk_weights.to(torch.float32)

        M, Kh = a_q.shape
        K = Kh * 2
        E, N, _ = b_q.shape
        top_k = topk_ids.shape[1]

        a = _dequant_mxfp4(a_q, a_scales)     # (M, K) float32
        b = _dequant_mxfp4(b_q, b_scales)     # (E, N, K) float32

        # Do the GEMM grouped by expert (mathematically equivalent to b[topk_ids]
        # followed by einsum("mek,menk->men"); writing back per group avoids
        # materializing a large intermediate)
        flat_ids = topk_ids.reshape(-1)                     # (M*top_k,)
        a_rep = a.unsqueeze(1).expand(M, top_k, K).reshape(-1, K)
        c = torch.empty((M * top_k, N), dtype=torch.float32, device=a_q.device)
        for e in range(E):
            mask = flat_ids == e
            if mask.any():
                c[mask] = a_rep[mask] @ b[e].t()
        c *= topk_weights.reshape(-1, 1)
        return c.reshape(M, top_k, N).to(torch.bfloat16)


def get_init_inputs():
    return []   # no hyperparameters: all dimensions come from the input tensor shapes


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials see diverse inputs. The magnitudes match make_inputs.
    return _gen_case(64, 128, 128, 8, 2, torch.bfloat16, None)


def make_inputs(M: int, N: int, K: int, E: int, top_k: int,
                dtype: str = "bfloat16", seed: int = 0):
    """Generate inputs from a public/hidden case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the
    device). The consumption order is fixed: a_raw -> b_raw -> values, all from
    the same torch.Generator(seed); bit-reproducible under the same seed."""
    gen = torch.Generator().manual_seed(seed)
    return _gen_case(M, N, K, E, top_k, getattr(torch, dtype), gen)
