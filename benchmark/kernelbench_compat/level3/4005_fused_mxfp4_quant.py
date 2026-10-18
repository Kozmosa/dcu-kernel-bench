# 4005_fused_mxfp4_quant -- fused residual + RMSNorm + MXFP4 dynamic
# quantization (feeding the downstream mxfp4 GEMM) model_class (KernelBench
# compatible) task file.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=4005, backend=triton.

import torch
import torch.nn as nn


def _rmsnorm(x, weight, eps):
    # Official semantics: row-wise RMS normalization in float32, then multiply
    # by the weight
    row_norm = torch.sum(x * x, dim=-1)
    norm_factor = torch.rsqrt((row_norm / x.shape[1]) + eps).reshape(-1, 1)
    return x * norm_factor * weight.reshape(1, -1)


def _mxfp4_quant(x):
    # MXFP4 dynamic quantization (block size 32, even rounding). See the Model
    # docstring for the bit-exact semantics: the bit-level algorithm matches the
    # reference implementation; the input is a float32 (M, N) matrix.
    MXFP4_QUANT_BLOCK_SIZE = 32
    x_shape = x.shape
    if x.shape[-1] % MXFP4_QUANT_BLOCK_SIZE != 0:
        shape = list(x_shape)
        shape = shape[:-1] + [
            ((shape[-1] - 1 + MXFP4_QUANT_BLOCK_SIZE) // MXFP4_QUANT_BLOCK_SIZE)
            * MXFP4_QUANT_BLOCK_SIZE
        ]
        shape = tuple(shape)
        x_padded = torch.zeros(shape, dtype=x.dtype, device=x.device)
        x_padded[..., : x.shape[-1]] = x
    else:
        x_padded = x

    # Block scale: round amax up to a power of two, then take floor(log2) - 2
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

    # Quantized values
    qx = x_padded * quant_scale.unsqueeze(-1)

    # blockscale_e8m0
    bs_e8m0 = scale_e8m0_unbiased.to(torch.uint8) + 127

    # e2m1 rounding: round-half-up, saturating past the range (take min(x, 7) of
    # ((E<<2 | M>>21) + 1) >> 1)
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

    # Pack 2 4-bit code words per byte: even indices occupy the low 4 bits
    x_mxfp4 = e2m1_value[..., ::2] | (e2m1_value[..., 1::2] << 4)
    x_mxfp4 = torch.flatten(x_mxfp4, -2, -1)

    # Drop the extra bytes produced by the zero padding
    if x.shape[-1] % MXFP4_QUANT_BLOCK_SIZE != 0:
        x_mxfp4 = x_mxfp4[..., : x.shape[-1] // 2]

    mxfp4_shape = list(x_shape)
    mxfp4_shape = tuple(mxfp4_shape[:-1] + [mxfp4_shape[-1] // 2])
    x_mxfp4 = x_mxfp4.reshape(mxfp4_shape)
    return x_mxfp4, bs_e8m0


class Model(nn.Module):
    """Fused residual add + RMSNorm + MXFP4 dynamic quantization (mxfp4 GEMM
    preprocessing).

    Mathematical definition (all intermediates in float32; inputs are randn in
    the bfloat16/float16 value range):
      1. s = inp1 + res1                                     element-wise add
      2. norm[m, j] = s[m, j] * rsqrt( (1/N1) * sum_j s[m, j]^2 + eps ) * weight1[j]
         eps = 1e-6, normalized along the last dimension
      3. MXFP4 dynamic quantization of every row of norm in 32-element blocks
         (definition below), yielding packed e2m1 code words (2 per byte) and
         e8m0 block scale bytes.

    MXFP4 quantization (block size fixed at 32 along the last dimension; a
    trailing block shorter than 32 zero padded and taking part):
      a. amax = max |x| (inside the block, padding included)
      b. round amax up to a power of two: interpret the IEEE754 bit pattern of
         amax as int32, add 0x200000, bitwise-AND with 0xFF800000, then
         reinterpret as float32 (unchanged when amax is already a power of two,
         otherwise it goes up to the next power of two)
      c. e = clamp( floor(log2(rounded amax)) - 2, -127, 127 )
      d. scale byte = e + 127 (uint8 e8m0 encoding)
      e. q = x * 2^(-e), rounded to the e2m1 code table
         {0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6}: round-half-up (0.5 rounds
         up), saturating to +-6 beyond the range; the equivalent bit-level form
         is ((E<<2 | M>>21) + 1) >> 1 on the fp32 bit pattern, then min(x, 7),
         with bit31 as the sign bit
      f. packing: the 2i-th and the (2i+1)-th code words share one byte, the
         even index occupying the low 4 bits

    Input/output specification:
      inp1    (M, N1)  bfloat16/float16  main input
      weight1 (N1,)    same as inp1      RMSNorm weight
      res1    (M, N1)  same as inp1      residual
      out     (M, N1//2 + ceil(N1/32)) uint8, single-tensor output contract:
              out[:, :N1//2]        packed e2m1 code words (2 per byte, even
                                    indices in the low 4 bits)
              out[:, N1//2:]        e8m0 block scale bytes (ceil(N1/32) per row)
      Constraints: M >= 1; N1 is even and N1 >= 2.
      Boundary behaviour: an all-zero block has amax=0, handled as
      log2(0)=-inf, so after the clamp e=-127, the scale byte is 0 and the code
      words are 0.

    Implementation constraints (violations are scored as failures):
      - The core computation of RMSNorm and quantization (amax/scale
        computation, e2m1 rounding, packing) must be completed inside the
        submitted file; calling torch.nn.functional.rms_norm /
        torch.nn.RMSNorm is forbidden, as is using torch built-in float4/mxfp4
        quantization or conversion APIs (including torch.float4_e2m1fn_x2,
        torch.quantize_per_tensor and other ready-made quantization
        operators), and wrapping or modifying any third-party operator
        library's existing implementation.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Intermediate computation uses float32; the output is uint8, and both
        the code words and the scale bytes must be bit-exact against the
        reference implementation (exact match, no tolerance).
      - The evaluator casts every input to fp32 before calling forward
        (bfloat16/float16 values are exact in fp32); just compute with those
        values directly.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: single-path residual + RMSNorm + MXFP4 dynamic quantization
    (block size 32, even rounding), without the downstream GEMM main
    computation.
    """

    def __init__(self, dtype: str = "bfloat16", eps: float = 1e-6):
        super().__init__()
        self.dtype = getattr(torch, dtype)
        self.eps = float(eps)

    def forward(self, inp1, weight1, res1):
        # The input may be the original bfloat16/float16 or the fp32 cast done
        # by the evaluator (values are lossless); promote everything to
        # float32 for the computation
        s = inp1.to(torch.float32) + res1.to(torch.float32)
        norm = _rmsnorm(s, weight1.to(torch.float32), self.eps)
        codes, scales = _mxfp4_quant(norm)
        return torch.cat([codes, scales], dim=1)


def output_segments(init_kwargs, numel, inputs=None):
    """For the evaluator: the boundaries of the two output segments (see
    tolerance_segments in task.yaml).

    codes segment = M * (N1//2) bytes (2 e2m1 code words per byte),
    scales segment = M * ceil(N1/32) e8m0 bytes.
    The segment lengths depend on the input shape, hence inputs is needed
    (inp1 has shape (M, N1)).
    """
    if not inputs:
        return None
    x = inputs[0]
    n1 = int(x.shape[-1])
    m = int(x.numel() // n1) if n1 else 0
    codes = m * (n1 // 2)
    return [("codes", 0, codes), ("scales", codes, int(numel))]


def get_init_inputs():
    return ["bfloat16"]   # dtype; eps defaults to 1e-6


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs. Magnitudes match make_inputs.
    m, n1 = 64, 512
    inp1 = torch.randn(m, n1, dtype=torch.bfloat16)
    weight1 = torch.randn(n1, dtype=torch.bfloat16)
    res1 = torch.randn(m, n1, dtype=torch.bfloat16)
    return [inp1, weight1, res1]


def make_inputs(m: int, n1: int, dtype: str = "bfloat16", seed: int = 0):
    """Generate inputs from a public/hidden case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the
    device).

    Consumption order is fixed: inp1 -> weight1 -> res1, all from the same
    torch.Generator(seed); bit-reproducible for the same seed.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    inp1 = torch.randn(m, n1, generator=gen, dtype=dt)
    weight1 = torch.randn(n1, generator=gen, dtype=dt)
    res1 = torch.randn(m, n1, generator=gen, dtype=dt)
    return inp1, weight1, res1
