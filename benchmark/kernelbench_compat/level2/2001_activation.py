# 2001_activation -- model_class (KernelBench compatible) task file for the fused
# gated activation x SwiGLU split-half multiply x dynamic MXFP4 quantization
# operator.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=2, problem_id=2001, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """Gated activation x SwiGLU split-half multiply x dynamic MXFP4 quantization
    fused operator.

    Input x has shape (M, N) (N is a multiple of 4). Split the last dimension in
    half: a = x[:, :N/2], b = x[:, N/2:], then first compute the gated product in
    float32 as out = act(a) * b, where act is selected by the __init__ argument
    activation (mathematical definition, all evaluated in float32):
        silu      : u * sigmoid(u)
        gelu      : 0.5 * u * (1 + erf(u * 0.7071067811865476))
        gelu_tanh : 0.5 * u * (1 + tanh(0.7978845608028654 * (u + 0.044715 * u^3)))

    Then split out along the last dimension into contiguous 32-element blocks and
    dynamically quantize each block to MXFP4 (float32 bit manipulation throughout):
      1. When N/2 is not a multiple of 32, zero-pad the right end of the last block
         up to 32 elements; the padding participates in amax.
      2. amax = max|block|; apply (bits + 0x200000) & 0xFF800000 to the float32 bit
         pattern of amax (even_round: clear the mantissa and round up to the power
         of two grid).
      3. Unbiased scale = clamp(floor(log2(amax)) - 2, -127, 127); the e8m0 byte =
         unbiased scale + 127, landing in [0, 254] (meaning 2^(byte - 127)). A
         fully zero block (including a zero-pad segment that makes amax 0) goes
         through log2(0) = -inf, is clamped to -127, and gets byte 0.
      4. q = block * 2^(-unbiased scale), each element rounded to the e2m1 grid
         {0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6}. Bit-level formula: take the
         float32 fields of q -- s (sign bit) / e (biased exponent) / m (23-bit
         mantissa):
             e' = max(e, 126) - 126
             m' = (e < 127) ? (0x400000 | (m >> 1)) >> (126 - e) : m (subnormal
                  rescaling)
             code = (((e' << 2) | (m' >> 21)) + 1) >> 1, then saturated to 0x7
             e2m1 code word (4 bits) = (s ? 8 : 0) | code
      5. Two adjacent e2m1 code words are packed into one uint8: the even index
         occupies the low 4 bits, the odd index the high 4 bits.

    The output is a single uint8 tensor (two segments concatenated along the last
    dimension):
        out (M, N/4 + ceil(N/64)) uint8
        first N/4 columns       : MXFP4 packed code words (two e2m1 values per byte)
        last ceil(N/64) columns : the e8m0 scale byte of each 32-element block
    Both segments are integer code words and must match exactly.

    Input/output specification:
        x    float16 / bfloat16 (M, N), contiguous, N % 4 == 0, M >= 1
             (the evaluator casts the input to fp32 before calling forward;
             fp16/bf16 -> fp32 is lossless)
        out  uint8 (M, N/4 + ceil(N/64))

    Implementation constraints (violations are scored as failures):
        - The core computation (activation, split-half multiply, per-block scale
          and e2m1 quantization, code-word packing) must be completed with a
          Triton kernel inside the submitted file; the output is uint8 code words
          with no floating-point output.
        - The __init__ and forward signatures of ModelNew must not be changed.
        - The activation, the multiply and the quantization bit manipulation all
          run in float32 and must be bit-identical to this reference implementation
          (code words and scale bytes exactly equal, no tolerance).
        - When N/2 is not a multiple of 32, handle the block tail with the zero-pad
          semantics described above.

    Forbidden: torch.nn.functional.silu / torch.nn.functional.gelu /
    torch.nn.SiLU / torch.nn.GELU / torch.sigmoid / torch.erf / torch.tanh.
    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    """

    def __init__(self, activation: str = "silu"):
        super().__init__()
        assert activation in ("silu", "gelu", "gelu_tanh"), "activation must be one of silu / gelu / gelu_tanh"
        self.activation = activation

    def forward(self, x):
        # The evaluator casts all inputs to fp32; fp16/bf16 -> fp32 is lossless, so
        # compute uniformly in float32
        x = x.to(torch.float32)
        M, N = x.shape
        assert N % 4 == 0, "N must be a multiple of 4"
        d = N // 2
        a, b = x.split([d, d], dim=-1)
        # Gated product act(a) * b (three mutually exclusive branches; __init__ has
        # already asserted that activation is valid)
        out = nn.functional.silu(a) * b
        if self.activation == "gelu":
            out = nn.functional.gelu(a) * b
        if self.activation == "gelu_tanh":
            out = nn.functional.gelu(a, approximate="tanh") * b

        # ---- dynamic MXFP4 quantization: per 32-element block, e8m0 block scale,
        # ---- float32 throughout
        BLOCK = 32
        n_blocks = (d + BLOCK - 1) // BLOCK
        blocks = out
        if d % BLOCK != 0:
            # zero-pad the block tail (the padding participates in amax)
            padded = torch.zeros(M, n_blocks * BLOCK, dtype=torch.float32, device=out.device)
            padded[:, :d] = out
            blocks = padded
        blocks = blocks.reshape(M, n_blocks, BLOCK)

        # even_round: apply (bits + 0x200000) & 0xFF800000 to the float32 bit
        # pattern of amax
        amax, _ = torch.max(torch.abs(blocks), dim=-1)
        amax = amax.view(torch.int32)
        amax = (amax + 0x200000) & 0xFF800000
        amax = amax.view(torch.float32)
        # e8m0 unbiased scale and byte (byte = unbiased scale + 127, in [0, 254])
        scale_unbiased = torch.clamp(torch.log2(amax).floor() - 2, min=-127, max=127)
        bs_e8m0 = (scale_unbiased + 127).to(torch.uint8)
        quant_scale = torch.exp2(-scale_unbiased)
        qx = blocks * quant_scale.unsqueeze(-1)

        # e2m1 rounding (round-to-nearest, saturated at +-6); the bit-level formula
        # matches the task statement
        qx = qx.contiguous().view(torch.int32)
        s = qx & 0x80000000
        e = (qx >> 23) & 0xFF
        m = qx & 0x7FFFFF
        adjusted = torch.clamp(127 - e - 1, min=0)   # the e >= 127 branch is discarded by the where
        m = torch.where(e < 127, (0x400000 | (m >> 1)) >> adjusted, m)
        e = torch.where(e > 126, e, 126) - 126
        combined = (((e << 2) | (m >> 21)) + 1) >> 1
        e2m1_tmp = torch.where(combined < 0x7, combined, 0x7)
        e2m1_value = (((s >> 28) & 0xF) | e2m1_tmp).to(torch.uint8)

        # pack adjacent code words: even index in the low 4 bits, odd index in the
        # high 4 bits
        x_fp4 = e2m1_value[..., ::2] | (e2m1_value[..., 1::2] << 4)
        x_fp4 = x_fp4.reshape(M, -1)[:, : d // 2]

        # single-tensor output: the first d//2 columns are FP4 code words, the last
        # n_blocks columns are e8m0 scale bytes
        return torch.cat([x_fp4, bs_e8m0], dim=-1)


def get_init_inputs():
    return ["silu"]   # activation


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the evaluator
    # calls this function after set_seed, so multiple correctness trials therefore
    # see diverse inputs.
    m, n = 128, 1024
    x = torch.randn(m, n).to(torch.float16)
    return [x]


def make_inputs(m: int, n: int, activation: str = "silu",
                dtype: str = "float16", seed: int = 0,
                name=None, seq_lens=None):
    """Generate inputs from a public/hidden case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the
    device).

    The consumption order is fixed: x (torch.randn, then cast to dtype); everything
    comes from the same torch.Generator(seed), so the same seed reproduces inputs
    bit-for-bit.
    name is accepted for case-name field pass-through compatibility (ignored);
    seq_lens is the placeholder calling convention of the general audit tool (this
    task's inputs have no notion of sequence length, so it is ignored).
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert activation in ("silu", "gelu", "gelu_tanh"), "activation must be one of silu / gelu / gelu_tanh"
    assert m >= 1 and n % 4 == 0, "requires M >= 1 and N % 4 == 0"
    x = torch.randn(m, n, generator=gen).to(dt)
    return (x,)
