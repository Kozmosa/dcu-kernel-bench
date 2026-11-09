# 4008_gemm_a16w4 -- AWQ W4A16 quantized GEMM (int4 packed weights dequantized
# inside the kernel) model_class (KernelBench compatible) task file.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=4008, backend=triton.

import torch
import torch.nn as nn


def _reverse_awq_order(tensor: torch.Tensor) -> torch.Tensor:
    """Reorder the 4-bit code-word columns in the AWQ reverse order
    [0,4,1,5,2,6,3,7] and keep the low 4 bits."""
    bits = 4
    awq_reverse_order = [0, 4, 1, 5, 2, 6, 3, 7]
    order = torch.arange(tensor.shape[-1], dtype=torch.int32)
    order = order.view(-1, 32 // bits)[:, awq_reverse_order].view(-1)
    return tensor[:, order] & 0xF


def _awq_reorder_and_repack(qweight: torch.Tensor, qzeros: torch.Tensor):
    """Input-side preprocessing (offline packing): reorder the raw int32 codes
    [K, N//8] / [K//G, N//8] and pack them into the layout consumed by the
    kernel -- qweight [N, K//2] (2 4-bit codes per byte along K, low nibble =
    even k) and qzeros [K//G, N//2] (along N, low nibble = even n), both int8.
    Only used by make_inputs/get_inputs to generate the task inputs; not part
    of the operator semantics proper."""
    bits = 4
    shifts = torch.arange(0, 32, bits)
    K = qweight.shape[0]
    N = qweight.shape[1] * 8
    G = K // qzeros.shape[0]

    iweights = torch.bitwise_right_shift(
        qweight[:, :, None], shifts[None, None, :]).to(torch.int8).view(K, -1)
    zeros = torch.bitwise_right_shift(
        qzeros[:, :, None], shifts[None, None, :]).to(torch.int8).view(K // G, -1)

    iweights = torch.bitwise_and(_reverse_awq_order(iweights), 0xF)
    zeros = torch.bitwise_and(_reverse_awq_order(zeros), 0xF)

    iweights = iweights.transpose(1, 0).contiguous().view(N, -1, 2)
    zeros = zeros.view(K // G, -1, 2)

    packed_weights = torch.zeros([N, K // 2], dtype=torch.int8)
    packed_zeros = torch.zeros([K // G, N // 2], dtype=torch.int8)
    for i in range(2):
        packed_weights |= iweights[:, :, i].to(torch.int8) << (i * bits)
        packed_zeros |= zeros[:, :, i].to(torch.int8) << (i * bits)
    return packed_weights, packed_zeros


class Model(nn.Module):
    """AWQ W4A16 quantized GEMM: int4 packed weights are dequantized inside the
    kernel and then multiplied with the activations.

    Operator semantics (mathematical definition and boundary behaviour):
      activation input [M, K] (float16), AWQ int4 packed weights
      qweight [N, K//2] (int8, 2 4-bit code words stored per byte consecutively
      along K, the low nibble corresponding to even k), grouped zeros
      qzeros [K//G, N//2] (int8, 2 4-bit code words stored per byte
      consecutively along N, the low nibble corresponding to even n), grouped
      scales scales [K//G, N] (float16). Dequantization and the matrix multiply
      are defined as
        w[n, k] = (qweight[n, k//2] >> (4*(k%2))) & 0xF
        z[g, n] = (qzeros[g, n//2]   >> (4*(n%2))) & 0xF
        W[k, n] = (w[n, k] - z[k//G, n]) * scales[k//G, n]
        out[m, n] = sum_{k=0}^{K-1} input[m, k] * W[k, n]
      A signed int8 right shift is an arithmetic right shift, so the & 0xF after
      the shift always extracts the stored 4-bit code word (result 0..15);
      w - z takes values in [-15, 15]. The element-wise multiply-accumulate runs
      in float32 throughout, and the output is cast back to the input dtype.
      Boundary behaviour: M, K and N need not be aligned to any tiling size
      (positions past the end along K and N take part in the masked
      accumulation as 0); G is 32/64/128 or G == K (per-channel), and G divides
      K.

    Input/output specification:
      forward(input, qweight, scales, qzeros) -> out
        input   [M, K]         float16, row-major contiguous
        qweight [N, K//2]      int8 packed weights (low nibble = even k)
        scales  [K//G, N]      float16, row-major contiguous
        qzeros  [K//G, N//2]   int8 packed zeros (low nibble = even n)
        out     [M, N]         same dtype as input
      __init__(in_features, out_features, group_size): correspond to K, N, G
      respectively and are instance metadata only; forward always follows the
      actual runtime tensor shapes (G = K // qzeros.shape[0]).

    Implementation constraints (violations are scored as failures):
      - The core computation (int4 code-word unpacking, grouped dequantization,
        the multiply-accumulate main loop) must be completed inside the
        submitted file as Triton kernels; calling torch.matmul / torch.bmm /
        torch.mm / torch.addmm / torch.einsum / torch.nn.functional.linear /
        F.linear / aiter or any other ATen GEMM shortcut is forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation; the output is cast back to
        the input dtype.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: single-card W4A16 GEMM, float16 activations and float16 scales;
    the scheduling strategy (data parallel / split-K / stream-K) and the tiling
    are up to the implementation and judged by the output tolerance; no bias,
    no quantization-aware activation transform, no cross-card communication
    fusion.
    """

    def __init__(self, in_features: int, out_features: int, group_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

    def forward(self, input, qweight, scales, qzeros):
        # The KernelBench evaluator casts every input tensor to the evaluation
        # precision (fp32 for the triton backend); the packed code words take
        # values |v| <= 128, exactly representable in fp32, so they are
        # recovered losslessly as integers here
        qweight = qweight.to(torch.int32)
        qzeros = qzeros.to(torch.int32)

        M, K = input.shape
        N = qweight.shape[0]                      # (N, K//2)
        group = K // qzeros.shape[0]
        assert qweight.shape[1] == K // 2 and qzeros.shape[1] == N // 2
        assert scales.shape[0] == K // group and scales.shape[1] == N

        # Unpack the int4 code words: the low nibble is the even index;
        # arithmetic right shift + & 0xF
        k_shift = (torch.arange(K, device=input.device) % 2) * 4
        w_int = (qweight.repeat_interleave(2, dim=1) >> k_shift) & 0xF    # [N, K]
        n_shift = (torch.arange(N, device=input.device) % 2) * 4
        z_int = (qzeros.repeat_interleave(2, dim=1) >> n_shift) & 0xF     # [K//G, N]

        # Grouped zeros/scales broadcast along K, then dequantized in float32
        z_full = z_int.repeat_interleave(group, dim=0)                    # [K, N]
        s_full = scales.to(torch.float32).repeat_interleave(group, dim=0)  # [K, N]
        w_deq = (w_int.to(torch.float32).t() - z_full.to(torch.float32)) * s_full

        # [M, K] @ [K, N] -> [M, N], accumulated once in float32, output cast
        # back to the input dtype
        out = torch.matmul(input.to(torch.float32), w_deq)
        return out.to(input.dtype)


def get_init_inputs():
    return [2048, 1536, 64]  # in_features(K), out_features(N), group_size(G)


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials therefore see diverse inputs. The data distribution follows the
    # operator's official test: activations and scales are drawn from U[0,1),
    # the raw code words are random int32 packed in the AWQ layout
    m = 64
    n = 1536
    k = 2048
    group_size = 64
    x = torch.rand((m, k)).to(torch.float16)
    qweight = torch.randint(0, torch.iinfo(torch.int32).max,
                            (k, n // 8), dtype=torch.int32)
    qzeros = torch.randint(0, torch.iinfo(torch.int32).max,
                           (k // group_size, n // 8), dtype=torch.int32)
    scales = torch.rand((k // group_size, n)).to(torch.float16)
    qweight, qzeros = _awq_reorder_and_repack(qweight, qzeros)
    return [x, qweight, scales, qzeros]


def make_inputs(m: int, n: int, k: int, group_size: int, seed: int = 0):
    """Generate inputs from a public/hidden case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the
    device). The generation order matches get_inputs: x -> raw qweight -> raw
    qzeros -> scales -> packing."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand((m, k), generator=gen).to(torch.float16)
    qweight = torch.randint(0, torch.iinfo(torch.int32).max,
                            (k, n // 8), dtype=torch.int32, generator=gen)
    qzeros = torch.randint(0, torch.iinfo(torch.int32).max,
                           (k // group_size, n // 8), dtype=torch.int32,
                           generator=gen)
    scales = torch.rand((k // group_size, n), generator=gen).to(torch.float16)
    qweight, qzeros = _awq_reorder_and_repack(qweight, qzeros)
    return x, qweight, scales, qzeros
