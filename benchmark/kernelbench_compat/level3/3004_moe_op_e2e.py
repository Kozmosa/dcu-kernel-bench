# 3004_moe_op_e2e -- fused MoE feed-forward (both grouped GEMMs + SwiGLU +
# routing weighting inside one kernel), as a model_class (KernelBench-compatible)
# task file.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=3004, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """The full fused Mixture-of-Experts (MoE) token feed-forward path: one call
    completes the whole chain "pick experts by routing -> first grouped GEMM ->
    SwiGLU activation -> second grouped GEMM -> routing weighting".

    Operator semantics (mathematical definition and boundary behaviour):
      Inputs: activation a [M, K]; expert weight w1 [E, N, K] for the first FFN
      layer (N = 2*hidden, the first N/2 channels are gate and the last N/2 are
      up), w2 [E, K, N//2] for the second FFN layer; the routing result
      topk_ids [M, top_k] holds the expert ids selected for each token and
      topk_weights [M, top_k] the matching routing weights. For each token m and
      its j-th expert e = topk_ids[m, j]:
        gate[m,j,:] = a[m,:] @ w1[e, :N//2, :].T          # [N//2]
        up[m,j,:]   = a[m,:] @ w1[e, N//2:, :].T          # [N//2]
        h[m,j,:]    = gate * sigmoid(gate) * up           # SwiGLU, SiLU(x)=x*sigmoid(x)
        out[m,j,k]  = topk_weights[m,j] * sum_{n<N//2} h[m,j,n] * w2[e,k,n]
      that is out[m,j,:] = topk_weights[m,j] * (h[m,j,:] @ w2[e].T). The (m, j)
      pairs are mutually independent; the outputs of all top_k slots are written
      to the result, with no merge, sum or normalization across experts.
      Boundary behaviour: N is even; top_k <= E and the topk_ids values lie in
      [0, E); the top_k experts of one token are generated distinct (even should
      a duplicate appear, the pair-independent computation still holds); M and K
      are arbitrary positive integers (1 included, i.e. the GEMV case); routing
      weighting is always enabled.

    Input/output specification:
      forward(a, w1, w2, topk_weights, topk_ids) -> out
        a            [M, K]        bfloat16/float32, row-major contiguous
        w1           [E, N, K]     same dtype as a, row-major contiguous
        w2           [E, K, N//2]  same dtype as a, row-major contiguous
        topk_weights [M, top_k]    same dtype as a, row-major contiguous
        topk_ids     [M, top_k]    int64 (losslessly recoverable after the
                                   evaluator's fp32 cast)
        out          [M, top_k, K] same dtype as a
      __init__(top_k, num_experts): instance metadata only; forward always goes
      by the actual runtime tensor shapes (top_k = topk_ids.shape[1],
      E = w1.shape[0]).

    Implementation constraints (violations are scored as failures):
      - Core computation (both grouped GEMMs, the SwiGLU activation, the routing
        weighting) must be done with a Triton kernel inside the submitted file;
        calling ATen shortcuts such as torch.matmul / torch.mm / torch.bmm /
        torch.baddbmm / torch.addmm / torch.einsum /
        torch.nn.functional.linear / F.linear / torch.nn.functional.silu /
        F.silu / torch.sigmoid / aiter is forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Use float32 for intermediate accumulation; the output is cast back to
        the input dtype.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: non-quantized bfloat16/float32 activations and weights; single
    card; no expert capacity limit, no shared experts, no routing weight
    normalization; the expert dispatch/scheduling strategy (sorted alignment,
    persistent grid, etc.) is up to the implementation and is judged by the
    output tolerance.
    """

    def __init__(self, top_k: int, num_experts: int):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, a, w1, w2, topk_weights, topk_ids):
        # The KernelBench evaluator casts every input tensor to the evaluation
        # precision (fp32 for the triton backend); expert ids are small
        # integers and exactly representable in fp32, so restore them here
        # losslessly as integer indices
        topk_ids = topk_ids.to(torch.long)

        M, K = a.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        H = N // 2
        assert 0 <= int(topk_ids.min()) and int(topk_ids.max()) < E, "topk_ids must fall inside [0, E)"

        a32 = a.to(torch.float32)
        tw32 = topk_weights.to(torch.float32)

        out = torch.empty((M, top_k, K), dtype=a.dtype, device=a.device)
        for e in range(E):
            sel = topk_ids == e
            if not sel.any():
                continue
            rows, slots = sel.nonzero(as_tuple=True)            # the (m, j) routed to expert e
            inter = a32[rows] @ w1[e].to(torch.float32).t()     # [T, N] first grouped GEMM
            gate, up = inter[:, :H], inter[:, H:]
            h = gate * torch.sigmoid(gate) * up                 # SwiGLU, float32 throughout
            o = h @ w2[e].to(torch.float32).t()                 # [T, K] second grouped GEMM
            o = o * tw32[rows, slots].unsqueeze(-1)             # routing weighting
            out[rows, slots] = o.to(a.dtype)                    # cast back to the input dtype
        return out


def get_init_inputs():
    return [2, 8]  # top_k, num_experts (metadata only; the runtime tensor shapes win)


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials see diverse inputs. The data distribution follows the official test
    # of this operator: activations ~ N(0,1), expert weights ~ U[0,1), routing
    # takes the top-k of softmax(logits)
    m = 32
    n = 1024
    k = 512
    top_k = 2
    e = 8
    dt = torch.bfloat16
    a = torch.randn((m, k), dtype=dt)
    w1 = torch.rand((e, n, k), dtype=dt)
    w2 = torch.rand((e, k, n // 2), dtype=dt)
    logits = torch.randn((m, e), dtype=dt)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=1), k=top_k, dim=1)
    return [a, w1, w2, topk_weights, topk_ids]


def make_inputs(m: int, n: int, k: int, top_k: int, num_experts: int,
                dtype: str = "bfloat16", seed: int = 0, seq_lens=None):
    """Generate inputs from a public/hidden case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the
    device). The generation order matches get_inputs: a -> w1 -> w2 -> routing
    logits -> softmax top-k; n is even and top_k <= num_experts. The parameter
    names align with task.yaml's io.init_inputs, so hidden/perf case fields can
    be passed through as they are. seq_lens is a reserved parameter kept for the
    uniform call form (this operator has no notion of sequence length; it is
    accepted and ignored)."""
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    a = torch.randn((m, k), dtype=dt, generator=gen)
    w1 = torch.rand((num_experts, n, k), dtype=dt, generator=gen)
    w2 = torch.rand((num_experts, k, n // 2), dtype=dt, generator=gen)
    logits = torch.randn((m, num_experts), dtype=dt, generator=gen)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=1), k=top_k, dim=1)
    return a, w1, w2, topk_weights, topk_ids
