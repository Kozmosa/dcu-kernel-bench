# 3009_routing -- MoE gating sigmoid top-1 routing (a compact version with a
# fixed config), as a model_class (KernelBench-compatible) task file.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=3009, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """Sigmoid top-1 routing of MoE (Mixture of Experts) gating: pick the expert
    with the highest routing score for each token out of N experts (top_k is
    always 1).

    Operator semantics: given the gating input x ([M, K], M tokens and K gating
    features) and the routing weight w ([K, N], one column per expert), for each
    token m and expert n:
        score[m, n] = sigmoid( sum_k x[m, k] * w[k, n] )
        sigmoid(z)  = 1 / (1 + exp(-z))
    top-1 routing output:
        topk_ids[m]     = argmax_n score[m, n], ties resolved to the smallest
                          expert index
        topk_weights[m] = max_n score[m, n]
    The inner product (matrix multiply) is evaluated with float32 accumulation;
    with top_k = 1 the normalization coefficient is always 1, so the weight is
    the raw sigmoid score and there is no need to divide by a sum.

    Fused shared expert column: with fused_shared_experts=True one fixed shared
    expert column is appended after the top-1 result -- the id column gets N
    (= the number of experts, outside the real expert id range) and the weight
    column gets 1.0. With False nothing is appended and the output has a single
    column.

    Input/output specification:
        x   [M, K] float16/bfloat16, contiguous; M/K/N >= 1, non powers of two
            allowed
        w   [K, N] same dtype as x, contiguous
        output  a single float32 tensor [M, 2*C], C = 1 + fused_shared_experts:
            out[:, :C] = the integers of topk_ids represented in fp32 (< 2^24,
                         exact)
            out[:, C:] = topk_weights (fp32 sigmoid scores; the fused column is
                         always 1.0)
        The evaluator casts all inputs to fp32 before passing them in (all
        inputs of this operator are floating point, so this is lossless).

    Implementation constraints (violations are scored as failures):
      - Core computation (the block-wise x * w matrix multiply, sigmoid, the
        top-1 selection and tie-break, writing back the fused shared expert
        column) must be done in a Triton kernel inside the submitted file; the
        official implementation fully fuses everything into a single kernel (the
        score matrix is never materialized), splitting it into several Triton
        kernels is not scored as a failure, but calling ATen shortcuts is.
      - Accumulation inside the matrix multiply is always float32 (inputs may be
        fp16/bf16/fp32).
      - The output is strictly float32 per the single-tensor packing protocol
        above.
      - The __init__ and forward signatures of ModelNew must not be changed.

    Disallowed list: aiter / torch.matmul / torch.bmm / torch.einsum /
    torch.nn.functional.linear / torch.topk / torch.argmax / torch.sigmoid.
    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    """

    def __init__(self, fused_shared_experts: bool = True):
        super().__init__()
        self.fused_shared_experts = bool(fused_shared_experts)

    def forward(self, x, w):
        assert x.dim() == 2 and w.dim() == 2, "x / w must be 2-D tensors"
        M, K = x.shape
        Kb, N = w.shape
        assert K == Kb, "the last dimension of x must equal the number of rows of w"

        # The reference implementation may use torch operators freely; scores are
        # evaluated in float32
        scores = torch.sigmoid(
            torch.matmul(x.to(torch.float32), w.to(torch.float32)))   # [M, N]

        # top-1: torch.max resolves ties to the first (smallest) index, matching
        # the tl.argmax(tie_break_left=True) semantics of the official kernel
        top1_weights, top1_ids = torch.max(scores, dim=1)

        ids = top1_ids.to(torch.int32).unsqueeze(1)
        weights = top1_weights.unsqueeze(1)
        if self.fused_shared_experts:
            # Fused shared expert column: append N to the ids (the expert count,
            # outside the real expert range) and 1.0 to the weights
            ids = torch.cat([ids, torch.full((M, 1), N, dtype=torch.int32, device=ids.device)], dim=1)
            weights = torch.cat([weights, torch.ones((M, 1), dtype=torch.float32, device=weights.device)], dim=1)

        # Single-tensor packing protocol: [ids (integers, exact in fp32) | weights]
        return torch.cat([ids.to(torch.float32), weights.to(torch.float32)], dim=1)


def get_init_inputs():
    return [True]   # fused_shared_experts (always on in the official test)


def get_inputs():
    # Fixed shape family (M=1024, K=5120, N=128, bf16, fused shared expert); the
    # random part consumes the global RNG -- the evaluator calls this function
    # after set_seed, so multiple correctness trials see diverse inputs. The value
    # distribution matches the official test's uniform integers over [-2, 2]
    # (integer scores are exactly reproducible along the fp32 accumulation chain,
    # so the tie-break is deterministic)
    M, K, N = 1024, 5120, 128
    x = torch.randint(-2, 3, (M, K)).to(torch.bfloat16)
    w = torch.randint(-2, 3, (K, N)).to(torch.bfloat16)
    return [x, w]


def make_inputs(m: int, k: int, n: int, fused_shared_experts: bool = True,
                dtype: str = "bfloat16", dist: str = "int", seed: int = 0):
    """Generate inputs from a public/hidden/perf case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the
    device). The fields map one-to-one onto the hidden/perf cases;
    fused_shared_experts is also the Model constructor hyperparameter and is
    passed to Model as it is.

    dist="int": uniform integers over [-2, 2] (the same distribution as the
    official test input generator; integer scores are exact in fp32, so the
    tie-break is deterministic). dist="bench": x ~ randn, w ~ randn*0.1 (the
    same distribution as the official benchmark input generator). The
    consumption order is fixed: x first, then w; bit-reproducible under the same
    seed.
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert m >= 1 and k >= 1 and n >= 1
    assert dist in ("int", "bench")
    if dist == "int":
        x = torch.randint(-2, 3, (m, k), generator=gen).to(dt)
        w = torch.randint(-2, 3, (k, n), generator=gen).to(dt)
    else:
        x = torch.randn(m, k, generator=gen).to(dt)
        w = (torch.randn(k, n, generator=gen) * 0.1).to(dt)
    return [x, w]
