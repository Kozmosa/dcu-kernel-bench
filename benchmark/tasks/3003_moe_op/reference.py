# 3003_moe_op -- MoE (Mixture-of-Experts) routed grouped GEMM (model_class
# form).
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here. The computation
# matches the upstream MoE grouped GEMM kernel: float32 intermediate math, the
# output cast back to the input dtype.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=3003, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """MoE (Mixture-of-Experts) routed grouped GEMM: non-quantized weights, with
    routing weighting.

    Each token's feature vector A[m] (K dimensions) is dispatched by the routing
    table topk_ids to top_k of the num_experts experts; expert e owns the weight
    matrix B[e] (N x K). For each token m and its j-th routing slot (write
    e = topk_ids[m, j]), the output is defined as

        C[m, j, n] = topk_weights[m, j] * sum_k A[m, k] * B[e, n, k]

    i.e. grouped GEMM followed by a per-slot multiply by the routing weight;
    intermediate accumulation is float32 and the output is cast back to the
    input dtype. Every (m, j) slot hits a legal expert inside [0, num_experts),
    so no output is undefined.

    The inputs also carry the block-aligned layout of grouped GEMM (produced by
    the upstream router; fetching data through it is exactly equivalent to
    computing the formula above directly):
      - sorted_token_ids: int32, length EM = M*top_k + num_experts*(block_m-1),
        with block_m always 64 in this task. Number the slots flat as
        p = m*top_k + j, regroup them by ascending expert id, and pad each group
        up to a multiple of block_m; the intra-group padding positions and the
        array tail hold the sentinel value M*top_k (that is num_valid_tokens;
        offs_token >= it means padding and produces no output).
      - expert_ids: int32, length M*top_k + num_experts; the expert id owning
        block t (covering sorted_token_ids[t*block_m : (t+1)*block_m]), values
        in [0, num_experts), no -1; positions past the number of blocks implied
        by num_tokens_post_padded are sentinel fill and must not be read.
      - num_tokens_post_padded: int32 scalar (length 1), the valid length of
        sorted_token_ids (intra-block padding included), always a multiple of
        block_m.
    The kernel's BLOCK_SIZE_M must divide block_m (that is, be a factor of 64)
    so that every program block sees a single expert.

    Input/output specification:
        A                       (M, K)             float16/bfloat16
        B                       (E, N, K)          same as A (E = num_experts)
        topk_weights            (M, top_k)         float32
        topk_ids                (M, top_k)         int32
        sorted_token_ids        (EM,)              int32
        expert_ids              (M*top_k + E,)     int32
        num_tokens_post_padded  (1,)               int32
        returns C               (M, top_k, N)      dtype same as A

    Implementation constraints (violations are scored as failures):
        - Core computation (layout-driven data fetch + grouped GEMM + weighted
          routing write-out) must be done in a Triton kernel inside the
          submitted file; calling torch.matmul / torch.mm / torch.bmm /
          torch.baddbmm / torch.addmm / torch.einsum /
          torch.nn.functional.linear or any precompiled operator library for
          any matrix multiply is forbidden.
        - The __init__ and forward signatures of ModelNew must not be changed.
        - Use float32 for intermediate accumulation; the output is cast back to
          the input dtype.
        - Sentinel padding slots must not contaminate any valid output.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: non-quantized fp16/bf16 weights and activations (no w8a8 / int4 /
    mxfp4 quantization), no fused activation (SiLU/GELU etc.), no bias,
    top_k <= 8, num_experts <= 256.
    """

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, A, B, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded):
        # The KernelBench evaluator casts every input tensor to the evaluation
        # precision (fp32 for the triton backend); expert ids and the sorted
        # tables are small integers and exactly representable in fp32, so
        # restore them here losslessly
        topk_ids = topk_ids.to(torch.long)
        sorted_token_ids = sorted_token_ids.to(torch.long)
        expert_ids = expert_ids.to(torch.long)
        num_tokens_post_padded = num_tokens_post_padded.to(torch.int32)

        M, K = A.shape
        E, N, _ = B.shape
        top_k = topk_ids.shape[1]

        a = A.to(torch.float32)
        b = B.to(torch.float32)
        w = topk_weights.to(torch.float32).reshape(-1)

        # The reference implementation does the GEMM grouped by expert (which is
        # equivalent to a per-slot dot product); sorted_token_ids / expert_ids /
        # num_tokens_post_padded are the block-aligned layout the kernel
        # consumes, and the reference implementation does not depend on them
        out = torch.zeros(M, top_k, N, dtype=torch.float32, device=A.device)
        flat_out = out.view(M * top_k, N)
        a_rp = a.repeat_interleave(top_k, dim=0)   # slot p = m*top_k + j -> a[m]
        flat_ids = topk_ids.reshape(-1)
        for e in range(E):
            mask = flat_ids == e
            if bool(mask.any()):
                flat_out[mask] = (a_rp[mask] @ b[e].t()) * w[mask][:, None]
        return out.to(A.dtype)


def get_init_inputs():
    return [16, 4]   # num_experts, top_k (consistent with the shape family of get_inputs)


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials see diverse inputs
    return _generate_inputs(
        num_tokens=64, num_experts=16, top_k=4,
        in_features=512, out_features=256,
        block_m=64, dtype=torch.float16, gen=None,
    )


def make_inputs(num_tokens: int, num_experts: int, top_k: int,
                in_features: int, out_features: int, block_m: int = 64,
                dtype: str = "float16", seed: int = 0):
    """Deterministic case generator (evaluation side only; generated on CPU and
    moved to the device by the evaluator).

    The parameter fields map one-to-one onto the hidden/perf cases; block_m is
    the block alignment granularity of the sorted layout, always 64 here.
    """
    gen = torch.Generator().manual_seed(seed)
    return _generate_inputs(num_tokens, num_experts, top_k, in_features,
                            out_features, block_m, getattr(torch, dtype), gen)


def _generate_inputs(num_tokens, num_experts, top_k, in_features,
                     out_features, block_m, dtype, gen):
    # gen=None consumes the global RNG (the get_inputs path); official input
    # distribution: activations randn/10, weights rand/10, routing weights are
    # the top-k of softmax over the logits
    a = torch.randn(num_tokens, in_features, dtype=dtype, generator=gen) / 10
    b = torch.rand(num_experts, out_features, in_features, dtype=dtype,
                   generator=gen) / 10
    values = torch.randn(num_tokens, num_experts, generator=gen)
    softmax_vals = torch.softmax(values, dim=1)              # float32
    topk_weights, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    sorted_token_ids, expert_ids, num_tokens_post_padded = _moe_align_block_size(
        topk_ids, num_experts, block_m)
    return [a, b, topk_weights, topk_ids, sorted_token_ids, expert_ids,
            num_tokens_post_padded]


def _moe_align_block_size(topk_ids, num_experts, block_m):
    """Expand the (M, top_k) routing table into the expert-grouped, block-aligned
    grouped GEMM layout.

    Alignment protocol: slots p = m*top_k + j are regrouped by ascending expert
    id; every group is padded up to a multiple of block_m, with the intra-group
    padding and the array tail filled with the sentinel value M*top_k;
    expert_ids records the owning expert per block, its tail filled with the
    sentinel as well; num_tokens_post_padded records the valid length.
    """
    M, top_k = topk_ids.shape
    num_valid = topk_ids.numel()

    expert_to_tokens = [[] for _ in range(num_experts)]
    ids = topk_ids.to(torch.long).tolist()
    for m in range(M):
        for j in range(top_k):
            expert_to_tokens[ids[m][j]].append(m * top_k + j)

    reordered_tokens = []
    reordered_experts = []
    for e in range(num_experts):
        tokens = expert_to_tokens[e]
        n_blocks = (len(tokens) + block_m - 1) // block_m
        reordered_tokens.extend(tokens)
        reordered_experts.extend([e] * n_blocks)
        reordered_tokens.extend([num_valid] * (n_blocks * block_m - len(tokens)))

    sorted_token_ids = torch.full(
        (num_valid + num_experts * (block_m - 1),), num_valid, dtype=torch.int32)
    expert_ids = torch.empty(num_valid + num_experts, dtype=torch.int32)
    num_tokens_post_padded = torch.empty(1, dtype=torch.int32)

    sorted_token_ids[: len(reordered_tokens)] = torch.tensor(
        reordered_tokens, dtype=torch.int32)
    expert_ids[: len(reordered_experts)] = torch.tensor(
        reordered_experts, dtype=torch.int32)
    if len(reordered_experts) < expert_ids.numel():
        expert_ids[len(reordered_experts):] = num_valid
    num_tokens_post_padded.fill_(len(reordered_tokens))
    return sorted_token_ids, expert_ids, num_tokens_post_padded
