# 3007_moe_op_silu_fused -- MoE grouped GEMM (bf16 dense and fp8 block-quantized
# variants) fused with a SiLU-and-multiply activation epilogue, as a model_class
# (KernelBench-compatible) task file.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=3007, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """SiLU-and-multiply fused into the epilogue of an MoE (Mixture of Experts)
    single-layer expert FFN grouped GEMM: one kernel performs the grouped matrix
    multiply of tokens against expert weights, the routing weight weighting and
    the SwiGLU-style activation together, so no intermediate activation is
    materialized.

    Operator semantics: given
        A             activations (M, K)
        B             expert weights (E, 2N, K), the transposed weight matrix of
                      the e-th expert
        topk_ids      routing result (M, top_k), elements in [0, E), distinct
                      within a row
        topk_weights  routing weights (M, top_k), always positive within a row,
                      no normalization needed
        and the three scheduling tensors precomputed from topk_ids by the
        upstream alignment operator
        sorted_token_ids / expert_ids / num_tokens_post_padded (see the input
        specification).
    For each token m and its j-th selected expert e = topk_ids[m, j]:
        x[m, j, :] = A[m, :] @ B[e, :, :]^T          in R^{2N}
        if mul_routed_weight = True: x[m, j, :] *= topk_weights[m, j]
        out[m*top_k + j, i] = SiLU(x[m, j, i]) * x[m, j, N + i],  i in [0, N)
    where SiLU(u) = u / (1 + exp(-u)) = u * sigmoid(u). The 2N output dimensions
    of B use the chunked gating layout: the first N rows are gate and the last N
    rows are up (distinct from the interleaved layout). Every row of the output
    (M*top_k, N) is written by exactly one expert (routing guarantees that every
    row has an owner).

    fp8 block-quantized mode: with quant="fp8_block", A / B are float8_e4m3 code
    words taking part in the same math after dequantization:
        A_deq[m, k]  = A[m, k]  * A_scale[m,  k // block_k]
        B_deq[e, j, k] = B[e, j, k] * B_scale[e, j // block_n, k // block_k]
    This mode requires K % block_k == 0 and 2N % block_n == 0 (no scale tail
    block). With quant="dense", A / B are the raw weights and A_scale / B_scale
    are ignored one-element placeholder tensors.

    Boundary behaviour:
      - slots in sorted_token_ids whose value is >= M*top_k are alignment padding
        sentinels: they take no part in the computation and produce no output
        row; expert_ids is valid only inside the first ceil(num_tokens_post_padded
        / block_size_m) blocks (the valid prefix takes values in [0, E), with no
        cross-card -1 sentinel, and the tail holds unconsumed fill values).
      - an expert selected by no token does not read its B rows.
      - the three scheduling tensors and topk_ids are mutually redundant (all
        derived from the same topk_ids); the core scheduling may use either
        information source, but the output must match the math defined above in
        terms of topk_ids.

    Input/output specification:
        A            (M, K)    float16/bfloat16 (e4m3 code words in fp8_block
                               mode)
        B            (E, 2N, K) same as A (e4m3 code words in fp8_block mode)
        A_scale      (M, K//block_k) float32 ((1,) placeholder in dense mode)
        B_scale      (E, 2N//block_n, K//block_k) float32 ((1,) placeholder in
                     dense mode)
        topk_weights (M, top_k) float16/bfloat16, same dtype as the output
        topk_ids     (M, top_k) int64
        sorted_token_ids (M*top_k + E*(block_size_m-1),) int32
        expert_ids   (M*top_k + E,) int32
        num_tokens_post_padded (1,) int32
        out          (M*top_k, N), dtype same as topk_weights.
        M, K, E, top_k, block_size_m >= 1; 2N is even; input magnitudes are the
        usual randn/rand scale with no inf/NaN. The evaluator casts all inputs
        to fp32 before passing them in (integer types restore exactly,
        fp16/bf16/fp8 -> fp32 is lossless), so when topk_weights is fp32 the
        output is fp32 as well.

    Implementation constraints (violations are scored as failures):
      - Core computation (the grouped GEMM, fp8 dequantization, the routing
        weight multiply, the SiLU-and-multiply epilogue) must be done with a
        Triton kernel inside the submitted file, with intermediate accumulation
        always float32 and the output cast back to the dtype of topk_weights.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Materializing the intermediate activation (the 2N-wide GEMM result) to
        device memory and then activating it is forbidden -- the whole point of
        this operator is epilogue fusion, so the activation is done in place
        block by block.

    Disallowed list: torch.matmul / torch.bmm / torch.einsum /
    torch._scaled_mm / torch.nn.functional.silu / torch.SiLU / torch.sigmoid.
    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    """

    def __init__(self, mul_routed_weight=True, quant="dense", block_n=128, block_k=128):
        super().__init__()
        assert quant in ("dense", "fp8_block"), "quant must be one of dense / fp8_block"
        assert block_n >= 1 and block_k >= 1, "block_n / block_k must be >= 1"
        self.mul_routed_weight = bool(mul_routed_weight)
        self.quant = quant
        self.block_n = int(block_n)
        self.block_k = int(block_k)

    def forward(self, A, B, A_scale, B_scale, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded):
        # The evaluator casts all inputs to fp32; the integer index / scheduling
        # tensors are restored losslessly here
        topk_ids = topk_ids.to(torch.long)
        sorted_token_ids = sorted_token_ids.to(torch.int32)
        expert_ids = expert_ids.to(torch.int32)
        num_tokens_post_padded = num_tokens_post_padded.to(torch.int32)

        M, K = A.shape
        E, N2, _ = B.shape
        assert N2 % 2 == 0, "the 2N output dimension of B must be even"
        N = N2 // 2
        top_k = topk_ids.shape[1]
        out_dtype = topk_weights.dtype
        # The three scheduling tensors are used only by the consuming side (the
        # Agent kernel); see the topk_ids path for the semantics
        _ = (sorted_token_ids, expert_ids, num_tokens_post_padded)

        # Dequantize / lift to float32
        a = A.to(torch.float32)
        b = B.to(torch.float32)
        if self.quant == "fp8_block":
            bk, bn = self.block_k, self.block_n
            assert A_scale.shape == (M, K // bk), "A_scale shape should be (M, K//block_k)"
            assert B_scale.shape == (E, N2 // bn, K // bk), \
                "B_scale shape should be (E, 2N//block_n, K//block_k)"
            a = a * A_scale.repeat_interleave(bk, dim=1)
            b = b * B_scale.repeat_interleave(bn, dim=1).repeat_interleave(bk, dim=2)

        tw = topk_weights.to(torch.float32)
        out = torch.zeros(M * top_k, N, dtype=torch.float32, device=A.device)
        for e in range(E):
            mask = topk_ids == e                      # (M, top_k)
            if not bool(mask.any()):
                continue
            rows = mask.nonzero(as_tuple=False)       # (cnt, 2)
            x = a[rows[:, 0]] @ b[e].t()              # (cnt, 2N) float32 accumulation
            if self.mul_routed_weight:
                x = x * tw[mask].unsqueeze(-1)
            gate, up = x[:, :N], x[:, N:]
            out[rows[:, 0] * top_k + rows[:, 1]] = gate / (1.0 + torch.exp(-gate)) * up
        return out.to(out_dtype)


def get_init_inputs():
    return [True, "dense"]   # mul_routed_weight / quant (block_n=block_k=128 default)


def get_inputs():
    # Fixed shape family (M=32, N=256, K=512, E=8, top_k=4, block_size_m=16,
    # dense bf16, routing weight applied); the random part consumes the global
    # RNG -- the evaluator calls this function after set_seed, so multiple
    # correctness trials see diverse inputs
    return list(_gen_case(m=32, n=256, k=512, e=8, top_k=4, block_size_m=16,
                          mul_routed_weight=True, quant="dense",
                          block_n=128, block_k=128, dtype="bfloat16", rng=None))


def make_inputs(m: int, n: int, k: int, e: int, top_k: int,
                block_size_m: int = 16, mul_routed_weight: bool = True,
                quant: str = "dense", block_n: int = 128, block_k: int = 128,
                dtype: str = "bfloat16", seed: int = 0):
    """Generate inputs from a public/hidden case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the
    device). The fields map one-to-one onto the hidden/perf cases;
    mul_routed_weight / quant / block_n / block_k are also the Model constructor
    hyperparameters and are passed to Model as they are.

    The consumption order is fixed (the same torch.Generator(seed)):
    A(randn) -> B(rand) -> [fp8_block: B_scale(rand)] -> routing logits(randn);
    the three alignment tensors are a deterministic function of topk_ids and
    consume no randomness; bit-reproducible under the same seed.
    """
    gen = torch.Generator().manual_seed(seed)
    return _gen_case(m=m, n=n, k=k, e=e, top_k=top_k, block_size_m=block_size_m,
                     mul_routed_weight=mul_routed_weight, quant=quant,
                     block_n=block_n, block_k=block_k, dtype=dtype, rng=gen)


def _gen_case(m, n, k, e, top_k, block_size_m, mul_routed_weight, quant,
              block_n, block_k, dtype, rng):
    """Deterministic generator shared by get_inputs (global RNG) and make_inputs
    (Generator).

    The distribution follows the family of the official input_helper (randn
    activations / rand positive weights / fp8 B as uniform code words over
    [-448, 448] plus a uniform scale in (0, 0.01) / A as per-token-group e4m3
    dynamic quantization / softmax-topk routing), with unit randn/rand
    magnitudes: dense output is O(10) and fp8_block output O(1), both at a
    magnitude the acceptance tolerance can judge.
    """
    dt = getattr(torch, dtype)
    assert m >= 1 and n >= 1 and k >= 1 and e >= 1 and top_k >= 1
    assert 1 <= top_k <= e, "top_k must not exceed the number of experts"
    assert quant in ("dense", "fp8_block")

    # 1) activation A: (m, k), randn (fp8_block mode quantizes it per token group afterwards)
    a = torch.randn(m, k, generator=rng, dtype=dt)

    # 2) expert weights B: (e, 2n, k)
    if quant == "dense":
        b = torch.rand(e, 2 * n, k, generator=rng, dtype=dt)
        a_scale = torch.zeros(1, dtype=torch.float32)
        b_scale = torch.zeros(1, dtype=torch.float32)
    else:
        assert k % block_k == 0, "fp8_block requires K % block_k == 0"
        assert (2 * n) % block_n == 0, "fp8_block requires 2N % block_n == 0"
        fp8_max = 448.0
        bf = (torch.rand(e, 2 * n, k, generator=rng, dtype=dt) - 0.5) * 2 * fp8_max
        b = bf.clamp(min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn)
        n_tiles = (2 * n) // block_n
        k_tiles = k // block_k
        b_scale = torch.rand(e, n_tiles, k_tiles, generator=rng,
                             dtype=torch.float32) * 1e-2
        a, a_scale = _per_token_group_quant_fp8(a / 10, block_k)   # a becomes e4m3 code words (official /10 magnitude)

    # 3) routing: the topk of softmax(logits)
    values = torch.randn(m, e, generator=rng, dtype=dt)
    softmax_vals = torch.softmax(values, dim=1)
    topk_weights, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)

    # 4) the three alignment tensors (a deterministic function of topk_ids, the product of the upstream scheduling prologue operator)
    sorted_token_ids, expert_ids, num_tokens_post_padded = _align_block_size(
        topk_ids, e, block_size_m)

    return (a, b, a_scale, b_scale, topk_weights, topk_ids,
            sorted_token_ids, expert_ids, num_tokens_post_padded)


def _per_token_group_quant_fp8(x, group_size, eps=1e-10):
    """e4m3 dynamic quantization grouped per token (the last dimension is cut
    into groups of group_size)."""
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_min, fp8_max = finfo.min, finfo.max
    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / fp8_max
    x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    return x_q.reshape(x.shape), x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))


def _align_block_size(topk_ids, num_experts, block_size):
    """Bucketize topk_ids per expert and align every bucket up to block_size,
    producing the three grouped GEMM scheduling tensors: sorted_token_ids
    (flattened indices plus sentinel padding), expert_ids (expert id per block,
    sentinel tail), num_tokens_post_padded ((1,), total padded slot count)."""
    M, top_k = topk_ids.shape
    ids = topk_ids.to(torch.int32).tolist()
    expert_to_tokens = [[] for _ in range(num_experts)]
    for token_id in range(M):
        for j in range(top_k):
            expert_to_tokens[ids[token_id][j]].append(token_id * top_k + j)

    reordered_token_ids, reordered_expert_ids = [], []
    for e_id in range(num_experts):
        tokens_for_expert = expert_to_tokens[e_id]
        num_tokens = len(tokens_for_expert)
        n_blocks = (num_tokens + block_size - 1) // block_size
        padded_size = n_blocks * block_size
        reordered_token_ids.extend(tokens_for_expert)
        reordered_expert_ids.extend([e_id] * n_blocks)
        if padded_size > num_tokens:
            reordered_token_ids.extend([topk_ids.numel()] * (padded_size - num_tokens))

    token_length = len(reordered_token_ids)
    sorted_token_ids = torch.full((topk_ids.numel() + num_experts * (block_size - 1),),
                                  topk_ids.numel(), dtype=torch.int32)
    sorted_token_ids[:token_length] = torch.tensor(reordered_token_ids, dtype=torch.int32)
    expert_ids = torch.full((topk_ids.numel() + num_experts,), topk_ids.numel(),
                            dtype=torch.int32)
    expert_ids[:len(reordered_expert_ids)] = torch.tensor(reordered_expert_ids,
                                                          dtype=torch.int32)
    num_tokens_post_pad = torch.tensor([token_length], dtype=torch.int32)
    return sorted_token_ids, expert_ids, num_tokens_post_pad
