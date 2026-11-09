# 3002_moe_align_block_size -- the scheduling prologue of MoE grouped GEMM
# (tokens bucketized per expert and laid out aligned to block_size), as a
# model_class (KernelBench-compatible) task file.
#
# This file must be self-contained: the KernelBench evaluator loads the task via
# exec(source), so local module imports are forbidden -- the reference
# implementation and input generation are all inlined here.
#
# How to run (PyramidKernel native model_class path):
#   kernelbench_root points at benchmark/kernelbench_compat/,
#   level=3, problem_id=3002, backend=triton.

import torch
import torch.nn as nn


class Model(nn.Module):
    """Scheduling prologue of MoE grouped GEMM: tokens bucketized per expert and
    laid out aligned to block_size.

    The input topk_ids ([num_tokens, top_k], elements are expert ids in
    [0, num_experts), duplicates allowed within a row) holds the experts the
    router selected for each token. Write the flattened index
    i = t * top_k + j (t is the token row, j the column), N = num_tokens * top_k,
    E = num_experts, B = block_size. For every expert e:
        T_e = { i : the i-th element of flattened topk_ids == e } (ascending i)
        n_blocks_e = ceil(|T_e| / B)        # aligned blocks occupied by e
        padded_e   = n_blocks_e * B         # aligned slot count of e
    Produce the three scheduling tensors of grouped GEMM:
        sorted_token_ids  length L1 = N + E*(B-1): for e = 0..E-1 in order, all
          indices of T_e (ascending), then padded_e - |T_e| fill sentinels N;
          the buffer tail [S, L1) is also sentinel N, where S = sum_e padded_e
        expert_ids        length L2 = ceil(L1/B): for e = 0..E-1 in order,
          n_blocks_e copies of e (together the first S/B positions), the tail
          [S/B, L2) filled with the -1 sentinel
        num_tokens_post_pad  scalar S (total padded slot count, always a
          multiple of B)
    grouped GEMM takes every B consecutive slots of sorted_token_ids as one
    M-tile, uses expert_ids as that tile's expert id to fetch weights, skips
    fill slots whose value is sentinel N, and num_tokens_post_pad gives the
    valid total length. This task is that scheduling prologue itself and does
    not include the GEMM.

    Boundary behaviour: an expert with |T_e| = 0 occupies 0 blocks and yields no
    sorted entry and no expert entry; when |T_e| is an exact multiple of B no
    padding is produced; the same row may select the same expert more than once
    (different flattened indices are different entries); entries within an
    expert follow ascending flattened index i.

    Input/output specification:
      topk_ids  int64/int32 (or fp32 as cast by the evaluator, integer values
                unchanged), contiguous 2D [num_tokens, top_k]; num_tokens /
                top_k >= 1 and top_k <= num_experts; E and B are __init__
                hyperparameters and >= 1, non powers of two allowed
      returns   a single 1-D float32 tensor of length L1 + L2 + 1, packed in
                three segments:
                out[:L1] = the L1 integers of sorted_token_ids (sentinel N
                           included)
                out[L1:L1+L2] = the L2 integers of expert_ids (tail -1)
                out[L1+L2] = num_tokens_post_pad (scalar S)
                -- all of them are non-negative / minus-one integers; the
                global-domain constraint L1 = N + E*(B-1) <= 2^24 guarantees
                that sentinel N, every index, every expert id and S <= L1 are
                exactly representable in fp32
      __init__ hyperparameters: num_experts (int), block_size (int).

    Implementation constraints (violations are scored as failures):
      - Core computation (bucket counting, block-count prefix sums, scatter
        writes of the layout, block expert table generation) must be done in a
        Triton kernel inside the submitted file; forward may only do input/
        output arrangement (dtype restoration, three-segment cat packing).
        Calling torch.sort / torch.argsort / torch.bincount / torch.cumsum /
        torch.histc / torch.unique / torch.repeat_interleave, or any
        third-party MoE/scheduling operator library, is forbidden.
      - The __init__ and forward signatures of ModelNew must not be changed.
      - Integer arithmetic throughout (no floating-point accumulation); the
        output is packed into float32 integers per the protocol above.

    Target hardware: Hygon DCU (gfx936). Implementation language: Triton.
    Task scope: the pure scheduling prologue (sorting / alignment / block table
    generation); no grouped GEMM body and no routing scores (softmax/topk are
    already done upstream).
    """

    def __init__(self, num_experts: int, block_size: int):
        super().__init__()
        self.num_experts = int(num_experts)
        self.block_size = int(block_size)

    def forward(self, topk_ids):
        # The KernelBench evaluator casts every input tensor to the evaluation
        # precision (fp32 for the triton backend); expert ids are small
        # integers and exactly representable in fp32, so restore them here
        # losslessly
        topk_ids = topk_ids.to(torch.long)
        E, B = self.num_experts, self.block_size
        assert int(topk_ids.min()) >= 0 and int(topk_ids.max()) < E, \
            "topk_ids must fall inside [0, num_experts)"

        N = topk_ids.numel()
        flat_ids = topk_ids.reshape(-1)

        counts = torch.bincount(flat_ids, minlength=E)          # |T_e|
        n_blocks = (counts + (B - 1)) // B                      # ceil(|T_e|/B)
        padded = n_blocks * B                                   # padded_e
        seg_start = torch.cumsum(counts, 0) - counts            # compact start
        slot_start = torch.cumsum(padded, 0) - padded           # aligned start
        S = int(padded.sum())                                   # num_tokens_post_pad
        total_blocks = int(n_blocks.sum())

        # Stable sort: expert id is the primary key, flattened index the
        # tie-break -- matches the bucket order of a per-token scan (ascending
        # i inside one expert)
        order = torch.argsort(flat_ids, stable=True)
        pos = torch.arange(N, dtype=torch.long, device=flat_ids.device)
        dst = pos - seg_start[flat_ids[order]] + slot_start[flat_ids[order]]

        L1 = N + E * (B - 1)
        L2 = (L1 + B - 1) // B
        sorted_ids = torch.full((L1,), N, dtype=torch.long, device=flat_ids.device)   # sentinel N prefill
        sorted_ids[dst] = order
        expert_ids = torch.full((L2,), -1, dtype=torch.long, device=flat_ids.device)  # tail -1 sentinel
        expert_ids[:total_blocks] = torch.repeat_interleave(
            torch.arange(E, device=flat_ids.device), n_blocks
        )

        # Single-tensor output protocol: [sorted_token_ids | expert_ids | num_tokens_post_pad]
        return torch.cat([
            sorted_ids.to(torch.float32),
            expert_ids.to(torch.float32),
            torch.tensor([S], dtype=torch.float32, device=flat_ids.device),
        ])


def get_init_inputs():
    return [16, 32]   # num_experts, block_size


def get_inputs():
    # Fixed shape family; the random part consumes the global RNG -- the
    # evaluator calls this function after set_seed, so multiple correctness
    # trials see diverse inputs. Scores are fp16 values, softmax is taken in
    # fp32 and then top_k is extracted (experts differ within a row, the same
    # distribution as MoE router output)
    num_tokens, num_experts, top_k = 256, 16, 4
    scores = torch.randn(num_tokens, num_experts).to(torch.float16)
    probs = torch.softmax(scores.to(torch.float32), dim=1)
    _, topk_ids = torch.topk(probs, k=top_k, dim=1)
    return [topk_ids]


def make_inputs(num_tokens: int, num_experts: int, top_k: int, block_size: int,
                ids_dtype: str = "int64", dist: str = "topk", seed: int = 0):
    """Generate inputs from a public/hidden/perf case description. Runs on the
    evaluation side only (generated on CPU; the evaluator moves them to the
    device). num_experts/block_size are used by the evaluation side to build
    the Model; here they only drive contract checks.

    dist="topk": fp16 randn scores, fp32 softmax, then top_k, experts differ
    within a row (the same distribution as the official test input generator);
    dist="uniform": uniform sampling, duplicates allowed within a row (used to
    stress the empty-expert / duplicate-expert boundaries). Consumption order
    is fixed: one randn or one randint, bit-reproducible under the same seed.
    """
    assert num_tokens >= 1 and num_experts >= 1 and block_size >= 1
    assert 1 <= top_k <= num_experts, "top_k must fall inside [1, num_experts]"
    assert dist in ("topk", "uniform")

    gen = torch.Generator().manual_seed(seed)
    if dist == "topk":
        scores = torch.randn(num_tokens, num_experts, generator=gen).to(torch.float16)
        probs = torch.softmax(scores.to(torch.float32), dim=1)
        _, topk_ids = torch.topk(probs, k=top_k, dim=1)
    else:
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), generator=gen)
    return [topk_ids.to(getattr(torch, ids_dtype))]
