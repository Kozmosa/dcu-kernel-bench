# aiter_impl.py — 3002_moe_align_block_size 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) -> (out, ctx)。
#
# 来源（本地 pinned 检出 .dcu_runs/aiter_pinned，commit c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#   aiter/ops/triton/moe_align_block_size.py
#     sha256 f950b961fff08ab75e3f29ac4c4788d0096a200883c7b949cf2b9350b485b279
#     （与 benchmark/sources/3002_moe_align_block_size.yaml 记录的哈希一致）
#   准入声明的核心计算即该文件的四个 stage kernel：
#     _moe_align_block_size_stage1_kernel（分桶计数，module:13）
#     _moe_align_block_size_stage2_kernel（chunk 内前缀和，module:34）
#     _moe_align_block_size_stage3_kernel（按 block_size 对齐的专家槽位前缀和
#       + num_tokens_post_pad，module:48）
#     _moe_align_block_size_stage4_kernel（散射写 sorted 排布 + 块专家表，module:65）
#
# 公开 host 入口（module:94，本适配器唯一调用的入口）：
#
#   moe_align_block_size_triton(
#       topk_ids: torch.Tensor,        # [num_tkns, top_k]，展平后按元素分桶
#       num_experts: int,
#       block_size: int,
#       sorted_token_ids: torch.Tensor,   # 输出，[L1]，调用前须填哨兵 numel
#       expert_ids: torch.Tensor,         # 输出，[L2]
#       num_tokens_post_pad: torch.Tensor,  # 输出，(1,)，int32
#   ) -> None
#
# 该入口是 **void + 三个预分配输出**，与题面「单个一维 float32 张量三段打包」的
# 契约不同形，因此在本适配器内自行分配输出缓冲、按题面协议打包（见下），
# 不改动 aiter 侧任何逻辑。官方测试 op_tests/triton_tests/test_moe_align_block_size.py
# 的 wrapper `triton_moe_align_block_size`（test:109-132）就是同样的"分配三缓冲 →
# 调 void 入口 → 返回三件套"用法，本适配器与之逐行同构，只是最后把三件套按题面
# 协议 cat 成一个 float32 张量。
#
# 语义要点（与 reference.py 逐位对齐的三处）：
#   1. sorted_token_ids 长度 L1 = numel + E*(B-1)，调用前必须整体填哨兵
#      numel（= num_tokens*top_k）。官方 wrapper 用 `sorted_ids.fill_(topk_ids.numel())`
#      （test:116），本适配器用等价的 torch.full 预填充。
#   2. expert_ids 长度 L2 = ceil(L1/B)。aiter 的 stage4 只写前 S/B 个位置
#      （每个专家写 n_blocks_e 个），尾部**不写**（官方 wrapper 里是
#      torch.empty 的未定义内存）。题面把尾部定义为 -1 哨兵
#      （reference.py:101 与 task.yaml io.invariants），故这里以 -1 预填充；
#      这是准入记录 admission.note 差异 (1) 明确登记的题面收严，aiter 写入的
#      前 S/B 个位置不受影响。
#   3. num_tokens_post_pad 由 stage3 写为标量 S（int32），题面把 S 放在打包
#      末尾一个元素。
#
# 布局：topk_ids 按行主序连续展平后使用（kernel 内为纯展平线性索引），题面
# io.inputs 亦声明 contiguous，故仅做一次 .contiguous() 保险 + 非整数 dtype
# 的整数恢复（KernelBench 在线路径会把输入 cast 成 fp32，专家编号是精确整数，
# 与 reference.py:75 的 `topk_ids.to(torch.long)` 同义）。
#
# 无 autotune 依赖：本模块四个 kernel 均为 @triton.jit（无 @triton.autotune，
# 也不读 AITER_TRITON_CONFIGS_PATH），缺 config JSON 不会失败或退化。
#
# 计时口径说明：记录基线时 run() 每次调用都会重新分配三个输出并打包，这部分
# 开销与题面单张量协议同源（reference 的 forward 同样 cat 打包），未额外加入
# 任何输入校验扫描（整型 range 校验会引入 16M 元素的全量归约，污染 perf 计时；
# 输入合法性由 make_inputs 契约与 reference 的 assert 负责）。

import torch


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方 moe_align_block_size。

    inputs      : [topk_ids]（单输入）。topk_ids 为 [num_tokens, top_k] 的
                  int64/int32（或可无损转回的整数 fp32）连续张量，元素 ∈ [0, E)。
    init_kwargs : {"num_experts": int, "block_size": int}（Model.__init__ 同名参数）。

    返回 (out, ctx)：
      out = cat([sorted_token_ids(L1), expert_ids(L2), [S]]) -> float32，长度
            L1 + L2 + 1，与 reference.py::Model.forward 的打包逐位同形同 dtype。
      ctx = 走了哪条 aiter 路径与关键 shape。
    """
    # aiter 顶层 import 很重，按部署约定在函数内最小导入
    from aiter.ops.triton.moe_align_block_size import moe_align_block_size_triton

    (topk_ids,) = inputs

    num_experts = int(init_kwargs.get("num_experts", 0))
    block_size = int(init_kwargs.get("block_size", 0))
    if num_experts < 1 or block_size < 1:
        raise ValueError(
            f"init_kwargs 非法：num_experts={num_experts}, block_size={block_size}"
            "（题面要求均 >= 1；aiter 的 grid=(num_experts,) 与 block_size 步长"
            "在越界时会给出无意义结果）"
        )

    if topk_ids.dim() != 2:
        raise ValueError(f"topk_ids 必须是 2D [num_tokens, top_k]，实得 {tuple(topk_ids.shape)}")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        # KernelBench 在线路径可能把输入 cast 成 fp32；专家编号是小整数，无损转回
        topk_ids = topk_ids.to(torch.long)
    topk_ids = topk_ids.contiguous()

    top_k = int(topk_ids.shape[1])
    if top_k > num_experts:
        raise ValueError(f"top_k={top_k} 超出 num_experts={num_experts}（题面约束 top_k <= num_experts）")

    numel = topk_ids.numel()                                    # N = num_tokens * top_k
    L1 = numel + num_experts * (block_size - 1)                 # sorted_token_ids 长度
    L2 = -(-L1 // block_size)                                   # ceil(L1 / block_size)
    if L1 > (1 << 24):
        raise ValueError(f"L1={L1} 超出题面全域约束 2^24（fp32 精确表示整数上界）")

    # 预填充哨兵：sorted 全域填 N，expert_ids 全域填 -1（aiter 只覆盖前 S/B 个）
    sorted_token_ids = torch.full(
        (L1,), numel, dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.full(
        (L2,), -1, dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty(
        (1,), dtype=torch.int32, device=topk_ids.device
    )

    # aiter 官方入口：void，四个 stage kernel 全部由它串行 launch
    moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
    )
    torch.cuda.synchronize()

    # 题面单张量协议：[sorted_token_ids | expert_ids | num_tokens_post_pad]
    out = torch.cat([
        sorted_token_ids.to(torch.float32),
        expert_ids.to(torch.float32),
        num_tokens_post_pad.to(torch.float32),
    ])

    ctx = {
        "aiter_path": "aiter.ops.triton.moe_align_block_size_triton (stage1-4 triton)",
        "num_tokens": int(topk_ids.shape[0]),
        "top_k": top_k,
        "num_experts": num_experts,
        "block_size": block_size,
        "numel": numel,
        "L1": L1,
        "L2": L2,
        "out_len": L1 + L2 + 1,
    }
    return out, ctx
