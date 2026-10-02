# 3002_moe_align_block_size — MoE grouped GEMM 调度前置（token 按 expert
# 分桶 + block_size 对齐排布）的 model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=3002, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """MoE grouped GEMM 的调度前置：token 按 expert 分桶并按 block_size 对齐排布。

    输入 topk_ids（[num_tokens, top_k]，元素为专家编号 ∈ [0, num_experts)，
    行内允许重复）是路由器为每个 token 选出的专家。记展平下标
    i = t * top_k + j（t 为 token 行号、j 为列号），N = num_tokens * top_k，
    E = num_experts，B = block_size。对每个专家 e：
        T_e = { i : topk_ids 展平后第 i 个元素 == e }（按 i 升序）
        n_blocks_e = ceil(|T_e| / B)        # e 占用的对齐块数
        padded_e   = n_blocks_e * B         # e 对齐后的槽位数
    产出 grouped GEMM 调度三件套：
        sorted_token_ids  长度 L1 = N + E*(B-1)：按 e = 0..E-1 依次存放 T_e
          的全部下标（升序），随后 padded_e - |T_e| 个填充哨兵 N；缓冲区末尾
          [S, L1) 也为哨兵 N，其中 S = sum_e padded_e
        expert_ids        长度 L2 = ceil(L1/B)：按 e = 0..E-1 依次写 n_blocks_e
          个 e（共占前 S/B 个位置），尾部 [S/B, L2) 填 -1 哨兵
        num_tokens_post_pad  标量 S（填充后总槽位数，恒为 B 的倍数）
    grouped GEMM 以 sorted_token_ids 中每 B 个连续槽位为一个 M-tile、以
    expert_ids 为该 tile 的专家编号取权重、跳过值为哨兵 N 的填充槽位，
    num_tokens_post_pad 给出有效总长度。本题为该调度前置本身，不含 GEMM。
    边界行为：|T_e| = 0 的专家占 0 块、不产生任何 sorted 条目与 expert 条目；
    |T_e| 恰为 B 的倍数时不产生填充；同一行允许重复选中同一专家（展平下标
    不同即不同条目）；专家内条目次序按展平下标 i 升序。

    输入输出规格：
      topk_ids  int64/int32（或被评测器 cast 成的 fp32，整数值不变），连续
                2D [num_tokens, top_k]；num_tokens/top_k >= 1 且
                top_k <= num_experts；E、B 为 __init__ 超参且 >= 1，允许非
                2 次幂
      返回      单个一维 float32 张量，长度 L1 + L2 + 1，三段打包：
                out[:L1] = sorted_token_ids 的 L1 个整数（含哨兵 N）
                out[L1:L1+L2] = expert_ids 的 L2 个整数（尾部 -1）
                out[L1+L2] = num_tokens_post_pad（标量 S）
                ——全部为非负/负一整数；全域约束 L1 = N + E*(B-1) <= 2^24
                保证哨兵 N、各下标、专家编号与 S <= L1 在 fp32 中精确表示
      __init__ 超参：num_experts（int）、block_size（int）。

    实现约束（违规判负）：
      - 分桶计数、块数前缀和、散射写排布、块专家表生成等核心计算必须在
        提交文件内的 Triton kernel 中完成；forward 里只允许做输入/输出
        整理（dtype 恢复、三段 cat 打包）。禁止调用 torch.sort /
        torch.argsort / torch.bincount / torch.cumsum / torch.histc /
        torch.unique / torch.repeat_interleave，以及任何第三方 MoE/调度
        算子库。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 全程整数算术（无浮点累加）；输出按上述协议打包成 float32 的整数。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：纯调度前置（排序/对齐/块表生成），不含 grouped GEMM 本体、
    不含路由打分（softmax/topk 已在上游完成）。
    """

    def __init__(self, num_experts: int, block_size: int):
        super().__init__()
        self.num_experts = int(num_experts)
        self.block_size = int(block_size)

    def forward(self, topk_ids):
        # KernelBench 评测器会把全部输入张量 cast 成评测 precision（triton 后端
        # 为 fp32）；专家编号是小整数，fp32 可精确表示，此处无损恢复
        topk_ids = topk_ids.to(torch.long)
        E, B = self.num_experts, self.block_size
        assert int(topk_ids.min()) >= 0 and int(topk_ids.max()) < E, \
            "topk_ids 必须落在 [0, num_experts) 内"

        N = topk_ids.numel()
        flat_ids = topk_ids.reshape(-1)

        counts = torch.bincount(flat_ids, minlength=E)          # |T_e|
        n_blocks = (counts + (B - 1)) // B                      # ceil(|T_e|/B)
        padded = n_blocks * B                                   # padded_e
        seg_start = torch.cumsum(counts, 0) - counts            # 无填充紧排起始
        slot_start = torch.cumsum(padded, 0) - padded           # 对齐排布起始
        S = int(padded.sum())                                   # num_tokens_post_pad
        total_blocks = int(n_blocks.sum())

        # 稳定排序：专家编号为主键、展平下标为次序 —— 与逐 token 扫描的
        # 分桶次序一致（专家内按 i 升序）
        order = torch.argsort(flat_ids, stable=True)
        pos = torch.arange(N, dtype=torch.long)
        dst = pos - seg_start[flat_ids[order]] + slot_start[flat_ids[order]]

        L1 = N + E * (B - 1)
        L2 = (L1 + B - 1) // B
        sorted_ids = torch.full((L1,), N, dtype=torch.long)     # 哨兵 N 预填充
        sorted_ids[dst] = order
        expert_ids = torch.full((L2,), -1, dtype=torch.long)    # 尾部 -1 哨兵
        expert_ids[:total_blocks] = torch.repeat_interleave(
            torch.arange(E), n_blocks
        )

        # 单张量输出协议：[sorted_token_ids | expert_ids | num_tokens_post_pad]
        return torch.cat([
            sorted_ids.to(torch.float32),
            expert_ids.to(torch.float32),
            torch.tensor([S], dtype=torch.float32),
        ])


def get_init_inputs():
    return [16, 32]   # num_experts, block_size


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性。打分用 fp16 值、fp32 softmax
    # 后取 top_k（行内专家互不相同，与 MoE 路由输出同分布）
    num_tokens, num_experts, top_k = 256, 16, 4
    scores = torch.randn(num_tokens, num_experts).to(torch.float16)
    probs = torch.softmax(scores.to(torch.float32), dim=1)
    _, topk_ids = torch.topk(probs, k=top_k, dim=1)
    return [topk_ids]


def make_inputs(num_tokens: int, num_experts: int, top_k: int, block_size: int,
                ids_dtype: str = "int64", dist: str = "topk", seed: int = 0):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器
    搬运到设备）。num_experts/block_size 供评测端构造 Model 用，这里只做
    契约校验。

    dist="topk"：fp16 randn 打分、fp32 softmax 后取 top_k，行内专家互不
    相同（与官方测试的输入生成同分布）；dist="uniform"：均匀采样，行内
    允许重复（用于空专家/重复专家边界压力）。消费顺序固定：randn 一次或
    randint 一次，同 seed 下逐位可复现。
    """
    assert num_tokens >= 1 and num_experts >= 1 and block_size >= 1
    assert 1 <= top_k <= num_experts, "top_k 必须落在 [1, num_experts] 内"
    assert dist in ("topk", "uniform")

    gen = torch.Generator().manual_seed(seed)
    if dist == "topk":
        scores = torch.randn(num_tokens, num_experts, generator=gen).to(torch.float16)
        probs = torch.softmax(scores.to(torch.float32), dim=1)
        _, topk_ids = torch.topk(probs, k=top_k, dim=1)
    else:
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), generator=gen)
    return [topk_ids.to(getattr(torch, ids_dtype))]
