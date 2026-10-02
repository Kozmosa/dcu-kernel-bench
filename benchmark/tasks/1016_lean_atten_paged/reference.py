# 1016_lean_atten_paged — 分页 KV Cache 的 ragged batch decode attention（KernelBench 兼容题目文件）。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=1016, backend=triton。

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """分页 KV Cache 的批量 decode attention：ragged batch，每请求 16 个 query，
    每个 head 一张独立的物理块表，物理块池中未被引用的块是垃圾数据。

    KV Cache 按固定 page_size 分页：k / v 的第 h 个 head 切片（形状
    [total_kv_len, head_size]）被划分成 total_kv_len / page_size 个物理块；
    第 h 个 head 的块表 kv_block_tables[h] 依次列出各请求引用的物理块编号，
    其中请求 b 引用的块为
      kv_block_tables[h, batch_num_block_n[b-1] : batch_num_block_n[b]]
    （b=0 时从 0 开始；batch_num_block_n 是每请求块数的累积和，末元素即
    全部已用块数 total_used_blocks）。未被任何表项引用的物理块为垃圾数据，
    不得参与计算。

    数学定义：记请求 b 的起始块偏移 s_b（b=0 为 0，否则 s_b =
    batch_num_block_n[b-1]），其逻辑 KV 长度 L_b = 块数 * page_size，
    第 j 个逻辑位置映射到物理行 row(j) = kv_block_tables[h, s_b + j //
    page_size] * page_size + j % page_size。对每个 head h 与请求 b
    （query 行区间 [b*n_ctx_q, (b+1)*n_ctx_q)）：
      scores[i, j] = sm_scale * <q[h, b*n_ctx_q+i, :], k[h, row(j), :]>
      p[i, :] = softmax_j(scores[i, :])
      out[h, b*n_ctx_q+i, :] = sum_j p[i, j] * v[h, row(j), :]
    softmax 采用数值稳定（在线）实现，中间累加为 float32，输出 cast 回输入
    dtype。边界行为：每条请求的 KV 长度是 page_size 的正整数倍（尾部不满的
    半块不在题面内），softmax 分母恒正；请求之间、head 之间互不影响。

    输入输出规格：
      q                 [num_heads, n_ctx_q * batch, head_size] float16/bfloat16
      k, v              [num_heads, total_kv_len, head_size] 与 q 同 dtype（物理块池）
      kv_block_tables   [num_heads, total_used_blocks] int32（0-based 物理块编号，
                        值 < total_kv_len / page_size）
      batch_num_block_n [batch] int32，每请求块数的累积和，单调递增
      out               [num_heads, n_ctx_q * batch, head_size] 与 q 同 dtype
    全域约束：head_size ∈ {16, 32, 64, 128, 256}；page_size 固定为 64；
    n_ctx_q = 16（decode 形态固定）；每请求 KV 长度为 page_size 的正整数倍
    且 >= page_size；num_kv_heads == num_q_heads（MHA，无 GQA）。

    实现约束（违规判负）：
      - 核心计算（分页 KV 收集 / QK^T / softmax / PV）必须在提交文件内完成，
        禁止调用 scaled_dot_product_attention / sdpa / flash_attn 等外部
        attention 算子库，以及 torch.matmul / torch.bmm / torch.mm /
        torch.baddbmm / torch.einsum / torch.softmax（含 F.softmax、
        Tensor.matmul 等 ATen 等价捷径与 @ 运算符）。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。
      - 调度策略（persistent 单 kernel 或常规 grid）自由选择，只考核语义
        正确性与性能。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：ragged batch decode、非量化 fp16/bf16、无因果掩码、无
    ALiBi/滑窗、无 GQA。
    """

    def __init__(self, head_size: int, page_size: int = 64, sm_scale=0.5):
        super().__init__()
        self.head_size = int(head_size)
        self.page_size = int(page_size)
        self.sm_scale = float(sm_scale)

    def forward(self, q, k, v, kv_block_tables, batch_num_block_n):
        # KernelBench 评测器会把全部输入张量 cast 成评测 precision（triton 后端
        # 为 fp32）；块编号与累积块数都是小整数，fp32 可精确表示，此处无损恢复
        kv_block_tables = kv_block_tables.to(torch.long)
        batch_num_block_n = batch_num_block_n.to(torch.int32)

        num_heads, m_total, head_dim = q.shape
        batch = int(batch_num_block_n.numel())
        n_ctx_q = m_total // batch
        page = self.page_size
        assert num_heads == k.shape[0] == v.shape[0], "q/k/v 的 head 数必须一致"
        assert k.shape[1] == v.shape[1], "k/v 的 KV 池长度必须一致"

        offs_in_block = torch.arange(page, device=q.device)
        out = torch.empty_like(q)
        for h in range(num_heads):
            for b in range(batch):
                s0 = 0 if b == 0 else int(batch_num_block_n[b - 1])
                s1 = int(batch_num_block_n[b])
                blocks = kv_block_tables[h, s0:s1]                     # [num_blocks_b]
                # 逻辑位置 j -> 物理行：row(j) = table[s0 + j//page]*page + j%page
                rows = (blocks[:, None] * page + offs_in_block[None, :]).reshape(-1)
                kb = k[h].index_select(0, rows).to(torch.float32)      # [L_b, D]
                vb = v[h].index_select(0, rows).to(torch.float32)      # [L_b, D]
                qb = q[h, b * n_ctx_q:(b + 1) * n_ctx_q, :].to(torch.float32)
                scores = torch.matmul(qb, kb.transpose(0, 1)) * self.sm_scale
                probs = torch.softmax(scores, dim=-1)                  # [n_ctx_q, L_b]
                out[h, b * n_ctx_q:(b + 1) * n_ctx_q, :] = torch.matmul(probs, vb).to(q.dtype)
        return out


def get_init_inputs():
    return [64, 64, 0.5]   # head_size, page_size（题面固定 64）, sm_scale（官方恒 0.5）


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    num_heads = 32
    batch = 4
    n_ctx_q = 16
    head_size = 64
    page_size = 64         # 题面固定
    extra_blocks = 4       # 物理块池大于实际用量：未引用槽位是垃圾数据

    ctx_lens = (torch.randint(1, 17, (batch,)) * page_size)   # 每请求 1..16 块
    blocks_per = ctx_lens // page_size
    used_blocks = int(blocks_per.sum())
    pool_blocks = used_blocks + extra_blocks
    # 每个 head 一张独立块表：取 pool 的随机排列前 used_blocks 个
    kv_block_tables = torch.stack(
        [torch.randperm(pool_blocks)[:used_blocks] for _ in range(num_heads)]
    ).to(torch.int32)
    batch_num_block_n = torch.cumsum(blocks_per, dim=0).to(torch.int32)

    total_kv_len = pool_blocks * page_size
    q = (torch.randn(num_heads, n_ctx_q * batch, head_size) * 0.5).to(torch.float16)
    k = (torch.randn(num_heads, total_kv_len, head_size) * 0.5).to(torch.float16)
    v = (torch.randn(num_heads, total_kv_len, head_size) * 0.5).to(torch.float16)
    return [q, k, v, kv_block_tables, batch_num_block_n]


def make_inputs(num_heads: int, batch: int, n_ctx_q: int, head_size: int,
                ctx_lens=None, dtype: str = "float16", extra_blocks: int = 0,
                page_size: int = 64, seed: int = 0, seq_lens=None):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    ctx_lens（或其别名 seq_lens——离线终审 harness 的惯例形参，二者等价）可
    显式指定（每请求 KV 长度，须为 page_size 的正整数倍）；两者均缺省时每
    请求随机取 1..16 个块。extra_blocks > 0 时物理块池追加等量垃圾块。
    题面固定 page_size=64。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    if ctx_lens is None:
        ctx_lens = seq_lens
    if ctx_lens is None:
        ctx_lens = (torch.randint(1, 17, (batch,), generator=gen) * page_size).tolist()
    assert len(ctx_lens) == batch, "ctx_lens 长度必须等于 batch"
    for s in ctx_lens:
        assert s > 0 and s % page_size == 0, "每请求 KV 长度必须是 page_size 的正整数倍"

    blocks_per = torch.tensor([s // page_size for s in ctx_lens], dtype=torch.int64)
    used_blocks = int(blocks_per.sum())
    pool_blocks = used_blocks + int(extra_blocks)
    kv_block_tables = torch.stack(
        [torch.randperm(pool_blocks, generator=gen)[:used_blocks] for _ in range(num_heads)]
    ).to(torch.int32)
    batch_num_block_n = torch.cumsum(blocks_per, dim=0).to(torch.int32)

    total_kv_len = pool_blocks * page_size
    q = (torch.randn(num_heads, n_ctx_q * batch, head_size, generator=gen) * 0.5).to(dt)
    k = (torch.randn(num_heads, total_kv_len, head_size, generator=gen) * 0.5).to(dt)
    v = (torch.randn(num_heads, total_kv_len, head_size, generator=gen) * 0.5).to(dt)
    return q, k, v, kv_block_tables, batch_num_block_n
