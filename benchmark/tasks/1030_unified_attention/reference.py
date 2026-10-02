# 1030_unified_attention —— 统一（unified）变长 attention：同一入口混合处理
# prefill（每序列 q_s > 1）与 decode（q_s == 1）序列，KV 以固定 block_size
# 分页存放于物理块池，支持 GQA、滑动窗口、logit softcap 与 attention sinks。
# KernelBench 兼容形态（Model / get_inputs / get_init_inputs）。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=1030, backend=triton。

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """统一变长 attention：prefill 与 decode 序列混合的分页 paged attention。

    batch 内 num_seqs 个序列共享一个 packed 的 query 张量：第 s 个序列的
    query 连续存放在行区间 [cu_seqlens_q[s], cu_seqlens_q[s+1])，共
    q_s = cu_seqlens_q[s+1] - cu_seqlens_q[s] 行（q_s == 1 为 decode 序列，
    q_s > 1 为 prefill 序列，二者在 batch 内可任意混合）；该序列的 KV 长度为
    kv_s = seq_lens[s]，且保证 kv_s >= q_s。KV 不连续存放，而是分页寻址：
    逻辑 token 位置 j（0 <= j < kv_s）的物理地址为
      key_cache[block_tables[s, j // block_size], j % block_size, g, :]
      value_cache[block_tables[s, j // block_size], j % block_size, g, :]
    其中 g 为 KV head 编号。物理块池中未被引用的槽位是垃圾数据，不得参与
    计算；block_tables 每行只有前 ceil(kv_s / block_size) 个表项有效。

    对第 s 个序列的第 i 个 query（0 <= i < q_s，记其在 KV 序列中的绝对位置
    T = kv_s - q_s + i，行号 r = cu_seqlens_q[s] + i）与第 h 个 query head
    （对应 KV head 为 g = h // query_group_size，
    query_group_size = num_q_heads // num_kv_heads）：
      logits[j] = scale * <query[r, h, :], K_phys(j, g, :)>
                  scale 缺省 1 / sqrt(head_size)
      可见性（causal，query 与 KV 右对齐）：j <= T；若 sliding_window > 0，
      还需 T - j < sliding_window（窗口左边界）
      若 use_sinks=True：额外引入一个虚拟 sink 列，其 logit 恒为 sinks[h]，
      value 恒为零向量，且永不被掩码
      若 softcap > 0：全部 logit（含 sink 列）先做
      c * tanh(logit / c)（c = softcap）再做掩码与 softmax
      p = softmax(可见 logits)
      out[r, h, :] = sum_j p[j] * V_phys(j, g, :)   （sink 列贡献零向量）
    causal 右对齐保证每个 query 至少可见自身位置，softmax 分母恒 > 0。

    输入输出规格（评测 dtype 取 float16 / bfloat16，张量均连续）：
      query        评测 dtype (total_q, num_q_heads, head_size)
                   total_q = cu_seqlens_q[-1]
      key_cache    评测 dtype (num_blocks, block_size, num_kv_heads, head_size)
      value_cache  评测 dtype，shape 同 key_cache
      block_tables int32 (num_seqs, max_blocks_per_seq) 逻辑块->物理块编号
      seq_lens     int32 (num_seqs) 每序列有效 KV 长度 kv_s（>= q_s）
      cu_seqlens_q int32 (num_seqs + 1) query 长度前缀和，首元素为 0
      sinks        评测 dtype (num_q_heads) 每 head 的 sink logit
                   （use_sinks=False 时忽略本张量）
      返回 out     评测 dtype (total_q, num_q_heads, head_size)，与 query 行对齐

    实现约束（违规判负）：
      - 核心计算（QK^T / softmax / PV）必须在提交文件内完成，禁止调用
        scaled_dot_product_attention / sdpa / flash_attn / flex_attention /
        aiter / torch.matmul / torch.bmm / torch.einsum / torch.softmax。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。
      - 评测器会把全部输入张量 cast 成 fp32 后传入 forward；块表编号、
        序列长度与位置前缀和都是小整数，fp32 可精确表示，入口处
        .to(torch.int32) / .to(torch.long) 无损恢复。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：分页 KV + prefill/decode 混合 ragged batch + GQA + causal
    右对齐 + 滑动窗口 + logit softcap + attention sinks；无 ALiBi、无
    query-query bias、无多模态双向前缀、无 fp8 量化 KV 与反量化 scale，
    这些变体留作后续独立题。
    """

    def __init__(self, head_size: int, sliding_window: int = 0, softcap: float = 0.0, use_sinks: bool = True):
        super().__init__()
        self.scale = 1.0 / math.sqrt(head_size)
        self.sliding_window = int(sliding_window)
        self.softcap = float(softcap)
        self.use_sinks = bool(use_sinks)

    def forward(self, query, key_cache, value_cache, block_tables, seq_lens, cu_seqlens_q, sinks):
        # KernelBench 评测器会把全部输入张量 cast 成评测 precision（triton
        # 后端为 fp32）；块表编号与长度类小整数 fp32 可精确表示，此处无损恢复
        block_tables = block_tables.to(torch.long)
        seq_lens = seq_lens.to(torch.int32)
        cu_seqlens_q = cu_seqlens_q.to(torch.int32)

        num_seqs = seq_lens.shape[0]
        block_size = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        head_size = key_cache.shape[3]
        num_q_heads = query.shape[1]
        group = num_q_heads // num_kv_heads
        assert num_q_heads == num_kv_heads * group, \
            "num_q_heads 必须是 num_kv_heads 的整数倍"

        outputs = []
        for s in range(num_seqs):
            start = int(cu_seqlens_q[s])
            q_len = int(cu_seqlens_q[s + 1]) - start
            kv_len = int(seq_lens[s])

            q = query[start:start + q_len].to(torch.float32) * self.scale

            # 按块表收集本序列的 K/V：[nblk, bs, H_KV, D] -> [kv_len, H_KV, D]
            num_kv_blocks = (kv_len + block_size - 1) // block_size
            physical = block_tables[s, :num_kv_blocks]
            k = key_cache[physical].reshape(
                num_kv_blocks * block_size, num_kv_heads, head_size
            )[:kv_len].to(torch.float32)
            v = value_cache[physical].reshape(
                num_kv_blocks * block_size, num_kv_heads, head_size
            )[:kv_len].to(torch.float32)

            # GQA：同一 KV head 的 K/V 复制给 group 内每个 query head
            if group > 1:
                k = k.repeat_interleave(group, dim=1)   # [kv_len, H_Q, D]
                v = v.repeat_interleave(group, dim=1)

            attn = torch.einsum("qhd,khd->hqk", q, k)   # [H_Q, q_len, kv_len]

            if self.use_sinks:
                # sink 列：logit 恒为 sinks[h]，value 恒为零向量
                sink_logits = sinks[:num_q_heads].to(torch.float32)[:, None, None]
                sink_logits = sink_logits.expand(-1, q_len, 1)
                attn = torch.cat([sink_logits, attn], dim=-1)
                v = torch.cat([torch.zeros(1, num_q_heads, head_size,
                                           dtype=v.dtype, device=v.device),
                               v], dim=0)

            # causal 掩码（query 与 KV 右对齐）：j - i > kv_len - q_len 不可见
            mask = torch.triu(torch.ones(q_len, kv_len, dtype=torch.bool,
                                         device=attn.device),
                              diagonal=kv_len - q_len + 1)
            if self.sliding_window > 0:
                sw_mask = ~torch.triu(
                    torch.ones(q_len, kv_len, dtype=torch.bool,
                               device=attn.device),
                    diagonal=kv_len - (q_len + self.sliding_window) + 1)
                mask = mask | sw_mask
            if self.softcap > 0:
                attn = self.softcap * torch.tanh(attn / self.softcap)
            if self.use_sinks:
                mask = torch.cat([torch.zeros(q_len, 1, dtype=torch.bool,
                                              device=mask.device), mask], dim=-1)

            attn = attn.masked_fill(mask, float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            outputs.append(torch.einsum("hqk,khd->qhd", attn, v))

        return torch.cat(outputs, dim=0).to(query.dtype)


def get_init_inputs():
    # head_size / sliding_window / softcap / use_sinks，与 get_inputs 的
    # shape 族一一对应（滑窗 64、softcap 50.0、启用 sinks）
    return [128, 64, 50.0, True]


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    num_seqs = 4
    num_kv_heads = 2
    query_group_size = 4      # num_q_heads = 8
    head_size = 128
    block_size = 16
    max_seq_len = 512

    num_q_heads = num_kv_heads * query_group_size
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    # 物理块池大于实际用量：未引用槽位是垃圾数据，用于检验掩码正确性
    num_blocks = num_seqs * max_blocks_per_seq + 8

    # 固定混合模式：2 个 decode（q=1）+ 2 个 prefill（q=24 / q=9）
    query_lens = torch.tensor([1, 24, 1, 9], dtype=torch.int32)
    ctx = torch.randint(0, max_seq_len - 24 + 1, (num_seqs,), dtype=torch.int32)
    seq_lens = (query_lens + ctx).to(torch.int32)

    perm = torch.randperm(num_blocks)
    block_tables = perm[: num_seqs * max_blocks_per_seq].to(torch.int32) \
        .reshape(num_seqs, max_blocks_per_seq)
    cu_seqlens_q = torch.tensor([0] + query_lens.tolist(),
                                dtype=torch.int32).cumsum(0, dtype=torch.int32)

    total_q = int(cu_seqlens_q[-1])
    query = torch.randn(total_q, num_q_heads, head_size).to(torch.bfloat16)
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads,
                            head_size).to(torch.bfloat16)
    value_cache = torch.randn_like(key_cache)
    sinks = torch.randn(num_q_heads).to(torch.bfloat16)
    return [query, key_cache, value_cache, block_tables, seq_lens,
            cu_seqlens_q, sinks]


def make_inputs(num_seqs: int, num_q_heads: int, num_kv_heads: int,
                head_size: int, block_size: int, max_seq_len: int,
                sliding_window: int = 0, softcap: float = 0.0,
                use_sinks: bool = True, dtype: str = "bfloat16",
                seed: int = 0, query_lens=None, kv_lens=None, name=None):
    """确定性 case 生成器（torch.Generator().manual_seed(seed)），CPU 生成。

    字段与 hidden/perf case 一一对应；sliding_window / softcap / use_sinks /
    name 是 init 侧配置（构题时传给 Model.__init__），不参与张量生成；
    sinks 张量始终生成（forward 入参个数固定），use_sinks=False 时被忽略。
    query_lens 缺省为混合模式（偶数位 decode q=1，奇数位 prefill q=17）；
    kv_lens 缺省在 [q_s, max_seq_len] 内采样；二者均可用单值广播到整个
    batch。仅在评测端运行（CPU 生成，评测器搬运到设备）。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert num_q_heads % num_kv_heads == 0, \
        "num_q_heads 必须是 num_kv_heads 的整数倍"

    if query_lens is None:
        query_lens = [1 if i % 2 == 0 else 17 for i in range(num_seqs)]
    if len(query_lens) == 1 and num_seqs > 1:
        query_lens = list(query_lens) * num_seqs
    assert len(query_lens) == num_seqs
    query_lens_t = torch.tensor(query_lens, dtype=torch.int32)

    if kv_lens is None:
        max_q = int(query_lens_t.max())
        ctx = torch.randint(0, max_seq_len - max_q + 1, (num_seqs,),
                            generator=gen, dtype=torch.int32)
        kv_lens_t = query_lens_t + ctx
    else:
        if len(kv_lens) == 1 and num_seqs > 1:
            kv_lens = list(kv_lens) * num_seqs
        kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
    assert len(kv_lens_t) == num_seqs
    assert bool((kv_lens_t >= query_lens_t).all()), "kv_len 必须 >= query_len"
    assert int(kv_lens_t.max()) <= max_seq_len

    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    # 物理块池大于实际用量：未引用槽位是垃圾数据，用于检验掩码正确性
    num_blocks = num_seqs * max_blocks_per_seq + 8
    perm = torch.randperm(num_blocks, generator=gen)
    block_tables = perm[: num_seqs * max_blocks_per_seq] \
        .reshape(num_seqs, max_blocks_per_seq).to(torch.int32)

    cu_seqlens_q = torch.cat([
        torch.zeros(1, dtype=torch.int32),
        query_lens_t.cumsum(0, dtype=torch.int32)])
    total_q = int(cu_seqlens_q[-1])

    query = torch.randn(total_q, num_q_heads, head_size, generator=gen).to(dt)
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size,
                            generator=gen).to(dt)
    value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size,
                              generator=gen).to(dt)
    sinks = torch.randn(num_q_heads, generator=gen).to(dt)
    return query, key_cache, value_cache, block_tables, kv_lens_t, \
        cu_seqlens_q, sinks
