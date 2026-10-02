# 1021 —— 分页 KV decode attention 的长上下文（分区两 pass）变体：题面与评测真值。
#
# 本文件必须自包含：评测器以 exec(source) 加载题目，禁止本地模块导入，
# reference 实现与输入生成全部内联。语义以官方测试的 PyTorch 参考实现为准
# （单 token decode 分页 attention；长序列 max_seq_len > 8192 时生产调度
# 切换为"逐 1024-token 分区并行 + 跨分区归并"的两 pass 形态），中间计算
# float32，输出 cast 回输入 dtype。与已入库的 1002 分界：1002 取短上下文
# 单 pass 工况（max_seq_len <= 8192），本题取长上下文分区两 pass 工况，
# 变体子集互不重叠。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=1021, backend=triton。

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """长上下文 decode 阶段（单 token query）的分页 attention，MHA 与 GQA 通用。

    算子语义：KV Cache 以固定 block_size 分页存放于物理块池 key_cache /
    value_cache，每个序列通过 block_tables 中的物理块编号间接寻址；每个序列
    实际参与 attention 的 KV 长度由 seq_lens[s] 给出（>=1）；物理块池中未被
    引用的槽位、以及尾块中超出 seq_lens[s] 的槽位均为垃圾数据，不得参与
    计算。对序列 s 的第 h 个 query head（其 KV head 为 h // query_group_size，
    记 bs = block_size，bt = block_tables）：
      logits[l] = scale * <query[s,h,:], key_cache[bt[s,l//bs], h_kv, l%bs, :]>
      p = softmax(logits[0:seq_lens[s]])
      out[s,h,:] = sum_l p[l] * value_cache[bt[s,l//bs], h_kv, l%bs, :]
    softmax 为对整个 [0, seq_lens[s]) 区间的数值稳定在线实现。

    本题处于长上下文工况：块表容量 max_seq_len（固定族为 16384）远超 8192，
    单序列 KV 无法由单个计算单元在单趟内高效处理。生产实现把 KV 序列切成
    1024-token 的分区并行计算（flash-decoding 式 split-KV：逐分区在线
    softmax 得到局部最大 logit m_p、指数和 l_p 与部分输出 o_p，再跨分区以
    log-sum-exp 归并：M = max_p m_p，
    out = sum_p exp(m_p-M)*l_p*o_p / sum_p exp(m_p-M)*l_p），数学上与整体
    softmax 完全等价；分区只是取得并行度的手段，不作为判分条件，实现路径
    不限。核心边界行为：尾分区部分填充（seq_len 不是 1024 整数倍）、
    seq_len 恰为分区长度整数倍、同一 batch 内序列长度悬殊（短至 1）、
    尾块槽位掩蔽——超出 seq_len 的位置不得贡献任何概率质量。

    输入输出规格：
      query       [num_seqs, num_q_heads, head_size]，fp16/bf16
      key_cache   [num_blocks, num_kv_heads, block_size, head_size]，fp16/bf16
      value_cache 同 key_cache
      block_tables[num_seqs, max_blocks_per_seq] int32，逻辑块 -> 物理块编号
      seq_lens    [num_seqs] int32，每序列有效 KV 长度，1 <= seq_lens[s] <= max_seq_len
      输出 out    [num_seqs, num_q_heads, head_size]，dtype 与 query 一致

    实现约束（违规判负）：
      - 核心计算（QK^T / softmax / PV）必须在提交文件内完成，禁止调用
        scaled_dot_product_attention / sdpa / flash_attn 等外部 attention
        算子库，以及 torch.matmul / torch.bmm / torch.einsum / torch.softmax。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。
      - 未引用物理块与超出 seq_len 的槽位不得影响输出。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：单 token decode、非量化 KV、无 ALiBi、无滑窗；长上下文
    （max(seq_lens) > 8192）的分区两 pass 工况。
    """

    def __init__(self, head_size: int, scale=None):
        super().__init__()
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_size)

    def forward(self, query, key_cache, value_cache, block_tables, seq_lens):
        # 在线评测器会把全部输入张量 cast 成评测 precision（triton 后端为
        # fp32）；块编号与序列长度都是小整数，fp32 可精确表示，此处无损恢复
        block_tables = block_tables.to(torch.long)
        seq_lens = seq_lens.to(torch.int32)

        B, H_Q, D = query.shape
        H_KV = key_cache.shape[1]
        bs = key_cache.shape[2]
        group = H_Q // H_KV
        assert H_Q == H_KV * group, "num_q_heads 必须是 num_kv_heads 的整数倍"

        out = torch.empty_like(query)
        for s in range(B):
            seq_len = int(seq_lens[s])
            num_blocks = (seq_len + bs - 1) // bs
            physical = block_tables[s, :num_blocks]

            # 按块表收集本序列的 K/V：[num_blocks, H_KV, bs, D] -> [H_KV, seq_len, D]
            k = key_cache[physical].permute(1, 0, 2, 3).reshape(H_KV, num_blocks * bs, D)[:, :seq_len].to(torch.float32)
            v = value_cache[physical].permute(1, 0, 2, 3).reshape(H_KV, num_blocks * bs, D)[:, :seq_len].to(torch.float32)

            # GQA：同一 KV head 的 K/V 复制给 group 内的每个 query head
            if group > 1:
                k = k.repeat_interleave(group, dim=0)   # [H_Q, seq_len, D]
                v = v.repeat_interleave(group, dim=0)

            q = query[s].to(torch.float32) * self.scale      # [H_Q, D]
            logits = torch.einsum("hd,hld->hl", q, k)   # [H_Q, seq_len]
            probs = torch.softmax(logits, dim=-1)
            out[s] = torch.einsum("hl,hld->hd", probs, v).to(query.dtype)
        return out


def get_init_inputs():
    return [128]  # head_size；scale 缺省 1/sqrt(head_size)


def get_inputs():
    # 固定 shape 族（长上下文工况：max_seq_len=16384 > 8192，分区两 pass 调度
    # 在 max(seq_lens) > 8192 时触发）；随机部分消费全局 RNG——评测器在
    # set_seed 后调用本函数，多轮 correctness trial 因此获得输入多样性
    num_seqs = 4
    num_kv_heads = 8
    query_group_size = 4      # num_q_heads = 32
    head_size = 128
    block_size = 16
    max_seq_len = 16384

    num_q_heads = num_kv_heads * query_group_size
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    # 物理块池大于实际用量：未引用槽位是垃圾数据，用于检验掩码正确性
    num_blocks = num_seqs * max_blocks_per_seq + 8

    seq_lens = torch.randint(1, max_seq_len + 1, (num_seqs,), dtype=torch.int32)
    perm = torch.randperm(num_blocks)
    block_tables = perm[: num_seqs * max_blocks_per_seq].to(torch.int32).reshape(num_seqs, max_blocks_per_seq)

    query = torch.randn(num_seqs, num_q_heads, head_size).to(torch.float16)
    key_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_size).to(torch.float16)
    value_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_size).to(torch.float16)
    return [query, key_cache, value_cache, block_tables, seq_lens]


def make_inputs(num_seqs: int, num_q_heads: int, num_kv_heads: int,
                head_size: int, block_size: int, max_seq_len: int,
                dtype: str = "float16", seed: int = 0, seq_lens=None):
    """确定性 case 生成器（隐藏/性能案例共用）。仅在评测端运行（CPU 生成，
    评测器搬运到设备），随机性全部来自 torch.Generator().manual_seed(seed)。

    参数字段与 hidden/perf case 一一对应；seq_lens 可显式指定（用于精确构造
    分区/尾块边界），缺省时在 [1, max_seq_len] 均匀采样。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    if seq_lens is None:
        seq_lens = torch.randint(1, max_seq_len + 1, (num_seqs,), generator=gen, dtype=torch.int32)
    else:
        if len(seq_lens) == 1 and num_seqs > 1:   # 单值广播到整个 batch
            seq_lens = seq_lens * num_seqs
        seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        assert len(seq_lens) == num_seqs and int(seq_lens.max()) <= max_seq_len
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size

    # 物理块池大于实际用量：未引用槽位是"垃圾数据"，用于检验掩码正确性
    num_blocks = num_seqs * max_blocks_per_seq + 8
    perm = torch.randperm(num_blocks, generator=gen)
    block_tables = perm[: num_seqs * max_blocks_per_seq].reshape(num_seqs, max_blocks_per_seq).to(torch.int32)

    query = torch.randn(num_seqs, num_q_heads, head_size, generator=gen).to(dt)
    key_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_size, generator=gen).to(dt)
    value_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_size, generator=gen).to(dt)
    return query, key_cache, value_cache, block_tables, seq_lens
