# 1002_paged_attention — 1001 的 model_class（KernelBench 兼容）变体。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。计算逻辑与 1001 的 reference.py
# 逐行一致（tests/test_1002_model_class.py 校验两者输出相等）。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=1002, backend=triton。

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """Decode 阶段（单 token query）的 paged attention，支持 MHA 与 GQA。

    KV Cache 以固定 block_size 分页存放于物理块池 key_cache / value_cache，
    每个序列通过 block_tables 中的物理块编号间接寻址；每个序列实际参与
    attention 的 KV 长度由 seq_lens[s] 给出，物理块池中其余槽位为垃圾数据，
    不得参与计算。对每个序列 s 的第 h 个 query head（其 KV head 为
    h // query_group_size）：
      logits[l] = scale * <query[s,h,:], key_cache[bt[s,l//bs], h_kv, l%bs, :]>
      p = softmax(logits[0:seq_lens[s]])
      out[s,h,:] = sum_l p[l] * value_cache[bt[s,l//bs], h_kv, l%bs, :]
    softmax 采用在线（数值稳定）实现，中间累加为 float32，输出 cast 回输入
    dtype。

    实现约束（违规判负）：
      - 核心计算（QK^T / softmax / PV）必须在提交文件内完成，禁止调用
        scaled_dot_product_attention / sdpa / flash_attn / aiter /
        torch.matmul / torch.bmm / torch.einsum / torch.softmax。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：单 token decode、非量化 KV、无 ALiBi、无滑窗。
    """

    def __init__(self, head_size: int, scale=None):
        super().__init__()
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_size)

    def forward(self, query, key_cache, value_cache, block_tables, seq_lens):
        # KernelBench 评测器会把全部输入张量 cast 成评测 precision（triton 后端
        # 为 fp32）；块编号与序列长度都是小整数，fp32 可精确表示，此处无损恢复
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
    return [64]  # head_size；scale 缺省 1/sqrt(head_size)


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    num_seqs = 8
    num_kv_heads = 8
    query_group_size = 4      # num_q_heads = 32
    head_size = 64
    block_size = 16
    max_seq_len = 1024

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
