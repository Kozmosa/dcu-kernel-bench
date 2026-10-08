# 1029_triton_decode_attention —— 两阶段 GQA flash decoding（单 token
# decode、分页 KV Cache、split-KV 两阶段归约）的自包含 KernelBench 题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=1029, backend=triton。

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """Decode 阶段（单 token query）的两阶段 flash decoding 注意力，支持分页 KV。

    KV Cache 以 page_size 粒度分页：key_cache / value_cache 第 p 个物理页的第
    s 个槽位对应扁平槽号 p * page_size + s。序列 b 通过 page_tables 间接寻址，
    其第 l 个 KV token（0 <= l < seq_lens[b]）位于物理页
    page_tables[b, l // page_size] 的槽位 l % page_size；页池中未被引用的槽位
    是垃圾数据，不得影响输出。query head h 对应 KV head h // query_group_size
    （query_group_size = num_q_heads // num_kv_heads，等于 1 即 MHA，等于
    num_q_heads 即 MQA）。对每个序列 b、每个 query head h：
      logits[l] = scale * <q[b,h,:head_dim_qk],
                            key_flat[slot(b,l), h//group, :head_dim_qk]>
      p[l]      = exp(logits[l] - max(logits)) / sum(exp(logits - max(logits)))
      out[b,h,:head_dim_v] = sum_l p[l] * value_flat[slot(b,l), h//group, :head_dim_v]
    其中 key_flat / value_flat 是页池的 [num_pages*page_size, H_KV, D] 扁平视图，
    scale 缺省 1/sqrt(head_dim_qk)，softmax 需数值稳定（在线 / 分块 max 减除）。

    参考结构（非强制，但它是长序列高并行度的关键）：两阶段 split-KV——
    stage1 把每条序列的 KV 长度切成 num_kv_splits 个连续切片，每个
    (batch, q_head, split) 并行块对切片做在线 softmax，产出部分加权 V（acc/e_sum）
    与部分 logsumexp（e_max + log(e_sum)）两组中间量；stage2 跨切片做 max 归约
    合并得到最终输出。num_kv_splits 只影响并行度，不影响数值结果；任何在容差内
    等价的实现均可。

    边界行为：
      - 仅前 seq_lens[b] 个 KV token 参与计算；seq_lens[b] >= 1（空序列无定义）。
      - page_size >= 1 任意取值（page_size == 1 时 page_tables 即逐 token 槽位表）。
      - head_dim_qk 与 head_dim_v 可以不同，且允许非 2 次幂（如 192/128、
        576/512）；不足 2 次幂的通道按掩码处理，不产生越界访问。
      - 页池可大于实际用量；page_tables 中超出 ceil(seq_lens[b]/page_size) 的
        表项指向有效但不属于本序列的垃圾页。

    输入：
      q           [num_seqs, num_q_heads, head_dim_qk]  fp16/bf16 —— 每 batch
                  一个新 token 的 query
      key_cache   [num_pages, page_size, num_kv_heads, head_dim_qk]
      value_cache [num_pages, page_size, num_kv_heads, head_dim_v]
      page_tables [num_seqs, max_pages_per_seq] int32 —— 逻辑页 -> 物理页编号
      seq_lens    [num_seqs] int32 —— 每序列实际有效 KV 长度
    输出：
      out [num_seqs, num_q_heads, head_dim_v]，dtype 与 q 一致。

    实现约束（违规判负）：
      - 核心计算（QK^T / softmax / PV / 跨切片归约）必须在提交文件内完成，
        禁止调用 scaled_dot_product_attention / sdpa / flash_attn 等外部
        attention 算子库，以及 torch.matmul / torch.bmm / torch.einsum /
        torch.softmax。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：单 token decode、非量化 KV、无 logit_cap 截断、无 ALiBi、无滑窗。
    """

    def __init__(self, page_size: int, head_dim_qk: int, head_dim_v: int, num_kv_splits: int = 8, scale=None):
        super().__init__()
        self.page_size = int(page_size)
        self.head_dim_qk = int(head_dim_qk)
        self.head_dim_v = int(head_dim_v)
        self.num_kv_splits = int(num_kv_splits)
        self.scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim_qk)

    def forward(self, q, key_cache, value_cache, page_tables, seq_lens):
        # KernelBench 评测器会把全部输入张量 cast 成评测 precision（triton 后端
        # 为 fp32）；页表与序列长度都是小整数，fp32 可精确表示，此处无损恢复
        page_tables = page_tables.to(torch.long)
        seq_lens = seq_lens.to(torch.int32)

        B, H_Q, D_QK = q.shape
        num_pages, page_size, H_KV, _ = key_cache.shape
        D_V = value_cache.shape[-1]
        group = H_Q // H_KV
        assert H_Q == H_KV * group, "num_q_heads 必须是 num_kv_heads 的整数倍"
        assert page_size == self.page_size, "key_cache 页大小与初始化 page_size 不一致"

        # 页池扁平视图：物理页 p 槽 s -> 扁平槽号 p*page_size+s
        key_flat = key_cache.view(num_pages * page_size, H_KV, D_QK)
        value_flat = value_cache.view(num_pages * page_size, H_KV, D_V)

        out = torch.empty(B, H_Q, D_V, dtype=q.dtype, device=q.device)
        for b in range(B):
            seq_len = int(seq_lens[b])
            offs = torch.arange(seq_len, device=q.device)
            slots = page_tables[b, offs // page_size] * page_size + offs % page_size

            # 按页表收集本序列的 K/V：[seq_len, H_KV, D]，gather 后升 float32
            k = key_flat[slots].to(torch.float32)
            v = value_flat[slots].to(torch.float32)

            # GQA：同一 KV head 的 K/V 复制给 group 内连续的 query heads
            if group > 1:
                k = k.repeat_interleave(group, dim=1)   # [seq_len, H_Q, D_QK]
                v = v.repeat_interleave(group, dim=1)   # [seq_len, H_Q, D_V]

            qf = q[b].to(torch.float32) * self.scale     # [H_Q, D_QK]
            logits = torch.einsum("hd,lhd->hl", qf, k)   # [H_Q, seq_len]
            probs = torch.softmax(logits, dim=-1)
            out[b] = torch.einsum("hl,lhd->hd", probs, v).to(q.dtype)
        return out


def get_init_inputs():
    # page_size / head_dim_qk / head_dim_v / num_kv_splits；
    # scale 缺省 1/sqrt(head_dim_qk)
    return [16, 192, 128, 8]


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    num_seqs = 4
    num_kv_heads = 8
    query_group_size = 4      # num_q_heads = 32 -> GQA（grouped 路径）
    head_dim_qk = 192
    head_dim_v = 128
    page_size = 16
    max_seq_len = 1024

    num_q_heads = num_kv_heads * query_group_size
    max_pages_per_seq = (max_seq_len + page_size - 1) // page_size
    # 物理页池大于实际用量：未引用槽位是垃圾数据，用于检验掩码正确性
    num_pages = num_seqs * max_pages_per_seq + 8

    seq_lens = torch.randint(1, max_seq_len + 1, (num_seqs,), dtype=torch.int32)
    perm = torch.randperm(num_pages)
    page_tables = perm[: num_seqs * max_pages_per_seq].to(torch.int32).reshape(num_seqs, max_pages_per_seq)

    q = torch.randn(num_seqs, num_q_heads, head_dim_qk).to(torch.bfloat16)
    key_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim_qk).to(torch.bfloat16)
    value_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim_v).to(torch.bfloat16)
    return [q, key_cache, value_cache, page_tables, seq_lens]


def make_inputs(num_seqs: int, num_q_heads: int, num_kv_heads: int,
                head_dim_qk: int, head_dim_v: int, page_size: int,
                max_seq_len: int, dtype: str = "bfloat16", seed: int = 0,
                seq_lens=None):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    seq_lens 可显式指定（用于精确构造页边界案例）；缺省时在 [1, max_seq_len]
    均匀采样。返回 [q, key_cache, value_cache, page_tables, seq_lens]，
    page_tables 行内是互不相同的物理页编号（randperm 抽取）。
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
    max_pages_per_seq = (max_seq_len + page_size - 1) // page_size

    # 物理页池大于实际用量：未引用槽位是"垃圾数据"，用于检验掩码正确性
    num_pages = num_seqs * max_pages_per_seq + 8
    perm = torch.randperm(num_pages, generator=gen)
    page_tables = perm[: num_seqs * max_pages_per_seq].reshape(num_seqs, max_pages_per_seq).to(torch.int32)

    q = torch.randn(num_seqs, num_q_heads, head_dim_qk, generator=gen).to(dt)
    key_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim_qk, generator=gen).to(dt)
    value_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim_v, generator=gen).to(dt)
    return [q, key_cache, value_cache, page_tables, seq_lens]
