# 1012_flash_attention_forward — 变长序列 FlashAttention v2 前向（KernelBench 兼容题目文件）。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=1012, backend=triton。

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """变长序列（varlen）FlashAttention v2 前向：稠密 Q/K/V 按 cu_seqlens 分段，
    每段独立计算 scaled dot-product attention（在线 softmax 语义），支持
    MHA/GQA 与因果掩码。

    数学定义：对 batch 内第 b 段（记 L_q = cu_seqlens_q[b+1]-cu_seqlens_q[b]、
    L_k = cu_seqlens_k[b+1]-cu_seqlens_k[b]，题面约束 0 < L_q <= L_k，
    g = num_q_heads // num_kv_heads）：
      scores[h,i,j] = sm_scale * <q[start_q+i,h,:], k[start_k+j,h//g,:]>
      causal=1 时施加右下对齐因果掩码：j > i + (L_k - L_q) 的位置记 -inf
      （即 query 位置 i 只 attend 到 key 位置 j <= i + L_k - L_q）；
      causal=0 时不施加任何掩码。
      p[h,i,:] = softmax_j(scores[h,i,:])
      out[start_q+i,h,:] = sum_j p[h,i,j] * v[start_k+j,h//g,:]
    softmax 采用数值稳定实现，中间累加为 float32，输出 cast 回输入 dtype。
    边界行为：题面保证每段 L_q >= 1，因此 softmax 分母恒正、无全掩码行；
    末段以外的段之间互不影响。

    输入输出规格：
      q            [total_q, num_q_heads, head_dim]  float16/bfloat16
      k, v         [total_k, num_kv_heads, head_dim] 与 q 同 dtype
      cu_seqlens_q [num_seqs+1] int32，单调非降、首元素 0、尾元素 total_q
      cu_seqlens_k [num_seqs+1] int32，单调非降、首元素 0、尾元素 total_k
      causal       0-dim int32 张量，1=因果掩码，0=无掩码
      out          [total_q, num_q_heads, head_dim]，与 q 同 dtype
    全域约束：num_q_heads 是 num_kv_heads 的整数倍（GQA；相等即 MHA）；
    1 <= head_dim <= 256；每条序列 0 < L_q <= L_k。

    实现约束（违规判负）：
      - 核心计算（QK^T / softmax / PV）必须在提交文件内完成，禁止调用
        scaled_dot_product_attention / sdpa / flash_attn / aiter /
        torch.matmul / torch.bmm / torch.einsum / torch.softmax。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：varlen 前向、非量化输入、无 bias/ALiBi、无 dropout、无 LSE 输出。
    """

    def __init__(self, head_dim: int, scale=None):
        super().__init__()
        self.sm_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)

    def forward(self, q, k, v, cu_seqlens_q, cu_seqlens_k, causal):
        # KernelBench 评测器会把全部输入张量 cast 成评测 precision（triton 后端
        # 为 fp32）；累积长度与 0/1 开关都是小整数，fp32 可精确表示，此处无损恢复
        cu_seqlens_q = cu_seqlens_q.to(torch.int32)
        cu_seqlens_k = cu_seqlens_k.to(torch.int32)
        if not torch.is_tensor(causal):
            causal = torch.tensor(causal)
        is_causal = bool(int(causal.to(torch.int32)))

        nheads_q = q.shape[1]
        nheads_k = k.shape[1]
        group = nheads_q // nheads_k
        assert nheads_q == nheads_k * group, "num_q_heads 必须是 num_kv_heads 的整数倍"
        num_seqs = int(cu_seqlens_q.numel()) - 1

        out = torch.empty_like(q)
        for b in range(num_seqs):
            start_q, end_q = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
            start_k, end_k = int(cu_seqlens_k[b]), int(cu_seqlens_k[b + 1])
            seq_len_q = end_q - start_q
            seq_len_k = end_k - start_k
            assert 0 < seq_len_q <= seq_len_k, "题面约束：每条序列 0 < L_q <= L_k"

            q_b = q[start_q:end_q].transpose(0, 1).to(torch.float32)   # [H_Q, Lq, D]
            k_b = k[start_k:end_k].transpose(0, 1).to(torch.float32)   # [H_K, Lk, D]
            v_b = v[start_k:end_k].transpose(0, 1).to(torch.float32)
            # GQA：同一 KV head 的 K/V 复制给 group 内的每个 query head
            if group > 1:
                k_b = k_b.repeat_interleave(group, dim=0)              # [H_Q, Lk, D]
                v_b = v_b.repeat_interleave(group, dim=0)

            scores = torch.matmul(q_b, k_b.transpose(-2, -1)) * self.sm_scale
            if is_causal:
                # 右下对齐因果掩码（与 L_q != L_k 时的 FlashAttention 语义一致）
                mask = torch.triu(
                    torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=q.device),
                    diagonal=seq_len_k - seq_len_q + 1,
                )
                scores = scores.masked_fill(mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            o_b = torch.matmul(probs, v_b)                             # [H_Q, Lq, D]
            out[start_q:end_q] = o_b.transpose(0, 1).to(q.dtype)
        return out


def get_init_inputs():
    return [64]  # head_dim；sm_scale 缺省 1/sqrt(head_dim)


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    num_seqs = 4
    num_kv_heads = 4
    query_group_size = 4      # num_q_heads = 16
    head_dim = 64
    max_seq_len_q = 256
    max_seq_len_k = 512

    num_q_heads = num_kv_heads * query_group_size
    seq_lens_q = torch.randint(1, max_seq_len_q + 1, (num_seqs,), dtype=torch.int64)
    seq_lens_k = torch.randint(1, max_seq_len_k + 1, (num_seqs,), dtype=torch.int64)
    seq_lens_k = torch.maximum(seq_lens_k, seq_lens_q)   # 每条序列 L_q <= L_k

    cu_seqlens_q = torch.zeros(num_seqs + 1, dtype=torch.int32)
    cu_seqlens_k = torch.zeros(num_seqs + 1, dtype=torch.int32)
    for i in range(num_seqs):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + int(seq_lens_q[i])
        cu_seqlens_k[i + 1] = cu_seqlens_k[i] + int(seq_lens_k[i])

    total_q = int(seq_lens_q.sum())
    total_k = int(seq_lens_k.sum())
    q = (torch.randn(total_q, num_q_heads, head_dim) * 0.5).to(torch.float16)
    k = (torch.randn(total_k, num_kv_heads, head_dim) * 0.5).to(torch.float16)
    v = (torch.randn(total_k, num_kv_heads, head_dim) * 0.5).to(torch.float16)
    causal = torch.tensor(1, dtype=torch.int32)
    return [q, k, v, cu_seqlens_q, cu_seqlens_k, causal]


def make_inputs(num_seqs: int, num_q_heads: int, num_kv_heads: int,
                head_dim: int, max_seq_len_q: int, max_seq_len_k: int,
                dtype: str = "float16", causal: int = 1, seed: int = 0,
                seq_lens_q=None, seq_lens_k=None):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    seq_lens_q / seq_lens_k 可显式指定（用于精确构造边界案例，单值广播到整个
    batch）；缺省时 L_q ~ U[1, max_seq_len_q]、L_k ~ U[1, max_seq_len_k] 取
    max(L_k, L_q) 后采样，恒满足题面约束 0 < L_q <= L_k。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    if seq_lens_q is None:
        seq_lens_q = torch.randint(1, max_seq_len_q + 1, (num_seqs,), generator=gen, dtype=torch.int64)
    else:
        if len(seq_lens_q) == 1 and num_seqs > 1:   # 单值广播到整个 batch
            seq_lens_q = seq_lens_q * num_seqs
        seq_lens_q = torch.tensor(seq_lens_q, dtype=torch.int64)
    if seq_lens_k is None:
        seq_lens_k = torch.randint(1, max_seq_len_k + 1, (num_seqs,), generator=gen, dtype=torch.int64)
        seq_lens_k = torch.maximum(seq_lens_k, seq_lens_q)
    else:
        if len(seq_lens_k) == 1 and num_seqs > 1:
            seq_lens_k = seq_lens_k * num_seqs
        seq_lens_k = torch.tensor(seq_lens_k, dtype=torch.int64)

    assert len(seq_lens_q) == num_seqs and len(seq_lens_k) == num_seqs
    assert int(seq_lens_q.max()) <= max_seq_len_q and int(seq_lens_k.max()) <= max_seq_len_k
    assert bool((seq_lens_q >= 1).all()), "每条序列 L_q >= 1"
    assert bool((seq_lens_q <= seq_lens_k).all()), "题面约束：每条序列 0 < L_q <= L_k"

    cu_seqlens_q = torch.zeros(num_seqs + 1, dtype=torch.int32)
    cu_seqlens_k = torch.zeros(num_seqs + 1, dtype=torch.int32)
    for i in range(num_seqs):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + int(seq_lens_q[i])
        cu_seqlens_k[i + 1] = cu_seqlens_k[i] + int(seq_lens_k[i])

    total_q = int(seq_lens_q.sum())
    total_k = int(seq_lens_k.sum())
    q = (torch.randn(total_q, num_q_heads, head_dim, generator=gen) * 0.5).to(dt)
    k = (torch.randn(total_k, num_kv_heads, head_dim, generator=gen) * 0.5).to(dt)
    v = (torch.randn(total_k, num_kv_heads, head_dim, generator=gen) * 0.5).to(dt)
    causal_flag = torch.tensor(int(causal), dtype=torch.int32)
    return q, k, v, cu_seqlens_q, cu_seqlens_k, causal_flag
