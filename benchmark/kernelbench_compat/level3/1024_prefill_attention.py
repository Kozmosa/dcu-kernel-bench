# 1024_prefill_attention — 非分页（连续 KV）ragged 变长 batch 的 prefill
# context attention：每条序列的多 token query 对本序列连续存放的 KV 做
# （可选 causal 的）在线 softmax 自注意力，支持 GQA 的 model_class
# （KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=1024, backend=triton。

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """ragged 变长 batch 的非分页 prefill 自注意力（连续 KV，GQA，causal 可选）。

    算子语义：batch 内 num_seqs 条序列的 token 展平存放于 q/k/v 的第 0 维，
    第 i 条序列占据行区间 [b_start_loc[i], b_start_loc[i] + b_seq_len[i])，
    b_start_loc 为序列长度的排他前缀和（b_start_loc[0] = 0，
    b_start_loc[i+1] = b_start_loc[i] + b_seq_len[i]），无 padding；每条序列的
    q/k/v 是同一批 token 的 query/key/value（自注意力）。对第 h 个 query head
    （其 KV head 为 g = h // query_group_size，
    query_group_size = num_q_heads // num_kv_heads）与第 i 条序列内相对位置 t
    （0 <= t < L_i，L_i = b_seq_len[i]，行号 s = b_start_loc[i] + t）：
      logits[j] = sm_scale * <q[s, h, :], k[b_start_loc[i]+j, g, :]>
                  （sm_scale 缺省 1 / sqrt(head_dim)）
      causal=True 时 j > t 的位置记 -inf；causal=False 时 j 全部可见
      p[j] = softmax_j(logits[j])
      out[s, h, :] = sum_{j=0}^{L_i-1} p[j] * v[b_start_loc[i]+j, g, :]
    softmax 采用数值稳定的在线实现，中间累加为 float32，输出 cast 回输入
    dtype。边界行为：b_seq_len[i] >= 1 恒成立，softmax 分母恒正、无全掩码行
    （causal 下第 t 行至少可见自身）；序列之间互不影响。

    输入输出规格（评测 dtype 取 float16 / bfloat16，张量均连续）：
      q            评测 dtype (total_tokens, num_q_heads, head_dim)
      k            评测 dtype (total_tokens, num_kv_heads, head_dim)
      v            评测 dtype (total_tokens, num_kv_heads, head_dim)
      b_start_loc  int32 (num_seqs) —— 序列长度的排他前缀和
      b_seq_len    int32 (num_seqs) —— 每条序列长度（>= 1，可各不相同）
      返回 out     评测 dtype (total_tokens, num_q_heads, head_dim)
    全域约束：num_q_heads 是 num_kv_heads 的整数倍（GQA；相等即 MHA）；
    Q/K/V 头维相同且 >= 16，允许非 2 次幂（越界通道需按 head_dim 掩码置零）；
    total_tokens = sum(b_seq_len)。

    实现约束（违规判负）：
      - 核心计算（QK^T / softmax / PV）必须在提交文件内完成，禁止调用
        scaled_dot_product_attention / sdpa / flash_attn / aiter /
        torch.matmul / torch.bmm / torch.einsum / torch.softmax。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。
      - 评测器会把全部输入张量 cast 成 fp32 后传入 forward；前缀和与序列
        长度都是小整数，fp32 可精确表示，入口处 .to(torch.int32) 无损恢复。
      - 最大序列长度不是输入：grid 沿序列内位置维的上界可在 forward 内由
        int(b_seq_len.max()) 推得，kernel 内按各序列 L_i 掩码越界行/列。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：连续非分页 KV、ragged 变长 batch、GQA、causal/非 causal、
    fp16/bf16 非量化输入；无分页 KV Cache、无 prefix 池间接寻址、无自定义
    mask、无滑动窗口、无 attention sink、无 dropout、无 LSE 输出、无 MLA
    吸收（absorb）形态，这些变体留作后续独立题。
    """

    def __init__(self, causal: bool = True, sm_scale=None):
        super().__init__()
        self.causal = bool(causal)
        self.sm_scale = None if sm_scale is None else float(sm_scale)

    def forward(self, q, k, v, b_start_loc, b_seq_len):
        # KernelBench 评测器会把全部输入张量 cast 成评测 precision（triton 后端
        # 为 fp32）；前缀和与序列长度都是小整数，fp32 可精确表示，此处无损恢复
        b_start_loc = b_start_loc.to(torch.int32)
        b_seq_len = b_seq_len.to(torch.int32)

        total, H_Q, D = q.shape
        H_KV = k.shape[1]
        group = H_Q // H_KV
        assert H_Q == H_KV * group, "num_q_heads 必须是 num_kv_heads 的整数倍"
        assert k.shape[-1] == D and v.shape[-1] == D, "Q/K/V 头维必须相同"
        scale = self.sm_scale if self.sm_scale is not None else 1.0 / math.sqrt(D)
        B = b_seq_len.shape[0]

        out = torch.empty_like(q)
        for i in range(B):
            start = int(b_start_loc[i])
            seq_len = int(b_seq_len[i])

            q_i = q[start:start + seq_len].to(torch.float32)   # [L, H_Q, D]
            k_i = k[start:start + seq_len].to(torch.float32)   # [L, H_KV, D]
            v_i = v[start:start + seq_len].to(torch.float32)

            # GQA：同一 KV head 的 K/V 复制给 group 内的每个 query head
            if group > 1:
                k_i = k_i.repeat_interleave(group, dim=1)      # [L, H_Q, D]
                v_i = v_i.repeat_interleave(group, dim=1)

            scores = torch.einsum("qhd,khd->hqk", q_i, k_i) * scale   # [H_Q, L, L]
            if self.causal:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1
                )
                scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
            p = torch.softmax(scores, dim=-1)
            out[start:start + seq_len] = torch.einsum("hqk,khd->qhd", p, v_i).to(q.dtype)
        return out


def get_init_inputs():
    return [True]   # causal；sm_scale 缺省 1/sqrt(head_dim)


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    num_seqs = 4
    num_kv_heads = 4
    query_group_size = 4      # num_q_heads = 16
    head_dim = 64
    max_seq_len = 256

    num_q_heads = num_kv_heads * query_group_size
    seq_lens = torch.randint(1, max_seq_len + 1, (num_seqs,), dtype=torch.int32)
    b_start_loc = torch.cat(
        [torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0, dtype=torch.int32)[:-1]]
    )
    total_tokens = int(b_start_loc[-1] + seq_lens[-1])

    q = torch.randn(total_tokens, num_q_heads, head_dim).to(torch.float16)
    k = torch.randn(total_tokens, num_kv_heads, head_dim).to(torch.float16)
    v = torch.randn(total_tokens, num_kv_heads, head_dim).to(torch.float16)
    return [q, k, v, b_start_loc, seq_lens]


def make_inputs(num_seqs: int, num_q_heads: int, num_kv_heads: int,
                head_dim: int, max_seq_len: int, dtype: str = "float16",
                seed: int = 0, seq_lens=None, causal: bool = True):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    seq_lens 可显式指定（精确构造边界 case；单值广播到整个 batch）；缺省时在
    [1, max_seq_len] 均匀采样。causal 不影响输入张量取值，仅记录该 case 对应
    的 Model 初始化超参。消费顺序固定：seq_lens → q → k → v，全部来自同一
    torch.Generator(seed)，同 seed 下逐位可复现。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    if seq_lens is None:
        seq_lens = torch.randint(1, max_seq_len + 1, (num_seqs,),
                                 generator=gen, dtype=torch.int32)
    else:
        if len(seq_lens) == 1 and num_seqs > 1:   # 单值广播到整个 batch
            seq_lens = list(seq_lens) * num_seqs
        seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        assert len(seq_lens) == num_seqs and int(seq_lens.min()) >= 1
        assert int(seq_lens.max()) <= max_seq_len

    b_start_loc = torch.cat(
        [torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0, dtype=torch.int32)[:-1]]
    )
    total_tokens = int(seq_lens.sum())

    q = torch.randn(total_tokens, num_q_heads, head_dim, generator=gen).to(dt)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, generator=gen).to(dt)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, generator=gen).to(dt)
    return q, k, v, b_start_loc, seq_lens
