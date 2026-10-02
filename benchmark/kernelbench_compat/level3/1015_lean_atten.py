# 1015_lean_atten — ragged batch 变长 KV 的 Lean Attention 前向（连续、非分页
# KV；KernelBench 兼容题目文件）。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=1015, backend=triton。

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """ragged batch 变长 KV 的 attention 前向（Lean Attention 语义：连续、
    非分页 KV，单 kernel 在线 softmax），覆盖 decode（各批 KV 长度独立）与
    prefill（方阵因果自注意力）两种形态，MHA（Q/K/V 头数一致）。

    数学定义：batch 共 num_seqs 条序列，第 b 条序列实际参与 attention 的
    KV 长度为 kv_lens[b]（>= 1），各批 query 段等长 q_len
    （total_q = num_seqs * q_len，q 按序列主序存放）。记
    start_k = kv_lens[0] + ... + kv_lens[b-1]（k/v 亦按序列主序连续存放），
    对每条序列 b、每个头 h、query 位置 i（0 <= i < q_len）与 KV 位置
    j（0 <= j < kv_lens[b]）：
      scores[i, j] = sm_scale * <q[b*q_len+i, h, :], k[start_k+j, h, :]>
      causal=1 时（题面约束该形态下 kv_lens[b] == q_len 对所有 b 成立）施加
      方阵因果掩码：j > i 的位置记 -inf；causal=0 时不施加任何掩码；
      p[i, :] = softmax_j(scores[i, :])
      out[b*q_len+i, h, :] = sum_j p[i, j] * v[start_k+j, h, :]
    softmax 采用数值稳定（在线）实现，中间累加为 float32，输出 cast 回输入
    dtype。边界行为：kv_lens[b] >= 1 恒成立，softmax 分母恒正、无全掩码行；
    序列之间互不影响；k/v 中超出各序列累计长度边界的槽位不参与计算。

    输入输出规格：
      q        [num_seqs*q_len, num_heads, head_dim]  float16/bfloat16，序列主序
      k, v     [total_kv, num_heads, head_dim]        与 q 同 dtype，total_kv = sum(kv_lens)
      kv_lens  [num_seqs] int32 —— 每条序列实际参与 attention 的 KV 长度
      causal   0-dim int32 张量，1=方阵因果掩码，0=无掩码
      out      [num_seqs*q_len, num_heads, head_dim]  与 q 同 dtype
    全域约束：Q/K/V 头数一致（MHA，无 GQA）；head_dim ∈ {16, 32, 64, 128, 256}
    且 Q/K/V 相同；q_len >= 1；causal=1 时 kv_lens[b] == q_len 对所有 b。

    实现约束（违规判负）：
      - 核心计算（QK^T / softmax / PV）必须在提交文件内完成，禁止调用
        scaled_dot_product_attention / sdpa / flash_attn / aiter /
        torch.matmul / torch.bmm / torch.einsum / torch.softmax。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：连续非分页 KV、变长 ragged batch、MHA、causal 两态、
    fp16/bf16 非量化输入；无 GQA、无 dropout、无 bias/ALiBi、无 LSE 输出。
    """

    def __init__(self, head_dim: int, scale=None):
        super().__init__()
        self.sm_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)

    def forward(self, q, k, v, kv_lens, causal):
        # KernelBench 评测器会把全部输入张量 cast 成评测 precision（triton 后端
        # 为 fp32）；KV 长度与 0/1 开关都是小整数，fp32 可精确表示，此处无损恢复
        kv_lens = kv_lens.to(torch.int32)
        if not torch.is_tensor(causal):
            causal = torch.tensor(causal)
        is_causal = bool(int(causal.to(torch.int32)))

        num_heads = q.shape[1]
        assert num_heads == k.shape[1] == v.shape[1], "MHA：Q/K/V 头数必须一致"
        num_seqs = int(kv_lens.numel())
        q_len = q.shape[0] // num_seqs
        assert q_len * num_seqs == q.shape[0], "total_q 必须是 num_seqs 的整数倍"
        if is_causal:
            assert bool((kv_lens == q_len).all()), "causal=1 时各批 kv_len 必须等于 q_len"

        out = torch.empty_like(q)
        start_k = 0
        for b in range(num_seqs):
            kv_len = int(kv_lens[b])
            q_b = q[b * q_len : (b + 1) * q_len].transpose(0, 1).to(torch.float32)  # [H, q_len, D]
            k_b = k[start_k : start_k + kv_len].transpose(0, 1).to(torch.float32)   # [H, kv_len, D]
            v_b = v[start_k : start_k + kv_len].transpose(0, 1).to(torch.float32)

            scores = torch.matmul(q_b, k_b.transpose(-2, -1)) * self.sm_scale
            if is_causal:
                mask = torch.triu(
                    torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device),
                    diagonal=1,
                )
                scores = scores.masked_fill(mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            o_b = torch.matmul(probs, v_b)                                          # [H, q_len, D]
            out[b * q_len : (b + 1) * q_len] = o_b.transpose(0, 1).to(q.dtype)
            start_k += kv_len
        return out


def get_init_inputs():
    return [128]  # head_dim；sm_scale 缺省 1/sqrt(head_dim)


def get_inputs():
    # 固定 shape 族（decode 形态：causal=0、各批 KV 长度独立）；随机部分消费
    # 全局 RNG——评测器在 set_seed 后调用本函数，多轮 correctness trial 因此
    # 获得输入多样性
    num_seqs = 4
    num_heads = 8
    head_dim = 128
    q_len = 16
    max_kv_len = 2048

    kv_lens = torch.randint(1, max_kv_len + 1, (num_seqs,), dtype=torch.int32)
    total_kv = int(kv_lens.sum())

    q = (torch.randn(num_seqs * q_len, num_heads, head_dim) * 0.5).to(torch.float16)
    k = (torch.randn(total_kv, num_heads, head_dim) * 0.5).to(torch.float16)
    v = (torch.randn(total_kv, num_heads, head_dim) * 0.5).to(torch.float16)
    causal = torch.tensor(0, dtype=torch.int32)
    return [q, k, v, kv_lens, causal]


def make_inputs(num_seqs: int, num_heads: int, head_dim: int, q_len: int,
                max_kv_len: int, dtype: str = "float16", causal: int = 0,
                seed: int = 0, kv_lens=None):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    causal=1（方阵因果自注意力）：各批 kv_len == q_len；kv_lens 缺省时自动填
    q_len，显式给出时必须逐批等于 q_len（单值 [q_len] 可广播）。
    causal=0（decode/非因果）：kv_lens 缺省时在 [1, max_kv_len] 均匀采样，
    显式给出（单值可广播到整个 batch）用于精确构造边界案例。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert head_dim in (16, 32, 64, 128, 256), "head_dim 必须在 {16,32,64,128,256} 内"

    if causal:
        if kv_lens is None:
            kv_lens = [q_len] * num_seqs
        else:
            if len(kv_lens) == 1 and num_seqs > 1:   # 单值广播到整个 batch
                kv_lens = kv_lens * num_seqs
            assert all(int(n) == q_len for n in kv_lens), "causal=1 时各批 kv_len 必须等于 q_len"
    else:
        if kv_lens is None:
            kv_lens = torch.randint(1, max_kv_len + 1, (num_seqs,), generator=gen, dtype=torch.int64).tolist()
        elif len(kv_lens) == 1 and num_seqs > 1:
            kv_lens = kv_lens * num_seqs
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32)
    assert int(kv_lens.numel()) == num_seqs
    assert 1 <= int(kv_lens.max()) <= max_kv_len, "kv_lens 超出 [1, max_kv_len]"

    total_kv = int(kv_lens.sum())
    q = (torch.randn(num_seqs * q_len, num_heads, head_dim, generator=gen) * 0.5).to(dt)
    k = (torch.randn(total_kv, num_heads, head_dim, generator=gen) * 0.5).to(dt)
    v = (torch.randn(total_kv, num_heads, head_dim, generator=gen) * 0.5).to(dt)
    causal_flag = torch.tensor(int(causal), dtype=torch.int32)
    return q, k, v, kv_lens, causal_flag
