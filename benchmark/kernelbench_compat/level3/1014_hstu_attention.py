# 1014_hstu_attention — HSTU 注意力前向（jagged 变长、无 softmax，KernelBench 兼容题目文件）。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=1014, backend=triton。

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """HSTU attention 前向（生成式推荐 HSTU 单元的注意力，无 softmax / 无归一化）：
    稠密 Q/K/V 按 seq_offsets 分段为 batch 条变长序列，每条序列独立计算
    out = (valid ∘ SiLU(alpha·Q·K^T) / N) · V，其中 N = max_seq_len。

    数学定义（第 b 条序列：s = seq_offsets[b]，L = seq_offsets[b+1] - seq_offsets[b]，
    g = num_targets[b]，c = contextual_seq_len，A = max_attn_len；c = 0 时 M = L、
    r[i] = i，c > 0 时 M = L - c + 1、r[i] = max(i - c + 1, 0)）：
      位置重编号 m[i] = min(r[i], M - g)
      d[i,j] = m[i] - m[j]；causal = False 时 d[i,j] = |d[i,j]|
      valid[i,j] = (i == j) 或 d[i,j] > 0
      A > 0 时 valid[i,j] 额外要求 d[i,j] <= A
      c > 0 时 valid[i,j] 额外放宽为真当 m[i] == 0 且 m[j] < M - g
      out[s+i, h, :] = Σ_j valid[i,j] · SiLU(alpha·<q[s+i,h,:], k[s+j,h,:]>) / N
                       · v[s+j,h,:]
    其中 SiLU(x) = x·sigmoid(x)。causal = True 时 valid 为下三角（含对角）；
    权重不做 softmax、不求和归一化，被掩码位置的贡献恰为 0。
    边界行为：序列末尾 g 个 token 为 target，经重编号截断从 key 侧排除，
    g = 0 等价于无 target 排除；c > 0 时前 c 个 token（重编号 m = 0）额外
    可见全部非 target key（含彼此）；空序列（L = 0）合法，不产生输出行；
    各序列长度可小于 N，缩放分母恒为 N（不是各序列自身长度）。

    输入输出规格：
      q            [total, num_heads, qk_dim]   float16 / bfloat16
      k            [total, num_heads, qk_dim]   与 q 同 dtype
      v            [total, num_heads, v_dim]    与 q 同 dtype，v_dim 可不等于 qk_dim
      seq_offsets  [batch + 1] int64，单调非降、首元素 0、尾元素 total
      num_targets  [batch] int32，0 <= num_targets[b] 且 c + num_targets[b] <= L_b
      out          [total, num_heads, v_dim]，与 v 同 dtype
    全域约束：0 <= L_b <= max_seq_len；qk_dim / v_dim 为 2 的幂（64~256 量级）。

    超参（__init__）：max_seq_len（缩放分母与长度上界 N）；alpha（logits 缩放，
    缺省 1/sqrt(qk_dim)）；causal（缺省 True）；max_attn_len（缺省 0 =
    不限距离）；contextual_seq_len（缺省 0 = 关闭）。

    实现约束（违规判负）：
      - 核心计算（QK^T / SiLU / 掩码 / PV）必须在提交文件内完成，禁止调用
        scaled_dot_product_attention / sdpa / flash_attn / torch.matmul /
        torch.bmm / torch.einsum / torch.softmax / torch.nn.functional.silu
        及其别名形式（F.silu / F.softmax / torch.mm / @ 矩阵乘算子；SiLU 以
        x·sigmoid(x) 自实现）。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回 v 的 dtype。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：前向、非量化输入、无 dropout、无反向传播。
    """

    # 注意：__init__/forward 签名必须单行（框架 loader 按首行提取签名）
    def __init__(self, max_seq_len: int, alpha=None, causal=True, max_attn_len=0, contextual_seq_len=0):
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.alpha = float(alpha) if alpha is not None else None
        self.causal = bool(causal)
        self.max_attn_len = int(max_attn_len)
        self.contextual_seq_len = int(contextual_seq_len)

    def forward(self, q, k, v, seq_offsets, num_targets):
        # KernelBench 评测器会把全部输入张量 cast 成评测 precision（triton 后端
        # 为 fp32）；偏移表与 target 计数都是小整数，fp32 可精确表示，此处无损恢复
        seq_offsets = seq_offsets.to(torch.long)
        num_targets = num_targets.to(torch.int32)

        B = int(seq_offsets.numel()) - 1
        alpha = self.alpha if self.alpha is not None else 1.0 / math.sqrt(q.shape[2])
        N = self.max_seq_len
        c = self.contextual_seq_len

        out = torch.empty_like(v)
        for b in range(B):
            s, e = int(seq_offsets[b]), int(seq_offsets[b + 1])
            L = e - s
            g = int(num_targets[b])
            assert 0 <= g and c + g <= L, "题面约束：0 <= g 且 contextual_seq_len + g <= L"

            # float32 中间计算；[H, L, *] 视图便于按头批量 matmul
            q_b = q[s:e].transpose(0, 1).to(torch.float32)            # [H, L, D_qk]
            k_b = k[s:e].transpose(0, 1).to(torch.float32)
            v_b = v[s:e].transpose(0, 1).to(torch.float32)            # [H, L, D_v]

            logits = torch.matmul(q_b, k_b.transpose(-2, -1)) * alpha  # [H, L, L]
            w = logits * torch.sigmoid(logits) / N                     # SiLU(logits)/N

            # 有效掩码：按重编号后的相对距离构造
            ids = torch.arange(L, device=q.device)
            M = L if c == 0 else L - c + 1
            if c > 0:
                ids = torch.clamp(ids - (c - 1), min=0)
            ids = torch.clamp(ids, max=M - g)
            dist = ids[:, None] - ids[None, :]
            if not self.causal:
                dist = torch.where(dist > 0, dist, -dist)
            valid = torch.eye(L, dtype=torch.bool, device=q.device) | (dist > 0)
            if self.max_attn_len > 0:
                valid = valid & (dist <= self.max_attn_len)
            if c > 0:
                valid = valid | ((ids[:, None] == 0) & (ids[None, :] < M - g))

            o_b = torch.matmul(w * valid[None].to(w.dtype), v_b)       # [H, L, D_v]
            out[s:e] = o_b.transpose(0, 1).to(v.dtype)
        return out


def get_init_inputs():
    return [256]  # max_seq_len；alpha 缺省 1/sqrt(qk_dim)，causal=True，
    # max_attn_len=0，contextual_seq_len=0


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    num_seqs = 8
    num_heads = 4
    qk_dim = 128
    v_dim = 128
    max_seq_len = 256

    lengths = torch.randint(1, max_seq_len + 1, (num_seqs,), dtype=torch.int64)
    num_targets = torch.minimum(
        torch.randint(1, 21, (num_seqs,), dtype=torch.int64), lengths
    ).to(torch.int32)

    seq_offsets = torch.zeros(num_seqs + 1, dtype=torch.int64)
    for i in range(num_seqs):
        seq_offsets[i + 1] = seq_offsets[i] + int(lengths[i])

    total = int(lengths.sum())
    q = torch.randn(total, num_heads, qk_dim).to(torch.bfloat16)
    k = torch.randn(total, num_heads, qk_dim).to(torch.bfloat16)
    v = torch.randn(total, num_heads, v_dim).to(torch.bfloat16)
    return [q, k, v, seq_offsets, num_targets]


def make_inputs(batch_size: int, max_seq_len: int, num_heads: int,
                qk_dim: int, v_dim: int, dtype: str = "bfloat16", seed: int = 0,
                lengths=None, sparsity=None, num_targets=None,
                target_size: int = 20, causal: bool = True, max_attn_len: int = 0,
                contextual_seq_len: int = 0, alpha=None):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    返回 (q, k, v, seq_offsets, num_targets)。causal / max_attn_len /
    contextual_seq_len / alpha 不参与张量生成，仅描述对应的 Model 超参
    （评测端据此构造 Model），其中 contextual_seq_len 参与长度与 target 的
    合法性约束。lengths / num_targets 可显式指定（单值广播到整个 batch），
    缺省时 lengths ~ U[1, max_seq_len]、num_targets ~ U[1, target_size] 截断到
    合法域；sparsity 给出稀疏度采样（官方口径）：0 -> 全 0 长度，
    1 -> 全 max_seq_len，>= 0.5 -> U[(2s-1)·N, N)，< 0.5 -> U[0, 2s·N)。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    if lengths is None:
        if sparsity is None:
            lo, hi = 1, max_seq_len + 1
        elif sparsity == 0.0:
            lo, hi = 0, 1
        elif sparsity == 1.0:
            lo, hi = max_seq_len, max_seq_len + 1
        elif sparsity >= 0.5:
            lo, hi = int((2 * sparsity - 1.0) * max_seq_len), max_seq_len
        else:
            lo, hi = 0, int(2 * sparsity * max_seq_len)
        if contextual_seq_len > 0:
            lo = max(lo, contextual_seq_len + 1)
            hi = max(hi, lo + 1)
        lengths = torch.randint(lo, hi, (batch_size,), generator=gen, dtype=torch.int64)
    else:
        if len(lengths) == 1 and batch_size > 1:      # 单值广播到整个 batch
            lengths = lengths * batch_size
        lengths = torch.tensor(lengths, dtype=torch.int64)
    assert lengths.numel() == batch_size
    assert bool((lengths >= 0).all()) and int(lengths.max()) <= max_seq_len
    if contextual_seq_len > 0:
        assert bool((lengths >= contextual_seq_len).all()), "题面约束：c + g <= L"

    cap = torch.clamp(lengths - contextual_seq_len, min=0)
    if num_targets is None:
        nt = torch.randint(1, target_size + 1, (batch_size,), generator=gen, dtype=torch.int64)
        nt = torch.minimum(nt, cap)
    else:
        if len(num_targets) == 1 and batch_size > 1:
            num_targets = num_targets * batch_size
        nt = torch.tensor(num_targets, dtype=torch.int64)
    assert nt.numel() == batch_size
    assert bool((nt >= 0).all()) and bool((nt <= cap).all()), "题面约束：0 <= g <= L - c"

    seq_offsets = torch.zeros(batch_size + 1, dtype=torch.int64)
    for i in range(batch_size):
        seq_offsets[i + 1] = seq_offsets[i] + int(lengths[i])

    total = int(lengths.sum())
    q = torch.randn(total, num_heads, qk_dim, generator=gen).to(dt)
    k = torch.randn(total, num_heads, qk_dim, generator=gen).to(dt)
    v = torch.randn(total, num_heads, v_dim, generator=gen).to(dt)
    return q, k, v, seq_offsets, nt.to(torch.int32)
