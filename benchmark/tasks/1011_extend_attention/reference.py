# 1011_extend_attention — extend（增量 prefill）段与 prefix 段混合的变长
# attention：prefix 段经 kv_indices 从 KV 池逐 token 收集、extend 段连续存放，
# 支持 GQA 与 causal 掩码的 model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=1011, backend=triton。

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """extend（增量 prefill）阶段与 prefix 段混合的变长 attention（GQA，causal 可选）。

    batch 内每个序列的 KV 分两段：prefix 段已写入 KV 池（k_buffer / v_buffer），
    通过 kv_indices 中的池行号逐 token 间接寻址；extend 段为本步新计算的 token，
    K/V 在 k_extend / v_extend 中按 token 连续存放，query 在 q_extend 中与之同序。
    对第 i 个序列记 E_i = qo_indptr[i+1] - qo_indptr[i]（extend token 数）、
    P_i = kv_indptr[i+1] - kv_indptr[i]（prefix token 数）。对 extend 段第 t 个
    query（全局位置 T = P_i + t，行号 s = qo_indptr[i] + t）与第 h 个 query head
    （对应 KV head 为 g = h // query_group_size，
    query_group_size = num_q_heads // num_kv_heads）：
      K_full = concat(k_buffer[kv_indices[kv_indptr[i]:kv_indptr[i+1]]][:, g, :],
                      k_extend[qo_indptr[i]:qo_indptr[i+1]][:, g, :])   共 P_i + E_i 行
      V_full 同理（取 head_dim_v 头维）
      logits[j] = sm_scale * <q_extend[s, h, :], K_full[j]>
                 （sm_scale 缺省 1/sqrt(head_dim_q)）
      可见性：prefix 全部可见；extend 段在 causal=True 时仅全局位置 <= T 的
      token 可见（即前 t+1 个 extend token），causal=False 时全部可见
      p = softmax(logits)（数值稳定的在线实现）
      out[s, h, :] = sum_j p[j] * V_full[j]
    causal 下 query 自身位置恒可见，softmax 分母恒 > 0。K 的头维等于 Q 的头维
    head_dim_q；V 的头维 head_dim_v 可与前者不同。

    输入输出规格（评测 dtype 取 float16 / bfloat16，张量均连续）：
      q_extend   评测 dtype (total_extend, num_q_heads, head_dim_q)
      k_extend   评测 dtype (total_extend, num_kv_heads, head_dim_q)
      v_extend   评测 dtype (total_extend, num_kv_heads, head_dim_v)
      k_buffer   评测 dtype (num_pool, num_kv_heads, head_dim_q)   prefix K 池
      v_buffer   评测 dtype (num_pool, num_kv_heads, head_dim_v)   prefix V 池
      qo_indptr  int32 (num_seqs + 1) —— extend 段长度的前缀和（每段 >= 1）
      kv_indptr  int32 (num_seqs + 1) —— prefix 段长度的前缀和（每段 >= 0，可全 0）
      kv_indices int32 (total_prefix) —— prefix token 的池行号，可为任意排列；
                 num_pool 可大于 total_prefix，未被引用的池行是垃圾数据，不得
                 参与计算
      返回 out   评测 dtype (total_extend, num_q_heads, head_dim_v)

    实现约束（违规判负）：
      - 核心计算（QK^T / softmax / PV）必须在提交文件内完成，禁止调用
        scaled_dot_product_attention / sdpa / flash_attn / aiter /
        torch.matmul / torch.bmm / torch.einsum / torch.softmax。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。
      - 评测器会把全部输入张量 cast 成 fp32 后传入 forward；indptr 与池行号
        都是小整数，fp32 可精确表示，入口处 .to(torch.int32) / .to(torch.long)
        无损恢复。
      - 最大 extend 长度不是输入：grid 第三维的上界可在 forward 内由 qo_indptr
        推得，或直接用 total_extend 做保守上界（kernel 内按各序列的
        cur_seq_len_extend 掩码越界 query 行）。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：prefix+extend 混合、变长 ragged batch、GQA、causal/非 causal、
    Q/K/V 头维关系如上；无自定义 mask、无滑动窗口、无 attention sink、无 KV
    反量化 scale、logit cap 未启用（恒 0）、prefix 段不做独立的 prefill kernel
    融合路径，这些变体留作后续独立题。
    """

    def __init__(self, causal: bool = True, sm_scale=None):
        super().__init__()
        self.causal = bool(causal)
        self.sm_scale = None if sm_scale is None else float(sm_scale)

    def forward(self, q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices):
        # 评测器把全部输入 cast 成 fp32；累积长度与池行号都是小整数，
        # fp32 可精确表示，此处无损恢复
        qo_indptr = qo_indptr.to(torch.int32)
        kv_indptr = kv_indptr.to(torch.int32)
        kv_indices = kv_indices.to(torch.long)

        total_extend, H_Q, Dq = q_extend.shape
        H_KV = k_extend.shape[1]
        Dv = v_extend.shape[-1]
        group = H_Q // H_KV
        assert H_Q == H_KV * group, "num_q_heads 必须是 num_kv_heads 的整数倍"
        scale = self.sm_scale if self.sm_scale is not None else 1.0 / math.sqrt(Dq)
        B = qo_indptr.shape[0] - 1

        out = torch.empty((total_extend, H_Q, Dv), dtype=q_extend.dtype,
                          device=q_extend.device)
        for i in range(B):
            qs, qe = int(qo_indptr[i]), int(qo_indptr[i + 1])
            ks, ke = int(kv_indptr[i]), int(kv_indptr[i + 1])
            prefix_len = ke - ks
            seq_len = qe - qs

            q = q_extend[qs:qe].to(torch.float32)                     # [E, H_Q, Dq]
            k_prefix = k_buffer[kv_indices[ks:ke]].to(torch.float32)  # [P, H_KV, Dq]
            v_prefix = v_buffer[kv_indices[ks:ke]].to(torch.float32)  # [P, H_KV, Dv]
            k_full = torch.cat([k_prefix, k_extend[qs:qe].to(torch.float32)], dim=0)
            v_full = torch.cat([v_prefix, v_extend[qs:qe].to(torch.float32)], dim=0)

            # GQA：同一 KV head 的 K/V 复制给 group 内的每个 query head
            if group > 1:
                k_full = k_full.repeat_interleave(group, dim=1)       # [P+E, H_Q, Dq]
                v_full = v_full.repeat_interleave(group, dim=1)       # [P+E, H_Q, Dv]

            scores = torch.einsum("qhc,khc->hqk", q, k_full) * scale  # [H_Q, E, P+E]
            if self.causal:
                pos_q = prefix_len + torch.arange(seq_len, device=q_extend.device)
                pos_k = torch.arange(prefix_len + seq_len, device=q_extend.device)
                causal_mask = pos_k.unsqueeze(0) > pos_q.unsqueeze(1)  # [E, P+E]
                scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
            p = torch.softmax(scores, dim=-1)
            out[qs:qe] = torch.einsum("hqk,khd->qhd", p, v_full).to(q_extend.dtype)
        return out


def get_init_inputs():
    return [True]   # causal；sm_scale 缺省 1/sqrt(head_dim_q)


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    num_seqs = 4
    num_kv_heads = 4
    query_group_size = 4      # num_q_heads = 16
    head_dim_q = 128
    head_dim_v = 64           # V 头维与 Q/K 头维不同
    max_prefix_len = 256
    max_extend_len = 192

    num_q_heads = num_kv_heads * query_group_size
    extend_lens = torch.randint(1, max_extend_len + 1, (num_seqs,), dtype=torch.int32)
    prefix_lens = torch.randint(1, max_prefix_len + 1, (num_seqs,), dtype=torch.int32)
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32),
                           extend_lens.cumsum(0, dtype=torch.int32)])
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32),
                           prefix_lens.cumsum(0, dtype=torch.int32)])
    total_extend = int(qo_indptr[-1])
    total_prefix = int(kv_indptr[-1])

    # KV 池大于实际用量：未引用行是垃圾数据，用于检验间接寻址正确性
    num_pool = total_prefix + 8
    kv_indices = torch.randperm(num_pool)[:total_prefix].to(torch.int32)

    q_extend = torch.randn(total_extend, num_q_heads, head_dim_q).to(torch.float16)
    k_extend = torch.randn(total_extend, num_kv_heads, head_dim_q).to(torch.float16)
    v_extend = torch.randn(total_extend, num_kv_heads, head_dim_v).to(torch.float16)
    k_buffer = torch.randn(num_pool, num_kv_heads, head_dim_q).to(torch.float16)
    v_buffer = torch.randn(num_pool, num_kv_heads, head_dim_v).to(torch.float16)
    return [q_extend, k_extend, v_extend, k_buffer, v_buffer,
            qo_indptr, kv_indptr, kv_indices]


def make_inputs(num_seqs: int, num_q_heads: int, num_kv_heads: int,
                head_dim_q: int, head_dim_v: int, max_prefix_len: int,
                max_extend_len: int, dtype: str = "float16", seed: int = 0,
                prefix_lens=None, extend_lens=None, causal: bool = True):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    extend_lens / prefix_lens 可显式指定（精确构造边界 case）；缺省时分别在
    [1, max_extend_len] / [1, max_prefix_len] 均匀采样（max_prefix_len 为 0 时
    prefix 全 0）。causal 不影响输入张量取值，仅记录该 case 对应的 Model 初始化
    超参。消费顺序固定：extend_lens → prefix_lens → randperm → q_extend →
    k_extend → v_extend → k_buffer → v_buffer，全部来自同一
    torch.Generator(seed)，同 seed 下逐位可复现。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    if extend_lens is None:
        extend_lens = torch.randint(1, max_extend_len + 1, (num_seqs,),
                                    generator=gen, dtype=torch.int32)
    else:
        if len(extend_lens) == 1 and num_seqs > 1:   # 单值广播到整个 batch
            extend_lens = list(extend_lens) * num_seqs
        extend_lens = torch.tensor(extend_lens, dtype=torch.int32)
        assert len(extend_lens) == num_seqs and int(extend_lens.min()) >= 1
        assert int(extend_lens.max()) <= max_extend_len

    if prefix_lens is None:
        if max_prefix_len <= 0:
            prefix_lens = torch.zeros(num_seqs, dtype=torch.int32)
        else:
            prefix_lens = torch.randint(1, max_prefix_len + 1, (num_seqs,),
                                        generator=gen, dtype=torch.int32)
    else:
        if len(prefix_lens) == 1 and num_seqs > 1:   # 单值广播到整个 batch
            prefix_lens = list(prefix_lens) * num_seqs
        prefix_lens = torch.tensor(prefix_lens, dtype=torch.int32)
        assert len(prefix_lens) == num_seqs and int(prefix_lens.min()) >= 0
        assert int(prefix_lens.max()) <= max_prefix_len

    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32),
                           extend_lens.cumsum(0, dtype=torch.int32)])
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32),
                           prefix_lens.cumsum(0, dtype=torch.int32)])
    total_extend = int(qo_indptr[-1])
    total_prefix = int(kv_indptr[-1])

    # KV 池大于实际用量：未引用行是垃圾数据，行号随机排列（间接寻址）
    num_pool = total_prefix + 8
    kv_indices = torch.randperm(num_pool, generator=gen)[:total_prefix].to(torch.int32)

    q_extend = torch.randn(total_extend, num_q_heads, head_dim_q, generator=gen).to(dt)
    k_extend = torch.randn(total_extend, num_kv_heads, head_dim_q, generator=gen).to(dt)
    v_extend = torch.randn(total_extend, num_kv_heads, head_dim_v, generator=gen).to(dt)
    k_buffer = torch.randn(num_pool, num_kv_heads, head_dim_q, generator=gen).to(dt)
    v_buffer = torch.randn(num_pool, num_kv_heads, head_dim_v, generator=gen).to(dt)
    return (q_extend, k_extend, v_extend, k_buffer, v_buffer,
            qo_indptr, kv_indptr, kv_indices)
