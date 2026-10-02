# 1020_mla_decode_rope — DeepSeek MLA 解码 attention 融合 RoPE（KernelBench 兼容题目文件）。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=1020, backend=triton。

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """DeepSeek MLA 解码阶段单 token query 的 attention，融合 RoPE 旋转，page size = 1。

    语义：q 为当前步解码 query [B, H, c + r]（c = kv_lora_rank 压缩 nope 部分
    在前、r = qk_rope_head_dim 的 rope 部分 q_pe 在后）。KV Cache 按_token_
    存放（page size 1）：kv_cache 共 total_tokens 行，每行 = [压缩隐向量
    （前 c 维）| k_pe（后 r 维）]；batch b 的第 j 个逻辑 token 位于物理行
    kv_indices[kv_indptr[b] + j]（kv_indptr 为序列长度累积和，从 0 起）；
    行池中未被引用的行是垃圾数据，不得参与计算。Value 即该行前 c 列：
    K 的压缩部分与 V 共享同一隐向量（MLA 结构，所有 query head 共用同一份
    逐 token 压缩 KV，无独立 KV head）。

    对每个 batch b（记 pos = positions[b]，rd = rotary_dim，half = rd // 2）：
      1. RoPE 旋转：按 pos 取 cos_sin_cache 的行 [cos(half) | sin(half)]。
         对 q_pe 的前 rd 维旋转、rd 之后的维原样直通；对逻辑最后一个 token
         （j = seq_len_b - 1）的 k_pe 同样只旋转前 rd 维（rd 之后直通），
         其余 token 的 k_pe 视为已旋转完毕、按 cache 原值使用。旋转由
         is_neox_style 选择风格：
           NEOX（rotate-half）: (x1, x2) = (前 half 维, 后 half 维)
           GPT-J（交错）:       (x1, x2) = (偶数维, 奇数维)
           旋转输出 = concat(x1·cos − x2·sin, x2·cos + x1·sin)（按各自顺序回排）。
      2. Attention（softmax 对 j = 0..seq_len_b-1 全体）：
           s[h, j] = sm_scale · ( <q_nope[b,h,:], k_lat[b,j,:]>
                                + <q_pe'[b,h,:], k_pe'[b,j,:]> )
         其中 ' 表示旋转后；k_pe' 仅最后一个 token 用旋转后的值，其余取 cache
         原值。
           p[h, :] = softmax_j(s[h, :])
           attn_out[b, h, :] = Σ_j p[h, j] · k_lat[b, j, :]
      3. k_pe_tokens[b, :] = 最后一个 token 旋转后的完整 k_pe（r 维，用于写回
         cache）。
    softmax 采用在线（数值稳定）实现，中间累加 float32，输出 cast 回输入
    dtype。边界行为：seq_len_b >= 1（等于 1 时该 token 即"最后一个 token"，
    其 k_pe 参与旋转）；KV 切片并行（num_kv_splits 等）只是实现细节，切片
    在线 softmax + log-sum-exp 归并与全量 softmax 数学等价，不改变输出；
    无 logit 封顶、无因果掩码（解码天然 attend 全部 cache）、无量化。

    输入输出规格：
      q              [B, H, c + r] float16/bfloat16
      kv_cache       [total_tokens, c + r] 与 q 同 dtype（token 级 KV 行池）
      cos_sin_cache  [max_positions, rd] 与 q 同 dtype（前 half 列 cos、后 half
                     列 sin；第 t 行对应位置 t = 0,1,...）
      positions      [B] int32（当前 query 的序列位置；取值 < max_positions，
                     与 seq_len 独立）
      kv_indptr      [B+1] int32（序列长度累积和：首元素 0、单调不减）
      kv_indices     [total_kv] int32（逻辑 token -> 物理行号，取值
                     < total_tokens；total_kv == kv_indptr[-1]）
      out            1-D 平铺拼接，长度 B·H·c + B·r，与 q 同 dtype：
                     前 B·H·c 个 = attn_out（[B, H, c] 行主序展平），
                     后 B·r 个 = k_pe_tokens（[B, r] 行主序展平）
    全域约束：rd 为 2 的幂且 rd <= r；total_tokens >= kv_indptr[-1]；
    每条序列 seq_len_b >= 1；B、H、c、r、max_positions 由 shape 族给定。

    实现约束（违规判负）：
      - 核心计算（RoPE 旋转 / KV 收集 / QK^T / softmax / PV）必须在提交文件内
        完成，禁止调用 scaled_dot_product_attention / sdpa / flash_attn /
        aiter / torch.matmul / torch.bmm / torch.einsum / torch.softmax。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。
      - 输出的 1-D 拼接布局必须与上述规格一致（评测器逐元素比较）。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：单 token decode、use_rope 恒真、非量化 fp16/bf16、无 logit
    封顶、无因果掩码 / ALiBi / 滑窗。
    """

    def __init__(self, kv_lora_rank: int, qk_rope_head_dim: int, rotary_dim: int, sm_scale=None, is_neox_style=True):
        super().__init__()
        self.kv_lora_rank = int(kv_lora_rank)
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.rotary_dim = int(rotary_dim)
        self.sm_scale = float(sm_scale) if sm_scale is not None else 1.0 / math.sqrt(self.kv_lora_rank + self.qk_rope_head_dim)
        self.is_neox_style = bool(is_neox_style)

    def forward(self, q, kv_cache, cos_sin_cache, positions, kv_indptr, kv_indices):
        # KernelBench 评测器会把全部输入张量 cast 成评测 precision（triton 后端
        # 为 fp32）；行号 / 累积长度 / 位置都是小整数，fp32 可精确表示，此处
        # 无损恢复
        positions = positions.to(torch.long).reshape(-1)
        kv_indptr = kv_indptr.to(torch.long)
        kv_indices = kv_indices.to(torch.long)

        c, rd = self.kv_lora_rank, self.rotary_dim
        half = rd // 2
        B, H, D = q.shape
        r = D - c
        assert r == self.qk_rope_head_dim, "q 末维必须等于 kv_lora_rank + qk_rope_head_dim"

        def apply_rope(x, cos, sin):
            # 对 float32 张量 x 的末维（宽度 r）就地旋转：前 rd 维旋转、其余直通
            rot = x[..., :rd]
            if self.is_neox_style:
                x1, x2 = rot[..., :half], rot[..., half:]
            else:
                x1, x2 = rot[..., ::2], rot[..., 1::2]
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            if self.is_neox_style:
                rot_out = torch.cat((o1, o2), dim=-1)
            else:
                rot_out = torch.stack((o1, o2), dim=-1).flatten(-2)
            return torch.cat((rot_out, x[..., rd:]), dim=-1)

        attn_out = torch.empty(B, H, c, dtype=q.dtype, device=q.device)
        k_pe_tokens = torch.empty(B, r, dtype=q.dtype, device=q.device)
        for b in range(B):
            s0, s1 = int(kv_indptr[b]), int(kv_indptr[b + 1])
            toks = kv_indices[s0:s1]                                 # 本批 token 物理行
            kv_b = kv_cache.index_select(0, toks).to(torch.float32)  # [S, c+r]（副本）
            k_lat = kv_b[:, :c]                                      # 压缩隐向量（K/V 共享）
            k_pe = kv_b[:, c:]

            pos = int(positions[b])
            cos = cos_sin_cache[pos, :half].to(torch.float32)
            sin = cos_sin_cache[pos, half:].to(torch.float32)

            q_b = q[b].to(torch.float32)                             # [H, c+r]
            q_pe = apply_rope(q_b[:, c:], cos, sin)                  # q_pe 旋转
            k_pe[-1] = apply_rope(k_pe[-1], cos, sin)                # 仅最后 token 的 k_pe 旋转
            k_pe_tokens[b] = k_pe[-1].to(q.dtype)

            qk = torch.matmul(q_b[:, :c], k_lat.transpose(0, 1))     # nope 部分内积
            qk += torch.matmul(q_pe, k_pe.transpose(0, 1))           # rope 部分内积
            qk *= self.sm_scale
            probs = torch.softmax(qk, dim=-1)                        # [H, S]
            attn_out[b] = torch.matmul(probs, k_lat).to(q.dtype)
        # 单张量输出：attn_out[B,H,c] ‖ k_pe_tokens[B,r]（行主序平铺拼接）
        return torch.cat((attn_out.reshape(-1), k_pe_tokens.reshape(-1)))


def get_init_inputs():
    return [512, 64, 64]   # kv_lora_rank, qk_rope_head_dim, rotary_dim；sm_scale
    # 缺省 1/sqrt(c+r)，is_neox_style 缺省 True


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    batch = 2
    num_heads = 128
    seq_len = 1024
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    rotary_dim = 64
    rope_base = 10.0
    max_positions = 16384

    total_kv = batch * seq_len
    q = torch.randn(batch, num_heads, kv_lora_rank + qk_rope_head_dim).to(torch.bfloat16)
    # 行池大于实际用量：未引用行是垃圾数据，用于检验间接寻址正确性
    kv_cache = torch.randn(total_kv + 8, kv_lora_rank + qk_rope_head_dim).to(torch.bfloat16)
    kv_indptr = (torch.arange(batch + 1) * seq_len).to(torch.int32)
    kv_indices = torch.randperm(total_kv + 8)[:total_kv].to(torch.int32)
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(max_positions, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)
    # 官方约定：当前 query 位置 == 该批 KV 长度
    positions = torch.full((batch,), seq_len, dtype=torch.int32)
    return [q, kv_cache, cos_sin_cache, positions, kv_indptr, kv_indices]


def make_inputs(batch: int, num_heads: int, seq_lens, kv_lora_rank: int,
                qk_rope_head_dim: int, rotary_dim: int, dtype: str = "bfloat16",
                is_neox_style: bool = True, positions=None, rope_base: float = 10.0,
                max_positions: int = 16384, extra_tokens: int = 8, seed: int = 0):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    seq_lens 显式给出每批 KV 长度（单个值广播到整个 batch；缺省每批随机
    1..1024）；positions 缺省取 seq_lens（官方约定：当前 query 位置 == 序列
    长度），可显式指定以解耦；is_neox_style 是随 case 传递的 Model init 超参
    （不影响输入张量内容，评测端用它构造 Model）；extra_tokens > 0 时物理
    行池追加等量垃圾行。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    assert rotary_dim % 2 == 0 and 0 < rotary_dim <= qk_rope_head_dim, \
        "rotary_dim 必须为 2 的倍数且 <= qk_rope_head_dim"
    if seq_lens is None:
        seq_lens = torch.randint(1, 1025, (batch,), generator=gen).tolist()
    if len(seq_lens) == 1 and batch > 1:      # 单值广播到整个 batch
        seq_lens = seq_lens * batch
    assert len(seq_lens) == batch and all(int(s) >= 1 for s in seq_lens), \
        "seq_lens 长度必须等于 batch 且每条 >= 1"
    if positions is None:
        positions = list(seq_lens)
    positions = [int(p) for p in positions]
    assert len(positions) == batch and max(positions) < max_positions, \
        "positions 长度必须等于 batch 且取值 < max_positions"

    total_kv = sum(int(s) for s in seq_lens)
    q = torch.randn(batch, num_heads, kv_lora_rank + qk_rope_head_dim, generator=gen).to(dt)
    kv_cache = torch.randn(total_kv + extra_tokens, kv_lora_rank + qk_rope_head_dim,
                           generator=gen).to(dt)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(torch.tensor([int(s) for s in seq_lens], dtype=torch.int64),
                                 dim=0).to(torch.int32)
    kv_indices = torch.randperm(total_kv + extra_tokens, generator=gen)[:total_kv].to(torch.int32)

    inv_freq = 1.0 / (rope_base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    t = torch.arange(max_positions, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dt)
    positions_t = torch.tensor(positions, dtype=torch.int32)
    return q, kv_cache, cos_sin_cache, positions_t, kv_indptr, kv_indices
