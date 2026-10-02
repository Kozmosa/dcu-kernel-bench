# 1027_sage_attention_qk_int8_per_block_causal — SageAttention 风格因果注意力前向（KernelBench 兼容题目文件）。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=1027, backend=triton。

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    """SageAttention2 风格的因果（causal）scaled dot-product attention 前向：
    Q/K 采用按块（per-block）INT8 量化，P 以 float16 与 V 相乘（块内 dot 输出
    float16、跨块累加 float32）的在线 softmax 实现（HND 布局，MHA/GQA 通用）。

    ① 算子语义（验收真值）：对 batch b、query head h（其 KV head 为
    h // g，g = num_qo_heads // num_kv_heads）与 query 位置 i（记
    L = qo_len = kv_len）：
      logits[i,j] = sm_scale * <q[b,h,i,:], k[b,h//g,j,:]>，
      sm_scale = 1/sqrt(head_dim)
      因果掩码：j > i 的位置记 -inf（query i 只 attend key 0..i）
      p[i,:]  = softmax_j(logits[i, 0..i])
      out[b,h,i,:] = sum_{j<=i} p[i,j] * v[b,h//g,j,:]
    softmax 用数值稳定（在线）实现，中间计算 float32，输出 cast 回输入 dtype。
    边界行为：i = 0 只 attend 自身（此时 out[b,h,0,:] = v[b,h//g,0,:]）；
    每个 query 行恒有可见 key，softmax 分母恒正，不存在全掩码行。

    ② 输入输出规格：
      q   [batch, num_qo_heads, qo_len, head_dim]  float16/bfloat16
      k   [batch, num_kv_heads,  qo_len, head_dim] 与 q 同 dtype
      v   [batch, num_kv_heads,  qo_len, head_dim] 与 q 同 dtype
      out [batch, num_qo_heads, qo_len, head_dim]  与 q 同 dtype
    全域约束：qo_len == kv_len（因果）；num_qo_heads 是 num_kv_heads 的整数倍
    （GQA，相等即 MHA）；head_dim ∈ {64, 128}；HND 布局、最后一维连续。

    ③ 实现约束（违规判负）：
      - 核心计算（INT8 量化 / QK^T / softmax / PV）必须在提交文件内完成。
      - 按题面算法实现：Q 每 128 行一块、K 每 64 行一块做 per-block INT8
        量化——Q 的量化把 sm_scale*log2(e) 折入被量化值，K 的量化不折入
        scale；每块 scale = max|x|/127，int8 取 round-half-away-from-zero；
        qk = int8_dot(q,k) * (q_scale * k_scale) 即 exp2 域 logits；在线
        softmax 在 exp2 域逐 KV 块更新；P cast 成 float16 与 float16 V 做
        dot（块内输出 float16），累加进 float32 acc；末块后除以 softmax
        分母。
      - 可先对 K 沿序列维去均值（smooth-k）再量化：对精确因果注意力的输出
        数学中性（每条 query 行的 logits 被减去同一常数），仅改善量化精度，
        不改变验收真值。
      - 验收容差 atol=rtol=2e-2 已涵盖上述量化方案的近似误差。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。

    ④ 禁用列表：scaled_dot_product_attention / sdpa / flash_attn /
      torch.matmul / torch.bmm / torch.einsum / torch.softmax，以及任何
      第三方算子库的同义封装。
      目标硬件：海光 DCU（gfx936），实现语言 Triton。
      本题范围：因果前向、HND 布局、无 attn_mask、无 LSE 输出、无 dropout；
      非因果分支与 NHD 布局不在本题内。
    """

    def __init__(self, head_dim: int, scale=None):
        super().__init__()
        self.sm_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)

    def forward(self, q, k, v):
        # 本题三个输入均为浮点张量（HND 布局），无整型索引/长度类输入，无需恢复
        B, Hq, L, D = q.shape
        Bk, Hkv, Lk, Dk = k.shape
        assert B == Bk and L == Lk and D == Dk, "题面约束：qo_len == kv_len（因果）"
        group = Hq // Hkv
        assert Hq == Hkv * group, "num_qo_heads 必须是 num_kv_heads 的整数倍"

        causal_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=q.device), diagonal=1
        )
        out = torch.empty_like(q)
        for b in range(B):
            qf = q[b].to(torch.float32) * self.sm_scale          # [Hq, L, D]
            kf = k[b].to(torch.float32)                           # [Hkv, L, D]
            vf = v[b].to(torch.float32)                           # [Hkv, L, D]
            # GQA：同一 KV head 的 K/V 复制给 group 内的每个 query head
            if group > 1:
                kf = kf.repeat_interleave(group, dim=0)           # [Hq, L, D]
                vf = vf.repeat_interleave(group, dim=0)           # [Hq, L, D]

            scores = torch.matmul(qf, kf.transpose(-2, -1))       # [Hq, L, L]
            scores = scores.masked_fill(causal_mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[b] = torch.matmul(probs, vf).to(q.dtype)
        return out


def get_init_inputs():
    return [128]  # head_dim；sm_scale 缺省 1/sqrt(head_dim)


def get_inputs():
    # 固定 shape 族（qo_len 除外）；随机部分消费全局 RNG——评测器在 set_seed
    # 后调用本函数，多轮 correctness trial 因此获得输入多样性
    batch = 2
    num_kv_heads = 4
    query_group_size = 4      # num_qo_heads = 16
    head_dim = 128
    max_qo_len = 512

    num_qo_heads = num_kv_heads * query_group_size
    qo_len = int(torch.randint(1, max_qo_len + 1, (1,)))

    q = torch.randn(batch, num_qo_heads, qo_len, head_dim).to(torch.float16)
    k = torch.randn(batch, num_kv_heads, qo_len, head_dim).to(torch.float16)
    v = torch.randn(batch, num_kv_heads, qo_len, head_dim).to(torch.float16)
    return [q, k, v]


def make_inputs(batch: int, num_qo_heads: int, num_kv_heads: int, qo_len: int,
                head_dim: int, dtype: str = "float16", seed: int = 0):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    因果题面约束 qo_len == kv_len，kv 张量与 q 共用同一 qo_len；GQA 要求
    num_qo_heads 是 num_kv_heads 的整数倍。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert num_qo_heads % num_kv_heads == 0, "num_qo_heads 必须是 num_kv_heads 的整数倍"

    q = torch.randn(batch, num_qo_heads, qo_len, head_dim, generator=gen).to(dt)
    k = torch.randn(batch, num_kv_heads, qo_len, head_dim, generator=gen).to(dt)
    v = torch.randn(batch, num_kv_heads, qo_len, head_dim, generator=gen).to(dt)
    return q, k, v
