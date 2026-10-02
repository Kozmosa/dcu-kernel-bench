# 1026_sage_attention_qk_int8_per_block — SageAttention 风格非 causal 前向
# （Q/K per-block int8 码字 + per-block scale，PV fp16）的 model_class
# （KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=1026, backend=triton。

import torch
import torch.nn as nn

# log2(e)：量化阶段把 softmax scale 与 log2(e) 一起折入 q_scale，kernel 侧
# 以 2 为底做在线 softmax；反量化回自然对数域需除以该常数
_LOG2E = 1.4426950408889634


def _quant_per_block_int8(x, blk):
    """对 [b, h, L, d] 沿 L 维每 blk 行一块做 absmax/127 对称 int8 量化。

    复刻源量化 kernel 的数值行为：块内全部元素的 absmax 作 scale（fp32），
    x/scale 后加 0.5*sign 再向零截断（round-half-away-from-zero）。
    返回 (int8 码字 [b,h,L,d], scale [b,h,ceil(L/blk)] fp32)。
    """
    b, h, length, d = x.shape
    n_blk = (length + blk - 1) // blk
    pad = n_blk * blk - length
    if pad:
        x = nn.functional.pad(x, (0, 0, 0, pad))
    blocks = x.reshape(b, h, n_blk, blk * d)
    scale = blocks.abs().amax(dim=-1) / 127.0
    t = blocks / scale[..., None]
    t = torch.trunc(t + 0.5 * torch.where(t >= 0, 1.0, -1.0))
    q = t.to(torch.int8).reshape(b, h, n_blk * blk, d)
    if pad:
        q = q[:, :, :length, :]
    return q.contiguous(), scale.contiguous()


def _sage_quantize(q, k, v, sm_scale=None):
    """把 HND 布局的 fp16 q/k/v 变换成本题前向的输入（量化阶段仿真）。

    smooth_k 默认开启：k 先减去沿序列维的均值（每行 logit 恒减 q·km，softmax
    语义不变）；q 的量化把 sm_scale*log2(e) 折入 scale，k 的量化不折 scale。
    Q 按 128 行分块，K 按 64 行分块。
    """
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = head_dim ** -0.5
    km = k.float().mean(dim=2, keepdim=True)          # [b, h_kv, 1, d]
    k_smooth = k.float() - km
    q_int8, q_scale = _quant_per_block_int8(q.float() * (sm_scale * _LOG2E), 128)
    k_int8, k_scale = _quant_per_block_int8(k_smooth, 64)
    return q_int8, k_int8, v, q_scale, k_scale


class Model(nn.Module):
    """SageAttention 风格的非 causal scaled dot-product attention 前向：
    Q/K 已按块做 int8 对称量化（per-block absmax/127 scale），V 保持 fp16。

    数学定义（g = num_q_heads // num_kv_heads，query 头 h 对应 KV 头 h//g；
    量化分块：Q 沿序列方向每 128 行一块、K 每 64 行一块，块编号分别为
    i//128 与 j//64，即 q_scale/k_scale 最后一维的索引）：
      dq[b,h,i,:]    = q_int8[b,h,i,:] * q_scale[b,h,i//128]    （float32 反量化）
      dk[b,hk,j,:]   = k_int8[b,hk,j,:] * k_scale[b,hk,j//64]
      logit[b,h,i,j] = <dq[b,h,i,:], dk[b,h//g,j,:]> / log2(e)
      （除以 log2(e)：量化阶段已把 softmax scale 与 log2(e) 折入 q_scale，
        源 kernel 以 2 为底做在线 softmax，此处统一回自然对数域）
      attn_mask[i,j] 为 False 的位置 logit 记 -inf（bool 掩码，
        形状 [qo_len, kv_len]，对全部 batch 与头共享）
      p[b,h,i,:]   = softmax_j(logit[b,h,i,:])
      out[b,h,i,:] = sum_j p[b,h,i,j] * v[b,h//g,j,:]
    softmax 采用数值稳定的在线实现语义；参考实现全程 float32（int8 码字的
    点积在 float32 下是精确整数算术），输出 cast 回 v 的 dtype。
    数值方案（源算子采用、提交可沿用）：p 转 fp16 与 v 做 fp16 矩阵乘，
    每个 KV 分片的结果累加进 float32 缓冲（"PV fp16 + fp32 分片累加"），
    int8 QK^T 用 int8 输入的矩阵乘直接整数累加——无论采用哪种数值方案，
    都必须在容差内匹配本参考实现。
    边界行为：qo_len 与 kv_len 可不相等，也无需是分块大小的整数倍（尾块
    照常参与反量化与 softmax）；题面保证每个 query 行至少有一个未掩码
    位置，softmax 分母恒为正。

    输入输出规格（tensor_layout="HND"/"NHD" 由 __init__ 指定）：
      q_int8   int8     HND [batch, num_q_heads, qo_len, head_dim]
                         NHD [batch, qo_len, num_q_heads, head_dim]
      k_int8   int8     HND [batch, num_kv_heads, kv_len, head_dim]
                         NHD [batch, kv_len, num_kv_heads, head_dim]
      v        float16  与 k_int8 同布局、同 KV 头数
      q_scale  float32  [batch, num_q_heads, ceil(qo_len/128)]（与布局无关）
      k_scale  float32  [batch, num_kv_heads, ceil(kv_len/64)]（与布局无关）
      attn_mask bool    [qo_len, kv_len]（与布局无关）
      out      与 q_int8 同形状，dtype 同 v（单张量输出）
    全域约束：num_q_heads 是 num_kv_heads 的整数倍（GQA；相等即 MHA）；
    head_dim ∈ {64, 128}；qo_len, kv_len >= 1。

    实现约束（违规判负）：
      - 核心计算（int8 QK^T / 反量化 / softmax / PV）必须在提交文件内
        完成，禁止调用 scaled_dot_product_attention / sdpa / flash_attn /
        torch.matmul / torch.bmm / torch.mm / torch.addmm / F.linear /
        torch.einsum / torch.softmax。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。
      - 评测器会把全部输入 cast 成 fp32 后传入 forward：int8 码字、fp16 的
        v 与 bool 掩码均可被 fp32 精确表示，入口处无损恢复原 dtype。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：非 causal 前向、bool 掩码、GQA、HND/NHD 两种布局；不含
    causal 变体、float 加性掩码、LSE 输出与量化阶段本身（int8 码字由
    输入直接给出）。
    """

    def __init__(self, tensor_layout="HND"):
        super().__init__()
        assert tensor_layout in ("HND", "NHD"), "tensor_layout 只支持 HND / NHD"
        self.tensor_layout = tensor_layout

    def forward(self, q_int8, k_int8, v, q_scale, k_scale, attn_mask):
        # 说明：forward 全部使用单行语句（条件用条件表达式），保证 loader 把
        # 逐条语句提取进 ModelNew scaffold 时不破坏缩进
        q_int8 = q_int8.to(torch.int8)
        k_int8 = k_int8.to(torch.int8)
        v = v.to(torch.float16)
        q_scale = q_scale.to(torch.float32)
        k_scale = k_scale.to(torch.float32)
        attn_mask = attn_mask.to(torch.bool)
        nhd = self.tensor_layout == "NHD"
        # NHD [b,l,h,d] 统一换轴成 HND [b,h,l,d] 视角计算
        q = (q_int8.permute(0, 2, 1, 3) if nhd else q_int8).to(torch.float32)
        k = (k_int8.permute(0, 2, 1, 3) if nhd else k_int8).to(torch.float32)
        vv = (v.permute(0, 2, 1, 3) if nhd else v).to(torch.float32)
        b, h_q, qo_len, head_dim = q.shape
        h_kv, kv_len = k.shape[1], k.shape[2]
        group = h_q // h_kv
        assert h_q == h_kv * group, "num_q_heads 必须是 num_kv_heads 的整数倍"
        # GQA：同一 KV head 的 K/V（及 k_scale）复制给 group 内的每个 query head
        k = k.repeat_interleave(group, dim=1)
        vv = vv.repeat_interleave(group, dim=1)
        k_scale = k_scale.repeat_interleave(group, dim=1)
        # per-block scale 展开到序列维：Q 块 128 行、K 块 64 行
        qs = q_scale.repeat_interleave(128, dim=-1)[..., :qo_len]   # [b,h_q,qo_len]
        ks = k_scale.repeat_interleave(64, dim=-1)[..., :kv_len]    # [b,h_q,kv_len]
        # int8 码字点积（fp32 下为精确整数算术）后按块反量化，除以 log2(e)
        # 换回自然对数域（常数内联，保证 scaffold 自包含）
        logits = torch.matmul(q, k.transpose(-2, -1))
        logits = logits * qs[:, :, :, None] * ks[:, :, None, :] / 1.4426950408889634
        logits = logits.masked_fill(~attn_mask, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        out = torch.matmul(probs, vv)
        out = out.permute(0, 2, 1, 3) if nhd else out
        return out.contiguous().to(v.dtype)


def get_init_inputs():
    return ["HND"]   # tensor_layout


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    batch = 2
    num_kv_heads = 4
    query_group_size = 2      # num_q_heads = 8（GQA）
    qo_len = 256
    kv_len = 512
    head_dim = 64

    num_q_heads = num_kv_heads * query_group_size
    q = torch.randn(batch, num_q_heads, qo_len, head_dim).to(torch.float16)
    k = torch.randn(batch, num_kv_heads, kv_len, head_dim).to(torch.float16)
    v = torch.randn(batch, num_kv_heads, kv_len, head_dim).to(torch.float16)
    q_int8, k_int8, v, q_scale, k_scale = _sage_quantize(q, k, v)

    # bool 掩码：随机丢弃约 15% 位置，第 0 列恒开，保证每行至少一个未掩码位
    attn_mask = torch.rand(qo_len, kv_len) > 0.15
    attn_mask[:, 0] = True
    return [q_int8, k_int8, v, q_scale, k_scale, attn_mask]


def make_inputs(batch: int, num_q_heads: int, num_kv_heads: int, qo_len: int,
                kv_len: int, head_dim: int, tensor_layout: str = "HND",
                mask_mode: str = "all_true", dtype: str = "float16",
                seed: int = 0):
    """确定性 case 生成器（CPU，torch.Generator 按 seed 初始化）。

    先按 HND 视角生成 fp16 q/k/v，经 smooth_k 与 per-block int8 量化
    （Q 块 128 行、K 块 64 行，sm_scale 折入 q_scale），再按需转到
    NHD 布局。mask_mode："all_true" 全开；"random" 随机掩掉约 15%
    位置并强制第 0 列为 True。字段与 hidden/perf case 一一对应。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)

    group = num_q_heads // num_kv_heads
    assert num_q_heads == num_kv_heads * group, "num_q_heads 必须是 num_kv_heads 的整数倍"
    assert head_dim in (64, 128), "head_dim 仅支持 64 / 128"
    assert tensor_layout in ("HND", "NHD"), "tensor_layout 只支持 HND / NHD"

    q = torch.randn(batch, num_q_heads, qo_len, head_dim, generator=gen).to(dt)
    k = torch.randn(batch, num_kv_heads, kv_len, head_dim, generator=gen).to(dt)
    v = torch.randn(batch, num_kv_heads, kv_len, head_dim, generator=gen).to(dt)
    q_int8, k_int8, v, q_scale, k_scale = _sage_quantize(q, k, v)

    if mask_mode == "random":
        attn_mask = torch.rand(qo_len, kv_len, generator=gen) > 0.15
        attn_mask[:, 0] = True
    else:
        attn_mask = torch.ones(qo_len, kv_len, dtype=torch.bool)

    if tensor_layout == "NHD":
        q_int8 = q_int8.permute(0, 2, 1, 3).contiguous()
        k_int8 = k_int8.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
    return [q_int8, k_int8, v, q_scale, k_scale, attn_mask]
