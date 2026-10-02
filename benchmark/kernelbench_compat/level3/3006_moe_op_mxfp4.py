# 3006_moe_op_mxfp4 — MoE 分组 GEMM（mxfp4 e2m1 打包激活 × mxfp4 e2m1 打包
# 专家权重，e8m0 块 scale，含路由权重乘）的 model_class（KernelBench 兼容）
# 题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=3006, backend=triton。

import torch
import torch.nn as nn

# e2m1 nibble（4 bit）解码表：bit3 为符号位，低 3 位查本表
_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _dequant_mxfp4(codes, scales):
    """mxfp4 反量化：(..., K//2) uint8 码字 + (..., K//32) uint8 e8m0 块 scale
    -> (..., K) float32。每字节低 4 位是偶下标元素、高 4 位是奇下标元素；
    x[..., k] = e2m1(nibble) * 2^(scales[..., k // 32] - 127)。"""
    lo = codes & 0xF
    hi = codes >> 4
    nib = torch.stack((lo, hi), dim=-1).flatten(-2)      # (..., K)
    x = _E2M1_TABLE[nib.long()]
    scale = torch.exp2(scales.to(torch.float32) - 127.0)
    return x * scale.repeat_interleave(32, dim=-1)


def _mxfp4_quant(x):
    """MXFP4 动态量化（块大小 32、even 圆整），与官方测试的输入生成器逐位
    一致。仅在输入生成侧使用，不属于 Agent 需要实现的计算。"""
    MXFP4_QUANT_BLOCK_SIZE = 32
    x_shape = x.shape
    if x.shape[-1] % MXFP4_QUANT_BLOCK_SIZE != 0:
        shape = list(x_shape)
        shape = shape[:-1] + [
            ((shape[-1] - 1 + MXFP4_QUANT_BLOCK_SIZE) // MXFP4_QUANT_BLOCK_SIZE)
            * MXFP4_QUANT_BLOCK_SIZE
        ]
        x_padded = torch.zeros(tuple(shape), dtype=x.dtype)
        x_padded[..., : x.shape[-1]] = x
    else:
        x_padded = x

    # 块 scale：amax 圆整到 2 的幂，e = clamp(floor(log2(amax)) - 2, -127, 127)
    x_padded = x_padded.reshape(
        -1, x_padded.shape[-1] // MXFP4_QUANT_BLOCK_SIZE, MXFP4_QUANT_BLOCK_SIZE
    ).to(torch.float32)
    amax, _ = torch.max(torch.abs(x_padded), dim=-1)
    amax = amax.view(torch.int32)
    amax = (amax + 0x200000) & 0xFF800000
    amax = amax.view(torch.float32)
    scale_e8m0_unbiased = torch.log2(amax).floor() - 2
    scale_e8m0_unbiased = torch.clamp(scale_e8m0_unbiased, min=-127, max=127)
    quant_scale = torch.exp2(-scale_e8m0_unbiased)

    qx = x_padded * quant_scale.unsqueeze(-1)
    bs_e8m0 = scale_e8m0_unbiased.to(torch.uint8) + 127

    # e2m1 舍入：round-half-up，超出饱和（对 (E<<2 | M>>21) + 1 >> 1 取 min(·,7)）
    qx = qx.view(torch.int32)
    s = qx & 0x80000000
    e = (qx >> 23) & 0xFF
    m = qx & 0x7FFFFF

    E8_BIAS = 127
    E2_BIAS = 1
    adjusted_exponents = E8_BIAS - e - 1
    m = torch.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)
    e = torch.where(e > E8_BIAS - E2_BIAS, e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

    combined_val = (((e << 2) | (m >> 21)) + 1) >> 1
    e2m1_tmp = torch.where(combined_val < 0x7, combined_val, 0x7)
    e2m1_value = (((s >> 28) & 0xF) | e2m1_tmp).to(torch.uint8)

    # 每字节打包 2 个 4-bit 码字：偶数下标在低 4 位
    x_mxfp4 = e2m1_value[..., ::2] | (e2m1_value[..., 1::2] << 4)
    x_mxfp4 = torch.flatten(x_mxfp4, -2, -1)

    if x.shape[-1] % MXFP4_QUANT_BLOCK_SIZE != 0:
        x_mxfp4 = x_mxfp4[..., : x.shape[-1] // 2]

    mxfp4_shape = tuple(list(x_shape)[:-1] + [x_shape[-1] // 2])
    x_mxfp4 = x_mxfp4.reshape(mxfp4_shape)
    bs_shape = tuple(list(x_shape)[:-1] + [x_shape[-1] // MXFP4_QUANT_BLOCK_SIZE])
    bs_e8m0 = bs_e8m0.reshape(bs_shape)
    return x_mxfp4, bs_e8m0


def _gen_case(M, N, K, E, top_k, dtype, generator):
    # 官方测试生成器：randn 激活/权重 -> 动态 mxfp4 量化；softmax+topk 路由。
    # 消费顺序固定：a_raw -> b_raw -> values。
    a_raw = torch.randn(M, K, dtype=dtype, generator=generator)
    b_raw = torch.randn(E, N, K, dtype=dtype, generator=generator)
    values = torch.randn(M, E, dtype=dtype, generator=generator)
    softmax_vals = torch.softmax(values, dim=1)
    topk_weights, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)
    a_q, a_scales = _mxfp4_quant(a_raw)
    b_q, b_scales = _mxfp4_quant(b_raw)
    return [a_q, a_scales, b_q, b_scales, topk_weights, topk_ids]


class Model(nn.Module):
    """MoE（Mixture of Experts）分组 GEMM：mxfp4 量化激活 × mxfp4 量化专家权重。

    路由已完成（topk_ids 给出每个 token 选中的专家），本算子对每个
    (token, 选中专家) 槽位做一次 GEMM 并乘以路由权重（mul_routed_weight
    恒为 True）：
      c[m, j, n] = topk_weights[m, j] * Σ_k adeq[m, k] * bdeq[topk_ids[m, j], n, k]
    求和沿 K 维、中间累加 float32，结果 cast 到 bfloat16 输出。

    mxfp4 反量化定义（对激活 a 与权重 b 相同）：
      - 码字 tensor 最后一维每字节含 2 个 e2m1 码：低 4 位是偶下标元素，
        高 4 位是奇下标元素，即 code[..., i] 的低/高 nibble 对应元素
        2i / 2i+1。
      - e2m1 nibble（4 bit）解码：bit3 为符号位，低 3 位查表
        {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 3.0, 6: 4.0, 7: 6.0}，
        符号位为 1 时取负（-0.0 视作 0 参与乘加）。
      - e8m0 块 scale：最后一维每 32 个元素共享 1 个 uint8 scale 字节
        s = scales[..., k // 32]，解码为 2^(s - 127)。
      - 反量化：x[..., k] = e2m1(nibble(..., k)) * 2^(scales[..., k // 32] - 127)。

    输入输出规格：
      a_q          (M, K//2)     uint8   激活码字（K 为逻辑特征维）
      a_scales     (M, K//32)    uint8   激活 e8m0 块 scale
      b_q          (E, N, K//2)  uint8   专家权重码字（E 个专家）
      b_scales     (E, N, K//32) uint8   专家权重 e8m0 块 scale
      topk_weights (M, top_k)    float16/bfloat16/float32  路由权重（>0）
      topk_ids     (M, top_k)    int64   选中专家编号，取值 ∈ [0, E)
      out          (M, top_k, N) bfloat16
      约束：M >= 1、E >= 1、N >= 1、K % 32 == 0、1 <= top_k <= E。
      边界行为：同一专家可被多个 token 选中，同一 token 的多个槽位也可选中
      同一专家；各 (m, j) 槽位独立计算，不做 expert 间的归约/加权合并，
      也不引入 dispatch 合并语义（那是下游算子的事）。

    实现约束（违规判负）：
      - 核心计算（nibble 解包、e2m1/e8m0 解码、按 topk_ids 分组的 GEMM、
        路由权重相乘）必须在提交文件内完成，禁止调用 torch.matmul /
        torch.bmm / torch.mm / torch.addmm / torch.einsum / F.linear /
        functional.linear / torch._scaled_mm，禁止使用 torch 内置
        float4/mxfp4 表示与现成量化转换 API（torch.float4_e2m1fn_x2、
        torch.quantize_per_tensor 等）。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 到 bfloat16。
      - 评测器会把全部输入 cast 成 fp32 后传入 forward（uint8 码字与 scale
        字节（0..255）、int64 专家编号（小整数）在 fp32 中均精确表示），
        需在入口无损恢复（.to(torch.uint8) / .to(torch.long)）。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：mxfp4 激活 × mxfp4 权重的 MoE 分组 GEMM + 路由权重乘；
    token-expert 对齐调度元数据（sorted_token_ids / expert_ids /
    num_tokens_post_padded）属实现细节，不在输入契约内。
    """

    def __init__(self):
        super().__init__()

    def forward(self, a_q, a_scales, b_q, b_scales, topk_weights, topk_ids):
        # 评测器把全部输入 cast 成 fp32：码字/scale 字节与专家编号在 fp32
        # 中精确表示，此处无损恢复
        a_q = a_q.to(torch.uint8)
        a_scales = a_scales.to(torch.uint8)
        b_q = b_q.to(torch.uint8)
        b_scales = b_scales.to(torch.uint8)
        topk_ids = topk_ids.to(torch.long)
        topk_weights = topk_weights.to(torch.float32)

        M, Kh = a_q.shape
        K = Kh * 2
        E, N, _ = b_q.shape
        top_k = topk_ids.shape[1]

        a = _dequant_mxfp4(a_q, a_scales)     # (M, K) float32
        b = _dequant_mxfp4(b_q, b_scales)     # (E, N, K) float32

        # 按 expert 分组做 GEMM（数学上等价于先 b[topk_ids] 再
        # einsum("mek,menk->men")，分组写回可避免物化大中间量）
        flat_ids = topk_ids.reshape(-1)                     # (M*top_k,)
        a_rep = a.unsqueeze(1).expand(M, top_k, K).reshape(-1, K)
        c = torch.empty((M * top_k, N), dtype=torch.float32)
        for e in range(E):
            mask = flat_ids == e
            if mask.any():
                c[mask] = a_rep[mask] @ b[e].t()
        c *= topk_weights.reshape(-1, 1)
        return c.reshape(M, top_k, N).to(torch.bfloat16)


def get_init_inputs():
    return []   # 无超参：全部维度由输入张量形状给出


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性。量级与 make_inputs 一致。
    return _gen_case(64, 128, 128, 8, 2, torch.bfloat16, None)


def make_inputs(M: int, N: int, K: int, E: int, top_k: int,
                dtype: str = "bfloat16", seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到
    设备）。消费顺序固定：a_raw -> b_raw -> values，全部来自同一
    torch.Generator(seed)，同 seed 下逐位可复现。"""
    gen = torch.Generator().manual_seed(seed)
    return _gen_case(M, N, K, E, top_k, getattr(torch, dtype), gen)
