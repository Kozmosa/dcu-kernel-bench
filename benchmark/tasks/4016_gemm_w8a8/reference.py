# 4016_gemm_w8a8 — int8 分块量化 GEMM（激活 per-token-group ×128 scale、
# 权重 128×128 块 scale）的 model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=4016, backend=triton。

import torch
import torch.nn as nn


def _per_token_group_quant_int8(x: torch.Tensor, group_k: int = 128):
    """按行分组（group_k 个元素一组）的 int8 量化：组内 absmax 定标。

    返回 (码字 int8 与 x 同形, scale float32 (m, ceil(k/group_k)))。
    尾组不足 group_k 时按有效元素取 absmax，补零元素量化后裁掉。
    """
    m, k = x.shape
    g = (k + group_k - 1) // group_k
    xp = torch.zeros(m, g * group_k, dtype=x.dtype)
    xp[:, :k] = x
    xg = xp.view(m, g, group_k).to(torch.float32)
    absmax = xg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = (absmax / 127.0).squeeze(-1)                       # (m, g) float32
    q = torch.clamp(torch.round(xg / scale.unsqueeze(-1)), -128, 127)
    q = q.to(torch.int8).view(m, g * group_k)[:, :k].contiguous()
    return q, scale


def _per_block_quant_int8(w: torch.Tensor, block_n: int = 128, block_k: int = 128):
    """按 (block_n × block_k) 块的 int8 量化：块内 absmax 定标。

    返回 (码字 int8 与 w 同形, scale float32 (ceil(n/block_n), ceil(k/block_k)))。
    右/下边缘的不完整块只按其有效元素取 absmax（补零不影响定标）。
    """
    n, k = w.shape
    gn = (n + block_n - 1) // block_n
    gk = (k + block_k - 1) // block_k
    wp = torch.zeros(gn * block_n, gk * block_k, dtype=w.dtype)
    wp[:n, :k] = w
    blocks = wp.view(gn, block_n, gk, block_k).to(torch.float32)   # (gn, bn, gk, bk)
    scale4d = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-10) / 127.0
    q = torch.clamp(torch.round(blocks / scale4d), -128, 127)
    scale = scale4d.squeeze(1).squeeze(-1)                      # (gn, gk) float32
    q = q.to(torch.int8).view(gn * block_n, gk * block_k)[:n, :k].contiguous()
    return q, scale


class Model(nn.Module):
    """int8 分块量化 GEMM（量化激活 × 量化权重，块 scale 反量化）。

    数学定义：给定 int8 激活码字 A (M,K)、int8 权重码字 B (N,K)、激活的
    per-token-group scale As (M, ceil(K/128))、权重的 128×128 分块 scale
    Bs (ceil(N/128), ceil(K/128))（均 float32），先反量化再做矩阵乘：
      Af[m,k] = A[m,k] * As[m, floor(k/128)]
      Bf[n,k] = B[n,k] * Bs[floor(n/128), floor(k/128)]
      out[m,n] = sum_k Af[m,k] * Bf[n,k]    （float32 累加，即 Af @ Bf^T）
    分块量化的等价计算顺序：沿 K 每 128 一组先做 int8 点积（可在 int32 中
    精确累加），再乘该组的 As[m,g] * Bs[n_blk,g] 后 float32 求和——与上式
    在 float32 精度内一致。输出 cast 到 __init__ 的 out_dtype。
    边界行为：K、N 不被 128 整除时按 floor 索引自然延拓——尾组/尾块只覆盖
    有效元素（越界元素不参与累加），As/Bs 的组数按 ceil(·/128) 给出。

    输入输出规格：
      aq      int8    (M, K)                激活码字，行主
      bq      int8    (N, K)                权重码字，行主（输出通道在行上）
      a_scale float32 (M, ceil(K/128))      每 token 按 128 元素一组的 scale
      b_scale float32 (ceil(N/128), ceil(K/128))   权重 128×128 块 scale
      out     out_dtype (M, N)              bfloat16 / float16
      M/N/K >= 1，均允许非 2 次幂与不对齐 128 的尾部。

    实现约束（违规判负）：
      - 核心计算（int8 GEMM 累加与 scale 乘）必须在提交文件内完成，禁止
        调用 torch.matmul / torch.bmm / torch.mm / torch.addmm /
        torch.einsum / torch._int_mm / F.linear / functional.linear 等
        库级矩阵乘捷径，以及题集静态审计禁用清单中的其余入口。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32（int8 乘积可在 int32 中精确累加后转 float32）；
        输出 cast 到 out_dtype。
      - 评测器会把全部输入 cast 成 fp32 后传入 forward；int8 码字值域
        [-128,127] 与 fp32 scale 均可精确表示，入口处 .to(torch.int8) /
        .to(torch.float32) 无损恢复。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：块形固定 [128,128]、A 为二维 (M,K)、无 bias、无 splitK；
    输出 bfloat16 / float16。
    """

    def __init__(self, out_dtype: str = "bfloat16"):
        super().__init__()
        self.out_dtype = getattr(torch, out_dtype)
        self.block_n = 128   # 权重分块量化块形固定 [128, 128]
        self.block_k = 128

    def forward(self, aq, bq, a_scale, b_scale):
        # 评测器把全部输入 cast 成 fp32；int8 码字与 fp32 scale 无损恢复
        aq = aq.to(torch.int8)
        bq = bq.to(torch.int8)
        a_scale = a_scale.to(torch.float32)
        b_scale = b_scale.to(torch.float32)

        M, K = aq.shape
        N = bq.shape[0]
        assert K == bq.shape[1], "aq 与 bq 的 K 维必须一致"

        # 反量化 A：A[m,k] * As[m, k//block_k]（每 token 按 128 一组）
        a = aq.to(torch.float32) * a_scale.repeat_interleave(self.block_k, dim=1)[:, :K]
        # 反量化 B：B[n,k] * Bs[n//block_n, k//block_k]（128×128 块 scale）
        b_exp = (
            b_scale.repeat_interleave(self.block_n, dim=0)[:N]
            .repeat_interleave(self.block_k, dim=1)[:, :K]
        )
        b = bq.to(torch.float32) * b_exp

        out = torch.matmul(a, b.t())          # (M, N) float32 累加
        return out.to(self.out_dtype)


def get_init_inputs():
    return ["bfloat16"]   # out_dtype


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性。生成口径与 make_inputs 一致：
    # 激活 fp16 randn → per-token-group int8；权重 randn → 128×128 块 int8。
    m, n, k = 96, 1024, 768

    x = torch.randn(m, k).to(torch.float16)
    aq, a_scale = _per_token_group_quant_int8(x)
    w = torch.randn(n, k)
    bq, b_scale = _per_block_quant_int8(w)
    return [aq, bq, a_scale, b_scale]


def make_inputs(m: int, n: int, k: int, dtype: str = "bfloat16", seed: int = 0,
                seq_lens=None, head_size=None):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：激活 randn(m,k) → 权重 randn(n,k)，来自同一
    torch.Generator(seed)，同 seed 下逐位可复现。dtype 为输出 dtype
    （Model 构造参数 out_dtype 的 case 侧键名），不参与输入生成。
    离线终审管线（audit_model_class.py）对所有 model_class 任务统一以
    make_inputs(seq_lens=..., **case_fields) 生成输入、以 case["head_size"]
    位置构造 Model、以 case["dtype"] 取容差键；本任务的 head_size 即 dtype
    的兼容别名（管线把它作为 Model 首个位置实参传入，落在本模型的
    out_dtype 上），make_inputs 仅校验其与 dtype 一致，不参与输入生成。
    seq_lens 为管线的统一调用兼容形参（分页注意力族题目专用关键字），
    本算子无该语义，传入任何值均被忽略。
    """
    if head_size is not None and head_size != dtype:
        raise ValueError(f"head_size({head_size!r}) 应与 dtype({dtype!r}) 一致（兼容别名）")
    gen = torch.Generator().manual_seed(seed)

    x = torch.randn(m, k, generator=gen).to(torch.float16)
    aq, a_scale = _per_token_group_quant_int8(x)
    w = torch.randn(n, k, generator=gen)
    bq, b_scale = _per_block_quant_int8(w)
    return aq, bq, a_scale, b_scale
