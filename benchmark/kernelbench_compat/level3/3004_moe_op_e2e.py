# 3004_moe_op_e2e — 融合 MoE 前馈（单 kernel 内完成双 grouped GEMM + SwiGLU +
# 路由加权）的 model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，参考实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=3004, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """融合 Mixture-of-Experts（MoE）token 前馈全流程：一次调用完成
    「按路由选专家 -> 第一段 grouped GEMM -> SwiGLU 激活 -> 第二段
    grouped GEMM -> 路由加权」的完整链路。

    算子语义（数学定义与边界行为）：
      输入激活 a [M, K]；专家权重 w1 [E, N, K] 为 FFN 第一层（N = 2*hidden，
      前 N/2 通道是 gate、后 N/2 通道是 up）、w2 [E, K, N//2] 为 FFN 第二层；
      路由结果 topk_ids [M, top_k] 为每个 token 选中的专家编号，
      topk_weights [M, top_k] 为对应路由权重。对每个 token m 与其第 j 个
      专家 e = topk_ids[m, j]：
        gate[m,j,:] = a[m,:] @ w1[e, :N//2, :].T          # [N//2]
        up[m,j,:]   = a[m,:] @ w1[e, N//2:, :].T          # [N//2]
        h[m,j,:]    = gate * sigmoid(gate) * up           # SwiGLU，SiLU(x)=x*sigmoid(x)
        out[m,j,k]  = topk_weights[m,j] * sum_{n<N//2} h[m,j,n] * w2[e,k,n]
      即 out[m,j,:] = topk_weights[m,j] * (h[m,j,:] @ w2[e].T)。各 (m, j) 对
      相互独立；top_k 个槽位的输出全部写入结果，不做专家间合并、求和或归一化。
      边界行为：N 为偶数；top_k <= E 且 topk_ids 取值在 [0, E)；同一 token 的
      top_k 个专家按互异生成（即使出现重复，逐对独立计算的语义依然成立）；
      M、K 为任意正整数（含 1，即 GEMV 情形）；路由加权恒启用。

    输入输出规格：
      forward(a, w1, w2, topk_weights, topk_ids) -> out
        a            [M, K]        bfloat16/float32，行主连续
        w1           [E, N, K]     与 a 同 dtype，行主连续
        w2           [E, K, N//2]  与 a 同 dtype，行主连续
        topk_weights [M, top_k]    与 a 同 dtype，行主连续
        topk_ids     [M, top_k]    int64（评测器 fp32 cast 后可无损恢复）
        out          [M, top_k, K] 与 a 同 dtype
      __init__(top_k, num_experts)：仅作实例元数据；forward 一律以运行期张量
      的实际形状为准（top_k = topk_ids.shape[1]，E = w1.shape[0]）。

    实现约束（违规判负）：
      - 核心计算（两段 grouped GEMM、SwiGLU 激活、路由加权）必须在提交文件内
        以 Triton kernel 完成，禁止调用 torch.matmul / torch.mm / torch.bmm /
        torch.baddbmm / torch.addmm / torch.einsum / torch.nn.functional.linear /
        F.linear / torch.nn.functional.silu / F.silu / torch.sigmoid / aiter
        等 ATen 捷径。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：非量化 bfloat16/float32 激活与权重；单卡；无专家容量限制、
    无共享专家、无路由权重归一化；专家分发/调度策略（排序对齐、persistent
    网格等）由实现自定，以输出容差判定。
    """

    def __init__(self, top_k: int, num_experts: int):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, a, w1, w2, topk_weights, topk_ids):
        # KernelBench 评测器会把全部输入张量 cast 成评测 precision（triton
        # 后端为 fp32）；专家编号是小整数，fp32 可精确表示，此处无损恢复为
        # 整型索引
        topk_ids = topk_ids.to(torch.long)

        M, K = a.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        H = N // 2
        assert 0 <= int(topk_ids.min()) and int(topk_ids.max()) < E, "topk_ids 必须落在 [0, E)"

        a32 = a.to(torch.float32)
        tw32 = topk_weights.to(torch.float32)

        out = torch.empty((M, top_k, K), dtype=a.dtype)
        for e in range(E):
            sel = topk_ids == e
            if not sel.any():
                continue
            rows, slots = sel.nonzero(as_tuple=True)            # 路由到专家 e 的 (m, j)
            inter = a32[rows] @ w1[e].to(torch.float32).t()     # [T, N] 第一段 grouped GEMM
            gate, up = inter[:, :H], inter[:, H:]
            h = gate * torch.sigmoid(gate) * up                 # SwiGLU，全程 float32
            o = h @ w2[e].to(torch.float32).t()                 # [T, K] 第二段 grouped GEMM
            o = o * tw32[rows, slots].unsqueeze(-1)             # 路由加权
            out[rows, slots] = o.to(a.dtype)                    # cast 回输入 dtype
        return out


def get_init_inputs():
    return [2, 8]  # top_k，num_experts（仅元数据，实际以运行期张量形状为准）


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性。数据分布沿用该算子官方测试：
    # 激活 ~ N(0,1)，专家权重 ~ U[0,1)，路由取 softmax(logits) 的 top-k
    m = 32
    n = 1024
    k = 512
    top_k = 2
    e = 8
    dt = torch.bfloat16
    a = torch.randn((m, k), dtype=dt)
    w1 = torch.rand((e, n, k), dtype=dt)
    w2 = torch.rand((e, k, n // 2), dtype=dt)
    logits = torch.randn((m, e), dtype=dt)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=1), k=top_k, dim=1)
    return [a, w1, w2, topk_weights, topk_ids]


def make_inputs(m: int, n: int, k: int, top_k: int, num_experts: int,
                dtype: str = "bfloat16", seed: int = 0, seq_lens=None):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到
    设备）。生成次序与 get_inputs 相同：a -> w1 -> w2 -> 路由 logits ->
    softmax top-k；n 为偶数且 top_k <= num_experts。参数名与 task.yaml
    io.init_inputs 对齐，hidden/perf case 字段可原样透传。seq_lens 是
    兼容统一调用形态的保留形参（本算子无序列长度概念，接受并忽略）。"""
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    a = torch.randn((m, k), dtype=dt, generator=gen)
    w1 = torch.rand((num_experts, n, k), dtype=dt, generator=gen)
    w2 = torch.rand((num_experts, k, n // 2), dtype=dt, generator=gen)
    logits = torch.randn((m, num_experts), dtype=dt, generator=gen)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=1), k=top_k, dim=1)
    return a, w1, w2, topk_weights, topk_ids
