# 3009_routing — MoE 门控 sigmoid top-1 路由（固定 config 简洁版）的
# model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=3009, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """MoE（Mixture of Experts）门控的 sigmoid top-1 路由：为每个 token 从
    N 个专家中选出路由得分最高的一個（top_k 恒为 1）。

    【算子语义】给定门控输入 x（[M, K]，M 个 token、K 维门控特征）与路由
    权重 w（[K, N]，每列一个专家），对每个 token m 与专家 n：
        score[m, n] = sigmoid( sum_k x[m, k] * w[k, n] )
        sigmoid(z)  = 1 / (1 + exp(-z))
    top-1 路由输出：
        topk_ids[m]     = argmax_n score[m, n]，平局取最小专家下标
        topk_weights[m] = max_n score[m, n]
    内积（矩阵乘）以 float32 累加求值；top_k = 1 时归一化系数恒为 1，
    权重即原 sigmoid 得分，无需再除以和。

    【融合共享专家列】fused_shared_experts=True 时在 top-1 结果之后追加一
    列固定的共享专家：id 列追加 N（= 专家数，落在真实专家编号区间之外），
    权重列追加 1.0。False 时不追加，输出仅一列。

    【输入输出规格】
        x   [M, K] float16/bfloat16，连续；M/K/N >= 1，允许非 2 次幂
        w   [K, N] 与 x 同 dtype，连续
        输出 单个 float32 张量 [M, 2*C]，C = 1 + fused_shared_experts：
            out[:, :C] = topk_ids 的整数以 fp32 表示（< 2^24，精确）
            out[:, C:] = topk_weights（fp32 sigmoid 得分；融合列恒 1.0）
        评测器会把全部输入 cast 成 fp32 传入（本算子输入均为浮点，无损）。

    实现约束（违规判负）：
      - 核心计算（x·w 的逐块矩阵乘、sigmoid、top-1 选择与平局裁决、融合
        共享专家列写回）必须在提交文件内的 Triton kernel 中完成；官方实现
        为单 kernel 全融合（得分矩阵不落存），拆成多个 Triton kernel 不判
        负，但调用 ATen 捷径判负。
      - 矩阵乘内积累加一律 float32（输入可为 fp16/bf16/fp32）。
      - 输出严格按上述单张量打包协议为 float32。
      - ModelNew 的 __init__ 与 forward 签名不可更改。

    禁用列表：aiter / torch.matmul / torch.bmm / torch.einsum /
    torch.nn.functional.linear / torch.topk / torch.argmax / torch.sigmoid。
    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    """

    def __init__(self, fused_shared_experts: bool = True):
        super().__init__()
        self.fused_shared_experts = bool(fused_shared_experts)

    def forward(self, x, w):
        assert x.dim() == 2 and w.dim() == 2, "x / w 必须是 2 维张量"
        M, K = x.shape
        Kb, N = w.shape
        assert K == Kb, "x 的最后一维必须等于 w 的行数"

        # 参考实现自由使用 torch 算子；得分在 float32 中求值
        scores = torch.sigmoid(
            torch.matmul(x.to(torch.float32), w.to(torch.float32)))   # [M, N]

        # top-1：torch.max 平局取首个（最小）下标，与官方 kernel 的
        # tl.argmax(tie_break_left=True) 语义一致
        top1_weights, top1_ids = torch.max(scores, dim=1)

        ids = top1_ids.to(torch.int32).unsqueeze(1)
        weights = top1_weights.unsqueeze(1)
        if self.fused_shared_experts:
            # 融合共享专家列：id 追加 N（专家数，真实专家区间之外）、权重追加 1.0
            ids = torch.cat([ids, torch.full((M, 1), N, dtype=torch.int32)], dim=1)
            weights = torch.cat([weights, torch.ones((M, 1), dtype=torch.float32)], dim=1)

        # 单张量打包协议：[ids（整数，fp32 精确）| weights]
        return torch.cat([ids.to(torch.float32), weights.to(torch.float32)], dim=1)


def get_init_inputs():
    return [True]   # fused_shared_experts（官方测试恒开）


def get_inputs():
    # 固定 shape 族（M=1024, K=5120, N=128，bf16，融合共享专家）；随机部分
    # 消费全局 RNG——评测器在 set_seed 后调用本函数，多轮 correctness trial
    # 因此获得输入多样性。取值分布对齐官方测试的 [-2, 2] 均匀整数（整数得
    # 分在 fp32 累加链上精确可复现，平局裁决确定）
    M, K, N = 1024, 5120, 128
    x = torch.randint(-2, 3, (M, K)).to(torch.bfloat16)
    w = torch.randint(-2, 3, (K, N)).to(torch.bfloat16)
    return [x, w]


def make_inputs(m: int, k: int, n: int, fused_shared_experts: bool = True,
                dtype: str = "bfloat16", dist: str = "int", seed: int = 0):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器
    搬运到设备）。字段与 hidden/perf case 一一对应；fused_shared_experts
    同时是 Model 构造超参，原样传给 Model。

    dist="int"：[-2, 2] 均匀整数（与官方测试的输入生成同分布；整数得分在
    fp32 中精确，平局裁决确定）。dist="bench"：x ~ randn、w ~ randn*0.1
    （与官方基准的输入生成同分布）。消费顺序固定：先 x 后 w，同 seed 下
    逐位可复现。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert m >= 1 and k >= 1 and n >= 1
    assert dist in ("int", "bench")
    if dist == "int":
        x = torch.randint(-2, 3, (m, k), generator=gen).to(dt)
        w = torch.randint(-2, 3, (k, n), generator=gen).to(dt)
    else:
        x = torch.randn(m, k, generator=gen).to(dt)
        w = (torch.randn(k, n, generator=gen) * 0.1).to(dt)
    return [x, w]
