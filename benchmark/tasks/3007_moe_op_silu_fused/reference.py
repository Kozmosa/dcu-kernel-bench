# 3007_moe_op_silu_fused —— MoE grouped GEMM（bf16 稠密 / fp8 分块量化两型）与
# SiLU-and-multiply 激活 epilogue 融合的 model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=3007, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """SiLU-and-multiply 融合进 epilogue 的 MoE（Mixture of Experts）单层专家
    FFN grouped GEMM：一次 kernel 同时完成 token×专家权重的分组矩阵乘、
    路由权重加权与 SwiGLU 式激活，避免中间激活落存。

    【算子语义】给定
        A            激活 (M, K)
        B            专家权重 (E, 2N, K)，第 e 个专家的转置权重矩阵
        topk_ids     路由结果 (M, top_k)，元素 ∈ [0, E)，行内互不相同
        topk_weights 路由权重 (M, top_k)，同一行恒正、无需归一
        以及上游对齐算子从 topk_ids 预计算的调度三件套
        sorted_token_ids / expert_ids / num_tokens_post_padded（见输入规格）。
    对每个 token m 与它的第 j 个选中专家 e = topk_ids[m, j]：
        x[m, j, :] = A[m, :] @ B[e, :, :]^T          ∈ R^{2N}
        若 mul_routed_weight = True：x[m, j, :] *= topk_weights[m, j]
        out[m*top_k + j, i] = SiLU(x[m, j, i]) * x[m, j, N + i],  i ∈ [0, N)
    其中 SiLU(u) = u / (1 + exp(-u)) = u * sigmoid(u)。B 的 2N 个输出维是
    chunked 门控布局：前 N 行为 gate、后 N 行为 up（与 interleaved 布局区分）。
    输出 (M*top_k, N) 的每一行恰被一个专家写出（路由保证行行有主）。

    【fp8 分块量化模式】quant="fp8_block" 时 A / B 为 float8_e4m3 码字，
    反量化后参与上述同一数学：
        A_deq[m, k]  = A[m, k]  * A_scale[m,  k // block_k]
        B_deq[e, j, k] = B[e, j, k] * B_scale[e, j // block_n, k // block_k]
    本模式约束 K % block_k == 0 且 2N % block_n == 0（scale 无尾块）。
    quant="dense" 时 A / B 即原始权重，A_scale / B_scale 为被忽略的
    1 元素占位张量。

    【边界行为】
      - sorted_token_ids 中值 >= M*top_k 的槽位是对齐填充哨兵，不参与计算、
        不产生输出行；expert_ids 只在前 ceil(num_tokens_post_padded /
        block_size_m) 个块内有效（有效前缀取值 ∈ [0, E)，无跨卡 -1 哨兵，
        尾部为不被消费的填充值）。
      - 未被任何 token 选中的专家不读其 B 行。
      - 调度三件套与 topk_ids 互相冗余（由同一 topk_ids 派生），核心调度
        用任一信息源均可，但输出必须与上述按 topk_ids 定义的数学一致。

    【输入输出规格】
        A            (M, K)    float16/bfloat16（fp8_block 模式为 e4m3 码字）
        B            (E, 2N, K) 同 A（fp8_block 模式为 e4m3 码字）
        A_scale      (M, K//block_k) float32（dense 模式为 (1,) 占位）
        B_scale      (E, 2N//block_n, K//block_k) float32（dense 模式为 (1,) 占位）
        topk_weights (M, top_k) float16/bfloat16，与输出 dtype 一致
        topk_ids     (M, top_k) int64
        sorted_token_ids (M*top_k + E*(block_size_m-1),) int32
        expert_ids   (M*top_k + E,) int32
        num_tokens_post_padded (1,) int32
        out          (M*top_k, N)，dtype 与 topk_weights 一致。
        M、K、E、top_k、block_size_m >= 1；2N 为偶数；输入值域常规 randn/rand
        量级，无 inf/NaN。评测器会把全部输入 cast 成 fp32 传入（整型精确
        恢复，fp16/bf16/fp8 -> fp32 无损），topk_weights 因此为 fp32 时输出
        即 fp32。

    实现约束（违规判负）：
      - 核心计算（分组 GEMM、fp8 反量化、路由权重乘、SiLU-and-multiply
        epilogue）必须在提交文件内以 Triton kernel 完成，中间累加一律
        float32，输出 cast 回 topk_weights 的 dtype。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 禁止把中间激活（GEMM 的 2N 维结果）物化到显存再激活——本算子的
        存在意义就是 epilogue 融合，逐块就地完成激活。

    禁用列表：torch.matmul / torch.bmm / torch.einsum /
    torch._scaled_mm / torch.nn.functional.silu / torch.SiLU / torch.sigmoid。
    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    """

    def __init__(self, mul_routed_weight=True, quant="dense", block_n=128, block_k=128):
        super().__init__()
        assert quant in ("dense", "fp8_block"), "quant 必须是 dense / fp8_block 之一"
        assert block_n >= 1 and block_k >= 1, "block_n / block_k 必须 >= 1"
        self.mul_routed_weight = bool(mul_routed_weight)
        self.quant = quant
        self.block_n = int(block_n)
        self.block_k = int(block_k)

    def forward(self, A, B, A_scale, B_scale, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded):
        # 评测器把全部输入 cast 成 fp32；索引/调度整型在此无损恢复
        topk_ids = topk_ids.to(torch.long)
        sorted_token_ids = sorted_token_ids.to(torch.int32)
        expert_ids = expert_ids.to(torch.int32)
        num_tokens_post_padded = num_tokens_post_padded.to(torch.int32)

        M, K = A.shape
        E, N2, _ = B.shape
        assert N2 % 2 == 0, "B 的输出维 2N 必须为偶数"
        N = N2 // 2
        top_k = topk_ids.shape[1]
        out_dtype = topk_weights.dtype
        # 调度三件套仅被消费方（Agent kernel）使用，语义见 topk_ids 路径
        _ = (sorted_token_ids, expert_ids, num_tokens_post_padded)

        # 反量化 / 上抛 float32
        a = A.to(torch.float32)
        b = B.to(torch.float32)
        if self.quant == "fp8_block":
            bk, bn = self.block_k, self.block_n
            assert A_scale.shape == (M, K // bk), "A_scale 形状应为 (M, K//block_k)"
            assert B_scale.shape == (E, N2 // bn, K // bk), \
                "B_scale 形状应为 (E, 2N//block_n, K//block_k)"
            a = a * A_scale.repeat_interleave(bk, dim=1)
            b = b * B_scale.repeat_interleave(bn, dim=1).repeat_interleave(bk, dim=2)

        tw = topk_weights.to(torch.float32)
        out = torch.zeros(M * top_k, N, dtype=torch.float32, device=A.device)
        for e in range(E):
            mask = topk_ids == e                      # (M, top_k)
            if not bool(mask.any()):
                continue
            rows = mask.nonzero(as_tuple=False)       # (cnt, 2)
            x = a[rows[:, 0]] @ b[e].t()              # (cnt, 2N) float32 累加
            if self.mul_routed_weight:
                x = x * tw[mask].unsqueeze(-1)
            gate, up = x[:, :N], x[:, N:]
            out[rows[:, 0] * top_k + rows[:, 1]] = gate / (1.0 + torch.exp(-gate)) * up
        return out.to(out_dtype)


def get_init_inputs():
    return [True, "dense"]   # mul_routed_weight / quant（block_n=block_k=128 缺省）


def get_inputs():
    # 固定 shape 族（M=32, N=256, K=512, E=8, top_k=4, block_size_m=16, dense
    # bf16, 加路由权重）；随机部分消费全局 RNG——评测器在 set_seed 后调用
    # 本函数，多轮 correctness trial 因此获得输入多样性
    return list(_gen_case(m=32, n=256, k=512, e=8, top_k=4, block_size_m=16,
                          mul_routed_weight=True, quant="dense",
                          block_n=128, block_k=128, dtype="bfloat16", rng=None))


def make_inputs(m: int, n: int, k: int, e: int, top_k: int,
                block_size_m: int = 16, mul_routed_weight: bool = True,
                quant: str = "dense", block_n: int = 128, block_k: int = 128,
                dtype: str = "bfloat16", seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到
    设备）。字段与 hidden/perf case 一一对应；mul_routed_weight / quant /
    block_n / block_k 同时是 Model 构造超参，原样传给 Model。

    消费顺序固定（同一 torch.Generator(seed)）：A(randn) -> B(rand) ->
    [fp8_block: B_scale(rand)] -> 路由 logits(randn)，对齐三件套是 topk_ids
    的确定函数不消费随机性；同 seed 下逐位可复现。
    """
    gen = torch.Generator().manual_seed(seed)
    return _gen_case(m=m, n=n, k=k, e=e, top_k=top_k, block_size_m=block_size_m,
                     mul_routed_weight=mul_routed_weight, quant=quant,
                     block_n=block_n, block_k=block_k, dtype=dtype, rng=gen)


def _gen_case(m, n, k, e, top_k, block_size_m, mul_routed_weight, quant,
              block_n, block_k, dtype, rng):
    """get_inputs（全局 RNG）/ make_inputs（Generator）共用的确定性生成器。

    分布取官方 input_helper 的族（randn 激活 / rand 正权重 / fp8 B 为
    [-448, 448] 均匀码字 + (0, 0.01) 均匀 scale / A 为按 token 分组的 e4m3
    动态量化 / 路由 softmax-topk），量级取单位 randn/rand：dense 输出
    O(10)、fp8_block 输出 O(1)，均落在验收容差可判定的量级内。
    """
    dt = getattr(torch, dtype)
    assert m >= 1 and n >= 1 and k >= 1 and e >= 1 and top_k >= 1
    assert 1 <= top_k <= e, "top_k 不能超过专家数"
    assert quant in ("dense", "fp8_block")

    # 1) 激活 A：(m, k)，randn（fp8_block 模式随后按 token 分组量化）
    a = torch.randn(m, k, generator=rng, dtype=dt)

    # 2) 专家权重 B：(e, 2n, k)
    if quant == "dense":
        b = torch.rand(e, 2 * n, k, generator=rng, dtype=dt)
        a_scale = torch.zeros(1, dtype=torch.float32)
        b_scale = torch.zeros(1, dtype=torch.float32)
    else:
        assert k % block_k == 0, "fp8_block 要求 K % block_k == 0"
        assert (2 * n) % block_n == 0, "fp8_block 要求 2N % block_n == 0"
        fp8_max = 448.0
        bf = (torch.rand(e, 2 * n, k, generator=rng, dtype=dt) - 0.5) * 2 * fp8_max
        b = bf.clamp(min=-fp8_max, max=fp8_max).to(torch.float8_e4m3fn)
        n_tiles = (2 * n) // block_n
        k_tiles = k // block_k
        b_scale = torch.rand(e, n_tiles, k_tiles, generator=rng,
                             dtype=torch.float32) * 1e-2
        a, a_scale = _per_token_group_quant_fp8(a / 10, block_k)   # a 变为 e4m3 码字（官方 /10 量级）

    # 3) 路由：softmax(logits) 的 topk
    values = torch.randn(m, e, generator=rng, dtype=dt)
    softmax_vals = torch.softmax(values, dim=1)
    topk_weights, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)

    # 4) 对齐三件套（topk_ids 的确定函数，上游调度前置算子的产物）
    sorted_token_ids, expert_ids, num_tokens_post_padded = _align_block_size(
        topk_ids, e, block_size_m)

    return (a, b, a_scale, b_scale, topk_weights, topk_ids,
            sorted_token_ids, expert_ids, num_tokens_post_padded)


def _per_token_group_quant_fp8(x, group_size, eps=1e-10):
    """按 token 分组（沿最后一维按 group_size 切组）的 e4m3 动态量化。"""
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_min, fp8_max = finfo.min, finfo.max
    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / fp8_max
    x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    return x_q.reshape(x.shape), x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))


def _align_block_size(topk_ids, num_experts, block_size):
    """把 topk_ids 按专家分桶并按 block_size 向上取整对齐，产出 grouped GEMM
    调度三件套：sorted_token_ids（展平下标 + 哨兵填充）、expert_ids（每块
    专家编号，尾部哨兵）、num_tokens_post_padded（(1,) 填充后总槽位数）。"""
    M, top_k = topk_ids.shape
    ids = topk_ids.to(torch.int32).tolist()
    expert_to_tokens = [[] for _ in range(num_experts)]
    for token_id in range(M):
        for j in range(top_k):
            expert_to_tokens[ids[token_id][j]].append(token_id * top_k + j)

    reordered_token_ids, reordered_expert_ids = [], []
    for e_id in range(num_experts):
        tokens_for_expert = expert_to_tokens[e_id]
        num_tokens = len(tokens_for_expert)
        n_blocks = (num_tokens + block_size - 1) // block_size
        padded_size = n_blocks * block_size
        reordered_token_ids.extend(tokens_for_expert)
        reordered_expert_ids.extend([e_id] * n_blocks)
        if padded_size > num_tokens:
            reordered_token_ids.extend([topk_ids.numel()] * (padded_size - num_tokens))

    token_length = len(reordered_token_ids)
    sorted_token_ids = torch.full((topk_ids.numel() + num_experts * (block_size - 1),),
                                  topk_ids.numel(), dtype=torch.int32)
    sorted_token_ids[:token_length] = torch.tensor(reordered_token_ids, dtype=torch.int32)
    expert_ids = torch.full((topk_ids.numel() + num_experts,), topk_ids.numel(),
                            dtype=torch.int32)
    expert_ids[:len(reordered_expert_ids)] = torch.tensor(reordered_expert_ids,
                                                          dtype=torch.int32)
    num_tokens_post_pad = torch.tensor([token_length], dtype=torch.int32)
    return sorted_token_ids, expert_ids, num_tokens_post_pad
