# 3003_moe_op — MoE（Mixture-of-Experts）路由 grouped GEMM（model_class 形态）。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。计算语义与上游 MoE grouped GEMM
# 内核一致：float32 中间计算，输出 cast 回输入 dtype。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=3003, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """MoE（Mixture-of-Experts）路由 grouped GEMM：非量化权重、含路由加权。

    每个 token 的特征向量 A[m]（K 维）按路由表 topk_ids 被发送到
    num_experts 个专家中的 top_k 个；专家 e 的权重是矩阵 B[e]（N×K）。对
    每个 token m 与其第 j 个路由槽位（记 e = topk_ids[m, j]），输出定义为

        C[m, j, n] = topk_weights[m, j] * Σ_k A[m, k] · B[e, n, k]

    即 grouped GEMM 后逐槽位乘以路由权重；中间累加 float32，输出 cast 回
    输入 dtype。所有 (m, j) 槽位都命中 [0, num_experts) 内的合法专家，不存在
    未定义输出。

    输入还携带 grouped GEMM 的块对齐布局（由上游 router 产出，按它取数与
    按上式直接计算完全等价）：
      - sorted_token_ids：int32，长度 EM = M*top_k + num_experts*(block_m-1)，
        本题 block_m 恒为 64。把槽位展平编号 p = m*top_k + j，按专家编号
        升序分组重排，每组内补齐到 block_m 的整数倍；组内补齐位与数组尾部
        填哨兵值 M*top_k（即 num_valid_tokens；offs_token ≥ 它即为 padding，
        不产生输出）。
      - expert_ids：int32，长度 M*top_k + num_experts；第 t 块（覆盖
        sorted_token_ids[t*block_m : (t+1)*block_m]）所属专家编号，取值
        [0, num_experts)，无 -1；超出 num_tokens_post_padded 对应块数的尾部
        为哨兵填充，不得读取。
      - num_tokens_post_padded：int32 标量（长度 1），sorted_token_ids 的
        有效长度（含块内 padding），恒为 block_m 的整数倍。
    kernel 的 BLOCK_SIZE_M 必须整除 block_m（即取 64 的因子），才能保证每个
    program 块内专家唯一。

    输入输出规格：
        A                       (M, K)             float16/bfloat16
        B                       (E, N, K)          同 A（E = num_experts）
        topk_weights            (M, top_k)         float32
        topk_ids                (M, top_k)         int32
        sorted_token_ids        (EM,)              int32
        expert_ids              (M*top_k + E,)     int32
        num_tokens_post_padded  (1,)               int32
        返回 C                  (M, top_k, N)      dtype 同 A

    实现约束（违规判负）：
        - 核心计算（按布局取数 + grouped GEMM + 路由加权写出）必须在提交
          文件内的 Triton kernel 中完成，禁止调用 torch.matmul /
          torch.mm / torch.bmm / torch.baddbmm / torch.addmm /
          torch.einsum / torch.nn.functional.linear 或任何预编译算子库
          完成任何矩阵乘。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 中间累加用 float32；输出 cast 回输入 dtype。
        - 哨兵 padding 槽位不得污染任何有效输出。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：非量化 fp16/bf16 权重与激活（无 w8a8 / int4 / mxfp4 量化），
    无激活融合（SiLU/GELU 等），无 bias，top_k ≤ 8，num_experts ≤ 256。
    """

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, A, B, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded):
        # KernelBench 评测器会把全部输入张量 cast 成评测 precision（triton
        # 后端为 fp32）；专家编号/排序表都是小整数，fp32 可精确表示，此处
        # 无损恢复
        topk_ids = topk_ids.to(torch.long)
        sorted_token_ids = sorted_token_ids.to(torch.long)
        expert_ids = expert_ids.to(torch.long)
        num_tokens_post_padded = num_tokens_post_padded.to(torch.int32)

        M, K = A.shape
        E, N, _ = B.shape
        top_k = topk_ids.shape[1]

        a = A.to(torch.float32)
        b = B.to(torch.float32)
        w = topk_weights.to(torch.float32).reshape(-1)

        # 参考实现按专家分组做 GEMM（与按槽位逐个点积等价）；
        # sorted_token_ids / expert_ids / num_tokens_post_padded 是供 kernel
        # 消费的块对齐布局，参考实现不依赖它们
        out = torch.zeros(M, top_k, N, dtype=torch.float32, device=A.device)
        flat_out = out.view(M * top_k, N)
        a_rp = a.repeat_interleave(top_k, dim=0)   # 槽位 p = m*top_k + j -> a[m]
        flat_ids = topk_ids.reshape(-1)
        for e in range(E):
            mask = flat_ids == e
            if bool(mask.any()):
                flat_out[mask] = (a_rp[mask] @ b[e].t()) * w[mask][:, None]
        return out.to(A.dtype)


def get_init_inputs():
    return [16, 4]   # num_experts, top_k（与 get_inputs 的 shape 族一致）


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    return _generate_inputs(
        num_tokens=64, num_experts=16, top_k=4,
        in_features=512, out_features=256,
        block_m=64, dtype=torch.float16, gen=None,
    )


def make_inputs(num_tokens: int, num_experts: int, top_k: int,
                in_features: int, out_features: int, block_m: int = 64,
                dtype: str = "float16", seed: int = 0):
    """确定性 case 生成器（评测端专用，CPU 生成，评测器搬运到设备）。

    参数字段与 hidden/perf case 一一对应；block_m 为 sorted 布局的块对齐
    粒度，本题恒为 64。
    """
    gen = torch.Generator().manual_seed(seed)
    return _generate_inputs(num_tokens, num_experts, top_k, in_features,
                            out_features, block_m, getattr(torch, dtype), gen)


def _generate_inputs(num_tokens, num_experts, top_k, in_features,
                     out_features, block_m, dtype, gen):
    # gen=None 时消费全局 RNG（get_inputs 路径）；官方输入分布：激活 randn/10、
    # 权重 rand/10，路由权重取 logits 过 softmax 后的 top-k
    a = torch.randn(num_tokens, in_features, dtype=dtype, generator=gen) / 10
    b = torch.rand(num_experts, out_features, in_features, dtype=dtype,
                   generator=gen) / 10
    values = torch.randn(num_tokens, num_experts, generator=gen)
    softmax_vals = torch.softmax(values, dim=1)              # float32
    topk_weights, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    sorted_token_ids, expert_ids, num_tokens_post_padded = _moe_align_block_size(
        topk_ids, num_experts, block_m)
    return [a, b, topk_weights, topk_ids, sorted_token_ids, expert_ids,
            num_tokens_post_padded]


def _moe_align_block_size(topk_ids, num_experts, block_m):
    """把 (M, top_k) 路由表展开为按专家分组、块对齐的 grouped GEMM 布局。

    对齐协议：槽位 p = m*top_k + j 按专家升序分组重排；每组补齐到 block_m
    的整数倍，组内 padding 与数组尾部填哨兵值 M*top_k；expert_ids 按块记录
    所属专家，尾部同样以哨兵填充；num_tokens_post_padded 记录有效长度。
    """
    M, top_k = topk_ids.shape
    num_valid = topk_ids.numel()

    expert_to_tokens = [[] for _ in range(num_experts)]
    ids = topk_ids.to(torch.long).tolist()
    for m in range(M):
        for j in range(top_k):
            expert_to_tokens[ids[m][j]].append(m * top_k + j)

    reordered_tokens = []
    reordered_experts = []
    for e in range(num_experts):
        tokens = expert_to_tokens[e]
        n_blocks = (len(tokens) + block_m - 1) // block_m
        reordered_tokens.extend(tokens)
        reordered_experts.extend([e] * n_blocks)
        reordered_tokens.extend([num_valid] * (n_blocks * block_m - len(tokens)))

    sorted_token_ids = torch.full(
        (num_valid + num_experts * (block_m - 1),), num_valid, dtype=torch.int32)
    expert_ids = torch.empty(num_valid + num_experts, dtype=torch.int32)
    num_tokens_post_padded = torch.empty(1, dtype=torch.int32)

    sorted_token_ids[: len(reordered_tokens)] = torch.tensor(
        reordered_tokens, dtype=torch.int32)
    expert_ids[: len(reordered_experts)] = torch.tensor(
        reordered_experts, dtype=torch.int32)
    if len(reordered_experts) < expert_ids.numel():
        expert_ids[len(reordered_experts):] = num_valid
    num_tokens_post_padded.fill_(len(reordered_tokens))
    return sorted_token_ids, expert_ids, num_tokens_post_padded
