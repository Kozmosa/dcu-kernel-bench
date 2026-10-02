# 4008_gemm_a16w4 — AWQ W4A16 量化 GEMM（int4 打包权重 kernel 内反量化）的
# model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，参考实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=4008, backend=triton。

import torch
import torch.nn as nn


def _reverse_awq_order(tensor: torch.Tensor) -> torch.Tensor:
    """把 4 位码字列按 AWQ 反序排列 [0,4,1,5,2,6,3,7] 重排并取低 4 位。"""
    bits = 4
    awq_reverse_order = [0, 4, 1, 5, 2, 6, 3, 7]
    order = torch.arange(tensor.shape[-1], dtype=torch.int32)
    order = order.view(-1, 32 // bits)[:, awq_reverse_order].view(-1)
    return tensor[:, order] & 0xF


def _awq_reorder_and_repack(qweight: torch.Tensor, qzeros: torch.Tensor):
    """输入侧预处理（离线打包）：int32 原始码 [K, N//8] / [K//G, N//8] 重排后
    打包为 kernel 消费的布局——qweight [N, K//2]（沿 K 方向每字节 2 个 4 位
    码，低半字节 = 偶数 k）、qzeros [K//G, N//2]（沿 N 方向，低半字节 =
    偶数 n），均为 int8。仅被 make_inputs/get_inputs 用于生成题面输入，
    不属于算子本体语义。"""
    bits = 4
    shifts = torch.arange(0, 32, bits)
    K = qweight.shape[0]
    N = qweight.shape[1] * 8
    G = K // qzeros.shape[0]

    iweights = torch.bitwise_right_shift(
        qweight[:, :, None], shifts[None, None, :]).to(torch.int8).view(K, -1)
    zeros = torch.bitwise_right_shift(
        qzeros[:, :, None], shifts[None, None, :]).to(torch.int8).view(K // G, -1)

    iweights = torch.bitwise_and(_reverse_awq_order(iweights), 0xF)
    zeros = torch.bitwise_and(_reverse_awq_order(zeros), 0xF)

    iweights = iweights.transpose(1, 0).contiguous().view(N, -1, 2)
    zeros = zeros.view(K // G, -1, 2)

    packed_weights = torch.zeros([N, K // 2], dtype=torch.int8)
    packed_zeros = torch.zeros([K // G, N // 2], dtype=torch.int8)
    for i in range(2):
        packed_weights |= iweights[:, :, i].to(torch.int8) << (i * bits)
        packed_zeros |= zeros[:, :, i].to(torch.int8) << (i * bits)
    return packed_weights, packed_zeros


class Model(nn.Module):
    """AWQ W4A16 量化 GEMM：int4 打包权重在 kernel 内反量化后与激活做矩阵乘。

    算子语义（数学定义与边界行为）：
      激活 input [M, K]（float16），AWQ int4 打包权重 qweight [N, K//2]（int8，
      沿 K 方向每字节连续存放 2 个 4 位码字，低半字节对应偶数 k），分组零点
      qzeros [K//G, N//2]（int8，沿 N 方向每字节连续存放 2 个 4 位码字，低半
      字节对应偶数 n），分组缩放 scales [K//G, N]（float16）。反量化与矩阵乘
      定义为
        w[n, k] = (qweight[n, k//2] >> (4*(k%2))) & 0xF
        z[g, n] = (qzeros[g, n//2]   >> (4*(n%2))) & 0xF
        W[k, n] = (w[n, k] - z[k//G, n]) * scales[k//G, n]
        out[m, n] = sum_{k=0}^{K-1} input[m, k] * W[k, n]
      有符号 int8 的右移为算术右移，移位后 & 0xF 恒提取存储的 4 位码字
      （结果 0..15）；w - z 取值 [-15, 15]。逐元素乘累加全程 float32，输出
      cast 回输入 dtype。边界行为：M、K、N 不要求与任何分块大小对齐（K、N
      方向越界部分按 0 参与掩码累加）；G 为 32/64/128 或 G == K（逐通道），
      且 G 整除 K。

    输入输出规格：
      forward(input, qweight, scales, qzeros) -> out
        input   [M, K]         float16，行主连续
        qweight [N, K//2]      int8 打包权重（低半字节 = 偶数 k）
        scales  [K//G, N]      float16，行主连续
        qzeros  [K//G, N//2]   int8 打包零点（低半字节 = 偶数 n）
        out     [M, N]         与 input 同 dtype
      __init__(in_features, out_features, group_size)：分别对应 K、N、G，
      仅作实例元数据；forward 一律以运行期张量的实际形状为准
      （G = K // qzeros.shape[0]）。

    实现约束（违规判负）：
      - 核心计算（int4 码字解包、分组反量化、乘累加主循环）必须在提交文件内
        以 Triton kernel 完成，禁止调用 torch.matmul / torch.bmm / torch.mm /
        torch.addmm / torch.einsum / torch.nn.functional.linear / F.linear /
        aiter 等 ATen GEMM 捷径。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 中间累加用 float32；输出 cast 回输入 dtype。

    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：单卡 W4A16 GEMM、float16 激活与 float16 scales；调度策略
    （数据并行 / split-K / stream-K）与分块由实现自行决定，以输出容差判定；
    无 bias、无量化感知的激活变换、无跨卡通信融合。
    """

    def __init__(self, in_features: int, out_features: int, group_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

    def forward(self, input, qweight, scales, qzeros):
        # KernelBench 评测器会把全部输入张量 cast 成评测 precision（triton
        # 后端为 fp32）；打包码字取值 |v| <= 128，fp32 可精确表示，此处按
        # 码字类无损恢复为整型
        qweight = qweight.to(torch.int32)
        qzeros = qzeros.to(torch.int32)

        M, K = input.shape
        N = qweight.shape[0]                      # (N, K//2)
        group = K // qzeros.shape[0]
        assert qweight.shape[1] == K // 2 and qzeros.shape[1] == N // 2
        assert scales.shape[0] == K // group and scales.shape[1] == N

        # 解包 int4 码字：低半字节对应偶数下标；算术右移 + & 0xF
        k_shift = (torch.arange(K) % 2) * 4
        w_int = (qweight.repeat_interleave(2, dim=1) >> k_shift) & 0xF    # [N, K]
        n_shift = (torch.arange(N) % 2) * 4
        z_int = (qzeros.repeat_interleave(2, dim=1) >> n_shift) & 0xF     # [K//G, N]

        # 分组 zeros/scales 沿 K 广播后在 float32 中反量化
        z_full = z_int.repeat_interleave(group, dim=0)                    # [K, N]
        s_full = scales.to(torch.float32).repeat_interleave(group, dim=0)  # [K, N]
        w_deq = (w_int.to(torch.float32).t() - z_full.to(torch.float32)) * s_full

        # [M, K] @ [K, N] -> [M, N]，float32 一次性累加，输出 cast 回输入 dtype
        out = torch.matmul(input.to(torch.float32), w_deq)
        return out.to(input.dtype)


def get_init_inputs():
    return [2048, 1536, 64]  # in_features(K)，out_features(N)，group_size(G)


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性。数据分布沿用该算子官方测试：
    # 激活与 scales 取 U[0,1)，原始码字取随机 int32 后按 AWQ 布局打包
    m = 64
    n = 1536
    k = 2048
    group_size = 64
    x = torch.rand((m, k)).to(torch.float16)
    qweight = torch.randint(0, torch.iinfo(torch.int32).max,
                            (k, n // 8), dtype=torch.int32)
    qzeros = torch.randint(0, torch.iinfo(torch.int32).max,
                           (k // group_size, n // 8), dtype=torch.int32)
    scales = torch.rand((k // group_size, n)).to(torch.float16)
    qweight, qzeros = _awq_reorder_and_repack(qweight, qzeros)
    return [x, qweight, scales, qzeros]


def make_inputs(m: int, n: int, k: int, group_size: int, seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到
    设备）。生成次序与 get_inputs 相同：x -> 原始 qweight -> 原始 qzeros ->
    scales -> 打包。"""
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand((m, k), generator=gen).to(torch.float16)
    qweight = torch.randint(0, torch.iinfo(torch.int32).max,
                            (k, n // 8), dtype=torch.int32, generator=gen)
    qzeros = torch.randint(0, torch.iinfo(torch.int32).max,
                           (k // group_size, n // 8), dtype=torch.int32,
                           generator=gen)
    scales = torch.rand((k // group_size, n), generator=gen).to(torch.float16)
    qweight, qzeros = _awq_reorder_and_repack(qweight, qzeros)
    return x, qweight, scales, qzeros
