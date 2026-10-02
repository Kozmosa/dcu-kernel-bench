# 3010_topk —— 行内 Top-K（值 + 原始列下标）的 model_class（KernelBench 兼容）
# 题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=3010, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """行内 Top-K 选择：对 2-D 得分矩阵的每一行取最大的 k 个值（严格降序）
    及其原始列下标。MoE 门控对 expert logits 做 top-k 路由的选择阶段
    即本算子。

    【算子语义】输入 x 形状 (B, M)，对每一行 b 独立地：
        (values[b, :], indices[b, :]) = TopK(x[b, :], k)
    values[b, :] 是 x[b, :] 中最大的 k 个元素、严格降序排列；indices[b, j]
    是 values[b, j] 在该行中的列下标（0 <= indices[b, j] < M，行内互不相
    同），即 out[b, k + j] 满足 x[b, out[b, k + j]] == out[b, j]。数学上
    等价于对每行做降序全排序后取前 k 个值及它们排序前的原始列位置。
    边界行为：k = 1 退化为行最大值与 argmax；k = M 退化为整行降序
    argsort；M = 1 时输出即该行唯一元素与下标 0。评测输入保证每行元素
    互不相等（由互异值集合的随机置换构造），因此 Top-K 结果唯一确定——
    相等值的平局顺序不属于本题语义。

    【实现路径提示（自由选择）】
      - 短行（M <= 1024）：单 kernel 逐轮选最大——每轮取当前最大值与其
        首次出现的列下标，写出后以 float32 最小值掩盖该位置，恰好 k 轮；
      - 长行（M > 1024）：两阶段——把行按定长 chunk 分块，每块局部 Top-K
        得到 chunk_num*k 个候选（值 + 全局列下标），再对候选归并排序取
        前 k。

    【输入输出规格】
        x   (B, M) float32（评测器把全部输入 cast 成 fp32 后传入），
            contiguous，行内元素互不相等；B >= 1，M >= 1 且 < 2^24（列下标
            在 fp32 打包中精确表示），可为非 2 次幂；题面规模域
            M ∈ [16, 131072]。
        k   构造超参（__init__ 传入），1 <= k <= M。
        out (B, 2*k) 单个 float32 张量（评测协议：单张量输出）：
            out[:, :k]   = values（严格降序，输入值的逐位拷贝）
            out[:, k:]   = indices（整数列下标，< 2^24，fp32 精确表示）
        输出是纯选择结果（无算术），任何一位偏差都是语义错误。

    实现约束（违规判负）：
      - 核心 Top-K 选择（扫描 / 分块 / 归并）必须在提交文件内以 Triton
        kernel 完成，值与下标都由 kernel 写出；数值比较在 float32 中进行。
      - ModelNew 的 __init__ 与 forward 签名不可更改。
      - 输出前 k 列必须严格降序；后 k 列必须是原始列下标（不是块内局部
        下标，也不是排序名次）。

    禁用列表：aiter / torch.topk / torch.sort / torch.argsort /
    torch.kthvalue / torch.mode，以及上述捷径的 Tensor 方法形式
    .topk( / .sort( / .argsort( / .kthvalue(——直接调用选择算子与全排序
    取前 k 属同一捷径，任何写法出现即判负（Triton 原语不受限）。
    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    """

    def __init__(self, k: int = 8):
        super().__init__()
        assert k >= 1, "k 必须 >= 1"
        self.k = int(k)

    def forward(self, x):
        # 评测器把全部输入 cast 成 fp32；x 为 (B, M) 行内互异的得分矩阵
        x = x.to(torch.float32).contiguous()
        values, indices = torch.topk(x, self.k, dim=-1, largest=True, sorted=True)
        # 单张量打包：前 k 列降序值，后 k 列原始列下标（整数，fp32 精确表示）
        return torch.cat([values, indices.to(torch.float32)], dim=-1)


def get_init_inputs():
    return [8]   # k


def get_inputs():
    # 固定 shape 族：B=8、M=128256（长行 -> 两阶段路径）、k=8；随机部分消费
    # 全局 RNG——评测器在 set_seed 后调用本函数，多轮 correctness trial 因此
    # 获得输入多样性
    return list(_gen_case(batch_size=8, row_len=128256, dtype=torch.float32, rng=None))


def make_inputs(batch_size: int, row_len: int, k: int = 8,
                dtype: str = "float32", seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到
    设备）。字段与 hidden/perf case 一一对应；k 同时是 Model 构造超参，原样
    传给 Model。

    消费顺序固定（同一 torch.Generator(seed)）：逐行 randperm；同 seed 下
    逐位可复现。
    """
    assert 1 <= k <= row_len, "k 必须满足 1 <= k <= row_len"
    gen = torch.Generator().manual_seed(seed)
    return _gen_case(batch_size=batch_size, row_len=row_len,
                     dtype=getattr(torch, dtype), rng=gen)


def _gen_case(batch_size, row_len, dtype, rng):
    """get_inputs（全局 RNG）/ make_inputs（Generator）共用的确定性生成器。

    分布取官方测试的族：每行是同一组互异值的独立随机置换（官方测试为
    arange 的行内 shuffle；此处对低精度 dtype 换用在该 dtype 下仍逐位精确
    且互异的值集合，见 _distinct_values），保证 Top-K 无平局歧义。
    """
    vals = _distinct_values(row_len, dtype)
    x = torch.empty(batch_size, row_len, dtype=dtype)
    for b in range(batch_size):
        x[b] = vals[torch.randperm(row_len, generator=rng)].to(dtype)
    return [x]


def _distinct_values(row_len, dtype):
    """返回 row_len 个升序、互不相等且在 dtype 下逐位精确表示的值。

    dtype 有效尾数有限（fp32 24 位 / fp16 11 位 / bf16 8 位）：row_len 不
    超过整数精确范围时直接用 arange（官方测试的值集合）；超出后按二进制
    位区间（binade）铺值——第 i 个值取 2^(i//P) * (1 + (i%P)/P)，P 为每个
    位区间内的尾数格数（2^(有效位数-1)），同区间内尾数不同、不同区间指数
    不同，任意两值在 dtype 中仍互异（fp16 与 bf16 的互异值上限均为 16384，
    覆盖题面规模域）。
    """
    sig_bits = {torch.float32: 24, torch.float16: 11, torch.bfloat16: 8}[dtype]
    if row_len <= 2 ** sig_bits:
        return torch.arange(row_len, dtype=torch.float64)
    per_binade = 2 ** (sig_bits - 1)
    idx = torch.arange(row_len)
    e, r = idx // per_binade, idx % per_binade
    return torch.pow(2.0, e.double()) * (1.0 + r.double() / per_binade)
