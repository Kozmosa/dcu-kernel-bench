# 5005_spconv_fprop —— 稀疏卷积前向（gathered 形式 spconv_fprop）的
# model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=5005, backend=cuda（HIP）。

import torch
import torch.nn as nn


class Model(nn.Module):
    """稀疏卷积前向（gathered 形式 spconv_fprop）：按索引对表在稀疏活跃点上
    做逐 kernel 偏移的 gather-GEMM-scatter 前向（spconv 风格）。

    语义定义（out 先置全零，再按下式累加）：
        对每个 kernel 偏移 k ∈ [0, kv)：
            n_k = indice_num_per_loc[k]
            对每个 j ∈ [0, n_k)：
                in_idx  = indice_pairs[0, k, j]      # 输入行号 ∈ [0, n)
                out_idx = indice_pairs[1, k, j]      # 输出行号 ∈ [0, num_act_out)
                out[out_idx, :] += features[in_idx, :] @ W[:, :, k].T
    即对每个索引对把输入行 gather 出来、与偏移 k 的权重矩阵做矩阵乘
    （(n_k, c_in) @ (c_in, c_out)），再按输出行号 scatter 累加进 out。
    同一输出行可被任意多个索引对命中（同一 k 内亦可重复），全部按 += 累加。

    filters 的布局映射（务必遵守，来自样例的 as_strided 重排）：设 F 为输入
    的 contiguous (c_out, c_in, kv) 张量，库端按步长 (c_in*kv, 1, c_in) 解释
    其存储，数学权重 W 与 F 满足：
        W[o, i, k] = F.reshape(c_out, kv, c_in)[o, k, i]
    等价于 W = F.reshape(c_out, kv, c_in).permute(0, 2, 1)（shape 仍为
    (c_out, c_in, kv)，contiguous 输入下是纯重解释视图）；kernel 偏移 k 的
    权重矩阵为 W[:, :, k]，形状 (c_out, c_in)。

    边界行为：n_k = 0 的 kernel 偏移整个跳过；全部 n_k = 0 时 out 为全零；
    kv = 1 退化为单偏移 gather-scatter；num_act_out 与 n 相互独立（可大于或
    小于 n）。j >= n_k 的索引对表项是填充垃圾（可能为任意值，包括重复的有
    效行号），必须依据 indice_num_per_loc 屏蔽，不得读入参与计算。

    输入输出规格：
        features           float32 / float16 / bfloat16 (n, c_in)，contiguous
        filters            同 features dtype (c_out, c_in, kv)，contiguous
        indice_pairs       int32 (2, kv, n)，contiguous
        indice_num_per_loc int32 (kv,)，contiguous
        num_act_out        0 维整数张量（在线评测把全部输入张量 cast 成
                           fp32：整型下标 <= 2^24 时精确，int() 恢复；
                           本题 shape 域保证下标 <= 2^24）
        out                单张量、同 features dtype (num_act_out, c_out)。
                           本算子天然单输出，无多输出打包分段。
    维度域：n, c_in, c_out, kv, num_act_out >= 1，允许非 2 的幂；
    indice_num_per_loc[k] ∈ [0, n]。输入值域为常规 randn 量级（无 inf/NaN）。

    实现约束（违规判负）：
        - 核心计算（按索引对 gather、与 W[:, :, k] 的矩阵乘、按输出行号的
          scatter 累加）必须在提交文件内以 HIP kernel 完成；ATen 只允许
          做 dtype 恢复 / 形状整理等非核心操作。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 累加全程 float32（fp16/bf16 输入先无损 cast 到 fp32），最终 cast
          回 features 的 dtype；填充表项不得参与计算（含越界读）。
        - forward 返回单张量 out（评测器按单输出比较 shape/dtype/allclose）。

    禁用列表：aiter / torch.sparse / torch.sparse_coo_tensor /
    torch.sparse_csr_tensor / torch.sparse_csc_tensor / torch.sparse_mm /
    torch.matmul / .matmul( / torch.mm / torch.bmm / torch.addmm /
    torch.baddbmm / torch.einsum / torch.tensordot /
    torch.nn.functional.linear / torch.nn.Linear / index_add_ /
    torch.index_add / index_put / torch.index_put / scatter_add_ /
    .scatter_( / torch.scatter / segment_reduce / miopen / hipdnn。
    目标硬件：海光 DCU（gfx936/gfx938），实现语言 HIP（经 hipcc 编译；在 CUDA_CU_SRC / CUDA_CPP_SRC 锚点区编写，设备侧必须 include <hip/hip_runtime.h>，本机无 <cuda_runtime.h>）。
    """

    def __init__(self):
        # 本题无构造超参（get_init_inputs 返回空列表）；PyramidKernel loader
        # 要求 Model 显式定义 __init__，故显式声明空构造
        super().__init__()

    def forward(self, features, filters, indice_pairs, indice_num_per_loc,
                num_act_out):
        # 在线评测会把全部输入张量 cast 成 fp32：整型索引张量在 <= 2^24 时
        # 精确，这里统一恢复；num_act_out 支持 0 维张量或 python int
        num_act_out = int(num_act_out)
        c_out, c_in, kv = filters.shape
        dt = features.dtype

        xf = features.to(torch.float32)
        # filters 布局映射（样例 as_strided(shape, [c_in*kv, 1, c_in]) 语义）：
        # W[o, i, k] = F.reshape(c_out, kv, c_in)[o, k, i]
        wf = filters.to(torch.float32).reshape(c_out, kv, c_in).permute(0, 2, 1)
        pairs = indice_pairs.to(torch.long)
        counts = indice_num_per_loc.to(torch.long)

        out = torch.zeros(num_act_out, c_out, dtype=torch.float32,
                          device=features.device)
        for k in range(kv):
            nvalid = int(counts[k])
            if nvalid == 0:
                continue
            in_idx = pairs[0, k, :nvalid]
            out_idx = pairs[1, k, :nvalid]
            contrib = xf[in_idx] @ wf[:, :, k].t()   # (nvalid, c_out)
            out.index_add_(0, out_idx, contrib)
        return out.to(dt)


def get_init_inputs():
    return []   # 本题无构造超参，全部语义输入走 forward


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性。
    # 消费顺序固定：features -> filters -> counts -> in_tab -> out_tab。
    n, c_in, c_out, kv, num_act_out = 1024, 16, 32, 9, 768
    features = torch.randn(n, c_in)
    filters = torch.randn(c_out, c_in, kv)
    counts = torch.randint(0, n + 1, (kv,), dtype=torch.int32)
    in_tab = torch.randint(0, n, (kv, n), dtype=torch.int32)
    out_tab = torch.randint(0, num_act_out, (kv, n), dtype=torch.int32)
    indice_pairs = torch.stack([in_tab, out_tab], dim=0)
    num_act_out_t = torch.tensor(num_act_out, dtype=torch.int64)
    return [features, filters, indice_pairs, counts, num_act_out_t]


def make_inputs(n: int, c_in: int, c_out: int, kv: int, num_act_out: int,
                dtype: str = "float32", counts_max=None, seed: int = 0):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器
    搬运到设备）。本题无构造超参，case 字段全部是这里的形参。

    消费顺序固定：features -> filters -> counts -> in_tab -> out_tab（均来
    自同一 torch.Generator(seed)，同 seed 下逐位可复现）。
        counts[k] = indice_num_per_loc[k] ~ randint[0, counts_max]
        （counts_max 缺省为 n；取 0 时全部 n_k = 0，out 为全零边界）
        in_tab / out_tab 覆盖全部 j ∈ [0, n)（有效前缀与填充垃圾同分布，
        填充为范围内的随机值，正确实现必须按 counts 屏蔽）。
    返回与 forward 形参同序的
    (features, filters, indice_pairs, indice_num_per_loc, num_act_out)。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert min(n, c_in, c_out, kv, num_act_out) >= 1, "各维均需 >= 1"
    cm = n if counts_max is None else int(counts_max)
    assert 0 <= cm <= n, "counts_max 需落在 [0, n]"
    assert n <= 2 ** 24 and num_act_out <= 2 ** 24, "下标须 <= 2^24（fp32 精确表示）"

    features = torch.randn(n, c_in, generator=gen).to(dt)
    filters = torch.randn(c_out, c_in, kv, generator=gen).to(dt)
    counts = torch.randint(0, cm + 1, (kv,), generator=gen, dtype=torch.int32)
    in_tab = torch.randint(0, n, (kv, n), generator=gen, dtype=torch.int32)
    out_tab = torch.randint(0, num_act_out, (kv, n), generator=gen,
                            dtype=torch.int32)
    indice_pairs = torch.stack([in_tab, out_tab], dim=0)
    return features, filters, indice_pairs, counts, \
        torch.tensor(num_act_out, dtype=torch.int64)
