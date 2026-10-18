# 5006_spconv_bwd —— 稀疏卷积反向（dgrad / wgrad，gathered 形式 spconv）的
# model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=5006, backend=cuda（HIP）。

import torch
import torch.nn as nn


class Model(nn.Module):
    """稀疏卷积反向（gathered 形式 spconv）：dgrad（对输入特征的梯度）与
    wgrad（对权值缓冲的梯度），是稀疏卷积前向（gather-GEMM-scatter）的精确
    伴随。

    前向语义（本题口径的唯一权威；out 先置全零，再按下式累加）：
        对每个 kernel 偏移 k ∈ [0, kv)：
            n_k = indice_num_per_loc[k]
            对每个 j ∈ [0, n_k)：
                in_idx  = indice_pairs[0, k, j]      # 输入行号 ∈ [0, n)
                out_idx = indice_pairs[1, k, j]      # 输出行号 ∈ [0, num_act_out)
                out[out_idx, :] += features[in_idx, :] @ W[:, :, k].T
    即对每个索引对把输入行 gather 出来、与偏移 k 的权重矩阵做矩阵乘
    （(n_k, c_in) @ (c_in, c_out)），再按输出行号 scatter 累加进 out。
    其中数学权重 W 与 contiguous filters 张量 F 满足布局映射：
        W[o, i, k] = F.reshape(c_out, kv, c_in)[o, k, i]
    （等价于 W = F.reshape(c_out, kv, c_in).permute(0, 2, 1)，shape 仍为
    (c_out, c_in, kv)）。

    给定上游梯度 out_bp = dL/dy（形状 (num_act_out, c_out)），两方向定义为
    上述前向对 features / filters 的伴随：
        dgrad：dx[in_idx, :] += out_bp[out_idx, :] @ W[:, :, k]，输出
               (n, c_in)，即 dL/dfeatures；
        wgrad：dW_math[:, :, k] += out_bp[out_idx, :, None] *
               features[in_idx, None, :]（按索引对外积累加），输出为按
               filters 同一布局映射解释的权值梯度 dF (c_out, c_in, kv)：
               dF.reshape(c_out, kv, c_in)[o, k, i] = dW_math[o, i, k]，
               contiguous 存储。
    __init__ 参数 variant 选择返回的方向：
        "dgrad"：返回 dx；此时 features 的数值不参与输出（仅形状约定）；
        "wgrad"：返回 dF；此时 filters 的数值不参与输出（仅形状约定）。
    forward(features, filters, out_bp, indice_pairs, indice_num_per_loc) 为
    固定五输入签名，两方向均天然单张量输出（无拼接打包）。

    边界行为：n_k = 0 的 kernel 偏移整个跳过；全部 n_k = 0 时输出为全零；
    kv = 1 退化为单偏移；num_act_out 与 n 相互独立（可大于或小于 n）。
    j >= indice_num_per_loc[k] 的索引对表项是填充垃圾（可能为任意值，包括
    重复的有效行号），必须依据 indice_num_per_loc 屏蔽，不得读入参与计算。
    同一输入/输出行可被任意多个索引对命中（同一 k 内亦可重复），全部按
    += 累加。

    输入输出规格：
        features           float32 / float16 / bfloat16 (n, c_in)，contiguous
        filters            同 features dtype (c_out, c_in, kv)，contiguous
        out_bp             同 features dtype (num_act_out, c_out)，contiguous
        indice_pairs       int32 (2, kv, n)，contiguous
        indice_num_per_loc int32 (kv,)，contiguous
        out                单张量、同 features dtype：dgrad 时 (n, c_in)，
                           wgrad 时 (c_out, c_in, kv)
        维度域：n, c_in, c_out, kv, num_act_out >= 1，允许非 2 的幂；
        indice_num_per_loc[k] ∈ [0, n]；行号下标 <= 2^24。输入值域为常规
        randn 量级（无 inf/NaN）。评测器会把全部输入张量 cast 成 fp32 传入
        （整型下标 <= 2^24 时精确，int()/long() 恢复）。

    实现约束（违规判负）：
        - 核心计算必须在提交文件内以 HIP kernel 完成：dgrad 的散射累加
          （out_bp 行经 W[:, :, k] 变换后按 in_idx 散射累加）或 wgrad 的
          按对外积相关累加，禁止回退到 torch 的 gather / matmul / scatter
          组合或 autograd。
        - ModelNew 的 __init__ 与 forward 签名不可更改；forward 返回单张量
          out（评测器按单输出比较 shape/dtype/allclose）。
        - 数值约定：累加全程 float32（fp16/bf16 输入先无损 cast 到 fp32），
          最终 cast 回 features 的 dtype；填充表项不得参与计算（含越界读）。

    禁用列表：aiter / torch.sparse / torch.sparse_coo_tensor /
    torch.sparse_csr_tensor / torch.sparse_csc_tensor / torch.sparse_mm /
    torch.matmul / .matmul( / torch.mm / torch.bmm / torch.addmm /
    torch.baddbmm / torch.einsum / torch.tensordot /
    torch.nn.functional.linear / torch.nn.Linear /
    index_add（含 .index_add( / index_add_ / torch.index_add 一切拼写）/
    index_put / torch.index_put / scatter_add（含 scatter_add_）/
    .scatter_( / torch.scatter / scatter_reduce / segment_reduce /
    torch.gather / index_select / torch.take / torch.sum /
    torch.ops.aten / torch.autograd / .backward( / as_strided；
    闭源库 MIOpen / hipDNN / cuDNN / rocBLAS / hipBLASLt 同样禁止（全局
    黑名单）。
    目标硬件：海光 DCU（gfx936/gfx938），实现语言 HIP（经 hipcc 编译；在 CUDA_CU_SRC / CUDA_CPP_SRC 锚点区编写，设备侧必须 include <hip/hip_runtime.h>，本机无 <cuda_runtime.h>）。
    """

    def __init__(self, variant: str = "dgrad"):
        # 注意：__init__/forward 的 def 签名必须保持单行（框架 loader 按首行
        # 提取签名生成 ModelNew scaffold）。
        super().__init__()
        assert variant in ("dgrad", "wgrad"), "variant 必须是 dgrad / wgrad 之一"
        self.variant = variant

    def forward(self, features, filters, out_bp, indice_pairs, indice_num_per_loc):
        # 对同族前向（gather-GEMM-scatter）构图后用 autograd 反传：保证
        # dgrad / wgrad 与前向语义（含 filters 布局映射）严格一致（reference
        # 不受禁用列表约束；数学上 dx 与 features 的数值无关、dF 与 filters
        # 的数值无关，autograd 自然给出该定义）。
        # 在线评测会把全部输入张量 cast 成 fp32：整型索引张量在 <= 2^24 时
        # 精确，这里统一恢复；浮点输入无损 cast 到 fp32 做全程 float32 累加，
        # 最终 cast 回 features 的 dtype。
        dt = features.dtype
        c_out, c_in, kv = filters.shape
        f32 = torch.float32

        xf = features.detach().to(f32).requires_grad_(True)
        wf0 = filters.detach().to(f32).requires_grad_(True)
        yf = out_bp.detach().to(f32)
        pairs = indice_pairs.to(torch.long)
        counts = indice_num_per_loc.to(torch.long)

        # 前向构图：out 先置全零，再按索引对 gather-GEMM-scatter 累加
        # （布局映射：W[o, i, k] = F.reshape(c_out, kv, c_in)[o, k, i]）。
        # 全部 n_k = 0 时前向与两片叶子均无依赖，乘 0.0 的钩子保持计算图
        # 连通且数值恰为零（输入域保证无 inf/NaN，0.0*有限值=0.0、x+0.0=x
        # 均精确），梯度自然为全零。
        wf = wf0.reshape(c_out, kv, c_in).permute(0, 2, 1)
        hook = xf.sum() * 0.0 + wf0.sum() * 0.0
        out = torch.zeros(yf.shape[0], c_out, dtype=f32,
                          device=features.device) + hook
        for k in range(kv):
            nvalid = int(counts[k])
            if nvalid == 0:
                continue
            in_idx = pairs[0, k, :nvalid]
            out_idx = pairs[1, k, :nvalid]
            contrib = xf[in_idx] @ wf[:, :, k].t()   # (nvalid, c_out)
            out = out.index_add(0, out_idx, contrib)

        if self.variant == "dgrad":
            (dx,) = torch.autograd.grad(out, (xf,), yf)
            return dx.to(dt)
        (dwf,) = torch.autograd.grad(out, (wf0,), yf)
        return dwf.to(dt)


def get_init_inputs():
    return ["dgrad"]   # variant


def get_inputs():
    # 固定 shape 族（与同族前向题目一致）；随机部分消费全局 RNG——评测器在
    # set_seed 后调用本函数，多轮 correctness trial 因此获得输入多样性。
    # 消费顺序固定：features -> filters -> out_bp -> counts -> in_tab -> out_tab。
    n, c_in, c_out, kv, num_act_out = 1024, 16, 32, 9, 768
    features = torch.randn(n, c_in)
    filters = torch.randn(c_out, c_in, kv)
    out_bp = torch.randn(num_act_out, c_out)
    counts = torch.randint(0, n + 1, (kv,), dtype=torch.int32)
    in_tab = torch.randint(0, n, (kv, n), dtype=torch.int32)
    out_tab = torch.randint(0, num_act_out, (kv, n), dtype=torch.int32)
    indice_pairs = torch.stack([in_tab, out_tab], dim=0)
    return [features, filters, out_bp, indice_pairs, counts]


def make_inputs(n: int, c_in: int, c_out: int, kv: int, num_act_out: int,
                variant: str = "dgrad", dtype: str = "float32",
                counts_max=None, seed: int = 0):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器
    搬运到设备）。variant 同时是构造 Model 的 init 超参（评测端按 task.yaml
    io.init_inputs 从 case 取），此处仅做合法性校验。

    消费顺序固定：features -> filters -> out_bp -> counts -> in_tab -> out_tab
    （均来自同一 torch.Generator(seed)，同 seed 下逐位可复现）。
        counts[k] = indice_num_per_loc[k] ~ randint[0, counts_max]
        （counts_max 缺省为 n；取 0 时全部 n_k = 0，输出为全零边界）
        in_tab / out_tab 覆盖全部 j ∈ [0, n)（有效前缀与填充垃圾同分布，
        填充为范围内的随机值，正确实现必须按 counts 屏蔽）。
    返回与 forward 形参同序的
    (features, filters, out_bp, indice_pairs, indice_num_per_loc)。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert variant in ("dgrad", "wgrad"), "variant 必须是 dgrad / wgrad 之一"
    assert min(n, c_in, c_out, kv, num_act_out) >= 1, "各维均需 >= 1"
    cm = n if counts_max is None else int(counts_max)
    assert 0 <= cm <= n, "counts_max 需落在 [0, n]"
    assert n <= 2 ** 24 and num_act_out <= 2 ** 24, "行号下标须 <= 2^24（fp32 精确表示）"

    features = torch.randn(n, c_in, generator=gen).to(dt)
    filters = torch.randn(c_out, c_in, kv, generator=gen).to(dt)
    out_bp = torch.randn(num_act_out, c_out, generator=gen).to(dt)
    counts = torch.randint(0, cm + 1, (kv,), generator=gen, dtype=torch.int32)
    in_tab = torch.randint(0, n, (kv, n), generator=gen, dtype=torch.int32)
    out_tab = torch.randint(0, num_act_out, (kv, n), generator=gen,
                            dtype=torch.int32)
    indice_pairs = torch.stack([in_tab, out_tab], dim=0)
    return features, filters, out_bp, indice_pairs, counts
