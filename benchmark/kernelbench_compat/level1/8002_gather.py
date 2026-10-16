# 8002_gather —— Gather / GatherElements（索引选择家族）的 model_class
# （KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=1, problem_id=8002, backend=cuda（HIP）。

import torch
import torch.nn as nn


class Model(nn.Module):
    """Gather / GatherElements：沿指定轴按整数索引选择元素的索引搬移家族。

    __init__ 参数 variant 选择两条路径之一，语义定义如下：
        "gather"（沿轴索引选择）：indices 为一维整数向量，长度 K 任意；
            out[i_0, ..., i_{axis-1}, j, i_{axis+1}, ...] =
                data[i_0, ..., i_{axis-1}, indices[j], i_{axis+1}, ...]
            即 indices[j] 一次指定第 axis 维上整条切片的选取；输出形状 =
            data 形状把第 axis 维替换为 K。
        "gather_elements"（同形索引逐元素选择）：indices 与 data 维数相同，
            除第 axis 维外各维长度与 data 一致（第 axis 维长度 K 可与 data
            不同）；out[i_0, ..., i_{n-1}] = data[i_0, ..., m, ..., i_{n-1}]，
            其中 m = indices[i_0, ..., i_{n-1}]——每个输出元素沿 axis 独立
            取下标；输出形状 = indices 形状。
    两者的差异：gather 的索引是一维向量，一次指定整条 axis 切片如何选取；
    gather_elements 的索引与输出同形，逐个输出元素独立指定沿 axis 的下标。
    __init__ 参数 axis 指定选择维（torch 惯例：非负从 0 起，负数从末尾
    计数），缺省 1。

    数值契约：本算子是纯索引搬移，无任何算术——正确实现的输出与输入中
    被选中元素逐位一致（位模式不变，不做 dtype 转换、不做数值运算）。

    输入输出规格：
        data    float32 / float16 / bfloat16，contiguous，ndim ∈ [1, 4]，
                各维 >= 1，元素总数 <= 2^23；值域为常规 randn 量级（无
                inf/NaN）
        indices 整数张量：variant="gather" 时 int64 一维（长度 K >= 1）；
                variant="gather_elements" 时 int32、与 data 同维数（形状
                约束见上）；所有索引值 ∈ [0, data.shape[axis])，输入不含
                越界索引
        out     单张量，与 data 同 dtype、contiguous：gather 时形状 =
                data 形状将 axis 维替换为 K；gather_elements 时形状 =
                indices 形状。forward 必须返回单个张量（评测器按单输出
                的 .shape / allclose 比较）。
        在线评测器会把全部输入 cast 成 fp32 传入（索引为 < 2^24 的小非负
        整数，fp32 可精确表示，无损恢复；fp16/bf16 -> fp32 亦无损）；
        离线终审按声明 dtype 原生传入——实现须对两种传入都正确。

    边界行为：K=1、K 不等于 data.shape[axis]、轴长 1、非 2 的幂 / 素数
    长度、索引重复命中同一元素都必须正确处理；负 axis 按 torch 惯例解释。

    实现约束（违规判负）：
        - 核心计算（索引展开 / 地址计算 + 按地址 load/store）必须在提交
          文件内以 HIP kernel 完成。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 纯搬移契约：输出与 data 中被选中元素逐位一致（含 float16 /
          bfloat16 的位模式），不得引入任何数值变换。
        - forward 返回单张量 out（评测器按单输出比较）。

    禁用列表（静态审计剥除 docstring 与注释后按字面子串匹配）：
    torch.gather / .gather( / torch.index_select /
    .index_select( / torch.take_along_dim / .take_along_dim( /
    torch.take( / .take(（含 Tensor.take 方法形式） /
    torch.embedding / nn.Embedding / .embedding( / hipdnn。
    目标硬件：海光 DCU（gfx936/gfx938），实现语言 HIP（经 hipcc 编译；在 CUDA_CU_SRC / CUDA_CPP_SRC 锚点区编写，设备侧必须 include <hip/hip_runtime.h>，本机无 <cuda_runtime.h>）。
    """

    def __init__(self, variant: str = "gather", axis: int = 1):
        super().__init__()
        assert variant in ("gather", "gather_elements"), "variant 必须是 gather / gather_elements 之一"
        self.variant = variant
        self.axis = int(axis)

    def forward(self, data, indices):
        # 在线评测器会把全部输入（含整数索引）cast 成 fp32；索引是小非负
        # 整数，fp32 可精确表示，此处无损恢复为 int64
        ndim = data.dim()
        axis = self.axis % ndim
        idx = indices.to(torch.long)

        if self.variant == "gather_elements":
            # 同形索引：输出形状 = indices 形状，沿 axis 逐元素取下标
            return torch.gather(data, axis, idx)

        # 一维索引向量：展开成与 data 同维数的索引张量后沿 axis 选取
        # （除 axis 维外广播 data 的既有下标），输出形状 = data 形状换
        # axis 维为 K
        k = idx.shape[0]
        view = [1] * ndim
        view[axis] = k
        full = idx.view(view).expand(data.shape[:axis] + (k,) + data.shape[axis + 1:])
        return torch.gather(data, axis, full)


def get_init_inputs():
    return ["gather", 1]   # variant, axis


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    b, c, h = 2, 3, 4
    k = 2
    data = torch.randn(b, c, h)          # float32 主 dtype
    indices = torch.randint(0, c, (k,))  # int64，值域 [0, c)
    return [data, indices]


def make_inputs(variant, data_shape, index_shape, axis, dtype="float32", seed=0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：data（torch.randn 后 cast 到 dtype）-> indices
    （torch.randint，上限为 data 在 axis 维的长度；dtype 按口径：
    gather 为 int64、gather_elements 为 int32），全部来自同一
    torch.Generator(seed)，同 seed 下逐位可复现。
    variant 与 axis 同时是构造 Model 的 init 超参，此处做合法性校验。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert variant in ("gather", "gather_elements"), "variant 必须是 gather / gather_elements 之一"
    ndim = len(data_shape)
    assert 1 <= ndim <= 4, "需 1 <= data ndim <= 4"
    assert all(s >= 1 for s in data_shape), "data 各维长度需 >= 1"
    d = int(axis) % ndim
    numel = 1
    for s in data_shape:
        numel *= s
    assert numel <= 2 ** 23, "data 元素总数需 <= 2^23"
    n = data_shape[d]
    if variant == "gather":
        assert len(index_shape) == 1, "gather 的 indices 需为一维"
        k = int(index_shape[0])
    else:
        assert len(index_shape) == ndim, "gather_elements 的 indices 需与 data 同维数"
        for i, (sd, si) in enumerate(zip(data_shape, index_shape)):
            if i != d:
                assert si == sd, "gather_elements 的 indices 除 axis 维外各维需与 data 一致"
        k = int(index_shape[d])
    assert k >= 1, "索引长度 K 需 >= 1"
    assert (numel // n) * k <= 2 ** 23, "输出元素总数需 <= 2^23"
    data = torch.randn(*data_shape, generator=gen).to(dt)
    idx_dtype = torch.int64 if variant == "gather" else torch.int32
    indices = torch.randint(0, n, list(index_shape), generator=gen, dtype=idx_dtype)
    return data, indices
