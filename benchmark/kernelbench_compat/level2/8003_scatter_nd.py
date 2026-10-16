# 8003_scatter_nd —— ScatterND（索引写入家族）的 model_class
# （KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=2, problem_id=8003, backend=cuda（HIP）。

import torch
import torch.nn as nn


class Model(nn.Module):
    """ScatterND：按整型索引元组将 updates 覆盖写入 data 的索引写入家族。

    语义定义：data 形状 (S_0, ..., S_{r-1})，r = data 的维数；indices 为
    int64 张量，形状 (K, r)——每一行是一个完整坐标元组，一次唯一指定
    data 中一个元素的位置；updates 形状 (K,)，与 data 同 dtype。输出
    out 与 data 同形状、同 dtype：
        out[indices[k,0], ..., indices[k,r-1]] = updates[k]
            （命中位置：覆盖写，原值被丢弃并由 updates 对应元素替代）
        其余位置 out[i_0, ..., i_{r-1}] = data[i_0, ..., i_{r-1}]
            （未命中：原值逐位保留）
    写入语义为覆盖写、非累加；K 个坐标元组两两互异（生成端无放回抽样
    保证），每个 data 元素至多被写一次。重复坐标元组下哪个写入生效是
    未定义行为（主流框架对重复索引的写入顺序均不做定义），重复元组不在
    本题输入域内——实现无需为其定义行为，也不得因去重 / 排序等预处理
    改变互异元组的覆盖结果。

    输入输出规格：
        data    float32 / float16 / bfloat16，contiguous，ndim ∈ [1, 6]，
                各维 >= 1，元素总数 <= 2^23；值域为常规 randn 量级（无
                inf/NaN）
        indices int64 (K, r)，r = data 维数，1 <= K <= data 元素总数；
                全部索引值 ∈ [0, S_d)，K 个坐标元组两两互异
        updates 同 data dtype (K,)，contiguous
        out     单张量，与 data 同形状、同 dtype、contiguous——本算子
                天然单输出，无需多输出打包。forward 必须返回单个张量
                （评测器按单输出的 .shape / allclose 比较）。
        在线评测器会把全部输入 cast 成 fp32 传入（索引为 < 2^24 的小
        非负整数、fp32 可精确表示，无损恢复；fp16/bf16 -> fp32 亦无损）；
        离线终审按声明 dtype 原生传入——实现须对两种传入都正确。

    边界行为：K=1、K = data 元素总数（全量打散，data 每个元素都被写）、
    ndim=1（indices 形状 (K, 1)）、非 2 的幂 / 素数长度 / 轴长 1 的形状
    都必须正确处理；输出须是新张量，不得原地修改 data / indices / updates。

    实现约束（违规判负）：
        - 核心计算（r 维坐标 -> 线性偏移的地址计算、updates 的读入与
          未命中位置原值的搬运写出）必须在提交文件内以 HIP kernel
          完成。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 纯搬移契约：未命中位置与 data、命中位置与 updates 均逐位一致
          （位模式不变），不得引入任何数值变换。
        - forward 返回单张量 out（评测器按单输出比较）。

    禁用列表（静态审计剥除 docstring 与注释后按字面子串匹配）：
    torch.scatter / .scatter( / .scatter_( / scatter_add / scatter_reduce /
    index_put / torch.put( / .put_( / index_copy / torch_scatter / hipdnn。
    目标硬件：海光 DCU（gfx936/gfx938），实现语言 HIP（经 hipcc 编译；在 CUDA_CU_SRC / CUDA_CPP_SRC 锚点区编写，设备侧必须 include <hip/hip_runtime.h>，本机无 <cuda_runtime.h>）。
    """

    def __init__(self):
        super().__init__()

    def forward(self, data, indices, updates):
        # 在线评测器会把全部输入（含整数索引）cast 成 fp32；索引是小非负
        # 整数，fp32 可精确表示，此处无损恢复为 int64
        idx = indices.to(torch.long)
        out = data.clone()
        out[tuple(idx.unbind(dim=-1))] = updates.to(out.dtype)
        return out


def get_init_inputs():
    return []   # 无构造超参：写入语义固定为覆盖写


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性
    data = torch.randn(4, 4)                          # float32 主 dtype
    flat = torch.randperm(16)[:2]                     # 无放回抽 2 个线性位置
    indices = torch.stack((flat // 4, flat % 4), dim=1)   # int64 (2, 2)
    updates = torch.randn(2)
    return [data, indices, updates]


def make_inputs(data_shape, num_updates, dtype="float32", seed=0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    消费顺序固定：data（torch.randn 后 cast 到 dtype）-> indices
    （torch.randperm(data 元素总数) 取前 K 个线性下标、按行主序逐维解
    坐标——无放回抽样保证 K 个坐标元组两两互异）-> updates（torch.randn
    后 cast 到 dtype），全部来自同一 torch.Generator(seed)，同 seed 下
    逐位可复现。
    """
    gen = torch.Generator().manual_seed(seed)
    assert dtype in ("float32", "float16", "bfloat16"), "dtype 需为 float32/float16/bfloat16"
    dt = getattr(torch, dtype)
    shape = tuple(int(s) for s in data_shape)
    ndim = len(shape)
    assert 1 <= ndim <= 6, "需 1 <= data ndim <= 6"
    assert all(s >= 1 for s in shape), "data 各维长度需 >= 1"
    numel = 1
    for s in shape:
        numel *= s
    assert numel <= 2 ** 23, "data 元素总数需 <= 2^23"
    k = int(num_updates)
    assert 1 <= k <= numel, "需 1 <= K <= data 元素总数（坐标互异的上界）"
    data = torch.randn(*shape, generator=gen).to(dt)
    flat = torch.randperm(numel, generator=gen)[:k]
    coords = []
    for d in range(ndim):
        stride_d = 1
        for j in range(d + 1, ndim):
            stride_d *= shape[j]
        coords.append(flat // stride_d % shape[d])
    indices = torch.stack(coords, dim=1).to(torch.long)   # int64 (K, r)
    updates = torch.randn(k, generator=gen).to(dt)
    return data, indices, updates
