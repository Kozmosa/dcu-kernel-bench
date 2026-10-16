# 5007_spconv_indice_pairs —— 稀疏卷积 indice pairs 生成（spconv 风格索引
# 对表构建）的 model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=2, problem_id=5007, backend=cuda（HIP）。

import torch
import torch.nn as nn


class Model(nn.Module):
    """稀疏卷积 indice pairs 生成：由活跃输入点表与卷积几何参数，构建稀疏卷
    积前向所需的索引对表（spconv 风格 kernel map，Native 算法、非 subm）。

    输入 indices 是 (n, ndim+1) 的活跃点表，每行 [batch_idx, x_0, ..., x_{ndim-1}]
    （列 0 为 batch 下标，其余为空间坐标），行互异、顺序任意。

    卷积几何由 6 个构造超参给出：batch_size（B）、input_dims（网格各维大小
    D）、ksize（K）、stride（S）、padding（P）、dilation（L），均为每维列表
    （1D/2D/3D）。派生量：
        kv = prod(K)
        output_dims[d] = (D[d] + 2*P[d] - L[d]*(K[d]-1) - 1) // S[d] + 1
    kernel 偏移 k ∈ [0, kv) 按混合进制（C-order）分解为每维偏移 o_d：第 0 维
    最慢、第 ndim-1 维最快，即 k = o_0*K_1*...*K_{n-1} + ... + o_{ndim-1}。

    生成规则（按此顺序枚举：外层 k 从 0 到 kv-1，内层输入行 i 从 0 到 n-1；
    每个有效对 (k, i) 依次处理）：
        对每维 d 令 val = x_d + P[d] - o_d*L[d]；
        对 (k, i) 有效当且仅当 0 <= batch_idx < B，且每维满足
            val % S[d] == 0（整除）且 0 <= val // S[d] < output_dims[d]，
        此时输出坐标为 q_d = val // S[d]（batch 不变）。
        (1) 输出点去重编号：有效输出坐标 (batch_idx, q_0..q_{ndim-1}) 首次
            出现时按枚举序赋予递增行号 r（从 0 开始）；out_indices 的第 r 行
            写入该坐标。有效输出数 m 的上界为
            cap = min(kv * n, B * prod(output_dims))（有效对数与输出网格容量
            的较小者），out_indices 形状固定 (cap, ndim+1)（容量式分配，有效
            行数 m <= cap），行号 >= m 的行全部填 -1。
        (2) indice_pairs 形状 (2, kv, n)，先全填 -1；对偏移 k，其第 c 个有效
            对（c 从 0 递增计数，密集前缀槽位）写入：
                indice_pairs[0, k, c] = i      # 输入行号（gather 源）
                indice_pairs[1, k, c] = r      # 输出行号（scatter 目标）
            槽位 c >= 该偏移有效对数的表项保持 -1。
        (3) indice_num_per_loc 形状 (kv,)：indice_num_per_loc[k] 为偏移 k 的
            有效对数（即 (2) 中的 c 计数终值）。
    一个 1D 示例（B=1，input_dims=[4]，ksize=[2]，stride=[2]，padding=[0]，
    dilation=[1]，output_dims=[2]，cap = min(kv*n, B*prod(output_dims)) =
    min(6, 2) = 2），indices = [[0,0],[0,1],[0,3]]：
        k=0（o=0）：i=0 命中输出 (0,0)→r=0；i=1（val=1 不整除）、i=2（val=3
        不整除）无效。k=1（o=1）：i=0（val=-1）无效；i=1（val=0）命中
        (0,0)→r=0；i=2（val=2）命中 (0,1)→r=1。
        out_indices = [[0,0],[0,1]]（m=2 恰好占满容量）；indice_num_per_loc =
        [1, 2]；indice_pairs[0] = [[0,-1,-1],[1,2,-1]]，indice_pairs[1] =
        [[0,-1,-1],[0,1,-1]]。

    单张量输出协议（forward 返回一维 int32 张量，三段拼接；cap =
    min(kv*n, B*prod(output_dims))）：
        段 1 [0, cap*(ndim+1))                         = out_indices 展平
        段 2 [cap*(ndim+1), cap*(ndim+1) + 2*kv*n)     = indice_pairs 展平
        段 3 尾部 kv 个元素                             = indice_num_per_loc
    输出总数恒为 cap*(ndim+1) + 2*kv*n + kv，由构造超参唯一决定。

    输入输出规格：
        indices  int32 (n, ndim+1)，contiguous；n >= 1，ndim ∈ [1, 3]；
                 行互异，0 <= batch_idx < B，0 <= x_d < D[d]。
        超参域   D[d] >= 1，K[d] >= 1，S[d] >= 1，P[d] >= 0，L[d] >= 1，
                 output_dims[d] >= 1；n <= B * prod(D)。
        out      一维 int32（见上打包协议）。索引语义为精确匹配（逐位一致），
                 无容差。
        评测器会把全部输入张量 cast 成 fp32 传入（索引值 <= 2^24 时 fp32 可
        精确表示），实现需先恢复 int32 再参与索引运算。

    实现约束（违规判负）：
        - 核心计算（坐标映射与整除/范围判定、输出点去重编号、三个输出表的
          填充）必须在提交文件内以 HIP kernel 完成；ATen 只允许 dtype
          恢复、torch.full/zeros/arange、形状整理与 torch.cat 等非核心操作。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 输出为索引：必须逐位一致（行号顺序、槽位顺序、-1 填充全部按上述
          规则），任何顺序差异都判负。
        - forward 返回单张量（评测器按单输出比较 shape/dtype/allclose）。

    禁用列表：aiter / miopen / hipdnn / torch.unique /
    torch.unique_consecutive / torch.sort / torch.argsort / torch.msort /
    .sort( / .argsort( / torch.mode / torch.histc / torch.bincount /
    torch.topk / scatter（含 scatter_ / scatter_add_ / scatter_reduce /
    torch.scatter 等全部形式）/ index_put（含 index_put_ 与 torch.index_put）/
    segment_reduce。
    目标硬件：海光 DCU（gfx936/gfx938），实现语言 HIP（经 hipcc 编译；在 CUDA_CU_SRC / CUDA_CPP_SRC 锚点区编写，设备侧必须 include <hip/hip_runtime.h>，本机无 <cuda_runtime.h>）。
    """

    def __init__(self, batch_size: int = 2, input_dims=None, ksize=None,
                 stride=None, padding=None, dilation=None):
        super().__init__()
        # 构造超参：网格与卷积几何（缺省值 = get_inputs 固定族）
        self.batch_size = int(batch_size)
        self.input_dims = list(input_dims if input_dims is not None else [4, 32, 32])
        self.ksize = list(ksize if ksize is not None else [3, 3, 3])
        self.stride = list(stride if stride is not None else [1, 1, 1])
        self.padding = list(padding if padding is not None else [1, 1, 1])
        self.dilation = list(dilation if dilation is not None else [1, 1, 1])
        assert self.batch_size >= 1, "batch_size 需 >= 1"
        ndim = len(self.input_dims)
        assert 1 <= ndim <= 3, "ndim 需在 [1, 3]"
        assert all(len(p) == ndim for p in
                   (self.ksize, self.stride, self.padding, self.dilation)), \
            "ksize/stride/padding/dilation 需与 input_dims 同维"
        for d in range(ndim):
            assert self.input_dims[d] >= 1 and self.ksize[d] >= 1 \
                and self.stride[d] >= 1 and self.padding[d] >= 0 \
                and self.dilation[d] >= 1, "各维超参非法"
            out_d = (self.input_dims[d] + 2 * self.padding[d]
                     - self.dilation[d] * (self.ksize[d] - 1) - 1) \
                // self.stride[d] + 1
            assert out_d >= 1, "output_dims 每维需 >= 1"

    def forward(self, indices):
        # 在线评测会把全部输入张量 cast 成 fp32：索引值 <= 2^24 时精确，
        # 这里恢复 int32
        rows = indices.to(torch.int32).tolist()
        n = len(rows)
        B = self.batch_size
        D, K = self.input_dims, self.ksize
        S, P, L = self.stride, self.padding, self.dilation
        ndim = len(D)

        kv = 1
        for k in K:
            kv *= k
        out_dims = [(D[d] + 2 * P[d] - L[d] * (K[d] - 1) - 1) // S[d] + 1
                    for d in range(ndim)]
        # out_indices 容量 = 有效输出数上界：min(有效对总数 kv*n, 输出网格容量)
        out_vol = B
        for od in out_dims:
            out_vol *= od
        cap = min(kv * n, out_vol)

        # kernel 偏移的 C-order 分解（第 0 维最慢、第 ndim-1 维最快）
        offs = []
        for k in range(kv):
            o = [0] * ndim
            t = k
            for d in range(ndim - 1, -1, -1):
                o[d] = t % K[d]
                t //= K[d]
            offs.append(o)

        out_flat = [-1] * (cap * (ndim + 1))
        pair_flat = [-1] * (2 * kv * n)
        num = [0] * kv
        table = {}   # 输出坐标 tuple -> 输出行号（首次出现序）
        m = 0
        for k in range(kv):
            o = offs[k]
            base = k * n
            cnt = 0
            for i in range(n):
                row = rows[i]
                b = int(row[0])
                if b < 0 or b >= B:
                    continue
                q = [0] * ndim
                valid = True
                for d in range(ndim):
                    val = int(row[d + 1]) + P[d] - o[d] * L[d]
                    if val % S[d] != 0:
                        valid = False
                        break
                    qd = val // S[d]
                    if qd < 0 or qd >= out_dims[d]:
                        valid = False
                        break
                    q[d] = qd
                if not valid:
                    continue
                key = (b, *q)
                r = table.get(key, -1)
                if r < 0:
                    r = m
                    table[key] = r
                    start = r * (ndim + 1)
                    out_flat[start] = b
                    for d in range(ndim):
                        out_flat[start + 1 + d] = q[d]
                    m += 1
                pair_flat[base + cnt] = i
                pair_flat[kv * n + base + cnt] = r
                cnt += 1
            num[k] = cnt

        # 单张量打包：out_indices 段 -> indice_pairs 段 -> indice_num_per_loc 段
        seg1 = torch.tensor(out_flat, dtype=torch.int32)
        seg2 = torch.tensor(pair_flat, dtype=torch.int32).reshape(2, kv, n)
        seg3 = torch.tensor(num, dtype=torch.int32)
        return torch.cat([seg1.reshape(-1), seg2.reshape(-1), seg3])


def get_init_inputs():
    # batch_size, input_dims, ksize, stride, padding, dilation（固定族）
    return [2, [4, 32, 32], [3, 3, 3], [1, 1, 1], [1, 1, 1], [1, 1, 1]]


def _sample_indices(n, batch_size, input_dims, gen):
    # 在 B×D 网格上无放回均匀采样 n 个活跃点（行互异），保持采样顺序（不排
    # 序，模拟真实点云的任意顺序）；gen 为 None 时消费全局 RNG
    sv = 1
    for d in input_dims:
        sv *= d
    vol = batch_size * sv
    assert 1 <= n <= vol, "n 需落在 [1, B*prod(D)]（活跃点行互异）"
    if gen is not None:
        perm = torch.randperm(vol, generator=gen)[:n]
    else:
        perm = torch.randperm(vol)[:n]
    ndim = len(input_dims)
    rows = []
    for v in perm.tolist():
        b, t = v // sv, v % sv
        row = [b]
        for d in range(ndim):
            sd = 1
            for dd in range(d + 1, ndim):
                sd *= input_dims[dd]
            row.append(t // sd % input_dims[d])
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int32)


def get_inputs():
    # 固定 shape 族（3D，B=2，D=[4,32,32]，3x3x3 卷积，stride/padding/dilation
    # 全 1）；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，多轮
    # correctness trial 因此获得输入多样性。唯一随机消费：torch.randperm。
    return [_sample_indices(1024, 2, [4, 32, 32], None)]


def make_inputs(n: int, batch_size: int, input_dims, ksize, stride, padding,
                dilation, seed: int = 0):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬
    运到设备）。

    消费顺序固定：唯一随机消费 torch.randperm(B*prod(D), torch.Generator(seed))
    的前 n 个位置，解码为 (batch_idx, x_0..x_{ndim-1}) 行（C-order，采样序即
    行序，不排序）。同 seed 下逐位可复现。6 个几何超参同时是构造 Model 的
    init 超参，此处仅做合法性校验（同 __init__ 的断言）。
    返回与 forward 形参同序的 (indices,)。
    """
    gen = torch.Generator().manual_seed(int(seed))
    # 合法性校验（与 Model.__init__ 同域，供评测端提前定位坏 case）
    ndim = len(input_dims)
    assert n >= 1 and batch_size >= 1 and 1 <= ndim <= 3, "n/B/ndim 非法"
    assert all(len(p) == ndim for p in (ksize, stride, padding, dilation)), \
        "ksize/stride/padding/dilation 需与 input_dims 同维"
    vol = 1
    for d in range(ndim):
        assert input_dims[d] >= 1 and ksize[d] >= 1 and stride[d] >= 1 \
            and padding[d] >= 0 and dilation[d] >= 1, "各维超参非法"
        out_d = (input_dims[d] + 2 * padding[d] - dilation[d] * (ksize[d] - 1)
                 - 1) // stride[d] + 1
        assert out_d >= 1, "output_dims 每维需 >= 1"
        vol *= input_dims[d]
    assert n <= batch_size * vol, "n 需 <= B*prod(D)（活跃点行互异）"
    return (_sample_indices(n, batch_size, input_dims, gen),)
