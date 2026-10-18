# 1032_deform_attention_bwd —— 多尺度可变形注意力反向（deform_attn_dgrad）
# 的 model_class（KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=1032, backend=cuda（HIP）。

import torch
import torch.nn as nn


def _deform_attn_forward(value_f, spatial_shapes, level_start_index,
                         sampling_locations_f, attention_weights_f):
    """多尺度可变形注意力前向（float32，与 1031 同口径手写实现）。

    与 1031 题目参考逐语义一致：坐标展开、四角点零填充双线性插值、采样权重
    沿 (levels×points) 展平一维 softmax 归一化、跨采样点加权归约。唯一差异
    是 floor 路径先 detach（floor 梯度为 0 的次梯度约定显式化，梯度只经
    插值分数 fx/fy 回传），数值与前向完全相同。
    输入均为 float32 张量（调用方负责 cast 与 requires_grad），返回未 cast
    的 float32 输出 (N, Q, heads*D)，供梯度链使用。
    """
    n, n_keys, num_heads, d = value_f.shape
    n_levels = spatial_shapes.shape[0]
    n_queries = sampling_locations_f.shape[1]
    n_points = sampling_locations_f.shape[4]

    # 采样权重归一化：沿 (levels × points) 展平一维做数值稳定 softmax
    wf = attention_weights_f.reshape(n, n_queries, num_heads,
                                     n_levels * n_points)
    wf = torch.softmax(wf, dim=-1).view(n, n_queries, num_heads,
                                        n_levels, n_points)

    n_idx = torch.arange(n, device=value_f.device).view(n, 1, 1)
    h_idx = torch.arange(num_heads, device=value_f.device).view(1, 1, num_heads)

    out = torch.zeros(n, n_queries, num_heads, d,
                      dtype=torch.float32, device=value_f.device)
    for l in range(n_levels):
        h_l = int(spatial_shapes[l, 0])
        w_l = int(spatial_shapes[l, 1])
        base = int(level_start_index[l])
        v_l = value_f[:, base: base + h_l * w_l]   # (n, H_l*W_l, heads, d)

        # 归一化坐标 × 层尺寸 → 实数采样坐标（第 0 通道沿 W、第 1 通道沿 H）
        sx = sampling_locations_f[:, :, :, l, :, 0] * float(w_l)
        sy = sampling_locations_f[:, :, :, l, :, 1] * float(h_l)
        # floor / 整数索引路径视为常数（显式 detach），梯度只经 fx/fy 回传
        ix0 = torch.floor(sx.detach()).to(torch.long)
        iy0 = torch.floor(sy.detach()).to(torch.long)
        fx = sx - ix0.to(torch.float32)
        fy = sy - iy0.to(torch.float32)

        def corner(iy, ix, wgt):
            # 单角点采样：越界角点贡献记 0（零填充语义）；clamp 仅防索引
            # 越界，乘 0 权重抵消
            valid = ((iy >= 0) & (iy <= h_l - 1)
                     & (ix >= 0) & (ix <= w_l - 1)).to(torch.float32)
            flat = iy.clamp(0, h_l - 1) * w_l + ix.clamp(0, w_l - 1)
            vals = v_l[n_idx, flat, h_idx]        # (n, q, heads, P, d)
            return vals * (valid * wgt).unsqueeze(-1)

        for p in range(n_points):
            iy0p, ix0p = iy0[..., p], ix0[..., p]      # (n, q, heads)
            fyp, fxp = fy[..., p], fx[..., p]
            sampled = (corner(iy0p, ix0p, (1.0 - fyp) * (1.0 - fxp))
                       + corner(iy0p, ix0p + 1, (1.0 - fyp) * fxp)
                       + corner(iy0p + 1, ix0p, fyp * (1.0 - fxp))
                       + corner(iy0p + 1, ix0p + 1, fyp * fxp))
            out = out + wf[:, :, :, l, p].unsqueeze(-1) * sampled

    return out.view(n, n_queries, num_heads * d)


class Model(nn.Module):
    """多尺度可变形注意力反向（Multi-Scale Deformable Attention Backward，
    deform_attn_dgrad）：给定前向输入与上游梯度 grad_output，一次调用同时
    计算 value、sampling_locations、attention_weights 三者的梯度。

    前向语义（中间计算全程 float32，与配套前向题同口径）：value (N, n_keys,
    heads, D) 按多尺度特征层拼接——第 l 层贡献 spatial_shapes[l] = (H_l, W_l)
    个键，层内按行主序（先 H 后 W）排布于 [level_start_index[l],
    level_start_index[l] + H_l*W_l)，level_start_index 必须是各层 H_l*W_l 的
    前缀和（首元素 0），n_keys = Σ_l H_l*W_l。sampling_locations
    (N, Q, heads, L, P, 2) 最后一维为归一化坐标 (x, y)：第 0 通道沿 W、第 1
    通道沿 H（生成域 [0,1)），实际采样坐标 loc_x = x * W_l、loc_y = y * H_l；
    采样值经四角点双线性插值（floor + 分数 fx/fy，角点越界贡献记 0 的零填充
    语义，无 clamp / 最近邻回拉）；attention_weights (N, Q, heads, L, P) 为
    非负未归一化分数，先沿 (levels × points) 展平一维做数值稳定 softmax
    （减最大值）归一化为凸组合权重，再与各采样点插值值加权求和，输出
    o (N, Q, heads*D)。

    反向语义（对上述 float32 前向、在输出 cast 回 value dtype 之前的结果
    反传；grad_output 先 cast 到 float32）：记 dO = grad_output 的 float32
    形式，w = softmax(attention_weights 沿 L*P 展平一维)，S_{l,p}[c] 为点
    (l,p) 的四角点零填充双线性采样值，t_{l,p} = <dO, S_{l,p}>，则三个梯度
    为 o 对三个浮点输入的解析梯度：
      grad_value[n, key, h, c] = Σ_{q, l, p} w_{l,p} * corner_weight *
                                 dO[n, q, h*D + c]
        （逐角点散射累加，越界角点不散射）
      grad_sampling_locations[n, q, h, l, p, 0] =
          Σ_c dO[n, q, h*D + c] * w_{l,p} * W_l *
          ((1-fy) * (v01 - v00) + fy * (v11 - v10))
      grad_sampling_locations[n, q, h, l, p, 1] =
          Σ_c dO[n, q, h*D + c] * w_{l,p} * H_l *
          ((1-fx) * (v10 - v00) + fx * (v11 - v01))
        （v00/v01/v10/v11 为该点四角点像素值，越界角点记 0；梯度只经分数
        fx/fy 回传，floor / 整数索引路径梯度为 0——双线性采样标准次梯度）
      grad_attention_weights[n, q, h, l, p] = w_{l,p} *
          (t_{l,p} - Σ_{l', p'} w_{l',p'} * t_{l',p'})
        （softmax 反传；L*P = 1 时恒为 0）
    softmax、插值、加权归约与全部梯度累加在 float32 中进行，三个梯度分别
    cast 回 value dtype 后打包输出。

    输出契约（单张量，一维拼接打包，段序固定不得交换）：
      out = cat([grad_value.reshape(-1),
                 grad_sampling_locations.reshape(-1),
                 grad_attention_weights.reshape(-1)])
      一维张量，dtype 与 value 一致，长度
          = n*n_keys*heads*D + n*n_queries*heads*L*P*2 + n*n_queries*heads*L*P
      段 0 grad_value：n*n_keys*heads*D 个元素，(N, n_keys, heads, D) 行主序
          展平；
      段 1 grad_sampling_locations：n*n_queries*heads*L*P*2 个元素，
          (N, Q, heads, L, P, 2) 行主序展平（末维 (x, y) 通道顺序保持）；
      段 2 grad_attention_weights：n*n_queries*heads*L*P 个元素，
          (N, Q, heads, L, P) 行主序展平。

    输入输出规格（张量均 contiguous）：
      value              float32 / float16 (N, n_keys, heads, D)
      spatial_shapes     int64 (L, 2)，每行 (H_l, W_l)，H_l / W_l >= 1
      level_start_index  int64 (L,)，各层键数前缀和
      sampling_locations 同 value dtype (N, Q, heads, L, P, 2)，生成域 [0,1)
      （fp16 输入下个别元素可因 cast 舍入到恰为 1.0，此时 loc_x = W_l 完全
      越界、采样值为 0，按零填充语义处理）
      attention_weights  同 value dtype (N, Q, heads, L, P)，非负
      grad_output        同 value dtype (N, Q, heads*D)，前向输出布局的上游
                         梯度，生成域 [0,1)
      out                单张量、同 value dtype，1-D 三段拼接（见上）
      各维度均为任意正整数（含 1、非 2 的幂、非 16 对齐；单层单像素
      (H_l, W_l) = (1, 1)、1×N / N×1 细条层、Q=1、P=1、L=1 均为合法边界，
      L*P=1 时 grad_attention_weights 恒为 0）。评测器会把全部输入 cast 成
      fp32 传入（低精度与 int64 索引无损——索引值 <= 2^24 时 fp32 可精确
      表示），实现须把 spatial_shapes / level_start_index 恢复为整数索引后
      使用。

    实现约束（违规判负）：
      - 核心计算（softmax 反传、四角点梯度散射与插值分数偏导、跨 query 的
        grad_value 散射累加）必须在提交文件内以 HIP kernel 手写完成。
      - 禁止借助 torch.autograd / autograd.grad / .backward() 反传求梯度，
        梯度须按上述解析式手写。
      - 全部梯度累加在 float32 中进行，三段梯度 cast 回 value dtype 后按
        上述段序拼接；forward 返回单张量 out（评测器按单输出比较）。
      - ModelNew 的 __init__ 与 forward 签名不可更改。

    禁用列表：torchvision（含 torchvision.ops.ms_deform_attn 等现成可变形
    注意力 / 可变形卷积原语）、mmcv、miopen、hipdnn、aiter、
    torch.nn.functional.grid_sample（F.grid_sample）、
    torch.nn.functional.interpolate（F.interpolate）、torch.matmul /
    torch.bmm / torch.mm / torch.einsum / torch.tensordot、
    torch.softmax / torch.nn.functional.softmax（F.softmax）、
    torch.autograd（含 autograd.grad）、.backward()。
    目标硬件：海光 DCU（gfx936/gfx938），实现语言 HIP（经 hipcc 编译；在 CUDA_CU_SRC / CUDA_CPP_SRC 锚点区编写，设备侧必须 include <hip/hip_runtime.h>，本机无 <cuda_runtime.h>）。
    """

    def __init__(self):
        super().__init__()

    def forward(self, value, spatial_shapes, level_start_index, sampling_locations, attention_weights, grad_output):
        # 评测器会把全部输入 cast 成 fp32（cuda 后端）；spatial_shapes /
        # level_start_index 是小整数索引，fp32 表示无损，此处恢复为 long
        spatial_shapes = spatial_shapes.to(torch.long)
        level_start_index = level_start_index.to(torch.long)

        n, n_keys, num_heads, d = value.shape
        n_levels = spatial_shapes.shape[0]
        n_queries = sampling_locations.shape[1]
        n_points = sampling_locations.shape[4]
        assert sampling_locations.dim() == 6 \
            and sampling_locations.shape[2] == num_heads \
            and sampling_locations.shape[3] == n_levels \
            and sampling_locations.shape[5] == 2, \
            "sampling_locations 须为 (N, Q, heads, L, P, 2)"
        assert attention_weights.shape == (n, n_queries, num_heads, n_levels, n_points), \
            "attention_weights 须为 (N, Q, heads, L, P)"
        assert value.shape[1] == int(spatial_shapes.prod(dim=1).sum()), \
            "value 第 1 维须等于各层 H_l*W_l 之和"
        assert grad_output.shape == (n, n_queries, num_heads * d), \
            "grad_output 须为 (N, Q, heads*D)"

        # 对 float32 中间前向构图后用 autograd 反传（reference 不受禁用列表
        # 约束；梯度定义 = float32 语义前向的解析梯度）
        vg = value.detach().to(torch.float32).requires_grad_(True)
        lg = sampling_locations.detach().to(torch.float32).requires_grad_(True)
        ag = attention_weights.detach().to(torch.float32).requires_grad_(True)
        out = _deform_attn_forward(vg, spatial_shapes, level_start_index, lg, ag)
        grad_value, grad_sampling_loc, grad_attn_weight = torch.autograd.grad(
            out, (vg, lg, ag), grad_output.to(torch.float32))

        # 单张量输出：三段按 grad_value | grad_sampling_locations |
        # grad_attention_weights 顺序一维拼接，各 cast 回 value dtype
        return torch.cat([grad_value.to(value.dtype).reshape(-1),
                          grad_sampling_loc.to(value.dtype).reshape(-1),
                          grad_attn_weight.to(value.dtype).reshape(-1)])


def get_init_inputs():
    return []   # 本题无构造超参（Model 无参构造）


def get_inputs():
    # 固定 shape 族（n=2、heads=2、D=32、L=2、P=2、Q=32；spatial_shapes 每层
    # (H_l, W_l) ~ randint[1,16)，n_keys/value 形状由其推导）；随机部分消费
    # 全局 RNG——评测器在 set_seed 后调用本函数，多轮 correctness trial 因此
    # 获得输入多样性。消费顺序固定：spatial_shapes -> value ->
    # sampling_locations -> attention_weights -> grad_output
    n, n_heads, d_head, n_levels, n_points, n_queries = 2, 2, 32, 2, 2, 32
    spatial_shapes = torch.randint(1, 16, (n_levels, 2), dtype=torch.int64)
    n_keys = int(spatial_shapes.prod(dim=1).sum())
    count_per_level = spatial_shapes.prod(dim=1)
    level_start_index = torch.zeros_like(count_per_level)
    level_start_index[1:] = torch.cumsum(count_per_level[:-1], dim=0)
    value = torch.rand(n, n_keys, n_heads, d_head, dtype=torch.float32)
    sampling_locations = torch.rand(
        n, n_queries, n_heads, n_levels, n_points, 2, dtype=torch.float32)
    attention_weights = torch.rand(
        n, n_queries, n_heads, n_levels, n_points, dtype=torch.float32)
    grad_output = torch.rand(n, n_queries, n_heads * d_head, dtype=torch.float32)
    return [value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights, grad_output]


def make_inputs(n: int, n_heads: int, embed_dims_per_head: int, n_queries: int,
                n_levels: int, n_points: int, spatial_shapes=None,
                max_size: int = 15, dtype: str = "float32", seed: int = 0):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器
    搬运到设备）。本题无构造超参，case 字段全部是这里的形参。

    消费顺序固定：spatial_shapes（未显式给出时逐元素 ~ randint[1, max_size]，
    缺省 max_size=15）-> value -> sampling_locations -> attention_weights ->
    grad_output（rand 张量后 cast 到 dtype），全部来自同一
    torch.Generator(seed)，同 seed 下逐位可复现；level_start_index 由
    spatial_shapes 推导（前缀和），不消费 RNG。
    返回与 forward 形参同序的
    (value, spatial_shapes, level_start_index, sampling_locations,
    attention_weights, grad_output)。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    assert min(n, n_heads, embed_dims_per_head, n_queries, n_levels, n_points) >= 1, \
        "各维均需 >= 1"
    max_size = int(max_size)
    assert max_size >= 1, "max_size 需 >= 1"
    if spatial_shapes is None:
        ss = torch.randint(1, max_size + 1, (n_levels, 2),
                           generator=gen, dtype=torch.int64)
    else:
        ss = torch.as_tensor(spatial_shapes, dtype=torch.int64).reshape(n_levels, 2)
        assert int(ss.min()) >= 1, "每层 H_l / W_l 需 >= 1"
    n_keys = int(ss.prod(dim=1).sum())
    assert n_keys <= 2 ** 24, "n_keys 须 <= 2^24（fp32 可精确表示的索引上限）"
    count_per_level = ss.prod(dim=1)
    level_start_index = torch.zeros_like(count_per_level)
    level_start_index[1:] = torch.cumsum(count_per_level[:-1], dim=0)
    value = torch.rand(n, n_keys, n_heads, embed_dims_per_head,
                       generator=gen).to(dt)
    sampling_locations = torch.rand(n, n_queries, n_heads, n_levels, n_points, 2,
                                    generator=gen).to(dt)
    attention_weights = torch.rand(n, n_queries, n_heads, n_levels, n_points,
                                   generator=gen).to(dt)
    grad_output = torch.rand(n, n_queries, n_heads * embed_dims_per_head,
                             generator=gen).to(dt)
    return (value, ss, level_start_index, sampling_locations, attention_weights,
            grad_output)
