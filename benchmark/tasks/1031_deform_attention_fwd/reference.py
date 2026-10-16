# 1031_deform_attention_fwd —— 多尺度可变形注意力前向的 model_class
# （KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=3, problem_id=1031, backend=cuda（HIP）。

import torch
import torch.nn as nn


class Model(nn.Module):
    """多尺度可变形注意力前向（Multi-Scale Deformable Attention）：value 采样 +
    手写双线性插值 + 采样权重 softmax 归一化。

    语义（中间计算全程 float32）：value (N, n_keys, heads, D) 按多尺度特征层
    拼接——第 l 层贡献 spatial_shapes[l] = (H_l, W_l) 个键，层内按行主序
    （先 H 后 W）排布于 [level_start_index[l], level_start_index[l] + H_l*W_l)，
    level_start_index 必须是各层 H_l*W_l 的前缀和（首元素 0），且
    n_keys = Σ_l H_l*W_l。sampling_locations (N, Q, heads, L, P, 2) 给出每个
    （batch, query, head）在每层每个采样点的归一化坐标，最后一维为 (x, y)：
    第 0 通道沿 W、第 1 通道沿 H（生成域 [0,1)），实际采样坐标为
        loc_x = x * W_l,  loc_y = y * H_l
    采样值经四角点双线性插值：
        ix0 = floor(loc_x), iy0 = floor(loc_y), fx = loc_x - ix0, fy = loc_y - iy0
        四角点 (iy0,ix0)、(iy0,ix0+1)、(iy0+1,ix0)、(iy0+1,ix0+1) 的权重依次为
        (1-fy)(1-fx)、(1-fy)fx、fy(1-fx)、fy*fx
    某角点的行或列越界（<0 或 >dim-1）时该角点贡献记 0、其余角点照常加权——
    即零填充语义，全程不做 clamp / 最近邻回拉；坐标完全越界时采样值为 0。
    attention_weights (N, Q, heads, L, P) 为非负未归一化分数：先沿 (levels ×
    points) 展平成一维做数值稳定 softmax（减最大值）归一化为凸组合权重，
    再与各采样点的插值值加权求和：
        out[n, q, h*D + d] = Σ_{l, p} w[n,q,h,l,p] * bilinear(value_l)[n,q,h,p,d]
    输出为单张量 out (N, Q, heads*D)，cast 回 value dtype（天然单输出，无多
    输出打包）。L*P = 1 时 softmax 后权重恒为 1。

    输入输出规格（张量均 contiguous）：
      value              float32 / float16 (N, n_keys, heads, D)
      spatial_shapes     int64 (L, 2)，每行 (H_l, W_l)，H_l / W_l >= 1
      level_start_index  int64 (L,)，各层键数前缀和
      sampling_locations 同 value dtype (N, Q, heads, L, P, 2)，生成域 [0,1)
      attention_weights  同 value dtype (N, Q, heads, L, P)，非负
      out                单张量、同 value dtype (N, Q, heads*D)
      各维度均为任意正整数（含 1、非 2 的幂、非 16 对齐；单层单像素
      (H_l, W_l) = (1, 1)、1×N / N×1 细条层均为合法边界）。评测器会把全部
      输入 cast 成 fp32 传入（低精度与 int64 索引无损——索引值 <= 2^24 时
      fp32 可精确表示），实现须把 spatial_shapes / level_start_index 恢复为
      整数索引后使用。

    实现约束（违规判负）：
      - 核心计算（坐标展开、四角点双线性插值、权重 softmax 归一化、跨层跨
        采样点加权归约）必须在提交文件内以 HIP kernel 手写完成。
      - softmax、插值与加权归约在 float32 中进行，输出 cast 回 value dtype。
      - forward 返回单张量 out (N, Q, heads*D)（评测器按单输出比较）。
      - ModelNew 的 __init__ 与 forward 签名不可更改。

    禁用列表：torchvision（含 torchvision.ops.ms_deform_attn 等现成可变形
    注意力 / 可变形卷积原语）、mmcv、miopen、hipdnn、aiter、
    torch.nn.functional.grid_sample（F.grid_sample）、
    torch.nn.functional.interpolate（F.interpolate）、torch.matmul /
    torch.bmm / torch.mm / torch.einsum / torch.tensordot、
    torch.softmax / torch.nn.functional.softmax（F.softmax）。
    目标硬件：海光 DCU（gfx936/gfx938），实现语言 HIP（经 hipcc 编译；在 CUDA_CU_SRC / CUDA_CPP_SRC 锚点区编写，设备侧必须 include <hip/hip_runtime.h>，本机无 <cuda_runtime.h>）。
    """

    def __init__(self):
        super().__init__()

    def forward(self, value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
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

        vf = value.to(torch.float32)                  # (n, n_keys, heads, d)
        loc = sampling_locations.to(torch.float32)    # (n, q, heads, L, P, 2)

        # 采样权重归一化：沿 (levels × points) 展平一维做数值稳定 softmax
        wf = attention_weights.to(torch.float32)
        wf = wf.reshape(n, n_queries, num_heads, n_levels * n_points)
        wf = torch.softmax(wf, dim=-1).view(n, n_queries, num_heads, n_levels, n_points)

        n_idx = torch.arange(n, device=vf.device).view(n, 1, 1)
        h_idx = torch.arange(num_heads, device=vf.device).view(1, 1, num_heads)

        out = torch.zeros(n, n_queries, num_heads, d,
                          dtype=torch.float32, device=vf.device)
        for l in range(n_levels):
            h_l = int(spatial_shapes[l, 0])
            w_l = int(spatial_shapes[l, 1])
            base = int(level_start_index[l])
            v_l = vf[:, base: base + h_l * w_l]       # (n, H_l*W_l, heads, d)

            # 归一化坐标 × 层尺寸 → 实数采样坐标（第 0 通道沿 W、第 1 通道沿 H）
            sx = loc[:, :, :, l, :, 0] * float(w_l)   # (n, q, heads, P)
            sy = loc[:, :, :, l, :, 1] * float(h_l)
            ix0 = torch.floor(sx).to(torch.long)
            iy0 = torch.floor(sy).to(torch.long)
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

        return out.view(n, n_queries, num_heads * d).to(value.dtype)


def get_init_inputs():
    return []   # 本题无构造超参（Model 无参构造）


def get_inputs():
    # 固定 shape 族（n=2、heads=2、D=32、L=2、P=2、Q=32；spatial_shapes 每层
    # (H_l, W_l) ~ randint[1,16)，n_keys/value 形状由其推导）；随机部分消费
    # 全局 RNG——评测器在 set_seed 后调用本函数，多轮 correctness trial 因此
    # 获得输入多样性。消费顺序固定：spatial_shapes -> value ->
    # sampling_locations -> attention_weights
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
    return [value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights]


def make_inputs(n: int, n_heads: int, embed_dims_per_head: int, n_queries: int,
                n_levels: int, n_points: int, spatial_shapes=None,
                max_size: int = 15, dtype: str = "float32", seed: int = 0):
    """按公开/隐藏/性能案例描述生成输入。仅在评测端运行（CPU 生成，评测器
    搬运到设备）。本题无构造超参，case 字段全部是这里的形参。

    消费顺序固定：spatial_shapes（未显式给出时逐元素 ~ randint[1, max_size]，
    缺省 max_size=15）-> value -> sampling_locations -> attention_weights
    （rand 张量后 cast 到 dtype），全部来自同一 torch.Generator(seed)，同
    seed 下逐位可复现；level_start_index 由 spatial_shapes 推导（前缀和），
    不消费 RNG。
    返回与 forward 形参同序的
    (value, spatial_shapes, level_start_index, sampling_locations,
    attention_weights)。
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
    return (value, ss, level_start_index, sampling_locations, attention_weights)
