# 2008_rope —— 稠密 sbhd 布局 RoPE（旋转位置编码）前向的 model_class
# （KernelBench 兼容）题目文件。
#
# 本文件必须自包含：KernelBench 评测器以 exec(source) 加载题目，禁止本地模块
# 导入，reference 实现与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=2, problem_id=2008, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """RoPE（Rotary Position Embedding，旋转位置编码）前向：稠密 sbhd 布局，
    旋转角度以 freqs 张量直接给出（非缓存 cos/sin 表），按序列位置对每个
    head 向量的通道做平面旋转。

    语义定义（三角函数与乘加全部以 float32 求值，输出 cast 回输入 dtype）：
      输入 x (S, B, H, D) 与 freqs (S, 1, 1, F)，freqs 在 B、H 维上广播。
      旋转宽度 R = F * 2（reuse_freqs_front_part=True）或 R = F（False）。
      将最后一维（长 D）拆为「旋转段」（长 R）与「直通段」（长 D - R）：
      nope_first=False 时旋转段在前 D 个通道，True 时在后 R 个通道。
      角度行 Θ（长 R）：reuse=False 时 Θ = freqs，每个旋转通道一个角度；
      reuse=True 时由 F = R/2 个角度扩展成 R 列——NEOX 风格按整段平铺
      （[a, b] -> [a, b, a, b]，前后半各用一遍），GPTJ 风格按相邻重复
      （[a, b] -> [a, a, b, b]，每个数对共用一个角度）。
      对旋转段 x_rot（长 R）计算：
        out_rot = x_rot * cos(Θ) + rotate_half(x_rot) * sin(Θ)
      rotate_half 由 __init__ 的 rotate_style 决定：
        NEOX（rotate_style=0）：rotate_half(x_rot) = concat(-x_rot[R/2:],
          x_rot[:R/2])，即前半 u 与后半 v 互换、u 段取负，等价于每对
          (u_j, v_j) 做平面旋转。
        GPTJ（rotate_style=1）：偶奇通道配对，rotate_half(x_rot)[2j] =
          -x_rot[2j+1]、rotate_half(x_rot)[2j+1] = x_rot[2j]。
      直通段不经任何变换接回：nope_first=False 时 out = concat(out_rot,
      直通段)；True 时 out = concat(直通段, out_rot)。
      边界行为：R = D 时直通段为空（全维旋转）；R < D 时直通段逐位原样
      保留；freqs 末维 F 只有 {D, D/2, D/4} 三种合法取值（与 reuse、R
      的组合见下），其余配置未定义。

    输入输出规格：
        x      (S, B, H, D)  float16 / bfloat16，contiguous；D 为 2 的幂
                             （4..256），S、B、H 为任意正整数
        freqs  (S, 1, 1, F)  与 x 同 dtype，contiguous；合法组合：
                             全维旋转 reuse=False：F = D；
                             全维旋转 reuse=True：F = D/2；
                             部分旋转（R=D/2）reuse=False：F = D/2；
                             部分旋转（R=D/2）reuse=True：F = D/4
        输出 out (S, B, H, D)，dtype 与 x 一致
        （评测器会把输入 cast 成 fp32 传入，fp16/bf16 -> fp32 无损）
        超参（__init__）：rotate_style（0=NEOX，1=GPTJ）、nope_first、
        reuse_freqs_front_part。

    实现约束（违规判负）：
        - 核心计算（角度扩展、rotate_half、cos/sin 与乘加、直通段拼接）
          必须在提交文件内以 Triton kernel 完成。
        - ModelNew 的 __init__ 与 forward 签名不可更改。
        - 中间计算用 float32；输出 cast 回输入 dtype。

    禁用列表：scaled_dot_product_attention / sdpa / flash_attn /
    torch.matmul / torch.bmm / torch.einsum / torch.softmax / torch.polar /
    torch.view_as_complex。
    目标硬件：海光 DCU（gfx936），实现语言 Triton。
    本题范围：稠密 sbhd 非缓存前向；变长 thd、cos/sin 缓存寻址、GQA 双
    输入、反向、二维图像旋转不在本题内。
    """

    def __init__(self, rotate_style: int, nope_first: bool, reuse_freqs_front_part: bool):
        super().__init__()
        self.rotate_style = int(rotate_style)
        self.nope_first = bool(nope_first)
        self.reuse_freqs_front_part = bool(reuse_freqs_front_part)

    def forward(self, x, freqs):
        # 评测器把全部输入 cast 成 fp32；fp16/bf16 -> fp32 无损，统一在
        # float32 计算，输出 cast 回入口 dtype（在线为 fp32、离线为原 dtype）。
        # 注：forward 内不用带 else 分支的语句，保证 loader 逐语句抽取
        # scaffold 后仍是合法 Python。
        dt = x.dtype
        x32 = x.to(torch.float32)
        f32 = freqs.to(torch.float32)

        d = x32.shape[-1]
        rotate_dim = f32.shape[-1] * (2 if self.reuse_freqs_front_part else 1)

        # ① 旋转段 / 直通段切分：nope_first=True 时旋转段在尾部 [d-R, d)，
        #    否则在前 [0, R)；直通段保持原顺序拼接（rotate_dim == D 时为空）
        start = (d - rotate_dim) if self.nope_first else 0
        x_rot = x32[..., start : start + rotate_dim]
        x_pass = torch.cat((x32[..., :start], x32[..., start + rotate_dim :]), dim=-1)

        # ② 角度行：reuse 模式把 F=R/2 列扩展成 R 列（NEOX 平铺 / GPTJ 相邻重复）
        theta = f32
        if self.reuse_freqs_front_part and self.rotate_style == 0:
            theta = f32.repeat([1] * (f32.dim() - 1) + [2])
        if self.reuse_freqs_front_part and self.rotate_style == 1:
            theta = f32.repeat_interleave(2, dim=-1)

        # ③ rotate_half + 平面旋转，全程 float32
        half = rotate_dim // 2
        rot_half = x_rot
        if self.rotate_style == 0:  # NEOX：前半 u / 后半 v 互换，u 段取负
            x1 = x_rot[..., :half]
            x2 = x_rot[..., half:]
            rot_half = torch.cat((-x2, x1), dim=-1)
        if self.rotate_style == 1:  # GPTJ：偶奇配对，偶位取负的奇元、奇位放偶元
            x1 = x_rot[..., ::2]
            x2 = x_rot[..., 1::2]
            rot_half = torch.stack((-x2, x1), dim=-1).flatten(-2)
        x_embed = x_rot * torch.cos(theta) + rot_half * torch.sin(theta)

        # ④ 直通段接回（nope_first 时直通段在前）后整体 cast 回输入 dtype
        pieces = (x_pass, x_embed) if self.nope_first else (x_embed, x_pass)
        out = torch.cat(pieces, dim=-1)
        return out.to(dt)


def get_init_inputs():
    return [0, False, False]  # rotate_style=NEOX, nope_first=False, reuse=False


def get_inputs():
    # 固定 shape 族；随机部分消费全局 RNG——评测器在 set_seed 后调用本函数，
    # 多轮 correctness trial 因此获得输入多样性。消费顺序：x / freqs。
    # 配置须与 get_init_inputs 一致：NEOX、直通段在后、reuse=False → 全维
    # 旋转，freqs 末维 F = D = 128
    s, b, h, d = 64, 2, 8, 128
    x = torch.randn(s, b, h, d).to(torch.float16)
    freqs = torch.randn(s, 1, 1, d).to(torch.float16)
    return [x, freqs]


def make_inputs(s: int, b: int, h: int, d: int, rotate_style: int, nope_first: bool,
                reuse_freqs_front_part: bool, rotary_percent: float = 1.0,
                dtype: str = "float16", seed: int = 0):
    """按公开/隐藏案例描述生成输入。仅在评测端运行（CPU 生成，评测器搬运到设备）。

    rotary_percent 为旋转宽度占 D 的比例（1.0 全维 / 0.5 部分旋转），与
    reuse_freqs_front_part 共同决定 freqs 末维 F = int(d * rotary_percent) //
    (2 if reuse else 1)；rotate_style / nope_first 不影响张量形状，仅供
    构造 Model 时使用（与 case 字段一一对应）。消费顺序固定：x（randn）、
    freqs（randn），全部来自同一 torch.Generator(seed)，同 seed 逐位可复现。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    rotate_dim = int(d * rotary_percent)
    assert d % 2 == 0 and rotate_dim % 2 == 0 and d % rotate_dim == 0, \
        "D 须为偶数且旋转宽度 R 为整除 D 的偶数"
    freqs_d = rotate_dim // (2 if reuse_freqs_front_part else 1)
    x = torch.randn(s, b, h, d, generator=gen).to(dt)
    freqs = torch.randn(s, 1, 1, freqs_d, generator=gen).to(dt)
    return (x, freqs)
