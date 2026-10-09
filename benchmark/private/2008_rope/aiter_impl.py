# aiter_impl.py — 2008_rope（稠密 sbhd 布局 RoPE 前向）的 aiter 官方实现适配器
#
# 来源（与 benchmark/sources/2008_rope.yaml 的准入记录一致）：
#   repo   OpenDAS/aiter
#   commit c39fff8c77df4e80617649e92fa3c2615f2c43d1
#   file   aiter/ops/triton/rope.py
#          sha256 34bcb120bcbb49bb49294a673fddea4ac03e05ca98021ad843d0a13a655c618a
#   test   op_tests/triton_tests/test_rope.py（test_rope_sbhd_fwd，:205-267）
#          op_tests/test_rope.py（ref_rope_sbhd_fwd，:479 —— 语义对照）
#
# 公开算子入口（host 侧，非 helper）：
#   aiter.ops.triton.rope.rope_fwd(x, freqs, rotate_style,
#                                  reuse_freqs_front_part, nope_first,
#                                  transpose_output=False) -> torch.Tensor
#   —— aiter/ops/triton/rope.py:2006；内部 _rope_fwd(:1948) 启动
#   _rope_kernel_sbhd_fwd(:99)，grid=(b, h, cdiv(s, 32))，BLOCK_S=32。
#
# 布局：**无需任何 layout 转换**。本评测集 make_inputs 产出的
#   x (S, B, H, D) / freqs (S, 1, 1, F) 正是 aiter 期望的 sbhd 非缓存布局，
#   与测试 generate_rope_inputs(layout="sbhd") 一致；rope_fwd 只读 x.shape，
#   输出 torch.empty((s, b, h, d), dtype=x.dtype, device=x.device)。
#   与 reference.py::Model.forward 的返回形状/dtype（(S,B,H,D)、x.dtype）逐个
#   对齐，是单张量契约，无需打包或拼接。
#
# 配置映射（reference.py::Model.__init__ → aiter rope_fwd）：
#   rotate_style            0=NEOX / 1=GPTJ → 同名参数（aiter IS_NEOX 比较
#                           等于 RotateStyle.NEOX，即 0；已验证 0 == NEOX）
#   nope_first              → 同名参数（真值段在前，旋转段在后）
#   reuse_freqs_front_part  → 同名参数（NEOX 整段平铺 / GPTJ 相邻重复）
#   rotate_dim R = F * (2 if reuse else 1) 由 freqs 末维隐式决定，与 reference
#   的 rotate_dim 定义完全相同。
#
# 已逐条静态核对（无 GPU，未运行验证）：
#   * aiter _rope_fwd(:1960-1975) 由 freqs.shape[-1] 反推 have_nope：
#       F == D/2 且 reuse      → have_nope=False（R=D，全维旋转）
#       F == D/2 且 not reuse  → have_nope=True （R=D/2，直通段 D/2）
#       F == D/4               → have_nope=True （R=D/2，reuse 的 F=D/4）
#       否则（F == D）         → have_nope=False（R=D）
#     仅覆盖 R ∈ {D, D/2}；R=D/4（rotary_percent=0.25, reuse=False）会被
#     aiter 静默当成 R=D/2 算错，故本适配器显式 raise（见 validate 段）。
#   * 角度索引与 reference 的角度扩展一致：reuse=False 取 freqs[..., d]；
#     reuse + NEOX 取 d_freqs_offs = d - D/2（后半段），即 [f, f] 平铺；
#     reuse + GPTJ 取 d//2，即相邻重复——分别等价于 reference 的
#     f32.repeat(...,2) 与 f32.repeat_interleave(2, dim=-1)。
#   * rotate_half 等价：_get_neox_rotated_x(:46) = cat(-x[R/2:], x[:R/2])；
#     _get_gptj_rotated_x(:73) = 偶奇配对 [-x2j+1, x2j]——与 reference 的
#     NEOX（torch.cat((-x2, x1))）与 GPTJ（stack+flatten）逐位同构。
#   * 直通段：HAVE_NOPE & NOPE_FIRST 时 nope_offs = BLOCK_D，旋转段写在
#     [R, D)、直通段原样拷回 [0, R)；NOPE_FIRST=False 时相反——与 reference 的
#     start = D - R if nope_first else 0 及 cat 顺序一致。
#   * 数值：kernel 内 freqs 先 to(tl.float32) 再 cos/sin，x 以原 dtype 载入后
#     与 fp32 三角值相乘（提升到 fp32），最后 .to(x_ptr.dtype.element_ty)；
#     与 reference「全程 float32 求值、末尾 cast 回输入 dtype」同一路径。
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。

import torch


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（sbhd 非缓存 RoPE 前向）。

    inputs     : [x (S, B, H, D), freqs (S, 1, 1, F)]，与 x 同 dtype，已在 device 上
    init_kwargs: {"rotate_style": 0/1, "nope_first": bool,
                  "reuse_freqs_front_part": bool}

    返回 (out, ctx)；out 为 (S, B, H, D)、dtype 与 x 一致的单个张量。
    """
    # aiter 顶层 import 很重：只在函数内做最小导入
    from aiter.ops.triton.rope import rope_fwd

    if len(inputs) < 2:
        raise ValueError(f"2008_rope 需要 (x, freqs) 两个输入，收到 {len(inputs)} 个")
    x, freqs = inputs[0], inputs[1]

    # init_kwargs 按名取值；case 未给出的超参用 get_init_inputs() 的默认值
    rotate_style = int(init_kwargs.get("rotate_style", 0))
    nope_first = bool(init_kwargs.get("nope_first", False))
    reuse_freqs_front_part = bool(init_kwargs.get("reuse_freqs_front_part", False))

    if rotate_style not in (0, 1):
        raise ValueError(f"rotate_style 只支持 0(NEOX)/1(GPTJ)，收到 {rotate_style}")
    if x.dim() != 4 or freqs.dim() != 4:
        raise ValueError(f"本题只覆盖 sbhd 4D 布局，收到 x{x.shape} freqs{freqs.shape}")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"不支持的 x.dtype: {x.dtype}")
    if freqs.dtype != x.dtype:
        raise ValueError(f"freqs.dtype({freqs.dtype}) 必须与 x.dtype({x.dtype}) 一致")

    s, b, h, d = (int(v) for v in x.shape)
    if freqs.shape[0] != s or tuple(freqs.shape[1:3]) != (1, 1):
        raise ValueError(
            f"freqs 必须是 (S, 1, 1, F) 且与 x 同 S；收到 freqs{freqs.shape} x{x.shape}"
        )
    # aiter 的 BLOCK_D/BLOCK_D_HALF 假设 D 为 2 的幂且 D/4 >= 1
    if d < 4 or (d & (d - 1)) != 0:
        raise ValueError(f"D 必须是 >= 4 的 2 的幂（aiter BLOCK_D 假设），收到 D={d}")

    f_cols = int(freqs.shape[-1])
    rotate_dim = f_cols * (2 if reuse_freqs_front_part else 1)

    # aiter _rope_fwd(rope.py:1960) 只按 freqs.shape[-1] 反推 have_nope，其表达力
    # 恰好是 R ∈ {D, D/2}；R=D/4（reuse=False）会被静默算成 R=D/2，必须拒绝。
    if rotate_dim not in (d, d // 2):
        raise ValueError(
            f"aiter rope_fwd 无法表达该配置：F={f_cols}, reuse={reuse_freqs_front_part} "
            f"→ R={rotate_dim}，而 aiter 只支持 R ∈ {{D, D/2}}（D={d}）"
        )

    # 用 aiter 自己的分派规则反推 have_nope，与本配置的 R 交叉校验，避免静默用错 BLOCK_D
    if f_cols == d // 2:
        have_nope = not reuse_freqs_front_part
    elif f_cols == d // 4:
        have_nope = True
    else:
        have_nope = False
    if have_nope != (rotate_dim != d):
        raise ValueError(
            f"aiter 分派与题面配置不一致：F={f_cols}, D={d}, reuse={reuse_freqs_front_part} "
            f"→ aiter have_nope={have_nope}，题面 R={rotate_dim}"
        )

    # 布局已经是 aiter 期望的 sbhd，contiguous 只是幂等保险（非 contiguous 时
    # kernel 亦按 stride 寻址，但输出用 empty 分配，保守起见先规整）
    x_c = x.contiguous()
    freqs_c = freqs.contiguous()

    out = rope_fwd(
        x_c,
        freqs_c,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
        transpose_output=False,
    )
    torch.cuda.synchronize()

    if out.shape != x.shape or out.dtype != x.dtype:
        raise RuntimeError(
            f"aiter rope_fwd 输出 {tuple(out.shape)}/{out.dtype} 与期望 "
            f"{tuple(x.shape)}/{x.dtype} 不一致"
        )

    ctx = {
        "path": "aiter.ops.triton.rope.rope_fwd/_rope_kernel_sbhd_fwd",
        "layout": "sbhd",
        "shape": [s, b, h, d],
        "freqs_shape": [int(v) for v in freqs.shape],
        "rotate_dim": rotate_dim,
        "have_nope": have_nope,
        "rotate_style": rotate_style,
        "nope_first": nope_first,
        "reuse_freqs_front_part": reuse_freqs_front_part,
        "block_s": 32,
        "dtype": str(x.dtype),
    }
    return out, ctx
