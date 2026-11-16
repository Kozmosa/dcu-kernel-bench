# aiter_impl.py — 2004_fused_qk_concat 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 契约：run(inputs, init_kwargs: dict, device) -> (out, ctx)，按名取参。
#
# 来源（准入记录 benchmark/sources/2004_fused_qk_concat.yaml）：
#   repo   : OpenDAS/aiter
#   commit : c39fff8c77df4e80617649e92fa3c2615f2c43d1
#   file   : aiter/ops/triton/fused_qk_concat.py
#   sha256 : 54d1f8c7152e28ed82ada497d3810c345dde4b7e1f517de41d9910fc57562a9e
#   test   : op_tests/triton_tests/test_fused_qk_concat.py
#            （sha256 73a86db3612516639cddddd2c7498ec8a05f766c18b7b9cfacb161000cc57810，
#             第 66 / 106 行给出两条入口的官方调用姿势）
#
# 本文件里出现的 helper（_unit_cat / _qk_cat_kernel / _unit_rope_cat /
# _qk_rope_cat_kernel）都是 device kernel，不是 host 入口；真正的公开入口是两个
# host 函数，源文件同文件内定义：
#
#   fused_qk_cat(q1, q2, k1, k2)                                  # 行 116（纯拼接）
#       -> (q_out [B, QH, D1+D2], k_out [B, KH, D1+D2])
#   fused_qk_rope_cat(q_nope, q_pe, k_nope, k_pe, pos, cos, sin,  # 行 346（旋转+拼接）
#                     is_neox)
#       -> (q_out [B, QH, D_nope+D_pe], k_out [B, KH, D_nope+D_pe])
#
# 布局与语义对齐（题面 reference.py::Model.forward 为准）：
#   * 两条入口都吃 BHD 布局 [B, H, D]（无 sbhd/permute 需求），与题面 make_inputs
#     逐维一致：q_nope/q_pe/k_nope/k_pe = [B, QH|KH, D_nope|D_pe]，cos/sin 为
#     连续 [max_pos, d_freq] 二维表（官方测试用 4 维表，kernel 只按
#     cos.stride(0)/cos.stride(-1) 行式索引，二维表等价），pos 为 [B]。
#   * head 维打包：reference 返回单张量 out [B, QH+KH, D_nope+D_pe]，前 QH 行是 Q
#     侧、后 KH 行是 K 侧。aiter 返回 (q_out, k_out) 二元组，故在 host 侧按 head 维
#     拼接 cat((q_out, k_out), dim=1)——纯打包，不改数值（与 1017 的 lse 拼列同理）。
#   * 去重写回：kernel 内 `if pid_hq % QH_PER_KH == 0`（行 94 / 319）保证每个 KV head
#     只写一次，与 reference 的 "K 侧去重" 语义一致；QH_PER_KH=qh//kh 由入口自算。
#   * cos/sin 表宽两档：d_freq == D_pe 走 REUSE_FREQS_FRONT_PART=False（行 388），
#     d_freq == D_pe//2 走 True，NEOX 取 [c, c]、GPTJ 取 d//2（行 273-287），与
#     reference 的 cat(c,c) / repeat_interleave(2) 扩展逐位同义。
#   * 旋转辅助复用 rope.py 的 _get_neox_rotated_x_1D / _get_gptj_rotated_x_1D
#     （行 212-218），与 reference 的 rotate_half（NEOX 前后半交换取负 / GPTJ 相邻
#     对交换取负）同义。
#
# 已知口径差异（不影响本适配器的构造，仅记录）：
#   * kernel 的 mul-add 在输入 dtype（bf16）下做，reference 提升 fp32 后 cast 回；
#     差值被 task.yaml bf16 容差（atol=rtol=2e-2）覆盖。官方测试同样只对 rope
#     变体开 bf16（test_fused_qk_concat.py:80-81 TODO 注明 fp16 会退化）。
#   * kernel 的 BLOCK_D_nope/BLOCK_D_pe 直接取维度值并喂给 tl.arange（行 270-271、
#     421-423），Triton 要求 arange 长度为 2 的幂，且 load 无掩码。故本适配器对
#     非 2 次幂维度直接 raise（见下），绝不静默算错——本题 perf case 的
#     D_nope=512 / D_pe=64|128 均满足。
#   * 无 @triton.autotune、无 AITER_TRITON_CONFIGS_PATH / config JSON 依赖
#     （rope.py 亦无），无需外部 autotune 配置。
#
# 调用约定来自官方测试（test_fused_qk_concat.py）：
#   q_triton, k_triton = fused_qk_cat(q_nope, q_pe, k_nope, k_pe)
#   q_triton, k_triton = fused_qk_rope_cat(q_nope, q_pe, k_nope, k_pe,
#                                          pos, cos, sin, is_neox)

import torch


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现。

    inputs     : [q_nope, q_pe, k_nope, k_pe, pos, cos, sin]
                 q_nope [B, QH, D_nope] / q_pe [B, QH, D_pe] /
                 k_nope [B, KH, D_nope] / k_pe [B, KH, D_pe] /
                 pos [B] int64 / cos, sin [max_pos, d_freq]
    init_kwargs: {"is_neox": bool, "apply_rope": bool}

    返回 (out, ctx)；out 为 [B, QH+KH, D_nope+D_pe]，dtype 与输入一致。
    """
    # 入口一律函数内 import：aiter 顶层 import 很重，真机部署走最小导入垫片
    from aiter.ops.triton.fused_qk_concat import fused_qk_cat, fused_qk_rope_cat

    q_nope, q_pe, k_nope, k_pe, pos, cos, sin = inputs

    is_neox = bool(init_kwargs.get("is_neox", True))       # 默认同 Model.__init__
    apply_rope = bool(init_kwargs.get("apply_rope", True))

    B, QH, D_nope = q_nope.shape
    KH = k_nope.shape[1]
    D_pe = q_pe.shape[-1]
    d_freq = cos.shape[-1]

    # —— 合法性校验：越界就 raise，绝不静默用错参数 ——
    if QH % KH != 0:
        raise ValueError(f"QH({QH}) 必须是 KH({KH}) 的整数倍（GQA），aiter 入口亦如此断言")
    if (QH, D_nope) != (q_pe.shape[1], k_nope.shape[2]):
        raise ValueError(
            f"q/k 的 head 数与 D_nope 必须各自一致：q_pe={tuple(q_pe.shape)}, "
            f"k_nope={tuple(k_nope.shape)}"
        )
    if D_pe != k_pe.shape[2]:
        raise ValueError(f"q_pe 与 k_pe 的 D_pe 必须一致：{D_pe} vs {k_pe.shape[2]}")
    if D_pe % 2 != 0:
        raise ValueError(f"D_pe({D_pe}) 必须为偶数（半宽旋转）")
    if d_freq not in (D_pe, D_pe // 2):
        raise ValueError(f"cos/sin 宽度 d_freq({d_freq}) 必须是 D_pe({D_pe}) 或 D_pe//2")
    # kernel 把维度值直接当 BLOCK 喂给 tl.arange 且 load 无掩码：rope 路径见
    # _qk_rope_cat_kernel 行 270-271 / 421-423，纯拼接路径见 _qk_cat_kernel 行 71-72
    # / 169-170（BLOCK_D1=d1、BLOCK_D2=d2）。两条路径都要求 2 的幂。
    for name, dim in (("D_nope", D_nope), ("D_pe", D_pe)):
        if dim & (dim - 1) != 0:
            raise ValueError(
                f"{name}({dim}) 不是 2 的幂：aiter 的 _qk_rope_cat_kernel / "
                f"_qk_cat_kernel 用 tl.arange(0, {name}) 直接覆盖该维（无掩码），"
                "无法表达非 2 次幂宽度的旋转/拼接，超出本题 aiter 基线范围"
            )
    if q_nope.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"aiter 入口要求 fp16/bf16，收到 {q_nope.dtype}")

    # pos 是位置索引（整型语义）：reference 里被 cast 成浮点后无损恢复为 long，
    # 此处同样无条件归一到 int64（aiter 官方测试也传 int64）
    pos = pos.to(torch.int64)

    if apply_rope:
        q_out, k_out = fused_qk_rope_cat(
            q_nope, q_pe, k_nope, k_pe, pos, cos, sin, is_neox
        )
        path = "fused_qk_rope_cat:" + ("neox" if is_neox else "gptj") + (
            "_reuse_front" if d_freq == D_pe // 2 else "_full"
        )
    else:
        # 纯拼接：pos/cos/sin 被忽略（与 reference 的 apply_rope=False 一致）
        q_out, k_out = fused_qk_cat(q_nope, q_pe, k_nope, k_pe)
        path = "fused_qk_cat:concat_only"

    # head 维打包成 reference 的单张量契约：前 QH 行 Q 结果、后 KH 行 K 结果
    out = torch.cat((q_out, k_out), dim=1)

    torch.cuda.synchronize()
    return out, {
        "path": path,
        "aiter_entry": "aiter.ops.triton.fused_qk_concat."
        + ("fused_qk_rope_cat" if apply_rope else "fused_qk_cat"),
        "shapes": {
            "B": int(B), "QH": int(QH), "KH": int(KH),
            "D_nope": int(D_nope), "D_pe": int(D_pe), "d_freq": int(d_freq),
            "out": tuple(out.shape),
        },
    }
