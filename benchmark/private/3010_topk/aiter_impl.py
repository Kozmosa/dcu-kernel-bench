# aiter_impl.py — 3010_topk 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) —— 按名取参（构造参数稀疏
# 给出时位置式会错位，见 audit_model_class.py::case_init_kwargs 的说明）。
#
# 来源（aiter pinned 检出 .dcu_runs/aiter_pinned/，commit
# c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#   device_kernel : aiter/ops/triton/topk.py
#                   sha256 0f27b4555b076c964c519a8c72ed23436357500bb1f1dc9ba251247f41fe88e5（已核对）
#     - _topk_kernel             topk.py:21-55    1-stage（短行，逐轮取最大 + 其最小列下标）
#     - one_stage_topk           topk.py:65-90    1-stage 宿主入口（M <= 1024）
#     - topk_stage1_kernel       topk.py:95-146   2-stage 第一段（行内分块局部 top-k + 全局列下标）
#     - _compare_and_swap/_bitonic_merge/argsort
#                                topk.py:149-255  双调排序（值 + 下标同步交换）
#     - topk_stage2_kernel       topk.py:259-296   2-stage 第二段（候选归并排序取前 k）
#     - two_stage_topk           topk.py:299-352   2-stage 宿主入口（M > 1024）
#   dispatcher    : aiter/ops/triton/topk.py:389-417（题目 io 说的"源算子 dispatcher topk()"）
#   official_test : op_tests/triton_tests/test_topk.py
#                   sha256 cfaca5ac882d449d5ad53cc391bb3bbddf1c49b3424b93e968341e09f08cef5e（已核对）
#
# 入口签名（topk.py:389-397；官方测试 test_topk.py:73 的调用口径
# `triton_topk(x, k, largest=largest)`）：
#
#   topk(x: torch.Tensor, k: int, *, dim: int = -1, largest: bool = True,
#        sorted: bool = True, tiny_row_thresh: int = MAX_TINY_ROW) -> (values, indices)
#
#   x       [B, M] 2-D（host 侧强制 contiguous，topk.py:409-410）
#   values  [B, k]，dtype == x.dtype（纯选择，逐位等于输入元素）
#   indices [B, k]，dtype == torch.int64（该行内的**原始**列下标，topk.py:132
#           stage1 加 chunk_offset、topk.py:50 1-stage 直接取 offs）
#   dim != -1 / 非 2-D / largest=False / sorted=False 一律 ValueError
#           （topk.py:398-407），本题面域（last-dim / largest / sorted）全在支持域内。
#
# 路径分派（topk.py:412-417，tiny_row_thresh = MAX_TINY_ROW = 1024，topk.py:356）：
#   M <= 1024 → one_stage_topk（单 kernel，BLOCK = _pick_block(M, k)，topk.py:58-62）
#   M >  1024 → two_stage_topk（chunk_size = 256/1024，chunk_size < k 时提升到
#               next_power_of_2(k)；chunk_num = cdiv(M, chunk_size)，topk.py:307-315）
# 与 reference.py 的【实现路径提示】两条路径阈值（M <= 1024 短行 / M > 1024 长行）
# 完全一致，故本题不需要绕开 dispatcher，直接调用它是权威口径。
#
# 布局说明（**无需任何 permute**）：题目 io 的 x 就是 aiter 要求的 [B, M] 行主
# contiguous 形态；reference 也是先在 fp32 上对最后一维做 topk（reference.py:70-71），
# aiter topk 亦只支持 last-dim。适配器只做两处 layout/dtype 归一：
#   ① `x.to(torch.float32).contiguous()`——与 reference.py:70 逐句同构。reference
#      恒先把输入 cast 成 fp32 再选择；aiter 输出 dtype 跟随输入 dtype
#      （topk.py:74、317），因此把输入统一喂 fp32，既与 reference 的值逐位一致，
#      又让 fp16/bf16 隐藏 case 也走官方测试**唯一启用**的 fp32 路径
#      （test_topk.py:8 `FLOAT_DTYPES = [torch.float32]`，fp16/bf16 档被注释掉
#      ——低精度分支在源库里无官方测试覆盖）。
#   ② 输出打包 `cat([values, indices.to(torch.float32)], dim=-1)`——与
#      reference.py:73 的 [values | indices] 单张量契约逐句同构，得 (B, 2k) fp32。
#      列下标 < 2^24（题目不变式）在 fp32 中精确表示，故打包零信息损失。
#
# ⚠️ 逐位精度前提：values 是输入元素的纯选择（无算术），indices 是精确整数，
# 故 aiter 输出应与 reference 在所有三个 dtype 档下 atol=rtol=0 逐位相等
# （task.yaml tolerance 正是 0/0）。aiter 内部一律以 fp32 比较
# （topk.py:37/120/287 的 `.to(tl.float32)`），与 reference 的 fp32 语义一致。
#
# autotune config：topk.py **没有** @triton.autotune，也不读
# AITER_TRITON_CONFIGS_PATH（仅 num_warps=4 / num_stages=2 硬编码，topk.py:87-88），
# 故 needs_autotune_config=False，无外部 JSON 依赖。
#
# 本文件只做 layout/dtype 归一与打包，核心 Top-K 选择全部由 aiter 官方 kernel 完成。

import torch

# 与 aiter/ops/triton/topk.py 的 MAX_TINY_ROW 保持一致（仅用于 ctx 记录，
# 真正的分派由被调用的 dispatcher 自己做）
_MAX_TINY_ROW = 1024
# 题目不变式：列下标在 fp32 打包中必须精确（M < 2^24）
_MAX_ROW_LEN = 1 << 24


def _pick_block(m, k):
    """复刻 topk.py:58-62 的 _pick_block，仅用于 ctx 记录 1-stage 的 BLOCK。"""
    blk = max(128, k)
    while blk < m and blk < 1024:
        blk <<= 1
    return blk


def _next_pow2(n):
    return 1 << max(0, int(n) - 1).bit_length() if n > 1 else 1


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（aiter.ops.triton.topk.topk）。

    inputs     : [x]（顺序同 reference.make_inputs 的返回）
                 x [B, M]，contiguous，行内元素互不相等；fp32/fp16/bf16
    init_kwargs: {"k": int}（Model.__init__(k=8) 的参数，io.init_inputs 声明的名字）
    device     : 目标设备（输入已在 device 上，仅用于 ctx 记录）

    返回 (out, ctx)；out 为 (B, 2*k) float32 打包张量
    （前 k 列 = 降序 values，后 k 列 = 原始列下标），与 reference.forward 同形同 dtype。
    """
    # aiter 顶层 import 很重，一律函数内 import
    from aiter.ops.triton.topk import topk as aiter_topk

    # ---- 输入解包 ----------------------------------------------------------
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 1:
        raise ValueError(f"3010_topk 的 make_inputs 只返回 [x]，实际 {type(inputs)}"
                         f"（len={len(inputs) if hasattr(inputs, '__len__') else '?'}）")
    x = inputs[0]
    if not torch.is_tensor(x):
        raise ValueError(f"inputs[0] 必须是张量，实际 {type(x)}")

    # ---- 构造参数（按名取参；缺/越界一律 raise，绝不静默用错）---------------
    k = init_kwargs.get("k", 8)
    if k is None:
        raise ValueError("init_kwargs 缺少 k（Model.__init__(k=8) 的构造超参）")
    k = int(k)

    # ---- shape / dtype 合法性（题目全域约束，违反即 raise）-----------------
    if x.dim() != 2:
        raise ValueError(f"x 必须是 2 维 (B, M)，实际 {tuple(x.shape)}")
    batch, row_len = int(x.shape[0]), int(x.shape[1])
    if batch < 1 or row_len < 1:
        raise ValueError(f"B/M 必须 >= 1，实际 (B={batch}, M={row_len})")
    if not (1 <= k <= row_len):
        raise ValueError(f"k={k} 越界（题目不变式 1 <= k <= M={row_len}）")
    if row_len >= _MAX_ROW_LEN:
        raise ValueError(
            f"M={row_len} 越界（题目不变式 M < 2^24：列下标须在 fp32 打包中精确）"
        )
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(
            f"x.dtype={x.dtype} 不受支持（题目 io 声明 float32/float16/bfloat16；"
            "aiter topk 为浮点选择，整型域不在题面）"
        )

    # ---- layout / dtype 归一（与 reference.py:70 同构；无需 permute）--------
    x32 = x.to(torch.float32).contiguous()

    # ---- 官方入口调用（dispatcher 内部按 M 选 1-stage / 2-stage）------------
    values, indices = aiter_topk(x32, k, dim=-1, largest=True, sorted=True)

    # ---- 输出形状核对（打包前先确认 aiter 给的是 (B, k)）--------------------
    if tuple(values.shape) != (batch, k) or tuple(indices.shape) != (batch, k):
        raise RuntimeError(
            f"aiter topk 返回形状异常：values{tuple(values.shape)} "
            f"indices{tuple(indices.shape)}，期望 ({batch}, {k})"
        )
    if values.dtype != torch.float32:
        raise RuntimeError(f"aiter topk values dtype={values.dtype}，期望 float32")
    if indices.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(f"aiter topk indices dtype={indices.dtype}，期望整型")

    torch.cuda.synchronize()

    # ---- 打包（与 reference.py:72-73 逐句同构）：[values | indices] -> (B, 2k)
    out = torch.cat([values, indices.to(torch.float32)], dim=-1)

    # ---- ctx：走了哪条 aiter 路径与关键 shape（按 topk.py:307-315 复算）-----
    use_one_stage = row_len <= _MAX_TINY_ROW
    if use_one_stage:
        chunk_size, chunk_num, stage1_cnt, stage2_block = None, None, None, None
    else:
        chunk_size = 256 if row_len < 1024 else 1024
        if chunk_size < k:
            chunk_size = _next_pow2(k)
        chunk_num = (row_len + chunk_size - 1) // chunk_size
        stage1_cnt = chunk_num * k
        stage2_block = _next_pow2(stage1_cnt)

    ctx = {
        "impl": "aiter",
        "module": "aiter.ops.triton.topk",
        "func": "topk",
        "path": "one_stage_topk" if use_one_stage else "two_stage_topk",
        "batch": batch,
        "row_len": row_len,
        "k": k,
        "block_1stage": _pick_block(row_len, k) if use_one_stage else None,
        "chunk_size": chunk_size,
        "chunk_num": chunk_num,
        "stage2_elem_cnt": stage1_cnt,
        "stage2_block": stage2_block,
        "input_dtype": str(x.dtype),
        "compute_dtype": "torch.float32",
        "out_dtype": str(out.dtype),
        "out_shape": tuple(out.shape),
        "device": str(device),
    }
    return out, ctx
