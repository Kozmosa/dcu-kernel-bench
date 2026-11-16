# aiter_impl.py — 4017_group_quant_int8 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) -> (out, ctx)
#   inputs      = [x]，x 已在 device 上（perf/hidden case 的 make_inputs 产物）
#   init_kwargs = {"group_size": int, "eps": float}（Model.__init__ 的那一套）
#   out         = 与 reference 同形同 dtype 的单张量（见下「输出打包」）
#
# ── aiter 来源 ────────────────────────────────────────────────────────────────
# repo   : OpenDAS/aiter
# commit : c39fff8c77df4e80617649e92fa3c2615f2c43d1
# file   : aiter/ops/triton/group_quant_int8.py
# sha256 : 707cd632743004355f5f1a97d6c81d24154c3e6302dd87eb60dc301ffe62aef4
#          （与 benchmark/sources/4017_group_quant_int8.yaml 的准入证据一致）
#
# ── 公开 host 入口（同模块；device kernel _per_token_group_quant_int8 才是
#    准入记录里的 device_kernel，但它不是 host 入口，不直接调用）──────────────
#   per_token_group_quant_int8(x, group_size, eps=1e-10, dtype=torch.int8)
#       -> (x_q, x_s)
#   x_q : 与 x 同形、dtype=torch.int8（源文件 L161 empty_like(x, dtype)）
#   x_s : torch.float32，shape = x.shape[:-1] + (x.shape[-1] // group_size,)
#         （源文件 L162-166）
#   调用约定与官方测试 op_tests/triton_tests/test_group_quant_int8.py:77 一致
#   （该测试的 native_per_token_group_quant_int8 即本题 reference 的语义权威）。
#
# ── 布局 ─────────────────────────────────────────────────────────────────────
# 题面 x 是 2D 连续 (rows, cols)，aiter 直接吃这个布局（断言 shape[-1]%gs==0
# 且 is_contiguous），无需 permute/reshape：kernel 按 BLOCK_SIZE 连续块处理，
# 块内再 reshape 成 (S_NUM=BLOCK_SIZE/group_size, group_size)，与 reference 的
# x.reshape(-1, group_size) 分组完全一致（前提 group_size | BLOCK_SIZE，见 run()
# 里的显式校验）。
#
# ── 输出打包 ─────────────────────────────────────────────────────────────────
# 题面是 KernelBench 单张量协议：out[:rows*cols] 为码字（int8 提升为 float32），
# out[rows*cols:] 为 scale（与 x_s 行主序展平一致）。故这里把 aiter 的两个返回
# 张量按同一顺序 cat 成一维 float32（与 reference.py L74-76 逐位对齐）。

import torch


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方 per_token_group_quant_int8，并按题面协议打包输出。"""
    from aiter.ops.triton.group_quant_int8 import (
        get_w8a8_group_quant_configs,
        per_token_group_quant_int8,
    )

    (x,) = inputs
    x = x if x.is_contiguous() else x.contiguous()

    group_size = int(init_kwargs.get("group_size", 128))
    eps = float(init_kwargs.get("eps", 1e-10))

    if x.dim() != 2:
        raise ValueError(f"4017 题面只覆盖 2D 输入，收到 {x.dim()}D")
    rows, cols = int(x.shape[0]), int(x.shape[1])
    if group_size < 1 or cols % group_size != 0:
        raise ValueError(
            f"group_size={group_size} 不整除 cols={cols}（题面不变式 cols % group_size == 0）"
        )
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"aiter kernel 只接受 fp16/bf16/fp32 输入，收到 {x.dtype}")
    if not (eps > 0.0):
        raise ValueError(f"eps 必须为正（除零保护），收到 {eps}")

    # aiter 的 kernel 先把整块 load 成 (S_NUM=BLOCK_SIZE//GROUP_SIZE, GROUP_SIZE)，
    # 因此 group_size 必须整除实际选中的 BLOCK_SIZE——否则 S_NUM 截断为 0（编译期
    # reshape 报错）或尾部分组跨块、与行主序分组不一致（静默算错）。配置 JSON 里的
    # BLOCK_SIZE 全是 128 的倍数、缺配置时默认 BLOCK_SIZE=128，故「group_size 整除
    # 128」既是必要也是充分条件（也顺带保证 S_NUM 是 2 的幂，tl.arange 合法）。
    # 本题 case 的 group_size ∈ {128, 64, 1} 均满足；越界直接 raise，不静默用错参数。
    if 128 % group_size != 0:
        raise ValueError(
            f"group_size={group_size} 不是 128 的约数：aiter group_quant_int8 的 Triton "
            "kernel 要求 group_size 整除 BLOCK_SIZE（配置里最小为 128），否则分组语义错误"
        )

    M = rows * cols
    num_groups = M // group_size

    # 性能配置（可选）：aiter 命中 device 配置 JSON 时用它选 BLOCK_SIZE，缺文件则回退
    # 默认 BLOCK_SIZE=128/num_warps=1/num_stages=1（源文件 L169-177），只影响性能不
    # 影响正确性。这里做一次只读诊断：设备名映射到的 JSON 键名若与 kernel 形参不匹配
    # （例如 K100_AI 的 JSON 实为 BLOCK_SIZE_M），triton 会抛难懂的 TypeError，提前
    # 给出可行动的报错。诊断本身异常（如驱动 target 未就绪）不改变调用路径。
    config_note = "default(BLOCK_SIZE=128,num_warps=1,num_stages=1)"
    config_error = None
    try:
        configs = get_w8a8_group_quant_configs(M, group_size)
        if configs:
            cfg = configs[min(configs.keys(), key=lambda k: abs(k - M))]
            missing = [k for k in ("BLOCK_SIZE", "num_warps", "num_stages") if k not in cfg]
            if missing:
                config_error = (
                    f"aiter group_quant 配置缺键 {missing}（M={M}, group_size={group_size}, "
                    f"config={cfg}）：该 device 的 JSON 与本 kernel 形参不匹配，"
                    "需人工确认（真机 gfx936→BW200 的 JSON 键名正确）"
                )
            else:
                config_note = (
                    f"json(BLOCK_SIZE={cfg['BLOCK_SIZE']},num_warps={cfg['num_warps']})"
                )
    except Exception as exc:  # 诊断失败不影响正确性，回退默认配置
        config_note = f"default(config-probe failed: {type(exc).__name__})"
    if config_error is not None:
        raise RuntimeError(config_error)

    x_q, x_s = per_token_group_quant_int8(x, group_size, eps, torch.int8)

    # 上游已知瑕疵（仅记录，不影响返回值）：kernel 的 scale store 无 mask
    # （源文件 L83 的 tl.store(y_s_ptr + s_cols, ...)），当 M % BLOCK_SIZE != 0 时
    # grid*S_NUM 会多于 M/group_size，多出的若干 fp32 写到 x_s 分配之外。x_s 由
    # aiter 内部按精确 group 数分配（L162-166），小张量落在 512B 分配器块内，
    # 返回的 x_s/x_q 取值不受影响（本题 hidden_nonpow2_3x192_gs64 即此情形）。
    # 单张量输出协议：[码字（行主序展平，int8→fp32）| scale（行主序展平 fp32）]
    codes = x_q.reshape(-1).to(torch.float32)
    scales = x_s.reshape(-1)
    out = torch.cat([codes, scales])

    torch.cuda.synchronize()

    ctx = {
        "path": "aiter.ops.triton.group_quant_int8.per_token_group_quant_int8",
        "rows": rows,
        "cols": cols,
        "group_size": group_size,
        "eps": eps,
        "M": M,
        "num_groups": num_groups,
        "x_dtype": str(x.dtype),
        "config": config_note,
        "out_len": int(out.numel()),
        "split": {"codes": M, "scales": num_groups},
    }
    return out, ctx
