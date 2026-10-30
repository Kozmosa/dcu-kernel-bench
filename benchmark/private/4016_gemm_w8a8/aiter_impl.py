# aiter_impl.py — 4016_gemm_w8a8 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 契约：run(inputs, init_kwargs: dict, device) -> (out, ctx)，按名取参。
#
# 来源（aiter pinned 检出 commit c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#   aiter/ops/triton/gemm_w8a8.py
#     sha256 c3f708da4acec0e31db29a7daf1ef351b10c17e8d8df36ed30ada39984fb9ac4
#     公开入口（:316）：
#       def gemm_w8a8(A, B, As, Bs, block_size: list[int],
#                     output_dtype: torch.dtype = torch.float16) -> torch.Tensor
#     主计算 kernel _w8a8_block_int8_matmul（:111，@triton.heuristics + @triton.jit）：
#     tl.dot int8 主计算 + 逐 128 组 As/Bs 反量化 scale 乘，全部在 kernel 内。
#     文件内其余符号（get_w8a8_block_int8_configs）是 autotune 配置查表 helper。
#
# 官方测试调用约定（op_tests/test_gemm_a8w8_blockscale_blaslt.py:66-68，
# sha256 9f1f73bb7a0d2ddc78165c5433a6f4a7ac849e4b683603d64e8ebd2df58b810c）：
#   gemm_w8a8(x, weight, x_scale, w_scale, block_size=(128, 128), dtype)
# 其中 x (m,k) int8、weight (n,k) int8、x_scale (m, ceil(k/128)) fp32、
# w_scale (ceil(n/128), ceil(k/128)) fp32——与本题 make_inputs 的
# [aq, bq, a_scale, b_scale] **逐项同序同形同 dtype**，无需 layout 转换。
#
# 布局/语义对齐（aiter gemm_w8a8 内部做的主机侧检查，与题目 reference 一致）：
#   - A (M,K) int8 contiguous；As (M, ceil(K/128)) → assert A.shape[:-1] == As.shape[:-1]
#   - B (N,K) int8 contiguous；Bs (ceil(N/128), ceil(K/128)) → assert cdiv 相等
#   - 输出 C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)，即 (M, N)，
#     与 reference 的 out (M,N) 同形同 dtype（bf16/fp16）——**非打包，单张量**。
#   - kernel 尾块语义与 reference 一致：K 侧越界 load mask（other=0，贡献 0），
#     scale 按 floor(k/128) 索引（BLOCK_SIZE_K == group_k = 128，每块恰好一组
#     scale）；N 侧越界列以 %N 计算但不 store（DIVISIBLE_N 为 False 时带 mask
#     写回），故 K/N 不被 128 整除的尾块与 reference 一致。
#
# 已知的基线口径注意点（只影响性能，不影响正确性）：
#   gemm_w8a8 用 get_w8a8_block_int8_configs(N, K, 128, 128)（:358）在
#   $AITER_TRITON_CONFIGS_PATH/gemm/block_w8a8/ 下按
#   "N={N},K={K},arch={arch},cu={num_cu},dtype=int8_w8a8,block_shape=[128, 128].json"
#   （缺失时退回 device_name=BW200 命名）查调优配置；查不到就返回 None，走文件内
#   默认 config（:366-375，BLOCK_SIZE_M=64/N=128/K=128/GROUP_SIZE_M=32，
#   COMBINE_SCALE_LOAD=False、USE_MLS_LOAD=False，num_warps=4/num_stages=3）并打
#   一条 warning。**不是 @triton.autotune，缺文件不会失败，只退化性能**。
#   pinned 检出实测（aiter/ops/triton/configs/gemm/block_w8a8/ 共 18 个 JSON）：
#     - perf case 1/2（N=7168, K=2048）：arch=gfx938,cu=64 命中同名文件；gfx936 上
#       arch 名不匹配则退回 device_name 命名，该 device_name=BW200 文件**存在**
#       （arch_info.py:5-10 把 gfx936 映射成 BW200）→ 这两条走 tuned config。
#     - perf case 3（N=4096, K=4096）：两种命名都不存在 → 回退默认 config，基线
#       数值偏保守（正确性不受影响）。
#   ctx["config_source"] / ["config_candidate_files"] 如实标注走的哪条。

import torch

# 题目范围：块形固定 [128, 128]（reference.Model.block_n / block_k），
# 且 task.yaml 的 a_scale/b_scale 形状定义即以此为准。
BLOCK_N = 128
BLOCK_K = 128

_ALLOWED_OUT_DTYPES = ("bfloat16", "float16")


def _resolve_out_dtype(raw) -> torch.dtype:
    """init_kwargs["out_dtype"] → torch.dtype；越界即 raise（绝不静默退回）。"""
    if isinstance(raw, torch.dtype):
        dtype = raw
    elif isinstance(raw, str):
        name = raw.strip()
        if name.startswith("torch."):
            name = name[len("torch."):]
        if name not in _ALLOWED_OUT_DTYPES:
            raise ValueError(
                f"4016_gemm_w8a8 只接受 out_dtype ∈ {_ALLOWED_OUT_DTYPES}（题目 io.outputs "
                f"声明 bfloat16/float16），收到 {raw!r}"
            )
        dtype = getattr(torch, name)
    else:
        raise TypeError(f"out_dtype 应为 str 或 torch.dtype，收到 {type(raw).__name__}: {raw!r}")

    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"4016_gemm_w8a8 的输出 dtype 只支持 bfloat16/float16，收到 {dtype}")
    return dtype


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def _probe_config_source(M: int, N: int, K: int) -> tuple[str, str, list]:
    """探测 aiter 会命中调优 config 还是回退默认 config（仅用于 ctx 记录）。

    复刻 gemm_w8a8(:358) → get_w8a8_block_int8_configs(:20-70) 的查表命名，并列出
    它实际会尝试的两个候选文件名（arch/cu 优先，其次 device_name）。失败只记录
    unknown，绝不影响主流程（triton runtime / 设备查询在异常环境下可能不可用）。
    """
    candidates: list = []
    try:
        import os

        from aiter.ops.triton.gemm_w8a8 import get_w8a8_block_int8_configs
        from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH

        configs = get_w8a8_block_int8_configs(N, K, BLOCK_N, BLOCK_K)

        try:  # 候选文件名纯粹是诊断信息，拿不到设备信息就留空
            import triton

            import aiter.ops.triton.utils.arch_info as arch_info
            from aiter.jit.utils.chip_info import get_cu_num

            arch = triton.runtime.driver.active.get_current_target().arch
            dev = arch_info.get_device()
            dev = "BW200" if dev.lower().startswith("bw") else dev
            candidates = [
                f"N={N},K={K},arch={arch},cu={get_cu_num()},dtype=int8_w8a8,"
                f"block_shape=[{BLOCK_N}, {BLOCK_K}].json",
                f"N={N},K={K},device_name={dev},dtype=int8_w8a8,"
                f"block_shape=[{BLOCK_N}, {BLOCK_K}].json",
            ]
            candidates = [
                os.path.join(AITER_TRITON_CONFIGS_PATH, "gemm/block_w8a8", name)
                for name in candidates
            ]
        except Exception:  # pragma: no cover
            pass

        if not configs:
            return (
                "default",
                "aiter 内部默认 config（gemm_w8a8:366-375，BLOCK_SIZE_M=64/N=128/K=128/"
                "GROUP_SIZE_M=32，COMBINE_SCALE_LOAD=False、USE_MLS_LOAD=False，"
                "num_warps=4/num_stages=3）——仅性能退化，数值语义不变",
                candidates,
            )
        key = min(configs.keys(), key=lambda x: abs(x - M))
        return "tuned", f"tuned config for bs={key}: {configs[key]}", candidates
    except Exception as exc:  # pragma: no cover - 诊断字段，不影响正确性
        return "unknown", f"config probe failed: {type(exc).__name__}: {exc}", candidates


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方 gemm_w8a8。

    inputs     : [aq, bq, a_scale, b_scale]
                 aq      int8    (M, K)                       激活码字，行主
                 bq      int8    (N, K)                       权重码字，行主
                 a_scale float32 (M, ceil(K/128))             每 token 每 128 一组
                 b_scale float32 (ceil(N/128), ceil(K/128))   权重 128x128 块 scale
    init_kwargs: {"out_dtype": "bfloat16" | "float16"}（Model(out_dtype=...) 的参数）

    返回 (out, ctx)：out 为 (M, N) 的 bfloat16/float16 单张量（与 reference 同形同
    dtype，无打包）；ctx 记录 aiter 入口、shape、块形与配置来源。

    注：评测器（KernelBench 语义）会把输入 cast 成 fp32 传入，码字值域 [-128,127]
    与 fp32 scale 均可精确表示，故入口的 .to(torch.int8) 无损——与 reference.Model
    forward 开头两行同口径。
    """
    from aiter.ops.triton.gemm_w8a8 import gemm_w8a8

    if len(inputs) != 4:
        raise ValueError(f"4016_gemm_w8a8 期望 4 个输入 [aq, bq, a_scale, b_scale]，收到 {len(inputs)} 个")
    aq, bq, a_scale, b_scale = inputs

    # 与 reference.Model.forward 一致：无损恢复 int8 码字与 fp32 scale
    aq = aq.to(torch.int8).contiguous()
    bq = bq.to(torch.int8).contiguous()
    a_scale = a_scale.to(torch.float32).contiguous()
    b_scale = b_scale.to(torch.float32).contiguous()

    if aq.dim() != 2 or bq.dim() != 2:
        raise ValueError(f"aq/bq 必须是二维 (M,K)/(N,K)，收到 {tuple(aq.shape)}/{tuple(bq.shape)}")
    if a_scale.dim() != 2 or b_scale.dim() != 2:
        raise ValueError(
            f"a_scale/b_scale 必须是二维，收到 {tuple(a_scale.shape)}/{tuple(b_scale.shape)}"
        )

    M, K = aq.shape
    N, K_b = bq.shape
    if K != K_b:
        raise ValueError(f"aq 与 bq 的 K 维必须一致，收到 {K} vs {K_b}")

    # aiter gemm_w8a8 内部 assert 的同款形状契约（:345-353），在此显式化并给出可读信息
    gk = _ceil_div(K, BLOCK_K)
    gn = _ceil_div(N, BLOCK_N)
    if tuple(a_scale.shape) != (M, gk):
        raise ValueError(f"a_scale 形状应为 (M, ceil(K/128)) = {(M, gk)}，收到 {tuple(a_scale.shape)}")
    if tuple(b_scale.shape) != (gn, gk):
        raise ValueError(
            f"b_scale 形状应为 (ceil(N/128), ceil(K/128)) = {(gn, gk)}，收到 {tuple(b_scale.shape)}"
        )

    out_dtype = _resolve_out_dtype(init_kwargs.get("out_dtype", "bfloat16"))

    config_source, config_note, config_candidates = _probe_config_source(M, N, K)

    out = gemm_w8a8(aq, bq, a_scale, b_scale, [BLOCK_N, BLOCK_K], out_dtype)

    if tuple(out.shape) != (M, N) or out.dtype != out_dtype:
        raise RuntimeError(
            f"aiter gemm_w8a8 返回 {tuple(out.shape)}/{out.dtype}，"
            f"与题目契约 (M,N)={(M, N)}/{out_dtype} 不符"
        )
    out = out.contiguous()

    torch.cuda.synchronize()

    ctx = {
        "path": "aiter.ops.triton.gemm_w8a8.gemm_w8a8",
        "M": M,
        "N": N,
        "K": K,
        "block_size": [BLOCK_N, BLOCK_K],
        "out_dtype": str(out_dtype).replace("torch.", ""),
        "config_source": config_source,
        "config_note": config_note,
        "config_candidate_files": config_candidates,
    }
    return out, ctx
