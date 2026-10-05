# aiter_impl.py — 1015_lean_atten 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线，
# 口径为 benchmark/private/1015_lean_atten/perf_cases.json。
#
# 来源（pinned 检出 .dcu_runs/aiter_pinned，commit c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#   aiter/ops/triton/lean_atten.py
#     sha256 aebd683a215c3cd271770c981127ce3bec8d992bbe9089de830d35e395d76b8b
#     （与 benchmark/sources/1015_lean_atten.yaml 记录的哈希一致）
#   官方语义/调用权威 op_tests/triton_tests/test_la.py
#     sha256 a123c4292831ae9cce92b05244dc462375a32c22b3869389e832cd2c4530d120
#
# 入口签名（lean_atten.py L27-L44 的 host 包装层；本文件只做 host 侧参数装配，
# 不碰 kernel 计算体）：
#   persistent_lean_attention(q, k, v, Mp, Lp, Op, locks, batch_num_block_n,
#                             total_programs, BLOCK_M, BLOCK_N, causal, batch_size,
#                             sm_scale, num_warps, waves_per_eu) -> o
#     q        [batch*n_ctx_q, H, D]  连续、fp16/bf16，序列主序（== 题面 q）
#     k, v     [sum(n_ctx), H, D]     同 dtype、同布局（== 题面 k/v）
#     Mp/Lp    [total_programs, BLOCK_M]      fp32  scratch（streamK 部分和）
#     Op       [total_programs, BLOCK_M, D]   fp32  scratch
#     locks    [total_programs] int32 —— **每次调用前必须清零**：kernel 用
#              atomic_xchg 置位、host CTA 用 atomic_cas 自旋等待跨 CTA 归约
#     batch_num_block_n [batch] int32 —— 各批 BLOCK_N **块数的前缀和**（kernel
#              既用它划每批的 lean-tile 边界 L336-345，也用它算 k/v 的块对齐
#              行起点 b_seq_size L367-372）
#     sm_scale float（kernel 内部乘 1.44269504 后走 exp2 在线 softmax）
#   返回 o = torch.empty_like(q, dtype=v.dtype)，形状/ dtype 与题面 out 一致。
#
# 布局结论：**不需要任何 permute**。题面 q/k/v 就是 [total, H, D] 序列主序连续，
# 正是入口要的布局；kv_lens / causal 只参与 host 侧推导，不进 kernel。与题面
# init 的映射：head_dim -> BLOCK/HEAD_DIM 约束；scale(缺省 1/sqrt(head_dim)) ->
# sm_scale。压缩/量化布局、GQA、LSE、alibi 在该入口都不存在，题面也没多给张量。
#
# 适用域（与题面的差异，越界一律 raise，绝不静默算错）：
#   1. kernel 的 k/v 行偏移是**块对齐**的：b_seq_size = batch_num_block_n[b-1]*
#      BLOCK_N（L367-372），且尾块没有任何掩码（源文件头 L13-17 的 TODO
#      "N_CTX with non-integer number of BLOCK_N (pad zeros or add mask)"）。
#      所以每个 kv_len 必须是 BLOCK_N 的整数倍，否则会把下一个序列的数据算进
#      softmax。题面 hidden case 故意覆盖非整块 kv_len（如 [1,200,333]），那种
#      shape 本适配器显式 raise；perf_cases.json 的三个 case 全满足（kv_lens
#      均为 2 的幂、≥2048）。
#   2. decode（causal=0）路径要求 num_m_blocks == 1：kernel 里 decode 的
#      q_idx = tile_batch_idx（L329-331、L395），不带 m 块下标，所以必须
#      BLOCK_M == q_len（q_len 为 2 的幂且 ≥16，tl.dot 的最小维度 16）。
#   3. decode 还要求 BLOCK_N >= BLOCK_M，即 MASKED_BLOCKS = BLOCK_M//BLOCK_N != 2：
#      收尾处那段 MASKED_BLOCKS==2 的 m_i/l_i/acc 重置（L458-465）是给 causal
#      分组用的（只在「CTA 的 chunk 恰好是该输出 tile 的最后一个 lean tile」时
#      触发，配合 host/非 host 的 streamK 归约才成立）；decode 每个 tile 对每行
#      都有贡献，落到该分支会丢贡献。
#   4. causal（causal=1）路径不支持 ragged batch（L310 注释 "Does not support
#      ragged batching"）：要求 kv_lens[b] == q_len（题面 task.yaml 的因果
#      invariant 本就如此），且 q_len 是 BLOCK_M 的整数倍。首选官方测试验证过的
#      (BLOCK_M=128, BLOCK_N=64, waves_per_eu=1)（test_la.py L56-70）；q_len 只
#      能整除 64 时退到 (64,64) 这个 MASKED_BLOCKS==1 分支（kernel L432-436
#      显式实现，但官方测试未覆盖，ctx 里标 not_officially_tested）。
#
# total_programs：既是 CTA 网格大小，也是 get_num_splits_and_buffer_sizes 的
# num_SMs。它只影响 streamK 的切分/占用，不影响数学正确性（host 侧用同一个值算
# num_splits 并据此分配 Mp/Lp/Op/locks），故按设备 CU 数取，并夹到 total_tiles
# 以内（grid > total_tiles 会让 max_tiles_per_tb==1 且 even_split=False，host 侧
# 出现 //0）。
#
# 已知噪声：lean_atten.py L145 在每次调用后 print 一行寄存器信息（aiter 自带，
# 本文件不改源文件），只为初始化时编译信息，对 cuda_event 计时影响可忽略。

import math

import torch

_HEAD_DIMS = (16, 32, 64, 128, 256)
# decode 的 BLOCK_N 候选（降序）：需整除全部 kv_len，且 >= BLOCK_M（见头注释 3）
_DECODE_BLOCK_N = (128, 64, 32, 16)
# causal 的 (BLOCK_M, BLOCK_N) 候选（降序）：需整除 q_len
_CAUSAL_BLOCKS = ((128, 64), (64, 64))
# 与官方测试用例一致的 launch 调优参数（test_la.py：decode 用 num_warps=4 /
# waves_per_eu∈{1,2}；causal 用 num_warps=4 / waves_per_eu=1）
_DECODE_NUM_WARPS, _DECODE_WAVES_PER_EU = 4, 2
_CAUSAL_NUM_WARPS, _CAUSAL_WAVES_PER_EU = 4, 1


def _multi_processor_count(device) -> int:
    """设备 CU 数；取不到返回 0（调用方退回 grid = total_tiles）。"""
    try:
        return int(torch.cuda.get_device_properties(device).multi_processor_count)
    except Exception:
        return 0


def _decode_total_tiles(q_len, total_kv, num_heads, block_m, block_n) -> int:
    """复刻 get_num_splits_and_buffer_sizes 的 total_tiles（仅用于夹 grid）。"""
    num_m_blocks = -(-q_len // block_m)
    num_n_blocks = -(-total_kv // block_n)
    return num_m_blocks * num_n_blocks * num_heads


def _causal_total_tiles(q_len, num_seqs, num_heads, block_m, block_n) -> int:
    num_m_blocks = -(-q_len // block_m)
    per_batch = sum(-(-(i + 1) * block_m // block_n) for i in range(num_m_blocks))
    return per_batch * num_seqs * num_heads


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方 Lean Attention（persistent_lean_attention）。

    inputs      : [q, k, v, kv_lens, causal]（make_inputs 的顺序，已在 device 上）
                  q [num_seqs*q_len, H, D] / k,v [sum(kv_lens), H, D] fp16|bf16
                  kv_lens [num_seqs] int32 / causal 0-dim int32（1=方阵因果）
    init_kwargs : {"head_dim": int, 可省 "scale": float|None}

    返回 (out, ctx)；out 与 reference 输出同形同 dtype（[num_seqs*q_len, H, D]）。
    """
    from aiter.ops.triton.lean_atten import persistent_lean_attention

    q, k, v, kv_lens, causal = inputs

    # ---- init_kwargs -> 入口参数（越界直接 raise） ----
    if "head_dim" not in init_kwargs:
        raise ValueError(
            "init_kwargs 缺 head_dim（Model(head_dim, scale=None) 的首个构造参数）"
        )
    head_dim = int(init_kwargs["head_dim"])
    if head_dim not in _HEAD_DIMS:
        raise ValueError(
            f"head_dim={head_dim} 不在 aiter lean_atten 支持的 {_HEAD_DIMS} 内"
            "（lean_atten.py L50 assert）"
        )
    scale = init_kwargs.get("scale", None)
    sm_scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)

    # ---- 题面张量合法性 ----
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"aiter lean_atten 只支持 fp16/bf16，收到 {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(f"q/k/v dtype 必须一致，收到 {q.dtype}/{k.dtype}/{v.dtype}")
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q/k/v 必须是 3 维 [total, H, D]")

    total_q, num_heads, dim_q = (int(s) for s in q.shape)
    if dim_q != head_dim or int(k.shape[-1]) != head_dim or int(v.shape[-1]) != head_dim:
        raise ValueError(
            f"head_dim={head_dim} 与 q/k/v 末维 {dim_q}/{k.shape[-1]}/{v.shape[-1]} 不一致"
        )
    if int(k.shape[1]) != num_heads or int(v.shape[1]) != num_heads:
        raise ValueError(
            "本题为 MHA（Q/K/V 头数一致，无 GQA），"
            f"收到 q {num_heads} / k {k.shape[1]} / v {v.shape[1]}"
        )

    if not torch.is_tensor(kv_lens):
        kv_lens = torch.as_tensor(kv_lens)
    kv_lens_i32 = (
        kv_lens.detach().to(device=q.device, dtype=torch.int32).reshape(-1).contiguous()
    )
    num_seqs = int(kv_lens_i32.numel())
    if num_seqs < 1:
        raise ValueError("kv_lens 不能为空")
    if total_q % num_seqs != 0:
        raise ValueError(f"total_q={total_q} 必须是 num_seqs={num_seqs} 的整数倍")
    q_len = total_q // num_seqs

    kv_list = [int(n) for n in kv_lens_i32.tolist()]
    if any(n < 1 for n in kv_list):
        raise ValueError(f"kv_lens 必须 >= 1，收到 {kv_list}")
    total_kv = sum(kv_list)
    if int(k.shape[0]) != total_kv or int(v.shape[0]) != total_kv:
        raise ValueError(
            f"k/v 首维应与 sum(kv_lens)={total_kv} 一致，收到 {k.shape[0]}/{v.shape[0]}"
        )

    if torch.is_tensor(causal):
        if causal.numel() != 1:
            raise ValueError(f"causal 必须是标量，收到 shape {tuple(causal.shape)}")
        is_causal = bool(int(causal.detach().to(torch.int32).reshape(-1)[0].item()))
    else:
        is_causal = bool(int(causal))

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()

    # ---- BLOCK_M / BLOCK_N 选取（含适用域校验） ----
    if is_causal:
        if any(n != q_len for n in kv_list):
            raise ValueError(
                "aiter lean_atten 的 causal 路径不支持 ragged batch"
                "（lean_atten.py L310：Does not support ragged batching），"
                f"要求 kv_lens[b] == q_len={q_len}，收到 {kv_list}"
            )
        chosen = None
        for block_m, block_n in _CAUSAL_BLOCKS:
            if q_len % block_m == 0 and q_len % block_n == 0:
                chosen = (block_m, block_n)
                break
        if chosen is None:
            raise ValueError(
                f"causal 路径要求 q_len 为 BLOCK_M 的整数倍"
                f"（候选 {_CAUSAL_BLOCKS}，MASKED_BLOCKS=BLOCK_M//BLOCK_N 决定掩码口径），"
                f"收到 q_len={q_len}"
            )
        block_m, block_n = chosen
        num_warps, waves_per_eu = _CAUSAL_NUM_WARPS, _CAUSAL_WAVES_PER_EU
        total_tiles = _causal_total_tiles(q_len, num_seqs, num_heads, block_m, block_n)
        officially_tested = (block_m, block_n) == _CAUSAL_BLOCKS[0]
    else:
        # q_len 必须是 2 的幂且 >= 16（tl.dot 最小维度）：BLOCK_M == q_len（头注释 2）
        if q_len < 16 or (q_len & (q_len - 1)) != 0:
            raise ValueError(
                "aiter lean_atten 的 decode 路径要求 num_m_blocks == 1"
                "（lean_atten.py L329-331/L395 的 q_idx 不含 m 块下标），"
                f"即 BLOCK_M == q_len 且 q_len 为 2 的幂、>=16，收到 q_len={q_len}"
            )
        block_m = q_len
        block_n = None
        for cand in _DECODE_BLOCK_N:
            if cand < block_m:  # 头注释 3：避免落到 MASKED_BLOCKS == 2
                continue
            if all(n % cand == 0 for n in kv_list):
                block_n = cand
                break
        if block_n is None:
            raise ValueError(
                "aiter lean_atten 的 k/v 行偏移按 BLOCK_N 块对齐、尾块无掩码"
                "（lean_atten.py L13-17 TODO / L367-372），因此每个 kv_len 必须是 "
                f"BLOCK_N（<= 128 且 >= BLOCK_M={block_m}，2 的幂）的整数倍；"
                f"kv_lens={kv_list} 不满足，本题面输入超出 aiter 该 kernel 的能力"
                "（hidden case 里的非整块 kv_len 即属此类）"
            )
        num_warps, waves_per_eu = _DECODE_NUM_WARPS, _DECODE_WAVES_PER_EU
        total_tiles = _decode_total_tiles(q_len, total_kv, num_heads, block_m, block_n)
        officially_tested = True

    if total_tiles < 1:
        raise ValueError(f"total_tiles={total_tiles} 非法（shape 过小）")

    # ---- total_programs（网格 = num_SMs，夹到 total_tiles 以免 host 侧 //0） ----
    num_sms = _multi_processor_count(q.device)
    grid = min(num_sms, total_tiles) if num_sms > 0 else total_tiles
    grid = max(1, int(grid))

    # ---- batch_num_block_n：各批 BLOCK_N 块数的前缀和（int32，device 上） ----
    cum = []
    running = 0
    for n in kv_list:
        running += -(-n // block_n)
        cum.append(running)
    batch_num_block_n = torch.tensor(cum, dtype=torch.int32, device=q.device)

    # ---- streamK scratch（尺寸与 total_programs 同源，由 aiter 自己读回） ----
    Mp = torch.empty((grid, block_m), dtype=torch.float32, device=q.device)
    Lp = torch.empty((grid, block_m), dtype=torch.float32, device=q.device)
    Op = torch.empty((grid, block_m, head_dim), dtype=torch.float32, device=q.device)
    locks = torch.zeros((grid,), dtype=torch.int32, device=q.device)

    out = persistent_lean_attention(
        q_c,
        k_c,
        v_c,
        Mp,
        Lp,
        Op,
        locks,
        batch_num_block_n,
        grid,
        block_m,
        block_n,
        bool(is_causal),
        num_seqs,
        sm_scale,
        num_warps,
        waves_per_eu,
    )
    torch.cuda.synchronize()

    ctx = {
        "aiter_module": "aiter.ops.triton.lean_atten",
        "aiter_symbol": "persistent_lean_attention",
        "path": "causal_prefill" if is_causal else "decode_ragged",
        "causal": is_causal,
        "num_seqs": num_seqs,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "q_len": q_len,
        "kv_lens": kv_list,
        "total_kv": total_kv,
        "dtype": str(q.dtype).replace("torch.", ""),
        "sm_scale": sm_scale,
        "block_m": block_m,
        "block_n": block_n,
        "masked_blocks": block_m // block_n,
        "num_warps": num_warps,
        "waves_per_eu": waves_per_eu,
        "total_programs": grid,
        "multi_processor_count": num_sms,
        "batch_num_block_n": cum,
        "total_tiles": total_tiles,
        "officially_tested_config": officially_tested,
        "layout": "q/k/v 已是入口要求的 [total, H, D] 序列主序连续，无 permute/pad",
        "out_shape": tuple(int(s) for s in out.shape),
    }
    return out, ctx
