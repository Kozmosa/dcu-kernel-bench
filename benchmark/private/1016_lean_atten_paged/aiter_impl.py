# aiter_impl.py — 1016_lean_atten_paged 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线，
# 口径为 benchmark/private/1016_lean_atten_paged/perf_cases.json。
#
# 来源（pinned 检出 .dcu_runs/aiter_pinned，commit c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#   aiter/ops/triton/lean_atten_paged.py
#     sha256 c103ace042f9c876ffa22afecd89ba30d839a32e60de17ee314631c4b3b47193
#     （与 benchmark/sources/1016_lean_atten_paged.yaml 记录的哈希一致）
#   官方语义/调用权威 op_tests/triton_tests/test_la_paged.py
#     sha256 a42f2d101a3be4d3550a7ea4744b0c1eee0dd8bd59353088f397f2cc4cab616f
#
# 入口签名（lean_atten_paged.py L23-L41 的 host 包装层；本文件只做 host 侧参数
# 装配与 tile 空间对齐，不碰 kernel 计算体）：
#   persistent_lean_attention_paged(q, k, v, kv_block_tables, Mp, Lp, Op, locks,
#                                   batch_num_block_n, total_programs, BLOCK_M,
#                                   BLOCK_N, batch_size, sm_scale, num_warps,
#                                   waves_per_eu) -> o
#     q        [H, n_ctx_q * batch_size, D]  fp16/bf16，请求主序（== 题面 q）
#     k, v     [H, pool_rows, D]             同 dtype；**物理块池**，每 head 一段
#              独立 slab，块 b 的行区间 = [b*BLOCK_N, (b+1)*BLOCK_N)
#     kv_block_tables [H, tiles_per_head] int32 —— kernel 用**扁平**指针访问它
#              （L273 `KV_block_tables_ptr = kv_block_tables + iter`，iter 是全局
#              lean-tile 号 = head*tiles_per_head + 局部块号），故每 head 步长必须
#              等于 tiles_per_head（本文件按 [H, tiles_per_head] 连续张量传入）
#     Mp/Lp    [total_programs, BLOCK_M]     fp32 streamK 部分和 scratch
#     Op       [total_programs, BLOCK_M, D]  fp32 scratch
#     locks    [total_programs] int32 —— **每次调用都必须从 0 开始**（kernel 用
#              atomic_xchg 置位、host CTA 用 atomic_cas 自旋等跨 CTA 归约），
#              故本文件每次调用重新 torch.zeros
#     batch_num_block_n [batch_size] int32 —— 各请求**块数的前缀和**（L249/L252
#              用它把 head 的 tile 区间切成请求；末元素同时是 host 侧 tiles_per_head
#              口径下"全部已用 tile 数"）
#     sm_scale float（kernel 内部乘 1.44269504 走 exp2 在线 softmax，L53）
#   返回 o = torch.empty_like(q, dtype=v.dtype)（L71），形状/ dtype 与题面 out 一致。
#
# 布局结论：**不需要任何 permute**。题面 q/k/v 就是入口要的 [H, N, D] head 主序
# 连续布局，块表也是每 head 一行的 [H, used_blocks] int32；页大小即 kernel 的
# BLOCK_N（L462-463 `tl.advance(..., kv_block_id*BLOCK_N, ...)`，块号->行号 =
# block*BLOCK_N + offset），因此 BLOCK_N 必须取题面的 page_size（题面固定 64），
# BLOCK_M 必须取 n_ctx_q（题面固定 16，见下"适用域"2）。
#
# ⚠ 本适配器的关键点——"块池 > 已用块数"的对齐（题面 extra_blocks > 0）：
#   host 入口把 k.shape[1] 直接当作**全部请求 KV 长度之和**（L50 注释
#   "This is the sum of all ctx_n in a batch"），据此推出
#     num_n_blocks = ceil(k.shape[1]/BLOCK_N)     (L131)
#     tiles_per_head = num_m_blocks * num_n_blocks (L136)  → 每 head 的 tile 空间
#   并把 [0, tiles_per_head) 的 tile 按 batch_num_block_n 切给各请求
#   （L249-L258）。官方测试因此让块表是**全部物理块的完整置换**
#   （test_la_paged.py L107/L120：num_kv_blocks == sum_n_ctx//BLOCK_N），
#   即池大小恒等于已用块数，tile 空间恰好被请求铺满。
#   题面允许物理块池更大、未被引用的块是垃圾（task.yaml shape.invariants；
#   get_inputs 即 extra_blocks=4，perf case 2 / 多个 hidden case 也有 extra）。
#   此时若直接把张量丢给入口：tile 空间是 pool_blocks 而请求只铺满前
#   used_blocks 个，尾部垃圾 tile 匹配不到任何请求（L254 的条件对
#   local_head_iter >= used_blocks 全不成立），于是 tile_iter 停在 head 起点、
#   tile_iter_end 落在 used_blocks，L261 的 local_iter_end 与 L260 的 local_iter
#   相等（甚至更小），L404 `iter = iter + (local_iter_end - local_iter)` 推进 0
#   → kernel 死循环（或回退），即使不挂也会把垃圾块算进请求 0。
#   修法（零数据搬运、不动 k/v、不依赖任何越界访问）：把**垃圾块归给一个哑请求**
#   —— 每个 head 的 q 末尾拼一段全 0 的 n_ctx_q 行（哑请求的 q），块表用合法块号
#   补齐到 pool_blocks 列，batch_num_block_n 末尾追加 pool_blocks。这样
#     batch_num_block_n 末元素 == tiles_per_head == pool_blocks（铺满，无未映射 tile）
#   被引用块全部落在 [0, pool) 内（无越界），哑请求只多算 extra 个块、
#   其输出（q 全 0 时的均匀加权 v）在返回前切掉丢弃。成本 = O(extra/pool) 的
#   额外 kernel 工作 + 一次 q 的拷贝（n_ctx_q*batch*D*H 元素，相对 KV 可忽略），
#   比"按引用顺序把块 gather 成紧凑池"（H*used*page*D 元素搬运）小两个数量级。
#   仅当 extra_blocks == 0 时走**与官方测试完全一致的直传路径**（perf case 1/3）。
#
# 适用域（与题面的差异，越界一律 raise，绝不静默算错）：
#   1. BLOCK_N = page_size 且 page_size 必须是 2 的幂、∈ {16,32,64,128,256}
#      （tl.dot 最小维度 16；kernel 的块号->行号按 BLOCK_N 粒度寻址，L462-463）。
#      题面固定 64 属官方测试主体配置（BLOCK_N=64）。
#   2. BLOCK_M 必须等于 n_ctx_q：kernel 的 decode 形态不含 m 块下标
#      （L279 `Q_base = Q + tile_idx*(stride_qh//batch_size)`，tile_idx 只到
#      head/请求两级），即要求 num_m_blocks == 1（L130）。题面固定 n_ctx_q=16，
#      官方测试全部 BLOCK_M=16=n_ctx_q；本文件要求 n_ctx_q 为 2 的幂且 >= 16，
#      取 BLOCK_M = n_ctx_q，并在 ctx 里标 block_m_officially_tested
#      （官方仅覆盖 16）。
#   3. 每个请求的 KV 长度必须是 page_size 的正整数倍：kernel 尾块无任何掩码
#      （源文件头 L13-17 TODO "N_CTX with non-integer number of BLOCK_N"），
#      尾部半块会把下一请求的数据算进 softmax。题面 invariant 亦如此
#      （make_inputs 里有断言），本文件对 k.shape[1] % page_size != 0 直接 raise。
#   4. MHA：入口以 H = q.shape[0] 同时索引 q/k/v（L51、L274），不支持 GQA
#      （源文件头 "To be added features: Add GQA"），故要求
#      k.shape[0] == v.shape[0] == q.shape[0]。
#   5. 无 causal/alibi/kv_scale 参数——题面也没有多给张量（无掩码、非量化、
#      无 ALiBi、无滑窗），不存在需要额外输入却被省略的情形。
#
# total_programs：aiter 官方测试固定传 912 = 3×304，源文件注释是"LeanAttention
# assign 3 CTAs per SM"（L142），故按设备 CU 数的 3 倍取，并夹到 total_tiles 以内：
# grid > total_tiles 会让 max_tiles_per_tb == 1 且 even_split == False，host 侧
# L156-158 出现整除 (max_tiles_per_tb - 1) == 0 崩溃。该值只影响 streamK 切分/
# 占用，不影响数学（host 用同一个值算 num_splits 并据此读回 Mp/Lp/Op）。
#
# launch 参数：waves_per_eu=2 / num_warps=4 与官方测试主体配置同值
# （test_la_paged.py 前 14 组：BLOCK_M=16, BLOCK_N=64, waves_per_eu=2,
# num_warps=4）。注意源文件 L112 把 `num_warps=waves_per_eu` 传进 launch，
# 即实际编译用的 num_warps 就是 waves_per_eu（=2），本文件不改 aiter 源码，
# 只在 ctx 里如实记录 effective_num_warps。

import torch

_HEAD_DIMS = (16, 32, 64, 128, 256)
_PAGE_SIZES = (16, 32, 64, 128, 256)  # BLOCK_N：2 的幂且 >= 16（tl.dot 最小维度）
_CTAS_PER_SM = 3  # lean_atten_paged.py L142 注释：LeanAttention assign 3 CTAs per SM
_NUM_WARPS = 4
_WAVES_PER_EU = 2
_N_CTX_Q_OFFICIAL = 16


def _multi_processor_count(device) -> int:
    """设备 CU 数；取不到返回 0（调用方退回 grid = total_tiles）。"""
    try:
        return int(torch.cuda.get_device_properties(device).multi_processor_count)
    except Exception:
        return 0


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方 Lean Attention（分页 KV 变体）。

    inputs      : [q, k, v, kv_block_tables, batch_num_block_n]（make_inputs 的顺序，
                  已在 device 上）
                  q [H, n_ctx_q*batch, D] / k,v [H, total_kv_len, D] fp16|bf16
                  kv_block_tables [H, used_blocks] int32 / batch_num_block_n [batch] int32
    init_kwargs : {"head_size": int, 可省 "page_size"=64, "sm_scale"=0.5}

    返回 (out, ctx)；out 与 reference 输出同形同 dtype（[H, n_ctx_q*batch, D]）。
    """
    from aiter.ops.triton.lean_atten_paged import persistent_lean_attention_paged

    q, k, v, kv_block_tables, batch_num_block_n = inputs

    # ---- init_kwargs -> 入口参数（越界直接 raise） ----
    head_size = int(init_kwargs.get("head_size", int(q.shape[-1])))
    page_size = int(init_kwargs.get("page_size", 64))
    sm_scale = float(init_kwargs.get("sm_scale", 0.5))
    if head_size not in _HEAD_DIMS:
        raise ValueError(
            f"head_size={head_size} 不在 aiter lean_atten_paged 支持的 {_HEAD_DIMS} 内"
            "（lean_atten_paged.py L47 assert）"
        )
    if page_size not in _PAGE_SIZES:
        raise ValueError(
            f"page_size={page_size} 必须 ∈ {_PAGE_SIZES}（入口的 BLOCK_N 按页粒度寻址，"
            "且 tl.dot 要求 >= 16；题面固定 64）"
        )

    # ---- 题面张量合法性 ----
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError("q/k/v 必须是 3 维 [H, N, D]（每 head 一段）")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"aiter lean_atten_paged 只支持 fp16/bf16，收到 {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError(f"q/k/v dtype 必须一致，收到 {q.dtype}/{k.dtype}/{v.dtype}")

    num_heads = int(q.shape[0])
    if int(k.shape[0]) != num_heads or int(v.shape[0]) != num_heads:
        raise ValueError(
            "本题为 MHA（入口以 H = q.shape[0] 统一索引 q/k/v，无 GQA），"
            f"收到 q {num_heads} / k {k.shape[0]} / v {v.shape[0]}"
        )
    dims = (int(q.shape[-1]), int(k.shape[-1]), int(v.shape[-1]))
    if dims != (head_size, head_size, head_size):
        raise ValueError(
            f"head_size={head_size} 与 q/k/v 末维 "
            f"{q.shape[-1]}/{k.shape[-1]}/{v.shape[-1]} 不一致"
        )
    pool_rows = int(k.shape[1])
    if int(v.shape[1]) != pool_rows:
        raise ValueError(f"k/v 的 KV 池长度必须一致，收到 {k.shape[1]}/{v.shape[1]}")
    if pool_rows % page_size != 0:
        raise ValueError(
            f"k 的池长度 {pool_rows} 必须是 page_size={page_size} 的整数倍"
            "（kernel 尾块无掩码，lean_atten_paged.py L13-17）"
        )
    pool_blocks = pool_rows // page_size

    cbn = batch_num_block_n.detach().to(device=q.device, dtype=torch.int32).reshape(-1).contiguous()
    batch = int(cbn.numel())
    if batch < 1:
        raise ValueError("batch_num_block_n 不能为空")
    total_q = int(q.shape[1])
    if total_q % batch != 0:
        raise ValueError(f"q 的行数 {total_q} 必须是请求数 batch={batch} 的整数倍")
    n_ctx_q = total_q // batch
    if n_ctx_q < 16 or (n_ctx_q & (n_ctx_q - 1)) != 0:
        raise ValueError(
            f"n_ctx_q={n_ctx_q} 必须是 2 的幂且 >= 16（BLOCK_M = n_ctx_q，"
            "kernel 的 decode 形态要求 num_m_blocks == 1，见 lean_atten_paged.py L279/L130）"
        )
    block_m = n_ctx_q

    tables = kv_block_tables.detach().to(device=q.device, dtype=torch.int32)
    if tables.dim() != 2 or int(tables.shape[0]) != num_heads:
        raise ValueError(
            f"kv_block_tables 必须是 [H={num_heads}, total_used_blocks]，收到 {tuple(tables.shape)}"
        )
    used_blocks = int(cbn[-1].item())
    if used_blocks < 1:
        raise ValueError(f"batch_num_block_n 末元素（已用块数）必须 >= 1，收到 {used_blocks}")
    if int(tables.shape[1]) < used_blocks:
        raise ValueError(
            f"kv_block_tables 列数 {tables.shape[1]} 不能小于已用块数 {used_blocks}"
        )
    tables = tables[:, :used_blocks].contiguous()
    if bool(((tables < 0) | (tables >= pool_blocks)).any()):
        raise ValueError(
            f"kv_block_tables 的值域必须在 [0, {pool_blocks})（池块数 = "
            f"k.shape[1]//page_size = {pool_rows}//{page_size}）"
        )
    extra_blocks = pool_blocks - used_blocks

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()

    # ---- tile 空间对齐：池 > 已用块数时补一个哑请求（见头注释） ----
    batch_size = batch
    padded = extra_blocks > 0
    if padded:
        # 块表补齐到 pool_blocks 列：垃圾 tile 只会被哑请求读到，块号取任一合法值即可
        filler = tables[:, :1].expand(num_heads, extra_blocks)
        tables = torch.cat([tables, filler], dim=1).contiguous()
        # 每 head 的 q 末尾追加哑请求（全 0），其输出在返回前切掉
        dummy_q = torch.zeros_like(q_c[:, :n_ctx_q, :])
        q_c = torch.cat([q_c, dummy_q], dim=1).contiguous()
        # 前缀和末尾追加 pool_blocks：让全部 tile 都被某个请求覆盖（否则 kernel 死循环）
        tail = torch.full((1,), pool_blocks, dtype=torch.int32, device=q_c.device)
        cbn = torch.cat([cbn, tail]).contiguous()
        batch_size = batch + 1

    # ---- total_programs（= CTAs/SM × CU 数，夹到 total_tiles 以免 host 侧 //0） ----
    num_m_blocks = -(-n_ctx_q // block_m)              # == 1
    num_n_blocks = -(-pool_rows // page_size)          # == pool_blocks（host L131 口径）
    tiles_per_head = num_m_blocks * num_n_blocks
    total_tiles = tiles_per_head * num_heads
    if total_tiles < 1:
        raise ValueError(f"total_tiles={total_tiles} 非法（shape 过小）")
    num_sms = _multi_processor_count(q.device)
    grid = _CTAS_PER_SM * num_sms if num_sms > 0 else total_tiles
    total_programs = max(1, min(int(grid), total_tiles))

    # ---- streamK scratch（尺寸与 total_programs 同源；locks 必须每次清零） ----
    Mp = torch.empty((total_programs, block_m), dtype=torch.float32, device=q_c.device)
    Lp = torch.empty((total_programs, block_m), dtype=torch.float32, device=q_c.device)
    Op = torch.empty((total_programs, block_m, head_size), dtype=torch.float32, device=q_c.device)
    locks = torch.zeros((total_programs,), dtype=torch.int32, device=q_c.device)

    out_full = persistent_lean_attention_paged(
        q_c,
        k_c,
        v_c,
        tables,
        Mp,
        Lp,
        Op,
        locks,
        cbn,
        total_programs,
        block_m,
        page_size,
        batch_size,
        sm_scale,
        _NUM_WARPS,
        _WAVES_PER_EU,
    )
    torch.cuda.synchronize()

    out = out_full[:, : batch * n_ctx_q, :].contiguous() if padded else out_full

    ctx = {
        "aiter_module": "aiter.ops.triton.lean_atten_paged",
        "aiter_symbol": "persistent_lean_attention_paged",
        "path": "dummy_request_padding" if padded else "official_pool_exact",
        "num_heads": num_heads,
        "batch": batch,
        "n_ctx_q": n_ctx_q,
        "head_size": head_size,
        "dtype": str(q.dtype).replace("torch.", ""),
        "sm_scale": sm_scale,
        "page_size": page_size,
        "block_m": block_m,
        "block_n": page_size,
        "block_m_officially_tested": block_m == _N_CTX_Q_OFFICIAL,
        "pool_blocks": pool_blocks,
        "used_blocks": used_blocks,
        "extra_blocks": extra_blocks,
        "tiles_per_head": tiles_per_head,
        "total_tiles": total_tiles,
        "total_programs": total_programs,
        "multi_processor_count": num_sms,
        "batch_num_block_n": [int(x) for x in cbn.tolist()],
        "num_warps": _NUM_WARPS,
        "waves_per_eu": _WAVES_PER_EU,
        # lean_atten_paged.py L112 实际把 waves_per_eu 当 num_warps 传进 launch
        "effective_num_warps": _WAVES_PER_EU,
        "layout": (
            "q/k/v 已是入口要求的 [H, N, D] head 主序连续，无 permute；"
            + (
                "池大于已用块数 → 块表补齐到 pool_blocks 列、q 末尾补 n_ctx_q 行全 0 "
                "哑请求、batch_num_block_n 末尾补 pool_blocks（tile 空间铺满，无越界）"
                if padded
                else "池大小 == 已用块数 → 与官方测试同构，直传（含块表/前缀和）"
            )
        ),
        "out_shape": tuple(int(s) for s in out.shape),
    }
    return out, ctx
