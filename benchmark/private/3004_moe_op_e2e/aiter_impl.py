# aiter_impl.py — 3004_moe_op_e2e 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约（按名取参，构造参数稀疏给出）：run(inputs, init_kwargs: dict, device)
# -> (out, ctx)，out 为与 reference 同形同 dtype 的单个张量。
#
# ════════════════════════════ 来源（已核 sha256）════════════════════════════
# 本地 pinned 检出 .dcu_runs/aiter_pinned，commit c39fff8c77df4e80617649e92fa3c2615f2c43d1
#   aiter/ops/triton/moe_op_e2e.py
#     sha256 ff02ef2ae74cfa3b52bed1e1947474b2847322c81eae353c12d5b83f7f366ea3
#     （与 benchmark/sources/3004_moe_op_e2e.yaml 记录一致）——本题核心计算：
#       e2e_moe_kernel（module:50，非 persistent 变体）
#       e2e_moe_persistent_kernel（module:351）
#     两 kernel 都在一次 launch 内完成「第一段 grouped GEMM → SwiGLU →
#     第二段 grouped GEMM → 路由加权」。
#   op_tests/triton_tests/test_moe_e2e.py
#     sha256 c14b8cb2e374d4f74eff9c5d18ac0c8a0488d1f507de99795ee7367d3c3d0062
#     （语义唯一权威：test_correctness，非量化子集 fp8_w8a8=False/int8_w8a16=False，
#       routed_weight 恒 True 的等价语义，atol=rtol=1e-1，test:451）
#   辅助入口（本题调度三件套的官方构造器，非核心计算）：
#   aiter/ops/triton/moe_align_block_size.py
#     sha256 f950b961fff08ab75e3f29ac4c4788d0096a200883c7b949cf2b9350b485b279
#     （与 sources/3002_moe_align_block_size.yaml 记录一致）
#
# ═══════════════════════ 公开 host 入口（本文件唯一调用的算子入口）══════════
#   aiter.ops.triton.moe_op_e2e.e2e_moe(              # moe_op_e2e.py:600-619
#       A, W1, W2, Intermediate, C,
#       A_scale, W1_scale, W2_scale,
#       topk_weights, sorted_token_ids, topk_ids, expert_ids,
#       num_tokens_post_padded,
#       mul_routed_weight, top_k, use_fp8_w8a8, use_int8_w8a16,
#       config=None) -> torch.Tensor                      # 返回 C（persistent 分支，
#                                                         # module:718）或 Out.to(dtype)
#
# 调用约定取自官方测试 test_moe_e2e.py:409-428（唯一非量化调用样板）。注意官方测试
# 那里是**位置传参**，而该处 config/use_fp8_w8a8/use_int8_w8a16 三个实参的顺序与
# 形参顺序不一致（测试把 config 传给了 use_fp8_w8a8、把 int8_w8a16 传给了 config），
# 在 c39fff8c 上会 `config["BLOCK_SIZE_M"]`（module:651）TypeError——本适配器因此
# **全部用关键字调用**，把官方测试的意图（非量化 + 默认 config）落实为正确绑定，
# 不复制该传参缺陷。
#
# ═════════════════════ 题面 io → 入口参数映射（逐项）════════════════════════
#   reference.py::make_inputs 返回 [a, w1, w2, topk_weights, topk_ids]：
#     a            (M, K)      bf16/fp32  → A            （原样，contiguous）
#     w1           (E, N, K)   same_as_a  → W1           （原样，contiguous）
#     w2           (E, K, N/2) same_as_a  → W2           （原样：见下方 stride 说明）
#     topk_weights (M, top_k)  same_as_a  → topk_weights （原样，stride(1)==1）
#     topk_ids     (M, top_k)  int64      → topk_ids（仅用 .numel() = 有效槽位数）
#   题面**没有**给调度三件套（sorted_token_ids / expert_ids /
#   num_tokens_post_padded）与 Intermediate：sources/*.yaml 的 admission.note 明确
#   「排序对齐/分块分发是官方 kernel 的调度中间量而非算子语义，不进题面接口，
#   实现策略自由」，故由本适配器按官方约定构造：
#     * sorted_token_ids / expert_ids / num_tokens_post_padded ← 官方 aiter 入口
#       aiter.ops.triton.moe_align_block_size.moe_align_block_size_triton
#       （官方 wrapper 用法见 test_moe_align_block_size.py:109-132：缓冲尺寸
#        L1 = numel + E*(block-1)、L2 = cdiv(L1, block)、sorted 全域预填哨兵 numel）。
#       本适配器逐行同构，不自己写 torch 排序。
#     * Intermediate ← 官方测试 test_moe_e2e.py:365-369 的
#       torch.zeros((M*top_k, N//2), float32)：第二段 GEMM 的 fp32 累加缓冲
#       （persistent kernel 对它做 atomic_add，故必须零初始化；fp32 与 output dtype
#        无关，是 kernel 侧约定的类型，module:367）。
#     * C ← torch.zeros((M, top_k, K), dtype=a.dtype)：与官方测试
#       test_moe_e2e.py:291 的 `c = torch.zeros((M, top_k, K), dtype=dtype)` 一致。
#   非量化：A_scale = W1_scale = W2_scale = None（官方测试 input_helper:279-281 同样
#   在非量化下传 None；module:626-648 的量化校验在 c39fff8c 上被整段注释掉，但
#   kernel 内 use_*_w8a* 全为 constexpr False，scale 指针不会被解引用）。
#   路由加权恒启用 → mul_routed_weight=True（官方测试用 routed_weight 参数，
#   本题语义见 reference.py:29/92，恒乘 topk_weights）。
#
# ═══════════════════════════ layout 约定（关键）════════════════════════════
# kernel 全程用「槽位」下标 p = m*top_k + j 索引（module:186-190、:275）：
#   * offs_token = sorted_token_ids[pid_m*BM + arange(BM)]，取值 ∈ [0, M*top_k)，
#     填充槽位被预填哨兵 numel = M*top_k，由 token_mask = offs_token < num_tokens
#     （module:188，num_tokens = topk_ids.numel()）屏蔽；
#   * A 的行：offs_token // top_k（module:207，persistent module:444）→ m；
#   * topk_weights：topk_weights_ptr + offs_token（module:323/:584）→ 逐槽位取值，
#     即把 (M, top_k) 视作展平的 (M*top_k,)，与 reference.py:92 的
#     tw32[rows, slots] 同义；host 侧断言 topk_weights.stride(1) == 1（module:623）；
#   * 输出：Out + stride_cm * offs_token + offs_k2，stride_cm = C.stride(1)
#     （module:667）；C 连续时为 K，等价于把 (M, top_k, K) 视作 (M*top_k, K)，
#     与 reference.py:93 的 out[rows, slots] 逐元素对应。
# 因此 (M,K) / (E,N,K) / (M,top_k) 三个用户级张量**无需任何 permute/transpose**：
# W1 走 stride(0,1,2)（module:687-689）、W2 走 stride(0,2,1)（module:692-694），
# 正好对应 w2 的 (E, K, N//2) 语义（w2 的 K 是第二段 GEMM 的归约维、N//2 是输出行）。
# 唯一要求：三者 contiguous（stride 按上述顺序读取），本适配器只做 .contiguous()。
#
# ═══════════════════════ 路径选择与 config（无 autotune JSON）═══════════════
# 走 **persistent** 变体，理由：
#   1) 官方测试只测这一支：test_moe_e2e.py:345 `@pytest.mark.parametrize("persistent",
#      [True])` + :361 moe_set_use_persistent_kernel(True)，且 Intermediate 只在
#      persistent 下分配（:365-369）；模块级缺省是 False（module:18），两者输出
#      等价但只有 persistent 支被官方测试覆盖。
#   2) 写出口更保守：persistent 支对 Intermediate 的归约是「每个 (槽位, 中间列)
#      恰好一个 CTA 写」（pid_m 分块互不相交，module:507-513），输出侧是纯
#      tl.store（module:596，每个 (槽位, K 列) 恰好一次）；非 persistent 支的
#      跨 pid_n 归约用的是 tl.atomic_add(sem="relaxed", scope="cta")（module:340-346），
#      而同一个输出元素会由 cdiv(N, BLOCK_SIZE_N) 个不同 CTA 并发累加，
#      cta 作用域的原子性对跨 workgroup 归约不是官方测试验证过的组合。
#   3) config 必须显式给：e2e_moe 在 module:651 无条件下标访问 config["BLOCK_SIZE_M"]，
#      config=None 会 TypeError；而 e2e_moe_kernel / e2e_moe_persistent_kernel 都只有
#      @triton.heuristics（module:43-48）**没有 @triton.autotune**，也不读
#      AITER_TRITON_CONFIGS_PATH（moe_config_utils 只服务 fused_moe 系列），
#      故不需要任何调优 JSON（needs_autotune_config = false）。本适配器用官方测试的
#      get_default_config(persistent=True)（test_moe_e2e.py:201-209）原值：
#      BLOCK_SIZE_M=64 / BLOCK_SIZE_N1=128 / BLOCK_SIZE_N2=64 /
#      BLOCK_SIZE_K1=64 / BLOCK_SIZE_K2=64。
#      BLOCK_SIZE_M 必须等于调度三件套的对齐粒度（对齐构造就按它做），否则
#      expert_ids[pid_m]（module:431）会跨专家错位——这是正确性硬约束，不是调参。
#
# ═══════════════════════════ dtype 与输出契约 ══════════════════════════════
# 题面 io 域为 bfloat16 / float32；kernel 的 compute dtype 由 C.dtype 决定
# （module:724 `dtype = C.dtype`，persistent 支 module:419
#  `dtype = Out.dtype.element_ty`），故 C 直接用 a.dtype ⇒ 计算精度档与题面一致，
# 返回的 C 与 reference 输出同形 (M, top_k, K)、同 dtype，无需打包（本算子是
# 单输出，reference 的 forward 也只返回一个张量）。
#
# ⚠️ 精度与容差（必须知道，实测证据）：本 kernel 在第二段 GEMM 之前把中间激活
#    h = SiLU(gate)*up 从 fp32 **cast 到 compute dtype**（非 persistent：
#    module:261 `acc = (silu_acc * mul_acc).to(dtype)`；persistent：module:505 同一行）。
#    本题 reference（reference.py:79-93）是 fp32 全程累加、只在输出处 cast 一次，
#    因此该中间 cast 的全部舍入落入两者之差，并在第二段 GEMM 上按 h 的量级放大。
#    本机以忠实数值仿真（h→bf16、bf16 乘积在 fp32 精确、fp32 累加、末尾 cast 一次，
#    对照同一输入的 fp32 reference）在真实 shape 族上测得：
#      * 在线/固定族 M=32,N=1024,K=512,top_k=2（32768 个输出元素，两个随机种子）：
#        max(diff) = 0.76 / 1.28，越界元素 39 / 244，最坏 diff/(atol+rtol*|e|)
#        = 13.0 / 43.7（越界点 |expected| ≈ 0.46~1.9，容差上限只有 0.03~0.06）；
#      * perf_mid/hidden_decode 形状（K=1024, N//2=1024，8192 元素）：最坏比值 10.8，
#        5 个越界元素。
#    即：**在 task.yaml 现有 bf16/float32 容差 atol=rtol=2e-2 下，aiter 官方 kernel
#    的原生 dtype 基线与 reference 比对不会通过**，record_baseline.py 会拒绝记录。
#    根因在 aiter 侧（中间 cast 是 kernel 语义的一部分，host 侧无法关闭：compute
#    dtype 由 C.dtype 唯一决定），不是本适配器的形状/dtype/打包对齐问题。
#    官方测试对此的处置是放宽容差到 atol=rtol=1e-1（test_moe_e2e.py:451），且其
#    torch 参考同样把中间值放在 bf16（test:64 的 silu_and_mul 直接吃 bf16），
#    两侧共享同一处舍入，所以在 1e-1 下成立；本题 reference 升为 fp32 全程后
#    不再共享该舍入。评测集已有同类先例：4008_gemm_a16w4 的 task.yaml 容差
#    正是为「aiter kernel 在 tl.dot 前把权重 cast 到 fp16」这一固有精度损失
#    放宽到 2e-1（见该题 task.yaml tolerance.note 的实测 worst(diff/tol) 论证）。
#    建议（任务侧、非适配器侧）：把本题 bf16/float32 容差放宽到官方测试的 1e-1
#    量级（按 4008 的做法做种子扫描定档），再采基线。
#    若维护者选择**不改容差**而宁愿要一个能过 2e-2 的基线，唯一办法是以该入口的
#    fp32 档调用（A/W1/W2 全部 .to(torch.float32)、C 用 fp32、返回前 .to(a.dtype)）：
#    此时 module:505/:261 的 cast 是 no-op、h 保持 fp32，与题面「fp32 全程累加、
#    仅输出 cast 一次」的语义一致，可过容差；但两段 GEMM 都退化为 fp32（DCU 上
#    fp32 tl.dot 远慢于 bf16 MFMA），性能基线会系统性偏慢、放大候选的加速比，
#    故本适配器**默认不启用**该档，仅在此登记为可选方案。
#
# 部署注意：本文件按约定在函数内做最小导入；被导入的 aiter 模块自身链式拉入
#   aiter/ops/triton/quant.py（moe_op_e2e.py:8）与
#   aiter/ops/triton/utils/types.py → utils/arch_info.py（:9），最小导入垫片需
#   同时带上这两个模块，均不读任何 config JSON。

import torch

# 调度三件套的对齐粒度 = 官方 get_default_config(persistent=True)["BLOCK_SIZE_M"]
_BLOCK_SIZE_M = 64

# 官方 get_default_config(persistent=True) 原值（op_tests/triton_tests/test_moe_e2e.py:201-209）
_PERSISTENT_CONFIG = {
    "BLOCK_SIZE_M": _BLOCK_SIZE_M,
    "BLOCK_SIZE_N1": 128,
    "BLOCK_SIZE_N2": 64,
    "BLOCK_SIZE_K1": 64,
    "BLOCK_SIZE_K2": 64,
}


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方 e2e_moe（persistent 变体，非量化，含路由加权）。

    inputs      : [a, w1, w2, topk_weights, topk_ids]（= reference.py::make_inputs
                  的返回顺序，元素已在 device 上）
                    a            (M, K)         bfloat16/float32
                    w1           (E, N, K)      同 a（前 N/2 通道 gate、后 N/2 up）
                    w2           (E, K, N//2)   同 a
                    topk_weights (M, top_k)     同 a
                    topk_ids     (M, top_k)     int64（或可无损转回的整数张量）
    init_kwargs : {"top_k": int, "num_experts": int}（Model.__init__ 的同名参数；
                  仅元数据，forward 以运行期张量形状为准，reference.py:44-45）

    返回 (out, ctx)；out = C，形状 (M, top_k, K)，dtype 同 a（与 reference 输出
    同形同 dtype 的单个张量，无需打包）。
    """
    # device 形参只作契约声明：新增张量一律按 a.device 分配（契约保证输入已在
    # device 上，用张量自身的 device 不会出现 cuda 与 cuda:0 的写法差异）。
    # aiter 顶层 import 很重，按部署约定在函数内做最小导入
    from aiter.ops.triton.moe_op_e2e import (
        e2e_moe,
        moe_set_use_persistent_kernel,
    )
    from aiter.ops.triton.moe_align_block_size import moe_align_block_size_triton

    if len(inputs) != 5:
        raise ValueError(
            f"3004 的 make_inputs 返回 5 个张量 [a, w1, w2, topk_weights, topk_ids]，"
            f"实得 {len(inputs)} 个"
        )
    a, w1, w2, topk_weights, topk_ids = inputs

    # ---------- shape 校验（越界直接 raise，绝不静默用错参数）----------
    if a.dim() != 2 or w1.dim() != 3 or w2.dim() != 3:
        raise ValueError(
            f"a 应为 2 维、w1/w2 应为 3 维，实际 {tuple(a.shape)} / "
            f"{tuple(w1.shape)} / {tuple(w2.shape)}"
        )
    if topk_weights.dim() != 2 or topk_ids.dim() != 2:
        raise ValueError(
            f"topk_weights/topk_ids 应为 2 维，实际 {tuple(topk_weights.shape)} / "
            f"{tuple(topk_ids.shape)}"
        )
    M, K = int(a.shape[0]), int(a.shape[1])
    E, N, K_w1 = int(w1.shape[0]), int(w1.shape[1]), int(w1.shape[2])
    if K_w1 != K:
        raise ValueError(f"w1 的末维 {K_w1} 与 a 的 K={K} 不一致")
    if N % 2 != 0:
        raise ValueError(f"w1 的 N={N} 必须为偶数（前 N/2 gate、后 N/2 up）")
    if N < 2:
        raise ValueError(f"N={N} 非法")
    if tuple(w2.shape) != (E, K, N // 2):
        raise ValueError(
            f"w2 形状应为 (E, K, N//2)=({E}, {K}, {N // 2})，实际 {tuple(w2.shape)}"
        )
    top_k = int(topk_ids.shape[1])
    if tuple(topk_weights.shape) != (M, top_k):
        raise ValueError(
            f"topk_weights 形状应为 (M, top_k)=({M}, {top_k})，实际 "
            f"{tuple(topk_weights.shape)}"
        )
    if E < 1 or K < 1 or top_k < 1:
        raise ValueError(f"E/K/top_k 必须 >= 1，实际 E={E}, K={K}, top_k={top_k}")
    if top_k > E:
        raise ValueError(f"top_k={top_k} 超出 num_experts={E}（题面不变量 top_k <= E）")

    # ---------- init_kwargs 与运行期形状的交叉校验 ----------
    # 题面不变式：case 键与 make_inputs 形参、io.init_inputs 名三方一致
    # （top_k / num_experts），且 reference.py:44-45 以运行期形状为准。
    kw_top_k = int(init_kwargs.get("top_k", top_k))
    kw_num_experts = int(init_kwargs.get("num_experts", E))
    if kw_top_k != top_k:
        raise ValueError(
            f"init_kwargs['top_k']={kw_top_k} 与运行期 topk_ids.shape[1]={top_k} 不一致"
        )
    if kw_num_experts != E:
        raise ValueError(
            f"init_kwargs['num_experts']={kw_num_experts} 与 w1.shape[0]={E} 不一致"
        )

    # ---------- dtype 校验（题面 io 域：bfloat16 / float32）----------
    if not (a.dtype == w1.dtype == w2.dtype == topk_weights.dtype):
        raise ValueError(
            f"a/w1/w2/topk_weights 必须同 dtype（io: same_as_a），实际 "
            f"{a.dtype} / {w1.dtype} / {w2.dtype} / {topk_weights.dtype}"
        )
    if a.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"题面 io 的 dtype 域为 bfloat16/float32，实际 {a.dtype}"
        )

    # ---------- 整数索引恢复 + 范围校验 ----------
    # KernelBench 在线路径会把输入统一 cast 成 fp32，专家编号是小整数可无损恢复
    # （reference.py:70-71 的 topk_ids.to(torch.long) 同义）；离线路径本就是 int64。
    if topk_ids.dtype not in (torch.int32, torch.int64):
        topk_ids = topk_ids.to(torch.long)
    topk_ids = topk_ids.contiguous()
    lo = int(topk_ids.min())
    hi = int(topk_ids.max())
    if lo < 0 or hi >= E:
        raise ValueError(f"topk_ids 取值 [{lo}, {hi}] 超出 [0, {E})（reference.py:77 同款断言）")

    # ---------- layout 归一（只做无拷贝/纯布局操作，不改数值）----------
    a = a.contiguous()
    w1 = w1.contiguous()                      # kernel 按 stride(0,1,2) 读 W1
    w2 = w2.contiguous()                      # kernel 按 stride(0,2,1) 读 W2
    topk_weights = topk_weights.contiguous()  # 官方入口断言 stride(1) == 1（module:623）

    num_valid = M * top_k                     # 有效槽位数 = topk_ids.numel()

    # ---------- 调度三件套：交给 aiter 官方 align 入口构造 ----------
    # 官方 wrapper 的缓冲尺寸（test_moe_align_block_size.py:112-121）与哨兵约定：
    # sorted_token_ids 全域预填 numel（module:188 的 token_mask 靠它屏蔽填充槽位）。
    L1 = num_valid + E * (_BLOCK_SIZE_M - 1)
    L2 = -(-L1 // _BLOCK_SIZE_M)
    sorted_token_ids = torch.full(
        (L1,), num_valid, dtype=torch.int32, device=a.device
    )
    # 官方 wrapper 用 torch.empty：stage4 只会写前 cdiv(num_tokens_post_pad, BM) 项，
    # 未写区不会被 e2e kernel 读（pid_m*BM >= num_tokens_post_padded 直接 return，
    # module:184 / persistent module:398 的 num_pid_m 上界）。这里用 0 预填作纯防御：
    # 万一被读到也是合法专家号，不会越界访问 W1/W2。
    expert_ids = torch.zeros((L2,), dtype=torch.int32, device=a.device)
    num_tokens_post_padded = torch.empty((1,), dtype=torch.int32, device=a.device)
    moe_align_block_size_triton(
        topk_ids,
        E,
        _BLOCK_SIZE_M,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )

    # ---------- 计算缓冲（与官方测试同款）----------
    # Intermediate：persistent 支第二段 GEMM 的 fp32 累加缓冲，必须零初始化
    # （module:513 对它 atomic_add）；shape (M*top_k, N//2)、fp32
    # （test_moe_e2e.py:365-369）。
    intermediate = torch.zeros(
        (num_valid, N // 2), dtype=torch.float32, device=a.device
    )
    # C：输出，persistent 支由 tl.store 全覆盖写（module:596），官方测试同样用
    # zeros（test_moe_e2e.py:291）。dtype = 题面输出 dtype ⇒ kernel 的 compute
    # dtype 也就是它（module:419）。
    out = torch.zeros((M, top_k, K), dtype=a.dtype, device=a.device)

    # ---------- 走官方测试的 persistent 配置 ----------
    moe_set_use_persistent_kernel(True)  # test_moe_e2e.py:361

    result = e2e_moe(
        A=a,
        W1=w1,
        W2=w2,
        Intermediate=intermediate,
        C=out,
        A_scale=None,          # 非量化（test_moe_e2e.py:279-281）
        W1_scale=None,
        W2_scale=None,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        topk_ids=topk_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=True,   # 本题语义恒启用（reference.py:29/92）
        top_k=top_k,
        use_fp8_w8a8=False,
        use_int8_w8a16=False,
        config=dict(_PERSISTENT_CONFIG),
    )
    torch.cuda.synchronize()

    if result is None:
        raise RuntimeError("e2e_moe 返回 None：本适配器走 persistent 分支，应返回 C")
    if tuple(result.shape) != (M, top_k, K) or result.dtype != a.dtype:
        raise RuntimeError(
            f"e2e_moe 返回 {tuple(result.shape)}/{result.dtype}，与题面输出契约 "
            f"({M}, {top_k}, {K})/{a.dtype} 不符"
        )

    ctx = {
        "aiter_path": "aiter.ops.triton.moe_op_e2e.e2e_moe -> "
                      "e2e_moe_persistent_kernel (persistent=True)",
        "dispatch_path": "aiter.ops.triton.moe_align_block_size.moe_align_block_size_triton",
        "compute_dtype": str(a.dtype),
        "config": dict(_PERSISTENT_CONFIG),
        "mul_routed_weight": True,
        "quant": "none (fp8_w8a8=False, int8_w8a16=False, scales=None)",
        "shape": {
            "M": M, "K": K, "E": E, "N": N, "hidden": N // 2, "top_k": top_k,
            "num_valid_tokens": num_valid,
            "sorted_token_ids_len": L1,
            "expert_ids_len": L2,
        },
        "out_shape": tuple(result.shape),
        "out_dtype": str(result.dtype),
        "precision_caveat": (
            "kernel 在第二段 GEMM 前把 h cast 到 compute dtype"
            "（moe_op_e2e.py:261/:505）；本题 reference 为 fp32 全程累加 + 末尾 cast 一次，"
            "故 bf16 档下 aiter 原生基线与 reference 的偏差在抵消点会被 atol 主导"
            "（本机忠实仿真：固定族 32768 元素最坏 diff/容差 = 13.0/43.7），"
            "record_baseline 在 task.yaml 现有 2e-2 容差下会被拒绝。详见文件头。"
        ),
    }
    return result, ctx
