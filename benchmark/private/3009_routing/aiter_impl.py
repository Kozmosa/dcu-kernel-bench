# aiter_impl.py — 3009_routing 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
#
# 适配器契约（按名取参，见 audit_model_class.py::case_init_kwargs 的说明）：
#     run(inputs, init_kwargs: dict, device) -> (out, ctx)
#
# ── 来源 ────────────────────────────────────────────────────────────────────
# aiter 本地 pinned 检出：.dcu_runs/aiter_pinned/
# commit c39fff8c77df4e80617649e92fa3c2615f2c43d1
#   aiter/ops/triton/routing.py
#     sha256 c911cac58a77d7bcc933c66c9cceb883dc9f7bda759963a1f645032b84839671
#   op_tests/triton_tests/test_routing.py（语义唯一权威）
#     sha256 def2b9cf549055a31b702362f33b121ffc6d868791ae34a43655325049df25f5
#   两个 sha256 均已用 Get-FileHash 与本机 pinned 检出核对，且与
#   benchmark/sources/3009_routing.yaml 记录一致。
#
# ── 公开入口（host 侧）──────────────────────────────────────────────────────
#   from aiter.ops.triton.routing import routing_sigmoid_top1
#
#   routing_sigmoid_top1(x, w, topk, fused_shared_experts=False)
#       -> (topk_ids: int32 [M, _topk], topk_weights: fp32 [M, _topk])   # routing.py:186-237
#     x: [M, K] 连续（内部先做 x.view(-1, x.shape[-1])，故更高维也可，本题恒 2D）
#     w: [K, N] 连续；要求 M/K/N >= 1 且 K == w.shape[0]（routing.py:191 assert topk == 1）
#     _topk = topk + (1 if fused_shared_experts else 0)，本题 topk 恒为 1
#     ⇒ 输出两路：ids int32 [M, C] 与 weights fp32 [M, C]，C = 1 + fused
#
#   官方测试的唯一调用方式（test_routing.py:28）与本适配器一致：
#       routing_sigmoid_top1(x, w, TOPK=1, fused_shared_experts=True)
#   官方测试同时给出融合列的对照值：dummy_ids = ones*N、dummy_weights = ones
#   （test_routing.py:22-23），与 kernel 的写回（routing.py:158-165）及题面
#   reference 的融合列（id=N、权重 1.0）逐位一致。
#
# ── 打包协议（题面单张量契约）───────────────────────────────────────────────
# reference.py:81 把两路输出打包成单个 fp32 张量：
#     out = cat([topk_ids.to(fp32), topk_weights.to(fp32)], dim=1)  # [M, 2*C]
# 本适配器复刻同一打包（id 列在前、权重列在后；ids 由 int32 精确转 fp32），
# 否则 record_baseline 会因 shape/dtype 不符拒绝记录基线。
#
# ── layout ─────────────────────────────────────────────────────────────────
# 本题输入就是 aiter 期望的 2D contiguous [M, K] / [K, N]，无需 permute/reshape；
# aiter 用显式 stride 寻址（routing.py:220-227），contiguous() 只是把 strides 归
# 一到最简形态（对已连续张量为 no-op）。唯一用到的 torch 操作是连续性保证、
# int32→fp32 精确提升与最终 cat 打包，核心计算（tl.dot fp32 累加 / tl.sigmoid /
# tl.argmax(tie_break_left=True) / 融合列写回）全部在 aiter 的 Triton kernel 内。
#
# ── init_kwargs ────────────────────────────────────────────────────────────
# io.init_inputs 只声明 fused_shared_experts（task.yaml:32-33），取值用 .get 带
# 默认值（与 reference.py:55 的 Model 默认值 True、官方测试恒 True 一致）；
# 出现任何未声明的构造参数即 raise（不静默忽略）。
#
# ── 合法性校验：越界就 raise（本文件不静默用错参数）────────────────────────
#   · dtype 必须是 fp16/bf16（task.yaml io.inputs; 官方 kernel 直接对原 dtype
#     做 tl.dot，fp32 输入会落到 tf32 通路，与题面「fp32 累加」参考链不等价）
#   · K == w.shape[0]（routing.py:196 assert K == Kb）
#   · N ∈ {16, 128}：aiter 的 get_config_heuristic 用 configs[N][m_bucket] 查表
#     （routing.py:27-44），且 routing.py:228 把 BLOCK_N 直接设为 N 供
#     tl.arange(0, BLOCK_N) 使用（Triton 要求 2 的幂）。这是官方 wrapper 的
#     实现限制而非语义限制（见 task.yaml:41 的 invariants），故对 N=24 这类
#     题面允许但官方 wrapper 不支持的取值，本适配器明确 raise 而不是硬凑。
#     本题的 3 个 perf case（private/3009_routing/perf_cases.json）N ∈ {16,128}、
#     M ∈ {256,4096,8192}、K=5120，全部落在查表范围内。
#
# ── 未采用的相邻入口 ────────────────────────────────────────────────────────
# aiter 另有一份同算子的 autotune 重复实现
# aiter/ops/triton/moe_routing_sigmoid_top1_fused.py（同名 host 入口），但其
# _get_config 必须读取外部 JSON
#   f"{AITER_TRITON_CONFIGS_PATH}/moe/{dev}-MOE_ROUTING_SIGMOID_TOPK1.json"
# （moe_routing_sigmoid_top1_fused.py:120-126，config=None 时无内存默认值），
# 缺文件即 FileNotFoundError；且它同样把 BLOCK_N 设为 N。故本题以准入记录
# 指定的 routing.py（固定 config 查表版，不依赖任何外部 config 文件）为唯一
# 路径，不引入 autotune/config 依赖。

import torch

# routing.py:27-44 的 configs 表键（get_config_heuristic 只覆盖这两个 N）
_HEURISTIC_N_CHOICES = (16, 128)

# reference.py:55 / test_routing.py:28：官方测试恒开融合共享专家列
_FUSED_SHARED_EXPERTS_DEFAULT = True

# 准入记录声明的构造参数集合（task.yaml io.init_inputs）
_DECLARED_INIT_KEYS = ("fused_shared_experts",)


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方 routing_sigmoid_top1（sigmoid top-1 路由 + 融合共享专家列）。

    inputs     : [x [M, K] fp16/bf16 连续, w [K, N] 同 dtype 连续]（已在 device 上）
    init_kwargs: {"fused_shared_experts": bool}（缺省 True，同 reference Model 默认值）
    返回        : (out fp32 [M, 2*C], ctx)，C = 1 + fused_shared_experts；
                  out[:, :C] = topk_ids（整数以 fp32 表示）、out[:, C:] = topk_weights
    """
    from aiter.ops.triton.routing import get_config_heuristic, routing_sigmoid_top1

    # ── init_kwargs：按名取参 + 未声明键 raise ──────────────────────────────
    init_kwargs = dict(init_kwargs or {})
    undeclared = sorted(set(init_kwargs) - set(_DECLARED_INIT_KEYS))
    if undeclared:
        raise ValueError(
            f"3009_routing 只声明了构造参数 {list(_DECLARED_INIT_KEYS)}"
            f"（task.yaml io.init_inputs），收到未声明的 init_kwargs={undeclared}"
        )
    fused_shared_experts = bool(
        init_kwargs.get("fused_shared_experts", _FUSED_SHARED_EXPERTS_DEFAULT)
    )

    # ── 输入解包与形状/dtype 校验 ──────────────────────────────────────────
    if len(inputs) != 2:
        raise ValueError(f"3009_routing 期望 2 个输入 (x, w)，收到 {len(inputs)} 个")
    x, w = inputs
    if not isinstance(x, torch.Tensor) or not isinstance(w, torch.Tensor):
        raise TypeError("3009_routing 的输入必须是 torch.Tensor")
    if x.dim() != 2 or w.dim() != 2:
        raise ValueError(
            f"aiter routing_sigmoid_top1 期望 2D 输入，收到 x{tuple(x.shape)} w{tuple(w.shape)}"
        )

    M, K = int(x.shape[0]), int(x.shape[1])
    Kb, N = int(w.shape[0]), int(w.shape[1])
    if K != Kb:
        raise ValueError(f"x 的最后一维 {K} 必须等于 w 的行数 {Kb}（routing.py:196）")
    if M < 1 or K < 1 or N < 1:
        raise ValueError(f"要求 M/K/N >= 1，收到 M={M} K={K} N={N}")
    if x.dtype != w.dtype:
        raise ValueError(f"x/w dtype 必须一致，收到 {x.dtype} / {w.dtype}")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"aiter routing_sigmoid_top1 的 kernel 直接对输入 dtype 做 tl.dot，本题域为 "
            f"fp16/bf16（task.yaml io.inputs）；收到 {x.dtype}"
        )
    if N not in _HEURISTIC_N_CHOICES:
        raise ValueError(
            f"aiter routing.py 的 get_config_heuristic 只覆盖 N ∈ {list(_HEURISTIC_N_CHOICES)}"
            f"（routing.py:27-44 的 configs 表；routing.py:228 又把 BLOCK_N 设为 N，Triton 的 "
            f"tl.arange 要求 2 的幂），收到 N={N}：该 shape 超出官方 wrapper 的实现范围，"
            f"不静默改用其它 config"
        )

    # 连续性保证（aiter 用显式 stride 寻址；已连续时为 no-op）
    x_c = x.contiguous()
    w_c = w.contiguous()

    # ── 调用 aiter 官方入口（topk 恒为 1，见 routing.py:191 assert topk == 1）──
    topk_ids, topk_weights = routing_sigmoid_top1(
        x_c, w_c, 1, fused_shared_experts=fused_shared_experts
    )

    C = 1 + int(fused_shared_experts)
    if tuple(topk_ids.shape) != (M, C) or topk_ids.dtype != torch.int32:
        raise RuntimeError(
            f"aiter routing_sigmoid_top1 的 topk_ids 异常：shape={tuple(topk_ids.shape)} "
            f"dtype={topk_ids.dtype}，期望 ({M}, {C}) / torch.int32"
        )
    if tuple(topk_weights.shape) != (M, C) or topk_weights.dtype != torch.float32:
        raise RuntimeError(
            f"aiter routing_sigmoid_top1 的 topk_weights 异常：shape={tuple(topk_weights.shape)} "
            f"dtype={topk_weights.dtype}，期望 ({M}, {C}) / torch.float32"
        )

    # ── 打包成题面单张量契约：out = [ids(int→fp32) | weights] fp32 [M, 2*C]
    #    （reference.py:81；id 列数值 < 2^24，int32 → fp32 精确无损）
    out = torch.cat(
        [topk_ids.to(torch.float32), topk_weights.to(torch.float32)], dim=1
    ).contiguous()

    # 记录本次实际派发的启发式 config（routing.py:6-58），作为基线口径证据
    cfg = get_config_heuristic(M, K, N)
    m_bucket = (
        "very_large"
        if M >= 8192
        else "large" if M >= 4096 else "medium" if M >= 2048 else "small"
    )

    torch.cuda.synchronize()

    ctx = {
        "impl": "aiter",
        "path": (
            "aiter.ops.triton.routing.routing_sigmoid_top1"
            "（固定 config 简洁版；get_config_heuristic 查表派发，无外部 config 文件）"
        ),
        "aiter_module": "aiter.ops.triton.routing",
        "aiter_symbol": "routing_sigmoid_top1",
        "commit": "c39fff8c77df4e80617649e92fa3c2615f2c43d1",
        "source_sha256": "c911cac58a77d7bcc933c66c9cceb883dc9f7bda759963a1f645032b84839671",
        "M": M,
        "K": K,
        "N": N,
        "topk": 1,
        "fused_shared_experts": bool(fused_shared_experts),
        "C": C,
        "in_dtype": str(x.dtype),
        "m_bucket": m_bucket,
        "heuristic_config": {
            **cfg.kwargs,
            "num_warps": cfg.num_warps,
            "num_stages": cfg.num_stages,
            "num_ctas": cfg.num_ctas,
        },
        "topk_ids_shape": tuple(topk_ids.shape),
        "topk_weights_shape": tuple(topk_weights.shape),
        "out_shape": tuple(out.shape),
        "packing": "cat([topk_ids.to(fp32), topk_weights], dim=1) → [M, 2*C]",
        "device": str(device),
    }
    return out, ctx
