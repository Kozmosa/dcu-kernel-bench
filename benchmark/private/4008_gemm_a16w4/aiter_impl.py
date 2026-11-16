# aiter_impl.py — 4008_gemm_a16w4 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) —— 按名取参（构造参数可能
# 稀疏给出，位置式取值会错位，见 audit_model_class.py 的 case_init_kwargs）。
#
# ── 来源（aiter pinned 检出 .dcu_runs/aiter_pinned，commit c39fff8c）─────────
#   文件 : aiter/ops/triton/gemm_a16w4.py
#   sha256: 8c47decb110ac484f9bf84f3827f446cf4b688d172ccf076a7c8a5dd4768285b
#           （与 benchmark/sources/4008_gemm_a16w4.yaml 的准入证据逐字一致）
#   入口 : gemm_a16w4（:696-734）
#   device kernel: awq_gemm_kernel（:557-571，按 SCHEDULER 分派 splitk/streamk）
#                  → awq_gemm_kernel_splitk（:447-467）/ awq_gemm_kernel_streamk
#                  （:413-445）→ awq_gemm_kernel_inner（:248-411，int4 解包 +
#                  分组反量化 + tl.dot(out_dtype=float32)）
#   宿主 : awq_gemm_triton_impl（:812-890，分配输出 / 建 grid / 必要时归约）
#
#   入口签名（逐字，源码 :696-701）：
#     gemm_a16w4(
#         input:   torch.Tensor,          # [M, K]          float16，行主连续
#         qweight: torch.Tensor,          # [N, K // 2]     int8（沿 K 每字节 2 个
#                                         #                 4 位码，低半字节 = 偶数 k）
#         scales:  torch.Tensor,          # [K // G, N]     float16
#         qzeros:  torch.Tensor,          # [K // G, N//2]  int8（沿 N 每字节 2 个
#                                         #                 4 位码，低半字节 = 偶数 n）
#         use_fused_kernel: int = 0,      # 源码 :730-732 已注释掉，实际恒走非融合路径
#         configs: Optional[Dict] = None, # None -> 宿主自己读 tuned JSON（见下）
#     ) -> torch.Tensor                   # [M, N] float16
#
#   官方调用约定（op_tests/triton_tests/test_gemm_a16w4.py:226-227，sha256
#   452e0c70b5e29ecec8edd6361477d4f72b6566c46181a590d56093e0de26a944）：
#     qweight_repack, qzeros_repack = awq_reorder_and_repack(qweight, qzeros)
#     output_triton = gemm_a16w4(input, qweight_repack, scales, qzeros_repack)
#   即官方测试与本题题面（reference.py::make_inputs 用同一套 AWQ 重排/打包，
#   reference.py:24-52 与源码 :54-118 逐行同构）传入的是**同一套布局**，故本题
#   适配器无需任何 permute / reshape / repack。
#
# ── 布局对齐（本题零转换）──────────────────────────────────────────────────
#   aiter 入口吃的正是题目 io 声明的 4 个张量，顺序也一致
#   （reference.forward(input, qweight, scales, qzeros)）：
#     input   [M, K]         -> 直接传（kernel 按 K*offs_m + offs_k 做行主寻址，
#                               :310，要求最后一维 stride == 1，显式 contiguous 兜住）
#     qweight [N, K//2]      -> 直接传（宿主 :821 assert qweight.is_contiguous()）
#     scales  [K//G, N]      -> 直接传（:353 offsets_s = N*offs_szk + offs_sn）
#     qzeros  [K//G, N//2]   -> 直接传（:340 offsets_z = N2*offs_szk + offsets_zn，
#                              其中 offsets_zn = (tile 内 int4 下标) // 2，配合 kernel
#                              内 zshifts = (offsets_bn % 2) * 4 解包，与题面 reference
#                              的 (qzeros >> n_shift) & 0xF 语义一致）
#   输出由宿主分配（:856 torch.empty((M, N), fp16) 或 :852/:854 zeros + 归约），
#   连续、dtype 恒为 float16（:883-890：3 维 splitk 结果走 awq_reduce_and_convert_triton
#   也返回 fp16；2 维结果 result.to(torch.float16)）。与题面 io.outputs 同形同 dtype，
#   reference 返回的是单个未打包张量，适配器无需打包。
#
# ── 构造参数（init_kwargs）的用法 ──────────────────────────────────────────
#   题面 Model.__init__(in_features, out_features, group_size) 的三个参数**仅作
#   实例元数据**，reference.forward 一律以运行期张量实际形状为准
#   （reference.py:81-83；组大小由 G = K // qzeros.shape[0] 现推）。
#   适配器同样以 shape 为准，并且**不能**拿 in_features/out_features 做校验：
#   audit_model_class.py::case_init_kwargs 会用 get_init_inputs()=[2048,1536,64]
#   补齐 case 未给的同名参数，而 private 的 perf/hidden case 形状族包含
#   m=4096,k=4096（perf）、n=1024/n=768/n=512（hidden）等，补齐值与实际 shape
#   必然不一致——这是该函数的既定行为，不是错误。
#   可校验的只有 group_size：所有 case 都显式给了 group_size，make_inputs 也按
#   qzeros=(k//group_size, n//8) 生成，故声明值与 shape 推导值必须相等，不等即 raise。
#
# ── autotune config 依赖（needs_autotune_config = True）────────────────────
#   gemm_a16w4 在 configs=None 时调 get_w4a16_awq_gemm_configs(N, K, GROUP_SIZE)
#   （:723-725 调用点，:667-689 定义），后者读
#     `{AITER_TRITON_CONFIGS_PATH}/gemm/awq_w4a16/`
#     `awq_gemm_N={N},K={K},device_name={dev},dtype=w4a16,group_size={G}.json`
#   （:653-665），dev 由 arch_info.get_device() 给出，arch_info.py:5-10 把 gfx936
#   映射为 "BW200"（gfx938 -> "BW200B"）。**文件不存在不报错**：:680-689 打一条
#   logger.warning 后返回 None，gemm_a16w4 退回到源码内置的 default_config
#   （:708-722，BLOCK_SIZE_M=16 / N=128 / K=32 / SPLITK=1 / USE_REDUCE_KERNEL=False /
#   D_DTYPE=16）——数值语义不变，只是**性能退化**。
#   本题 3 个 perf case 的命中情况（对 pinned 检出 aiter/ops/triton/configs/gemm/
#   awq_w4a16/ 逐名核对）：
#     perf_prefill_m4096_k4096_n8192_g64 → N=8192,K=4096 **缺文件**（无任何 N=8192 文件）
#     perf_decode_m128_k7168_n7168_g64  → N=7168,K=7168 命中
#     perf_mid_m512_k2048_n1536_g64     → N=1536,K=2048 命中
#   故缺文件的 case 会走内置 default_config，其 us 是"未调优 aiter"的口径。
#   本适配器**不自造 config**（那样记下的就不是 aiter 官方口径）：把探测结果如实
#   写进 ctx["config_probe"] / ctx["config_file"]，绝不静默换参。
#
# ── 输出对齐（与 reference 的差异是算子语义，已在 task.yaml 容差中裁定）────
#   kernel 在 tl.dot 前把反量化权重 cast 到 a_ptr 的元素类型（:383
#   b = b.to(a_ptr.type.element_ty)，即 fp16），而 reference 全程 fp32 累加；
#   该 fp16-W 舍入即 task.yaml tolerance=2.0e-1 的来由（见 task.yaml 的 note 与
#   sources/4008 的准入记录实测）。

import os

import torch

# 与 aiter/ops/triton/gemm_a16w4.py:29 AWQ_TRITON_SUPPORTED_GROUP_SIZES 一致
# （源码 :829 允许 `group_size in [-1, 32, 64, 128] or group_size == K`）
_AWQ_SUPPORTED_GROUP_SIZES = (-1, 32, 64, 128)


def _probe_config_file(N, K, group_size):
    """只读探测 gemm_a16w4 会去读的 tuned config 文件（不构造、不注入 config）。

    返回 (probe, fpath, device_name)；probe ∈
      "official_autotune_json"  —— 文件在，宿主会用官方 tuned 配置；
      "aiter_default_config"    —— 文件不在，宿主退到源码内置 default_config；
      "probe_failed(<Exc>)"     —— 探测本身失败（如无 triton driver），无结论。
    aiter 一律函数内 import（顶层 import 很重，真机部署走最小导入垫片）。
    """
    try:
        import aiter.ops.triton.utils.arch_info as arch_info  # noqa: PLC0415
        from aiter.ops.triton.gemm_a16w4 import (  # noqa: PLC0415
            get_w4a16_awq_gemm_config_filepath,
        )

        device_name = arch_info.get_device()  # gfx936 -> "BW200"、gfx938 -> "BW200B"
        fpath = get_w4a16_awq_gemm_config_filepath(N, K, group_size)
        probe = "official_autotune_json" if os.path.exists(fpath) else "aiter_default_config"
        return probe, fpath, device_name
    except Exception as exc:  # pragma: no cover - 仅探测，失败不影响调用
        return f"probe_failed({type(exc).__name__})", None, None


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（aiter.ops.triton.gemm_a16w4.gemm_a16w4）。

    inputs     : [input, qweight, scales, qzeros]（顺序同 reference.make_inputs）
                 input   [M, K]         float16
                 qweight [N, K//2]      int8（低半字节 = 偶数 k）
                 scales  [K//G, N]      float16
                 qzeros  [K//G, N//2]   int8（低半字节 = 偶数 n）
    init_kwargs: {"in_features": int, "out_features": int, "group_size": int}
                 （Model.__init__(in_features, out_features, group_size)；
                  前两个仅作实例元数据，见文件头，不参与校验）
    device     : 目标设备（张量已在 device 上，仅用于 ctx 记录）

    返回 (out, ctx)；out 为 [M, N] 连续 float16 张量，与 reference.forward 同形
    同 dtype（reference 返回单个未打包张量，本题无需打包）。
    """
    from aiter.ops.triton.gemm_a16w4 import gemm_a16w4  # noqa: PLC0415

    if len(inputs) != 4:
        raise ValueError(
            f"4008 需要 4 个输入 [input, qweight, scales, qzeros]，实际 {len(inputs)} 个"
        )
    x, qweight, scales, qzeros = inputs

    # ---- 构造参数（按名取参；越界一律 raise，绝不静默用错）-------------------
    declared_group = init_kwargs.get("group_size")
    # 源码 :730-732 的融合路径已被注释掉（use_fused_kernel 是非融合入口的 inert
    # 形参），传非 0 只会让人以为走了融合 kernel。
    use_fused_kernel = int(init_kwargs.get("use_fused_kernel", 0))
    if use_fused_kernel != 0:
        raise ValueError(
            f"use_fused_kernel={use_fused_kernel} 不在本题范围：gemm_a16w4 的融合"
            "分支在源码 :730-732 已被注释掉，官方入口恒走 awq_gemm_kernel"
            "（awq_gemm_kernel_fused 的 streamk 分支更是 assert False，:600-602）"
        )

    # ---- shape / dtype 校验（题目 io 与不变式；不满足则 raise，不猜）--------
    if x.dim() != 2 or qweight.dim() != 2 or scales.dim() != 2 or qzeros.dim() != 2:
        raise ValueError(
            "input/qweight/scales/qzeros 必须都是 2 维，实际 "
            f"{tuple(x.shape)}/{tuple(qweight.shape)}/{tuple(scales.shape)}/{tuple(qzeros.shape)}"
        )
    M, K = x.shape
    N = qweight.shape[0]
    if min(M, N, K) < 1:
        raise ValueError(f"M/N/K 必须 >= 1，实际 {(M, N, K)}")
    if qweight.shape[1] != K // 2:
        raise ValueError(
            f"qweight 应为 [N, K//2]=[{N}, {K // 2}]（宿主 :825 断言），实际 {tuple(qweight.shape)}"
        )
    if qzeros.shape[0] < 1 or K % qzeros.shape[0] != 0:
        raise ValueError(
            f"qzeros.shape[0]={qzeros.shape[0]} 必须整除 K={K}（G = K // qzeros.shape[0]）"
        )
    group_size = K // qzeros.shape[0]
    if tuple(scales.shape) != (K // group_size, N):
        raise ValueError(
            f"scales 应为 [K//G, N]=[{K // group_size}, {N}]（宿主 :827 断言），实际 {tuple(scales.shape)}"
        )
    if tuple(qzeros.shape) != (K // group_size, N // 2):
        raise ValueError(
            f"qzeros 应为 [K//G, N//2]=[{K // group_size}, {N // 2}]（宿主 :826 断言），实际 {tuple(qzeros.shape)}"
        )
    if not (group_size in _AWQ_SUPPORTED_GROUP_SIZES or group_size == K):
        raise ValueError(
            f"由 shape 推出的 group_size={group_size} 不在 aiter 白名单 "
            f"{_AWQ_SUPPORTED_GROUP_SIZES} 内，且 != K（源码 :829）"
        )
    if declared_group is not None and int(declared_group) != group_size:
        raise ValueError(
            f"init_kwargs['group_size']={declared_group} 与 shape 推导的 G={group_size}"
            f"（K={K} // qzeros.shape[0]={qzeros.shape[0]}）不一致——"
            "make_inputs 保证二者相等，不一致说明输入不是本题题面的张量"
        )
    if x.dtype != torch.float16:
        raise ValueError(
            f"input 必须是 float16（io.inputs 声明；输出 dtype 由它决定），实际 {x.dtype}"
        )

    # ---- dtype / 布局规整（允许的 torch 用途：layout 转换与打包）-------------
    # 四个张量都已在题面声明的行主布局上，这里只做 contiguous 兜底（宿主 :821
    # 对 qweight 有硬断言，其余三处按行主 stride 寻址）。
    # 码字类张量的 .to(int8)：题面 make_inputs 产出的就是 int8（含高半字节置位后的
    # 负值）；若上层把 int8 升成 fp32/int32，取值 |v| <= 128，回 cast 无损。
    x = x.contiguous()
    qweight = qweight.to(torch.int8).contiguous()
    qzeros = qzeros.to(torch.int8).contiguous()
    scales = scales.to(torch.float16).contiguous()

    # ---- autotune config 依赖（只探测，不自造）------------------------------
    config_probe, config_fpath, device_name = _probe_config_file(N, K, group_size)

    # ---- 调用官方入口（configs=None：宿主按 N/K/G/device 自己找 tuned JSON）--
    out = gemm_a16w4(
        x,
        qweight,
        scales,
        qzeros,
        0,      # use_fused_kernel：本题恒 0（融合分支在源码里已注释掉）
        None,   # configs：None -> get_w4a16_awq_gemm_configs(N, K, G) / 内置 default
    )
    torch.cuda.synchronize()

    # ---- 输出契约自检（与题面 io.outputs / reference 对齐）------------------
    if tuple(out.shape) != (M, N):
        raise RuntimeError(f"aiter 输出 shape {tuple(out.shape)} != reference 的 {(M, N)}")
    if out.dtype != x.dtype:
        raise RuntimeError(
            f"aiter 输出 dtype {out.dtype} != input dtype {x.dtype}"
            "（宿主 :883-890 恒返回 float16；不一即说明 config/路径异常）"
        )

    return out, {
        "path": "aiter.ops.triton.gemm_a16w4",
        "entry": "gemm_a16w4",
        "layout": "AWQ 打包布局原样传入（qweight [N,K//2] / qzeros [K//G,N//2] int8，低半字节=偶数下标），零 permute/repack",
        "device": str(device),
        "M": M,
        "N": N,
        "K": K,
        "group_size": group_size,
        "num_groups_tiles": K // group_size,
        "out_shape": tuple(out.shape),
        "out_dtype": str(out.dtype),
        "use_fused_kernel": 0,
        "scheduler": "由 config 决定（SCHEDULER=0 走 awq_gemm_kernel_splitk，SPLITK=1 即 DP；SCHEDULER=1 为 streamk，本题 configs 未使用）",
        "config_probe": config_probe,
        "config_file": config_fpath,
        "device_name": device_name,
        "declared_init_kwargs": {
            "in_features": init_kwargs.get("in_features"),
            "out_features": init_kwargs.get("out_features"),
            "group_size": declared_group,
        },
    }
