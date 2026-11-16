# aiter_impl.py — 1020_mla_decode_rope 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约：run(inputs, init_kwargs: dict, device) —— 按名取参（构造参数在 case
# 里是稀疏给出的，位置式取值会错位，见 audit_model_class.case_init_kwargs 的说明）。
#
# 来源（aiter pinned 检出 .dcu_runs/aiter_pinned/，commit
# c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#   device_kernel  : aiter/ops/triton/mla_decode_rope.py
#                    sha256 4e818ae25333e14451d7420e2da1e9f29d84fb4c44f60bf64a5ab362e60fe3cd
#                    （两段式：_fwd_grouped_kernel_stage1_rope 逐 (batch, head 块,
#                      KV 切片) 在线 softmax 并写回旋转后的 k_pe，_fwd_kernel_stage2
#                      以 log-sum-exp 归并各 KV 切片。核心计算全在源文件内）
#   official_test  : op_tests/triton_tests/test_mla_decode_rope.py
#                    sha256 d6f50e3a554fd28f627831563729fdeaba7926a6585a76c474a2105396b51837
#                    （语义权威；test_op_fwd_rope / test_op_fwd_rope_neox /
#                      test_op_fwd_rope_integration，含 PyTorch 参考实现
#                      ref_compute_full_fwd 与 ref_preprocess）
#
# 公开入口（宿主函数，mla_decode_rope.py:477-553）：
#
#   decode_attention_fwd_grouped_rope(
#       q, k_buffer, v_buffer, o, kv_indptr, kv_indices, k_pe_tokens,
#       kv_lora_rank, rotary_dim, cos_sin_cache, positions, attn_logits,
#       num_kv_splits, sm_scale, logit_cap=0.0, use_rope=False,
#       is_neox_style=False, config=None) -> None
#
#   **返回值是 None**：真正的输出 o（[B, H, c]）与 k_pe_tokens（[B, r]）由调用方
#   预分配、kernel 就地写入（mla_decode_rope.py:526-553；stage1 的
#   tl.store(Att_Out ...) :284 / tl.store(k_pe_t_out ...) :282，stage2 的
#   tl.store(O ...) :427）。官方测试的 integration 用例正是这样调用的
#   （test_mla_decode_rope.py:443-466）。
#
# 布局（题面 io 与 aiter 期望形态逐项对齐，只需一次 unsqueeze/narrow）：
#   - q [B, H, c+r] 无转换：kernel 用 stride_qb/stride_qh + 列偏移取址，
#     nope 部分在列 [0, c)、rope 部分在列 [c, c+r)（:86-103、:87 的 offs_qk_r），
#     与题面 q = [Q_NOPE; Q_PE] 完全一致。
#   - k_buffer **必须**是 3 维 [N, 1, c+r]、v_buffer [N, 1, c]：宿主用
#     `kv_group_num = q.shape[1] // k_buffer.shape[1]` 推 GQA 分组（:331），
#     MLA 所有 query head 共用同一份逐 token 压缩 KV，即 k_buffer.shape[1] == 1
#     （官方 ref_preprocess 就是 `latent_cache.unsqueeze(1)`，测试 :68）。
#     kernel 只用 k_buffer.stride(0) 做行步长（:356），故 [N, c+r] 的 kv_cache
#     直接 unsqueeze(1) 即可；v_buffer 取官方同款 `kv_cache[:, :c]` 连续视图
#     （:66-67 的 v_input）——题面不变式「V ≡ kv_cache[:, :c]」正对应这条。
#   - cos_sin_cache [max_positions, rd]：kernel 按 stride(0) 取行、行内前 rd/2 列
#     当 cos、后 rd/2 列当 sin（:146-158 的 offs_rotary 与 +rotary_dim//2），与
#     题面布局逐位一致，无需转置。
#   - positions [B] int32 / kv_indptr [B+1] int32 / kv_indices [total_kv] int32：
#     kernel 按 positions.stride(0) 逐 batch 取当前位置（:145），按
#     kv_indptr[cur_batch]、kv_indptr[cur_batch+1] 取本批 KV 逻辑区间（:97-98），
#     再经 kv_indices 间接寻址物理行（:185-187、:206-210）——题面的 page size=1
#     行池 + 垃圾行语义天然成立。
#   - 中间缓存 attn_logits = [B, H, num_kv_splits, c+1]（最后一列放
#     e_max + log(e_sum)），dtype 同输入（官方测试 :42-44、:111-113 同款）。
#
# 语义对齐（与 sources/1020_mla_decode_rope.yaml 的准入说明一致）：
#   - RoPE：NEOX 用 (i + rd/2) % rd 取配对维、前 rd/2 个索引取负（:116-122），
#     GPT-J 用 ((i+1)%2)*2-1+i 取相邻奇偶配对、偶数索引取负（:125-133）；
#     输出 = x*cos + x_rot*sin，与 reference.py:103-116 的
#     o1 = x1*cos - x2*sin / o2 = x2*cos + x1*sin 同式（rotary_dim 之外的维
#     cos=1/sin=0 直通，:135-143）。
#   - 只旋转每个序列**最后一个逻辑 token** 的 k_pe（LAST_SPLIT + kv_loc =
#     kv_indices[start + seq_len - 1]，:178-197、:221-226），其余 token 用行池原值，
#     与 reference.py:133 逐行一致。
#   - 打分 = sm_scale * (q_nope·k_lat + q_pe'·k_pe')，softmax 在线（fp32 累加），
#     V 取该行前 c 列（:230-266），与 reference.py:136-140 一致；无 logit 封顶
#     （logit_cap=0.0）、无因果掩码、无量化。
#   - num_kv_splits 只影响 KV 流程切分（切片内在线 softmax + stage2
#     log-sum-exp 归并），数学等价于全量 softmax —— 题面不变式已显式声明。
#     取值 2 = 官方测试全部三个参数化用例的默认（test_mla_decode_rope.py:213、
#     :315、:420），也是唯一有官方用例覆盖的取值。
#
# 输出打包：reference.forward 返回单个 1-D 平铺张量
#   out = cat(attn_out.reshape(-1), k_pe_tokens.reshape(-1))
#   （reference.py:141-142，长度 B*H*c + B*r，dtype 同输入）。适配器按**同一顺序**
#   把两个就地写好的缓冲展平后 cat，形状/dtype/元素序与 reference 逐位对应。
#
# autotune config 依赖（needs_autotune_config=True）：宿主在 config=None 时读
#   AITER_TRITON_CONFIGS_PATH/{dev}-MLA_DECODE_ROPE-DEFAULT.json 的
#   "fwd_grouped_kernel_stage1_rope" / "fwd_kernel_stage2" 两个键
#   （_get_config，mla_decode_rope.py:464-474；调用点 :523-524、:543、:552）。
#   该 JSON **不在** aiter 源码树里：`git ls-tree -r HEAD
#   aiter/ops/triton/configs/ | grep MLA` 为空（同 commit 的 configs 目录只有
#   GROUPED_DECODE_ATTENTION / EXTEND_ATTENTION 等），arch_info 把 gfx936/gfx938
#   映射为 BW200/BW200B（arch_info.py:5-10），故 DCU 真机走 config=None 会直接
#   FileNotFoundError。因此本适配器：**设备同名 JSON 存在且两个键齐全** →
#   config=None（完全走官方 config 通道）；否则显式传 config（宿主形参 config
#   就是给外部供 config 用的，官方 API 与官方测试都这么用），取值逐字段抄自
#   同族算子的官方默认分块 get_stage1_default_config / get_stage2_default_config
#   （grouped_decode_attention.py:431-443、630-638：mla_decode_rope 的 stage1/2
#   与它有同一套 BLOCK_C/BLOCK_R/BLOCK_H/BLOCK_N/NUM_KV_SPLITS 约束，且同样按
#   Lk = c+r >= 576 收窄 BLOCK_N）。config 只影响分块/占用（性能），不影响数值；
#   ctx["config_source"] 显式标注来源，绝不静默换参。注意②下的绝对性能不是官方
#   tuned 值（官方 JSON 缺失，基线偏保守），拿到该 JSON 后无需改代码即自动切回①。
#
# 不允许的算子：本文件不含任何 torch 高层计算，只用 contiguous / unsqueeze /
# reshape / to / empty / cat 做布局、索引恢复与打包，核心计算全在 aiter kernel 内。
#
# 注意：本机（无 GPU、无 aiter）只做过静态检查（py_compile），**未运行验证**。

import functools
import math
import os

import torch

# 官方 config JSON 读不到时的显式回退（逐字段抄自 grouped_decode_attention.py 的
# 官方默认分块）：stage1 见 get_stage1_default_config（:431-443），stage2 见
# get_stage2_default_config（:630-638）。BLOCK_C / BLOCK_R / NUM_KV_SPLITS 由宿主
# 自行注入，不在此列出（mla_decode_rope.py:333-336、:445-447）。
_FALLBACK_STAGE1_CONFIG = {
    "BLOCK_H": 16,
    "waves_per_eu": 1,
    "matrix_instr_nonkdim": 16,
    "kpack": 2,
    "num_warps": 4,
    "num_stages": 1,
}
_FALLBACK_STAGE2_CONFIG = {
    "waves_per_eu": 4,
    "matrix_instr_nonkdim": 16,
    "kpack": 2,
    "num_warps": 4,
    "num_stages": 2,
}
# 官方默认分块里 BLOCK_N 的分档阈值（grouped_decode_attention.py:441-442）
_BLOCK_N_NARROW_LK = 576


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _fallback_config(lk: int) -> dict:
    """官方 {dev}-MLA_DECODE_ROPE-DEFAULT.json 缺失时的內联兜底分块。

    BLOCK_N 沿用官方 get_stage1_default_config 的规则：Lk = c + r >= 576 时
    收窄到 16（本题 c=512，r ∈ {64,127,128} → Lk ∈ {576,639,640}，恒走 16）。
    """
    stage1 = dict(_FALLBACK_STAGE1_CONFIG)
    stage1["BLOCK_N"] = 16 if lk >= _BLOCK_N_NARROW_LK else 32
    return {
        "fwd_grouped_kernel_stage1_rope": stage1,
        "fwd_kernel_stage2": dict(_FALLBACK_STAGE2_CONFIG),
    }


@functools.lru_cache(maxsize=4)
def _resolve_config(lk: int):
    """返回 (config, source)。

    config=None 表示走官方 _get_config() 通道（设备同名 JSON 存在且键齐全）；
    否则为文件内联兜底 dict。source 写进 ctx，便于终审核对基线用的是哪一套。
    进程内只探测一次（lru_cache），不把文件 IO 计入基线计时。
    """
    try:
        from aiter.ops.triton.utils import arch_info
        from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH

        fpath = os.path.join(
            AITER_TRITON_CONFIGS_PATH,
            f"{arch_info.get_device()}-MLA_DECODE_ROPE-DEFAULT.json",
        )
        if os.path.exists(fpath):
            import json

            with open(fpath, "r") as fh:
                doc = json.load(fh)
            keys = ("fwd_grouped_kernel_stage1_rope", "fwd_kernel_stage2")
            if all(k in doc for k in keys):
                return None, f"official_autotune_json:{fpath}"
            return (
                _fallback_config(lk),
                f"fallback_config({os.path.basename(fpath)} 缺 {keys} 键)",
            )
        return (
            _fallback_config(lk),
            f"fallback_config(no {os.path.basename(fpath)})",
        )
    except Exception as exc:  # noqa: BLE001 - 仅探测；FileNotFoundError/ImportError 等一律回退
        return _fallback_config(lk), f"fallback_config(probe {type(exc).__name__})"


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现（mla_decode_rope.decode_attention_fwd_grouped_rope）。

    inputs     : [q, kv_cache, cos_sin_cache, positions, kv_indptr, kv_indices]
                 （顺序同 reference.py::make_inputs 的返回，已在 device 上）
                 q             [B, H, c+r]        fp16/bf16，contiguous
                 kv_cache      [N, c+r]          与 q 同 dtype（token 级 KV 行池）
                 cos_sin_cache [max_pos, rd]     与 q 同 dtype（前 rd/2 cos、后 rd/2 sin）
                 positions     [B]               int32（评测端可能 cast fp32，需无损恢复）
                 kv_indptr     [B+1]             int32（序列长度累积和）
                 kv_indices    [total_kv]        int32（逻辑 token -> 物理行号）
    init_kwargs: {"kv_lora_rank": int, "qk_rope_head_dim": int, "rotary_dim": int,
                  "sm_scale": float | None, "is_neox_style": bool}
                 （Model.__init__ 的参数名；sm_scale 缺省 1/sqrt(c+r)、
                  is_neox_style 缺省 True，与 reference.py:81-87 同口径）
    device     : 目标设备（张量已在 device 上，此处仅用于记录）

    返回 (out, ctx)；out 为 reference 同形的单张量
    [B*H*c + B*r]、dtype 同输入（前 B*H*c 个 = attn_out 行主序，后 B*r 个 =
    k_pe_tokens 行主序）。
    """
    # aiter 顶层 import 很重：一律函数内 import（真机部署走最小导入垫片）。
    from aiter.ops.triton.mla_decode_rope import decode_attention_fwd_grouped_rope

    if len(inputs) != 6:
        raise ValueError(
            f"1020 需要 6 个输入 [q, kv_cache, cos_sin_cache, positions, kv_indptr, "
            f"kv_indices]，实际 {len(inputs)} 个"
        )
    q, kv_cache, cos_sin_cache, positions, kv_indptr, kv_indices = inputs

    # ---- 构造参数（按名取参，缺必需项/越界即 raise，绝不静默用错）-----------
    c_raw = init_kwargs.get("kv_lora_rank")
    r_raw = init_kwargs.get("qk_rope_head_dim")
    rd_raw = init_kwargs.get("rotary_dim")
    if c_raw is None or r_raw is None or rd_raw is None:
        raise KeyError(
            "init_kwargs 缺 kv_lora_rank / qk_rope_head_dim / rotary_dim 之一："
            "Model.__init__ 的必需参数，不能猜"
        )
    c, r, rd = int(c_raw), int(r_raw), int(rd_raw)
    sm_scale_raw = init_kwargs.get("sm_scale", None)
    sm_scale = (
        float(sm_scale_raw) if sm_scale_raw is not None else 1.0 / math.sqrt(c + r)
    )
    is_neox_style = bool(init_kwargs.get("is_neox_style", True))

    # ---- shape / dtype / 约束校验（全部无 device->host 同步，不污染基线计时）--
    if q.dim() != 3:
        raise ValueError(f"q 必须是 3 维 [B, H, c+r]，实际 {tuple(q.shape)}")
    B, H, D = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    if D != c + r:
        raise ValueError(
            f"q 末维必须等于 kv_lora_rank + qk_rope_head_dim = {c + r}，实际 {D}"
        )
    if kv_cache.dim() != 2 or int(kv_cache.shape[1]) != c + r:
        raise ValueError(
            f"kv_cache 必须是 2 维 [total_tokens, c+r] = [N, {c + r}]，"
            f"实际 {tuple(kv_cache.shape)}"
        )
    if cos_sin_cache.dim() != 2 or int(cos_sin_cache.shape[1]) != rd:
        raise ValueError(
            f"cos_sin_cache 必须是 [max_positions, rotary_dim] = [*, {rd}]"
            f"（前 rd/2 列 cos、后 rd/2 列 sin），实际 {tuple(cos_sin_cache.shape)}"
        )
    if positions.numel() != B:
        raise ValueError(f"positions 长度必须等于 batch({B})，实际 {positions.numel()}")
    if kv_indptr.numel() != B + 1:
        raise ValueError(
            f"kv_indptr 长度必须等于 batch+1({B + 1})，实际 {kv_indptr.numel()}"
        )
    if kv_indices.dim() != 1:
        raise ValueError(f"kv_indices 必须是 1 维，实际 {tuple(kv_indices.shape)}")
    if B < 1 or H < 1:
        raise ValueError(f"batch / num_heads 必须 >= 1，实际 {B}/{H}")
    # 题面不变式：rd 为 2 的幂且 rd <= r（kernel 的配对索引按 % rotary_dim / % 2
    # 构造，非 2 次幂直接错配）。
    if not _is_pow2(rd) or rd < 2 or rd > r:
        raise ValueError(
            f"rotary_dim 必须是 >= 2 的 2 的幂且 <= qk_rope_head_dim，实际 "
            f"rotary_dim={rd} / qk_rope_head_dim={r}"
        )
    # tl.dot 的 K/N 维下限为 16：nope 部分 K=BLOCK_C=next_pow2(c)，rope 部分
    # K=BLOCK_R=next_pow2(r) 且 N=BLOCK_N>=16（mla_decode_rope.py:230、240、
    # :333-334）。
    if c < 16 or r < 16:
        raise ValueError(
            f"kv_lora_rank / qk_rope_head_dim 必须 >= 16（tl.dot 的 K 维下限），"
            f"实际 {c}/{r}"
        )
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"本题 dtype 域为 fp16/bf16（kernel 走 tl.dot 半精度路径），实际 {q.dtype}"
        )
    if kv_cache.dtype != q.dtype or cos_sin_cache.dtype != q.dtype:
        raise ValueError(
            f"kv_cache / cos_sin_cache 的 dtype 必须与 q 一致：q={q.dtype} "
            f"kv_cache={kv_cache.dtype} cos_sin_cache={cos_sin_cache.dtype}"
        )

    # ---- layout 归一 + 索引 dtype 恢复 --------------------------------------
    # 题面 io 已是 aiter 期望的连续布局；索引张量统一 int32（题面声明值域小，
    # 与 1002/1021 适配器同口径）。评测端若把整数张量 cast 成 fp32，小整数可
    # 无损还原（reference.py:90-95 亦按此约定）。
    q_c = q.contiguous()
    kv = kv_cache.contiguous()
    # k_buffer 必须 3 维 [N, 1, c+r]：宿主用 shape[1] 当 KV head 数推 GQA 分组
    # （mla_decode_rope.py:331），MLA 无独立 KV head，故为 1（官方 ref_preprocess
    # 即 latent_cache.unsqueeze(1)，测试 :68）。
    k_buffer = kv.unsqueeze(1)
    # v_buffer = kv_cache[:, :c] 的连续副本（官方 ref_preprocess :66-67 同款；
    # 题面不变式：V ≡ 该行前 c 列压缩隐向量，K/V 共享）。
    v_buffer = kv[:, :c].contiguous().unsqueeze(1)
    cos_sin = cos_sin_cache.contiguous()
    pos = positions.reshape(-1).to(torch.int32).contiguous()
    indptr = kv_indptr.reshape(-1).to(torch.int32).contiguous()
    indices = kv_indices.reshape(-1).to(torch.int32).contiguous()

    # ---- 输出预分配（kernel 就地写入，宿主函数返回 None）--------------------
    num_kv_splits = 2  # 官方测试全部用例的默认（见文件头说明）
    logit_cap = 0.0    # 题面无 logit 封顶：必须为 0，>0 会走 tanh 封顶分支
    use_rope = True    # 本题恒为 rope 融合解码（use_rope=False 是另一道题）

    o = torch.empty(B, H, c, dtype=q.dtype, device=q.device)
    k_pe_tokens = torch.empty(B, r, dtype=q.dtype, device=q.device)
    # 中间缓存：最后一列放 e_max + log(e_sum)，供 stage2 做 log-sum-exp 归并
    attn_logits = torch.empty(
        B, H, num_kv_splits, c + 1, dtype=q.dtype, device=q.device
    )

    config, config_source = _resolve_config(c + r)

    # ---- 调用 aiter 官方入口（实参顺序对齐 mla_decode_rope.py:477-496）------
    decode_attention_fwd_grouped_rope(
        q_c,             # q            [B, H, c+r]
        k_buffer,        # k_buffer     [N, 1, c+r]（含 k_pe）
        v_buffer,        # v_buffer     [N, 1, c]  （= k_buffer 行前 c 列）
        o,               # o            就地写入 [B, H, c]
        indptr,          # kv_indptr    [B+1]
        indices,         # kv_indices   [total_kv]
        k_pe_tokens,     # k_pe_tokens  就地写入 [B, r]（旋转后的最后一个 token）
        c,               # kv_lora_rank
        rd,              # rotary_dim
        cos_sin,         # cos_sin_cache [max_positions, rd]
        pos,             # positions    [B]
        attn_logits,     # attn_logits  [B, H, num_kv_splits, c+1]
        num_kv_splits,   # num_kv_splits
        sm_scale,        # sm_scale
        logit_cap=logit_cap,
        use_rope=use_rope,
        is_neox_style=is_neox_style,
        config=config,
    )

    torch.cuda.synchronize()

    # ---- 打包成单张量（顺序与 reference.py:141-142 完全一致）---------------
    out = torch.cat((o.reshape(-1), k_pe_tokens.reshape(-1)))

    # 仅用于 ctx 记录：宿主 grid 的 head 块大小（官方 JSON 通道下取该 JSON 的值）
    block_h = int(_FALLBACK_STAGE1_CONFIG["BLOCK_H"])
    if config is not None:
        block_h = int(config["fwd_grouped_kernel_stage1_rope"].get("BLOCK_H", block_h))

    ctx = {
        "impl": "aiter",
        "module": "aiter.ops.triton.mla_decode_rope",
        "entry": "decode_attention_fwd_grouped_rope",
        "path": (
            "_fwd_grouped_kernel_stage1_rope + _fwd_kernel_stage2 "
            "(USE_ROPE=True, logit_cap=0.0)"
        ),
        "batch": B,
        "num_heads": H,
        "kv_lora_rank": c,
        "qk_rope_head_dim": r,
        "rotary_dim": rd,
        "rope_half": rd // 2,
        "is_neox_style": is_neox_style,
        "sm_scale": sm_scale,
        "scale_from_init": sm_scale_raw is not None,
        "dtype": str(q.dtype),
        "num_kv_splits": num_kv_splits,
        "logit_cap": logit_cap,
        "total_tokens": int(kv.shape[0]),
        "total_kv": int(indices.numel()),
        # 宿主 grid：stage1 = (batch, cdiv(H, min(BLOCK_H, kv_group_num)), splits)，
        # stage2 = (batch, H)（mla_decode_rope.py:337-341、:449）
        "stage1_grid": (
            B,
            -(-H // min(block_h, H)),
            num_kv_splits,
        ),
        "stage1_block_h": block_h,
        "stage2_grid": (B, H),
        "attn_logits_shape": [B, H, num_kv_splits, c + 1],
        "out_numel": int(out.numel()),
        "out_dtype": str(out.dtype),
        "config_source": config_source,
        "config": config,
        "device": str(device),
    }
    return out, ctx
