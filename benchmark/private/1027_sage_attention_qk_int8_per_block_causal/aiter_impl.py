# aiter_impl.py — 1027_sage_attention_qk_int8_per_block_causal 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 适配器契约统一为「按名取参」：run(inputs, init_kwargs: dict, device) -> (out, ctx)。
#
# 来源（pinned 检出 .dcu_runs/aiter_pinned，OpenDAS/aiter @
# c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#
#   公开算子入口   aiter/ops/triton/sage_attention.py::sageattn_qk_int8_pv_fp16
#                  sha256 ee7c0585fbd2c8ef8d18dc0bdac66f3870f96f140dfbf485356be32e84684f2a
#                  （签名 L34-46；is_causal=True 分支 L177-186）
#   因果 kernel    aiter/ops/triton/sage_attention_qk_int8_per_block_causal.py
#                  sha256 c0ef2f2ca09132996e8ad7fbd64a9794b7935d61883f102a52b50b9493ecb11b
#                  （准入记录的 device_kernel：_attn_fwd_inner L26 / _attn_causal_fwd L73 /
#                    host 入口 forward L176；per-block INT8 量化 + exp2 在线 softmax +
#                    fp16 PV 块内 dot / fp32 跨块累加，HND）
#   QK 量化 helper aiter/ops/triton/sage_attention_quant_per_block.py::per_block_int8
#                  sha256 a44fc9f09225e42799c5546f3d0880791ef6c66fba172166d44709792fb140b0
#                  （经 sage_attention.py L12-17 do_quant_qk 调用；BLKQ/BLKK 取自因果
#                    kernel config 的 BLOCK_M/BLOCK_N）
#   官方测试       op_tests/triton_tests/test_sage_attention_qk_int8_pv_fp16.py
#                  sha256 3dd4b84eae019ebb109863070dbb4242107d53fd48f1e9057726c92986b28648
#                  （语义唯一权威：L67 调 sageattn_qk_int8_pv_fp16(q, k, v,
#                    tensor_layout=..., is_causal=...)，真值取 PyTorch 原生注意力，
#                    L84 验收 torch.allclose(atol=2e-2, rtol=2e-2)）
#
# 入口签名（sage_attention.py L34-46）：
#   sageattn_qk_int8_pv_fp16(q, k, v, tensor_layout="HND",
#                            quantization_backend="triton", is_causal=False,
#                            attn_mask=None, sm_scale=None, smooth_k=True,
#                            return_lse=False, **kwargs) -> torch.Tensor
#
# 与题面语义的对齐（题面 = 本入口 is_causal=True / HND / triton 量化分支：
# 因果掩码、qo_len == kv_len、GQA/MHA、head_dim ∈ {64,128}）：
#   - 题面 q/k/v 本就是 HND [B, H, L, D] 且「最后一维连续」（task.yaml io.inputs），
#     与 aiter 的 HND stride 约定一致 —— 无需 permute/flatten 重排。这里只做一次
#     contiguous 兜底（上面可能给出非连续 view），不改变数值。
#   - 题面不提供 q_scale / k_scale：per-block INT8 量化由本入口内部完成
#     （do_quant_qk -> per_block_int8；Q 折入 sm_scale*log2(e)、K 折入 1.0，
#      每块 scale = max|x|/127，round-half-away-from-zero），与题面 ③ 的算法描述一致。
#   - 量化块大小 = 因果 kernel config 的 BLOCK_M / BLOCK_N（默认 128 / 64，
#     即题面的 Q 每 128 行、K 每 64 行一块）。
#   - smooth_k 取入口默认 True（官方测试亦未显式传参）：对 K 沿序列维去均值后再量化。
#     题面 docstring 明确允许（"可先对 K 沿序列维去均值（smooth-k）再量化"），
#     且对精确因果注意力数学中性。
#   - return_lse=False：入口返回**单个** [B, Hq, L, D] 张量，dtype = output_dtype = q.dtype
#     与 reference 输出同形同 dtype，无需任何打包/拼接。
#   - sm_scale：init_kwargs 无 scale 时用 1/sqrt(head_dim)（与 Model.__init__ 缺省一致，
#     也与入口 sm_scale=None 的缺省一致）；显式给了 scale 则原样透传。
#
# 环境注意：
#   1) 本入口链上的三个 triton 模块都在模块头 `from triton.utils.hcutuner import get_gpu_label`，
#      依赖 DTK Triton 的 hcutuner 扩展（1002 的 pa_decode 链路不需要它）。缺失则导入即失败，
#      属部署环境问题（真机 DTK Triton 自带）。
#   2) autotune JSON 依赖（AITER_TRITON_CONFIGS_PATH；落点见文末 _CONFIG_FILES）：
#      - 因果 kernel 的 {configs}/_attn_causal_fwd-device=<gpu_label>-dtype=f16_f16_f16_f16_f32.json
#        **在 c39fff8c 的 git tree 里不存在**（用
#        `git ls-tree -r HEAD -- aiter/ops/triton/configs` 核对：configs/sage_attention/
#        下只有非因果的 _attn_fwd-... 与 quant_per_block_int8_kernel-...，没有
#        _attn_causal_fwd）→ _get_config 必然走 except 分支退回模块内置 default_config：
#        BLOCK_M=128 / BLOCK_N=64 / STAGE=3 / waves_per_eu=1 / matrix_instr_nonkdim=16 /
#        kpack=2 / num_warps=4(head_dim=64)|8 / num_stages=2。
#        数值语义不受影响（BLOCK_M/BLOCK_N 正是题面的 Q 每 128 行 / K 每 64 行一块），
#        仅仅是因果 kernel 的性能永远停在「未调优默认」。
#        （另注：这些 JSON 文件名里的 gpu_label 形如 "gfx936:cu_80"，带冒号，本机
#          Windows 侧的 sparse-checkout `!**/*:*` 不会把它们 checkout 出来。）
#      - 量化 kernel 的 {configs}/sage_attention/quant_per_block_int8_kernel-device=...json
#        在 tree 里**存在**（Linux 检出即命中；只调 num_warps/num_stages 等，不含分块）。
#      故本适配器不 raise，只在 ctx 里如实记录 config 来源与期望文件，供基线采集者
#      判断是否需要补一份因果 kernel 的调优 JSON。

import math
import os

import torch

# 这两个 JSON 是因果 kernel / 量化 kernel 的官方 autotune 产物落点
# （aiter/ops/triton/sage_attention_qk_int8_per_block_causal.py L149、
#   aiter/ops/triton/sage_attention_quant_per_block.py L69）；
# 相对路径基准 = aiter/ops/triton/configs（aiter/ops/triton/utils/core.py L6）。
# 路径里 {gpu} 是 triton.utils.hcutuner.get_gpu_label()，gfx936 上形如 "gfx936:cu_80"。
_CONFIG_FILES = {
    "attn": "_attn_causal_fwd-device={gpu}-dtype=f16_f16_f16_f16_f32.json",
    "quant": "sage_attention/quant_per_block_int8_kernel-device={gpu}-dtype=f16_i8_f32.json",
}


def _probe_config(qo_len, kv_len, h_qo, num_kv_groups, head_dim):
    """记录因果 kernel 实际会用的 BLOCK_M/BLOCK_N 与 autotune config 是否就位（诊断用）。

    只读、不改行为：入口自己也是用同一把 key 调 _get_config 的
    （sage_attention.py L179-180 -> L211-212）。任何异常都被吞掉——
    诊断信息缺失不得影响基线采集。
    """
    key = str((qo_len, kv_len, h_qo, num_kv_groups))
    info = {"config_query_key": key, "config_files_present": {}}
    try:
        from aiter.ops.triton.sage_attention_qk_int8_per_block_causal import _get_config
        from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
        from triton.utils.hcutuner import get_gpu_label

        gpu = get_gpu_label()
        cfg = _get_config(key, head_dim)
        for kind, name in _CONFIG_FILES.items():
            rel = name.format(gpu=gpu)
            info["config_files_present"][kind] = os.path.exists(
                os.path.join(AITER_TRITON_CONFIGS_PATH, rel)
            )
        info["gpu_label"] = gpu
        info["block_m"] = int(cfg["BLOCK_M"])
        info["block_n"] = int(cfg["BLOCK_N"])
        info["stage"] = int(cfg.get("STAGE", 3))
        # 因果 kernel 的 JSON 缺失时，_get_config 返回模块内置 default_config
        info["attn_config_source"] = (
            "tuned_json" if info["config_files_present"]["attn"] else "module_default"
        )
    except Exception as exc:  # noqa: BLE001 — 纯诊断
        info["config_probe_error"] = repr(exc)
    return info


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方 SageAttention2 因果前向。

    inputs     : [q, k, v] —— 已在 device 上的 HND 浮点张量
                 q [B, Hq, L, D] / k, v [B, Hkv, L, D]，q/k/v 同 dtype（fp16/bf16），
                 qo_len == kv_len（因果），Hq % Hkv == 0（GQA/MHA）。
    init_kwargs: {"head_dim": int[, "scale": float]}（与 Model.__init__ 同名同义；
                 scale 缺省 = 1/sqrt(head_dim)）

    返回 (out, ctx)：out 为单个 [B, Hq, L, D] 张量，dtype 与 q 相同。
    """
    from aiter.ops.triton.sage_attention import sageattn_qk_int8_pv_fp16

    if len(inputs) != 3:
        raise ValueError(f"1027 需要 (q, k, v) 三个输入，实际收到 {len(inputs)} 个")
    q, k, v = inputs

    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.dim() != 4:
            raise ValueError(f"{name} 必须是 HND 4D 张量，实际 {tuple(tensor.shape)}")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"q/k/v dtype 必须相同，实际 {q.dtype}/{k.dtype}/{v.dtype}")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"aiter sageattn_qk_int8_pv_fp16 只支持 float16/bfloat16，实际 {q.dtype}"
        )

    b, h_qo, qo_len, head_dim = q.shape
    b_k, h_kv, kv_len, head_dim_k = k.shape
    if (b, qo_len, head_dim) != (b_k, kv_len, head_dim_k):
        raise ValueError(f"k 形状与 q 不匹配：q={tuple(q.shape)}, k={tuple(k.shape)}")
    if v.shape != k.shape:
        raise ValueError(f"v 形状必须与 k 相同：v={tuple(v.shape)}, k={tuple(k.shape)}")
    if qo_len != kv_len:
        raise ValueError(
            f"1027 是因果题面，要求 qo_len == kv_len，实际 {qo_len} != {kv_len}"
        )
    if h_kv <= 0 or h_qo % h_kv != 0:
        raise ValueError(f"num_qo_heads({h_qo}) 必须是 num_kv_heads({h_kv}) 的整数倍")

    # init_kwargs 校验：绝不静默用错参数
    declared_head_dim = int(init_kwargs.get("head_dim", head_dim))
    if declared_head_dim != head_dim:
        raise ValueError(
            f"init_kwargs['head_dim']={declared_head_dim} 与 q.shape[-1]={head_dim} 不一致"
        )
    if not (16 <= head_dim <= 128):
        raise ValueError(
            f"head_dim={head_dim} 超出 aiter SageAttention 支持范围（<=128；"
            "head_dim<64 时 aiter 内部 pad 到 64，属题面 head_dim ∈ {64,128} 之外）"
        )

    scale = init_kwargs.get("scale", None)
    if scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    else:
        sm_scale = float(scale)
        if not math.isfinite(sm_scale) or sm_scale <= 0.0:
            raise ValueError(f"init_kwargs['scale'] 非法：{scale!r}")

    # HND 已是 aiter 的布局约定；contiguous 只兜底上层给的非连续 view
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = sageattn_qk_int8_pv_fp16(
        q,
        k,
        v,
        tensor_layout="HND",
        quantization_backend="triton",
        is_causal=True,          # 题面子集：因果分支（准入记录 admission.note）
        attn_mask=None,          # is_causal=True 时 aiter 断言 mask 必须为 None
        sm_scale=sm_scale,
        smooth_k=True,           # 入口/官方测试的默认行为；题面 docstring 允许
        return_lse=False,        # 单张量返回，与 reference 输出同形同 dtype
    )
    torch.cuda.synchronize()

    # 自检：与 reference 的契约（形状/dtype）不符时立即失败，别让错基线落盘
    if out.shape != q.shape or out.dtype != q.dtype:
        raise RuntimeError(
            f"aiter 输出与题面契约不符：out={tuple(out.shape)}/{out.dtype}，"
            f"期望 {tuple(q.shape)}/{q.dtype}"
        )

    ctx = {
        "path": "sageattn_qk_int8_pv_fp16/is_causal=True/HND/triton-quant",
        "entry": "aiter.ops.triton.sage_attention::sageattn_qk_int8_pv_fp16",
        "kernel_module": "aiter.ops.triton.sage_attention_qk_int8_per_block_causal",
        "tensor_layout": "HND",
        "is_causal": True,
        "quantization_backend": "triton",
        "smooth_k": True,
        "return_lse": False,
        "batch": b,
        "num_qo_heads": h_qo,
        "num_kv_heads": h_kv,
        "num_kv_groups": h_qo // h_kv,
        "qo_len": qo_len,
        "kv_len": kv_len,
        "head_dim": head_dim,
        "dtype": str(q.dtype).replace("torch.", ""),
        "sm_scale": sm_scale,
        "out_shape": tuple(out.shape),
        "head_dim_pad_path": head_dim not in (64, 128),
    }
    ctx.update(_probe_config(qo_len, kv_len, h_qo, h_qo // h_kv, head_dim))
    return out, ctx
