# aiter_impl.py — 1026_sage_attention_qk_int8_per_block 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用，采集离线终审用的 aiter 基线。
# 契约：run(inputs, init_kwargs: dict, device) -> (out, ctx)，按名取构造参数
# （init_kwargs 由 audit_model_class.case_init_kwargs 产出，稀疏参数下位置式取值
# 会错位，故一律 .get(name, default)）。
#
# 来源（pinned 检出 .dcu_runs/aiter_pinned，commit c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#   aiter/ops/triton/sage_attention_qk_int8_per_block.py
#       sha256 3fbd63579aac5530b1f603742776f88ce8f363b32d7ec0d9378caff97d020bf1
#     · host 入口（模块内公开函数，非 kernel）——文件:186
#         forward(q, k, v, q_scale, k_scale, tensor_layout="HND", attn_mask=None,
#                 output_dtype=torch.float16, return_lse=False, config=None)
#         -> (o, lse)
#       return_lse=False 时 lse 是 CPU 上的空张量（文件:219），故本题只取 o
#       （单张量输出契约，与 reference 的 out 对齐）。
#     · device kernel：_attn_fwd（文件:79）/ _attn_fwd_inner（文件:25）
#   aiter/ops/triton/sage_attention.py:186-205（wrapper）
#       调用约定的直接依据：attn_mask 用 expand 广播到 [b, h_qo, qo_len, kv_len]
#       （前两维 stride=0），并把 _get_config 解析出的 config 原样传给 forward。
#   官方测试（语义权威，验收容差来源）：
#     op_tests/triton_tests/test_sage_attention_qk_int8_pv_fp16.py
#       sha256 3dd4b84eae019ebb109863070dbb4242107d53fd48f1e9057726c92986b28648
#       （sageattn_qk_int8_pv_fp16 vs F.scaled_dot_product_attention，atol=rtol=2e-2）
#
# 布局：题面的 HND [b,h,l,d] / NHD [b,l,h,d] 就是 aiter 的两种 tensor_layout
# （kernel 直接吃 stride），**无需任何 permute**；q_scale [b,h_q,ceil(qo/128)] 与
# k_scale [b,h_kv,ceil(kv/64)] 与布局无关，kernel 的 offset 公式
# （文件:118-119，cdiv(qo_len, BLOCK_M) / cdiv(kv_len, BLOCK_N)）与之逐字对应。
#
# 与 reference 的数值口径：reference 把 q_scale 展开回序列维后除 log2(e) 回到自然
# 对数域；kernel 折入 scale 后走 exp2 在线 softmax，两者数学等价（exp2(x)=exp(x·ln2)）。
#
# 本机无 GPU、未安装 aiter，本文件**未做运行验证**，只做了 py_compile 静态检查。

import torch

# 题面 scale 的分块行数（reference.py::_quant_per_block_int8 的 blk）：
# Q 每 128 行一块、K 每 64 行一块。kernel 侧它们必须等于 BLOCK_M / BLOCK_N，
# 否则 q_scale/k_scale 的块索引会被解读成另一套分块（静默错值，甚至越界读）。
_Q_BLOCK = 128
_K_BLOCK = 64


def _module_default_config(head_dim):
    """模块内置默认 config（sage_attention_qk_int8_per_block.py:170-179 逐字复刻）。

    其 BLOCK_M=128 / BLOCK_N=64 与题面 scale 的分块一致，是「调优 JSON 不可用或
    分块不匹配」时唯一正确的选择。
    """
    return {
        "BLOCK_M": _Q_BLOCK,
        "BLOCK_N": _K_BLOCK,
        "STAGE": 1,
        "waves_per_eu": 1,
        "matrix_instr_nonkdim": 16,
        "kpack": 2,
        "num_warps": 4 if head_dim == 64 else 8,
        "num_stages": 2,
    }


def _resolve_config(qo_len, kv_len, h_qo, num_kv_groups, head_dim):
    """返回 (config, source)。

    source == "tuned_json"：模块自己的 _get_config() 命中了
      {AITER_TRITON_CONFIGS_PATH}/sage_attention/
      _attn_fwd-device=<gpu_label>-dtype=f16_f16_f16_f32_f32_f16_f32.json
      里该 shape 的 key，且其 BLOCK_M/BLOCK_N 与题面 scale 的 128/64 一致；
      key 格式取自官方 wrapper（sage_attention.py:20-31、:200）。
      （该文件在 aiter 仓里确实存在，实测 gfx936:cu_80 版含 (4352,4352,24,1) ->
       BLOCK_M=128/BLOCK_N=64，与 perf case 1 对应。）
    source == "module_default"：JSON 缺失/无此 key/分块不一致时退回模块默认 config。

    注意：**不能**无条件采用调优 JSON——官方 wrapper 是用 config["BLOCK_M"] 去量化
    Q 的（sage_attention.py:203，BLKQ=config['BLOCK_M']），即调优 config 同时决定
    分块；本题的量化阶段不在题面内（int8 码字与 scale 由输入直接给出，分块固定为
    128/64），所以只有分块匹配的 config 才能正确消费给定 scale。JSON 里其它 shape
    的 key 确有 BLOCK_M=64 的条目，故此处必须校验而不是照搬。
    """
    default = _module_default_config(head_dim)
    cfg = None
    try:
        # 模块内私有符号，但 forward 自己也是这么取 config 的（文件:223-226），
        # 且它是官方 wrapper 的调用套路（sage_attention.py:8、:201）。
        from aiter.ops.triton.sage_attention_qk_int8_per_block import _get_config

        cfg = _get_config(str((qo_len, kv_len, h_qo, num_kv_groups)), head_dim)
    except Exception:
        cfg = None
    if (
        isinstance(cfg, dict)
        and int(cfg.get("BLOCK_M", -1)) == _Q_BLOCK
        and int(cfg.get("BLOCK_N", -1)) == _K_BLOCK
    ):
        return dict(cfg), "tuned_json"
    return default, "module_default"


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方 sage attention（int8 QK per-block + fp16 PV）前向。

    inputs（顺序即 reference.make_inputs 的返回顺序，已在 device 上）：
      q_int8  int8    HND [b, h_q, qo_len, d] / NHD [b, qo_len, h_q, d]
      k_int8  int8    HND [b, h_kv, kv_len, d] / NHD [b, kv_len, h_kv, d]
      v       fp16    与 k_int8 同布局同头数
      q_scale fp32    [b, h_q,  ceil(qo_len/128)]
      k_scale fp32    [b, h_kv, ceil(kv_len/64)]
      attn_mask bool  [qo_len, kv_len]（对 batch/头共享）
    init_kwargs：{"tensor_layout": "HND"|"NHD"}（Model.__init__ 的唯一参数）

    返回 (out, ctx)：out 与 q_int8 同形、dtype=fp16（= v.dtype），单张量；
    ctx 记录走的 aiter 路径、config 来源与关键 shape。
    """
    # aiter 顶层 import 很重，按契约在函数内做最小导入
    from aiter.ops.triton.sage_attention_qk_int8_per_block import forward as sage_attn_fwd

    if len(inputs) != 6:
        raise ValueError(
            f"期望 6 个输入 (q_int8, k_int8, v, q_scale, k_scale, attn_mask)，收到 {len(inputs)}"
        )
    q_int8, k_int8, v, q_scale, k_scale, attn_mask = inputs

    tensor_layout = str(init_kwargs.get("tensor_layout", "HND"))
    if tensor_layout not in ("HND", "NHD"):
        raise ValueError(f"tensor_layout 只支持 HND / NHD，收到 {tensor_layout!r}")

    # dtype 复原：与 reference.forward 的入口 cast 一致（评测器可能把输入统一 cast
    # 成 fp32，int8 码字 / fp16 的 v / bool 掩码在 fp32 下均可无损还原）。
    q_int8 = q_int8.to(torch.int8)
    k_int8 = k_int8.to(torch.int8)
    v = v.to(torch.float16)          # 官方 wrapper 同样把非 fp16 的 v 统一成 fp16
    q_scale = q_scale.to(torch.float32).contiguous()
    k_scale = k_scale.to(torch.float32).contiguous()
    attn_mask = attn_mask.to(torch.bool)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q_int8.shape
        h_kv, kv_len = k_int8.shape[1], k_int8.shape[2]
    else:
        b, qo_len, h_qo, head_dim = q_int8.shape
        kv_len, h_kv = k_int8.shape[1], k_int8.shape[2]

    # kernel 把 head_dim 维当 stride=1 隐式索引（_attn_fwd 文件:124-129 用
    # offs_k[None,:] / offs_k[:,None]）；官方 wrapper 也有同样的断言
    # （sage_attention.py:148）。题面 make_inputs 产出的 q/k/v 均为最后一维连续的
    # contiguous（NHD 也是 permute 后 contiguous），这里显式确认而不是默默出错。
    for name, t in (("q_int8", q_int8), ("k_int8", k_int8), ("v", v)):
        if t.stride(-1) != 1:
            raise ValueError(f"{name} 的最后一维必须连续（stride(-1)==1），实际 stride={t.stride()}")

    if head_dim not in (64, 128):
        # 官方 wrapper 对 head_dim<64 / 64<d<128 会先 pad 到 64/128（sage_attention.py:136-145），
        # 那会改变 v 的语义（补 0）并需要裁回输出；题面域已限定 head_dim∈{64,128}，
        # 越界直接 raise，不静默 pad。
        raise ValueError(f"head_dim 只支持 64 / 128，收到 {head_dim}")
    if h_kv <= 0 or h_qo % h_kv != 0:
        raise ValueError(f"num_q_heads({h_qo}) 必须是 num_kv_heads({h_kv}) 的整数倍（GQA）")
    num_kv_groups = h_qo // h_kv

    # per-block scale 的分块必须与 kernel 的 BLOCK_M/BLOCK_N 一致（见 _Q_BLOCK/_K_BLOCK 说明）
    want_q_scale = (b, h_qo, (qo_len + _Q_BLOCK - 1) // _Q_BLOCK)
    want_k_scale = (b, h_kv, (kv_len + _K_BLOCK - 1) // _K_BLOCK)
    if tuple(q_scale.shape) != want_q_scale:
        raise ValueError(f"q_scale 期望 {want_q_scale}（Q 每 {_Q_BLOCK} 行一块），实际 {tuple(q_scale.shape)}")
    if tuple(k_scale.shape) != want_k_scale:
        raise ValueError(f"k_scale 期望 {want_k_scale}（K 每 {_K_BLOCK} 行一块），实际 {tuple(k_scale.shape)}")

    # 掩码：题面是 [qo_len, kv_len] 的 bool，对 batch/头共享；官方 wrapper 的做法是
    # expand 到 [b, h_qo, qo_len, kv_len]（前两维 stride=0）后传给 kernel
    # （sage_attention.py:188-198；kernel 侧按 stride_mask* 索引，文件:133、:39）。
    # 前两维 stride=0 不构成问题：mask=None 时 kernel 的 stride 全是 0、同样会执行
    # 文件:105-108 的 tl.assume(stride_mask*>0)，即该情形在官方测试里每次都在跑。
    if attn_mask.dim() == 2:
        if tuple(attn_mask.shape) != (qo_len, kv_len):
            raise ValueError(
                f"attn_mask 期望 [qo_len, kv_len]=({qo_len}, {kv_len})，实际 {tuple(attn_mask.shape)}"
            )
    elif attn_mask.dim() != 4:
        raise ValueError(f"attn_mask 期望 2 维（题面）或可广播的 4 维，实际 {attn_mask.dim()} 维")
    mask4 = attn_mask.expand(b, h_qo, qo_len, kv_len)

    config, config_source = _resolve_config(qo_len, kv_len, h_qo, num_kv_groups, head_dim)

    # 入口参数顺序取自 aiter/ops/triton/sage_attention_qk_int8_per_block.py:186
    out, lse = sage_attn_fwd(
        q_int8.contiguous(),
        k_int8.contiguous(),
        v.contiguous(),
        q_scale,
        k_scale,
        tensor_layout=tensor_layout,
        attn_mask=mask4,
        output_dtype=v.dtype,
        return_lse=False,
        config=config,
    )
    torch.cuda.synchronize()

    ctx = {
        "aiter_path": "aiter.ops.triton.sage_attention_qk_int8_per_block.forward",
        "aiter_symbol": "_attn_fwd (kernel)",
        "tensor_layout": tensor_layout,
        "layout_convert": "none（HND/NHD 原生支持，kernel 直接吃 stride）",
        "config_source": config_source,
        "config": config,
        "block_q": _Q_BLOCK,
        "block_k": _K_BLOCK,
        "shape": {
            "b": b,
            "h_qo": h_qo,
            "h_kv": h_kv,
            "num_kv_groups": num_kv_groups,
            "qo_len": qo_len,
            "kv_len": kv_len,
            "head_dim": head_dim,
        },
        "mask": "expand -> [b, h_qo, qo_len, kv_len]（b/h 维 stride=0，官方 wrapper 同款）",
        "return_lse": False,
        "lse_shape": tuple(lse.shape),
        "out_dtype": str(out.dtype),
    }
    return out, ctx
