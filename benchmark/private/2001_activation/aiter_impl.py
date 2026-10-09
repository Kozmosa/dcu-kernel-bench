# aiter_impl.py — 2001_activation 的 aiter 官方实现适配器
#
# 供 benchmark/evaluator/record_baseline.py 调用（离线终审 perf 基线采集）。
# 契约：run(inputs, init_kwargs: dict, device) -> (out, ctx)，按名取参。
#
# 来源（aiter pinned 检出 .dcu_runs/aiter_pinned，commit c39fff8c77df4e80617649e92fa3c2615f2c43d1）：
#   aiter/ops/triton/activation.py
#     sha256 f17cda5d62cd06e1ae2b8ca0bb818b6449821797291f7bf26b4361c3a00430bc
#   官方测试 op_tests/triton_tests/test_activation.py::test_act_mul_and_mxfp4_quant
#     （torch_act_mul_and_mxfp4_quant 为语义权威，逐位断言 uint8 码字相等）
#
# 真正的公开算子入口（host 入口，activation.py:174）：
#   act_mul_and_mxfp4_quant(x, activation, scaling_mode="even", shuffle=False)
#         -> (x_fp4, blockscale_e8m0)
#   activation.py:174 是 host 包装；其下的 _act_mul_and_dynamic_mxfp4_quant_kernel
#   (activation.py:59) 与 helper（_silu:9 / _gelu:24 / _gelu_tanh:31 /
#   _get_activation_from_str:43）都不是可调用入口。
#   aiter_entries 里排前面的 cdiv/next_power_of_2/get_*_config 类 helper 本题不涉及；
#   该入口无 @triton.autotune，块大小是 activation.py:230-256 里按 shape 硬编码的，
#   不依赖 AITER_TRITON_CONFIGS_PATH，无需 config JSON。
#
# 语义与布局对齐（reference.py 的 Model.forward）：
#   1. 输入 x (M, N) fp16/bf16 contiguous，N % 4 == 0。aiter 内部取 N_half = N // 2，
#      一次 load 同时拿到 a = x[:, :N/2] 与 b = x[:, N/2:]（后者靠
#      `x_offs + stride_x_n * N`，activation.py:103），再算 act(a) * b
#      （activation.py:116，是**乘**；入口 docstring 里"adds"是笔误，代码为准），
#      与 reference 的 `act(a) * b` 一致。故**不需要**任何 split/permute。
#   2. 量化块：MXFP4_QUANT_BLOCK_SIZE = 32（activation.py:207），块尾不足 32 时
#      由 masked load（other=0，activation.py:107-114）补零并参与 amax，
#      与 reference 的 `torch.zeros + padded[:, :d] = out` 补零语义一致。
#   3. even_round amax / clamp(log2-2, -127, 127) / e2m1 位级舍入 / 相邻码字打包
#      全在 _mxfp4_quant_op（aiter/ops/triton/quant.py:364-433，实现与 reference 同行级
#      一致；打包为 reshape[..., 16, 2] 后 tl.split，偶低奇高）。
#   4. 返回值：x_fp4 (M, N//4) uint8；blockscale_e8m0 在 shuffle=False 下是
#      torch.empty((scaleN, scaleM)).T（activation.py:223-227）——形状
#      (M, ceil(N/64))、uint8、**非 contiguous 转置视图**（stride (1, M)），
#      这正是 reference 用的 blockscale 语义，直接按最后一维 cat 即可。
#
# 输出打包（题面单张量契约，task.yaml io.outputs）：
#   out = cat([x_fp4, blockscale_e8m0], dim=-1)，uint8
#       (M, N//4 + ceil(N/64))：前 N//4 列 e2m1 打包码字、后 ceil(N/64) 列 e8m0 字节。
#   与 reference.py:125 的 `torch.cat([x_fp4, bs_e8m0], dim=-1)` 逐列对应。
#   题面未取 shuffle=True 的预排布 scale 布局（需 M%256==0/N%512==0），恒传 False。
#
# 已知残留（low，无法在本机验证，记录以免误判）：
#   e8m0 字节在 aiter 是 `scale_unbiased.to(tl.uint8) + 127`（quant.py:386），
#   reference 是 `(scale_unbiased + 127).to(torch.uint8)`（reference.py:104）。
#   无偏 scale 为 [-127,127] 内整数时二者逐位等价（负 float -> uint8 依赖平台
#   wrap 语义，admission 记录已核）；唯一分歧点是**全零块**（amax==0 ->
#   scale 被钳到 -127 -> aiter 走 fptoui(-127.0)），而 randn 输入下 32 元素块
#   的 amax 恒非零，该路径在 perf case 中不会被触发。

import torch

# 入口宿主返回值与题面形状契约的对应关系（见 reference.py:86-125）
_MXFP4_QUANT_BLOCK_SIZE = 32


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def run(inputs, init_kwargs, device):
    """执行一次 aiter 官方实现。

    inputs     : [x]  —— x (M, N) fp16/bf16（评测器已搬到 device），N % 4 == 0
    init_kwargs: {"activation": "silu" | "gelu" | "gelu_tanh"}

    返回 (out, ctx)；out 为 uint8 (M, N//4 + ceil(N/64))，与 reference 输出同形同 dtype。
    """
    # aiter 一律函数内 import：顶层 import 很重（真机走最小导入垫片）
    from aiter.ops.triton.activation import act_mul_and_mxfp4_quant

    if len(inputs) != 1:
        raise ValueError(f"2001_activation 只需 1 个输入 x，收到 {len(inputs)} 个")

    x = inputs[0]
    if not isinstance(x, torch.Tensor) or x.dim() != 2:
        raise ValueError(f"x 必须是 2D 张量 (M, N)，收到 {type(x)} / {getattr(x, 'shape', None)}")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"x dtype 必须是 fp16/bf16/fp32，收到 {x.dtype}")

    # 构造参数按名取；缺省与 Model.__init__ 默认值一致（reference.py:67）
    activation = init_kwargs.get("activation", "silu")
    if activation not in ("silu", "gelu", "gelu_tanh"):
        raise ValueError(
            f"activation 必须是 silu / gelu / gelu_tanh 之一，收到 {activation!r}"
            "（越界即报错，绝不静默退化到其它激活）"
        )

    M, N = int(x.shape[0]), int(x.shape[1])
    if M < 1 or N % 4 != 0:
        raise ValueError(f"需 M >= 1 且 N % 4 == 0，收到 M={M}, N={N}")

    # x 需按 (M, N) 行主序送进 kernel（入口用 x.stride() 取 stride，activation.py:266）
    x_in = x.contiguous()

    # shuffle=False：题面取的是未预排布的 (M, ceil(N/64)) e8m0 布局；
    # scaling_mode 只在 host 侧被忽略（kernel 恒 SCALING_MODE=0），显式写 "even"。
    x_fp4, blockscale_e8m0 = act_mul_and_mxfp4_quant(
        x_in, activation=activation, scaling_mode="even", shuffle=False
    )

    # 单张量契约：前 N//4 列 FP4 码字、后 ceil(N/64) 列 e8m0 scale 字节
    # （blockscale 是转置视图，cat 会自动落到新的 contiguous 缓冲）
    out = torch.cat([x_fp4, blockscale_e8m0], dim=-1)

    expected_cols = N // 4 + _cdiv(N // 2, _MXFP4_QUANT_BLOCK_SIZE)
    if out.shape != (M, expected_cols) or out.dtype != torch.uint8:
        raise RuntimeError(
            f"aiter 输出与题面契约不符：得到 {tuple(out.shape)}/{out.dtype}，"
            f"期望 ({M}, {expected_cols})/torch.uint8"
        )

    torch.cuda.synchronize()
    return out, {
        "path": "act_mul_and_mxfp4_quant",
        "aiter_module": "aiter.ops.triton.activation",
        "aiter_symbol": "act_mul_and_mxfp4_quant",
        "activation": activation,
        "scaling_mode": "even",
        "shuffle": False,
        "input_shape": [M, N],
        "x_fp4_shape": list(x_fp4.shape),
        "blockscale_shape": list(blockscale_e8m0.shape),
        "out_shape": list(out.shape),
        "out_dtype": str(out.dtype),
    }
