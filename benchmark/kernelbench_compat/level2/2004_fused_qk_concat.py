# 2004_fused_qk_concat — MLA 风格 QK 预处理：pe 段 RoPE 旋转 + nope/pe 拼接（KernelBench 兼容 model_class 题目）。
#
# 本文件必须自包含：评测器以 exec(source) 加载题目，禁止本地模块导入，
# 参考实现（Model.forward）与输入生成全部内联。
#
# 运行方式（PyramidKernel 原生 model_class 路径）：
#   kernelbench_root 指向 benchmark/kernelbench_compat/，
#   level=2, problem_id=2004, backend=triton。

import torch
import torch.nn as nn


class Model(nn.Module):
    """MLA 风格 QK 预处理融合算子：pe 段旋转位置编码（RoPE）+ nope/pe 拼接。

    ① 算子语义（数学定义与边界行为）：
    输入 Q 侧 q_nope [B, QH, D_nope]、q_pe [B, QH, D_pe] 与 K 侧
    k_nope [B, KH, D_nope]、k_pe [B, KH, D_pe]，约束 QH 为 KH 的整数倍
    （GQA，记 QH_PER_KH = QH // KH），D_pe 为偶数。输出打包为单张量
    out [B, QH + KH, D_nope + D_pe]：head 维前 QH 行是 Q 结果、后 KH 行
    是 K 结果（每个 KV head 只计算并写回一次，即 GQA 去重写回语义）。

    nope 段原样拷贝：out[:, :QH, :D_nope] = q_nope、out[:, QH:, :D_nope]
    = k_nope。pe 段（x 指 q_pe 或 k_pe，计算前提升 float32）：
    - apply_rope=False（纯拼接模式）：out[..., D_nope:] = x 原样拷贝，
      pos / cos / sin 被忽略；
    - apply_rope=True（rope 模式）：对每个 batch b 取 c = cos[pos[b]]、
      s = sin[pos[b]]（宽 d_freq；d_freq == D_pe 为全宽表，
      d_freq == D_pe//2 为前半复用表），按旋转风格扩展到宽 D_pe：
        全宽：c~ = c；
        前半复用 + NEOX：c~ = [c, c]（整段重复一次，前后两半同频）；
        前半复用 + GPTJ：c~ = [c0, c0, c1, c1, ...]（每元素紧邻重复两次）。
      rotate_half(x) 定义：
        NEOX（is_neox=True）：rotate_half(x)[i] = -x[i + D_pe//2]
          （i < D_pe//2），= x[i - D_pe//2]（i >= D_pe//2）；
        GPTJ（is_neox=False）：rotate_half(x)[2j] = -x[2j+1]，
          rotate_half(x)[2j+1] = x[2j]。
      pe 段输出 = x * c~ + rotate_half(x) * s~，cast 回输入 dtype。
    边界行为：pos[b] ∈ [0, max_pos)（越界为未定义行为）；本算子是
    memory-bound 预处理，不计算任何 attention 分数；nope 段与纯拼接
    模式的 pe 段为原样拷贝，须逐位一致；整个前向 out-of-place，
    不得原地改写任何输入。

    ② 输入输出规格：
    - q_nope / q_pe / k_nope / k_pe：float16 / bfloat16，contiguous，
      形状如上（QH == KH * QH_PER_KH，四张量 batch 一致）；
    - pos：int64，[B]，每 batch 一个位置索引（评测精度下被 cast 成
      浮点，入口无损恢复为整型）；
    - cos / sin：与数据同 dtype，[max_pos, d_freq]，contiguous，只按
      cos[pos[b], d] 行式索引使用；
    - 返回：out [B, QH + KH, D_nope + D_pe]，dtype 与 q_nope 一致；
    - 超参（__init__）：is_neox —— True 为 NEOX 旋转风格 / False 为
      GPTJ；apply_rope —— True 对 pe 段做旋转 / False 纯拼接。

    ③ 实现约束（违规判负）：
    - 核心计算（cos/sin 按位置索引、pe 段旋转、nope/pe 拼接写回、
      GQA 下 K 侧去重）必须在提交文件内以自写 Triton kernel 完成：
      单个 kernel（一次 launch）、单次读入-计算-写回 pass；
    - ModelNew 的 __init__ 与 forward 签名不可更改；
    - 中间计算用 float32；输出 cast 回输入 dtype。

    ④ 禁用列表与目标硬件：
    - 禁止调用 torch.cat / torch.concat（含张量方法 .cat）、index_select
      等 ATen 捷径直接完成本题计算；禁用清单以 task.yaml forbidden 列表
      为准（含来源算子库与其它闭源加速库，终审静态审计对 docstring /
      注释之外的代码做子串匹配）；
    - 目标硬件：海光 DCU（gfx936），实现语言 Triton。
    """

    def __init__(self, is_neox: bool = True, apply_rope: bool = True):
        super().__init__()
        self.is_neox = bool(is_neox)
        self.apply_rope = bool(apply_rope)

    def forward(self, q_nope, q_pe, k_nope, k_pe, pos, cos, sin):
        # 评测器会把全部输入张量统一 cast 成评测精度（triton 后端为
        # fp32）；pos 是位置索引（整型语义），max_pos < 2^24 时 fp32 可
        # 精确表示，此处无损恢复为整型。注：forward 的语句保持单行
        # （框架 scaffold 逐语句重排缩进，多行块会被破坏）
        pos = pos.to(torch.long)
        B, QH, D_nope = q_nope.shape
        KH = k_nope.shape[1]
        D_pe = q_pe.shape[-1]
        assert q_pe.shape[0] == B and k_nope.shape[0] == k_pe.shape[0] == B, "batch 维必须一致"
        assert q_pe.shape[1] == QH and k_pe.shape[1] == KH, "Q/K 各自的 head 数必须一致"
        assert k_nope.shape[2] == D_nope and k_pe.shape[2] == D_pe, "q/k 的 D_nope、D_pe 必须一致"
        assert QH % KH == 0, "QH 必须是 KH 的整数倍（GQA）"
        assert D_pe % 2 == 0, "D_pe 必须为偶数（半宽旋转）"
        out = torch.empty((B, QH + KH, D_nope + D_pe), dtype=q_nope.dtype, device=q_nope.device)
        out[:, :QH, :D_nope] = q_nope
        out[:, QH:, :D_nope] = k_nope
        # rope 模式：按位置取 cos/sin 行并扩展到宽 D_pe（float32）；纯拼接
        # 模式下 cos_f/sin_f 不参与结果（下方三元式直接取原样 pe 段）
        d_freq = cos.shape[-1]
        assert d_freq in (D_pe, D_pe // 2), "cos/sin 宽度必须是 D_pe 或 D_pe//2"
        cos_b = cos.to(torch.float32)[pos]
        sin_b = sin.to(torch.float32)[pos]
        cos_f = (torch.cat((cos_b, cos_b), dim=-1) if self.is_neox else cos_b.repeat_interleave(2, dim=-1)) if d_freq == D_pe // 2 else cos_b
        sin_f = (torch.cat((sin_b, sin_b), dim=-1) if self.is_neox else sin_b.repeat_interleave(2, dim=-1)) if d_freq == D_pe // 2 else sin_b
        cos_f = cos_f.unsqueeze(1)                # [B, 1, D_pe] 广播到 head 维
        sin_f = sin_f.unsqueeze(1)
        rotate = (lambda x: torch.cat((-x[..., D_pe // 2:], x[..., : D_pe // 2]), dim=-1)) if self.is_neox else (lambda x: torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2))
        q32, k32 = q_pe.to(torch.float32), k_pe.to(torch.float32)
        out[:, :QH, D_nope:] = (q32 * cos_f + rotate(q32) * sin_f).to(q_nope.dtype) if self.apply_rope else q_pe
        out[:, QH:, D_nope:] = (k32 * cos_f + rotate(k32) * sin_f).to(q_nope.dtype) if self.apply_rope else k_pe
        return out


def get_init_inputs():
    return [True, True]   # is_neox=NEOX 旋转风格；apply_rope=rope+拼接（旗舰路径）


def get_inputs():
    # 固定 shape 族（与 public_cases.json 一致）；随机部分消费全局 RNG——
    # 评测器在 set_seed 后调用本函数，多轮 correctness trial 因此获得输入多样性
    B, KH, qh_per_kh = 8, 4, 4
    QH = KH * qh_per_kh                 # 16
    D_nope, D_pe, d_freq = 512, 64, 32  # 前半复用表（MLA 常用配置）
    max_pos = 1024
    dt = torch.bfloat16

    q_nope = torch.randn((B, QH, D_nope), dtype=dt)
    q_pe = torch.randn((B, QH, D_pe), dtype=dt)
    k_nope = torch.randn((B, KH, D_nope), dtype=dt)
    k_pe = torch.randn((B, KH, D_pe), dtype=dt)
    pos = torch.randint(0, max_pos, (B,), dtype=torch.int64)
    freqs = torch.randn((max_pos, d_freq), dtype=dt)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return [q_nope, q_pe, k_nope, k_pe, pos, cos, sin]


def make_inputs(B, KH, QH_PER_KH, D_nope, D_pe, max_pos, d_freq=None,
                dtype="bfloat16", seed=0, is_neox=True, apply_rope=True):
    """确定性 case 生成器：torch.Generator().manual_seed(seed)，CPU 生成。

    字段与隐藏/性能 case 一一对应（B / KH / QH_PER_KH / D_nope / D_pe /
    max_pos / d_freq / dtype / seed）；is_neox 与 apply_rope 是 Model 初始
    化配置，case 一并记录以便评测端构造 Model，不参与张量生成。
    仅在评测端运行（生成 [q_nope, q_pe, k_nope, k_pe, pos, cos, sin]）。
    """
    gen = torch.Generator().manual_seed(seed)
    dt = getattr(torch, dtype)
    QH = KH * QH_PER_KH
    if d_freq is None:
        d_freq = D_pe
    assert D_pe % 2 == 0, "D_pe 必须为偶数"
    assert d_freq in (D_pe, D_pe // 2), "cos/sin 宽度必须是 D_pe 或 D_pe//2"

    q_nope = torch.randn((B, QH, D_nope), dtype=dt, generator=gen)
    q_pe = torch.randn((B, QH, D_pe), dtype=dt, generator=gen)
    k_nope = torch.randn((B, KH, D_nope), dtype=dt, generator=gen)
    k_pe = torch.randn((B, KH, D_pe), dtype=dt, generator=gen)
    pos = torch.randint(0, max_pos, (B,), generator=gen)
    freqs = torch.randn((max_pos, d_freq), dtype=dt, generator=gen)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return [q_nope, q_pe, k_nope, k_pe, pos, cos, sin]
