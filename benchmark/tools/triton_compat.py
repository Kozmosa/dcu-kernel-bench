"""triton 版本兼容垫片（由 aiter 包导入时自动生效）。

本机正式栈是 DTK 26.04 自带的 **triton 3.3.0+das.opt1.dtk2604**；aiter 的部分内核按
更新的 DAS triton（3.5.1/3.6.0）编写，用到 3.3.0 没有的 API。这些 API 全部是
「语言层 / 宿主层」的东西，可以在 3.3.0 上等价补出来，**不需要换 triton 版本**。

> 为什么不该换版本：曾尝试装 DAS triton 3.5.1（cp310）做全量迁移，实测不可行——
> 该 wheel 的 DCU 代码生成改为 shell 出去调外部 clang，需要带 HCU 补丁的 clang-18
> （DTK 里的 `aillvm`），而本机是 dcc 25.10 / clang 17，矩阵核在链接期报
> `undefined symbol: llvm.hcu.mmac.f32.16x16x16.f16`。反过来 3.3.0 的 codegen 在
> libtriton 内部完成，`tl.dot` 正常（实测 max_diff 4e-5）。所以正确做法是在 3.3.0
> 上补齐这些语言层 API。

已补四项，均在本机实测验证：

1. `tl.matrix_load`                 —— 占位，解开 1010 / 1022 的 aiter 基线；
                                       1029 走真 matrix_load 路径时会显式报错（设计如此）
2. `tl.constexpr_function`          —— 等价实现，解开 moe_op_mxfp4 的编译期求值
3. `triton.autotune(perf_debug=)`   —— 吞掉 3.3.0 不认识的调优开关，解开
                                       1012_flash_attention_forward 的 aiter 基线
4. `triton.utils.hcutuner`          —— 3.3.0 里连 `triton.utils` 包都不存在，重建它，
                                       解开 1026 / 1027 sage_attention 的 aiter 基线
"""

import functools
import inspect
import sys
import types

__all__ = ["install"]


# --------------------------------------------------------------------------- #
# 1. tl.matrix_load
# --------------------------------------------------------------------------- #
def _install_matrix_load(tl) -> None:
    """补 `tl.matrix_load` 占位。

    aiter 部分内核在 `if USE_MATRIX_LOAD:` 分支里引用它，而宿主会把 USE_MATRIX_LOAD
    显式设为 False。致命点在于 3.3.0 的 `DependencyFinder` 会在 JIT 之前**遍历整个
    函数 AST** 解析所有被引用的属性（不区分分支是否会被执行），因此即使分支永不执行，
    也会抛 `AttributeError: module 'triton.language' has no attribute 'matrix_load'`。

    这里补一个「一旦被真正调用就报错」的占位：既让 AST 解析通过，又保证若某条路径
    真的走回 matrix_load，会显式失败而不是静默算错。
    """
    if hasattr(tl, "matrix_load"):
        return

    def matrix_load(*_args, **_kwargs):
        raise NotImplementedError(
            "tl.matrix_load 在本机 triton 3.3.0 中不可用（缺少 HCU matrix 指令支持）。"
            "请将该内核的 USE_MATRIX_LOAD 设为 False，走普通 tl.load 路径。"
        )

    tl.matrix_load = matrix_load   # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# 2. tl.constexpr_function
# --------------------------------------------------------------------------- #
def _install_constexpr_function(tl) -> None:
    """补 `tl.constexpr_function`（3.5.1 的等价实现）。

    3.5.1 的实现（`triton/runtime/jit.py`）：把函数包成 `ConstexprFunction`，调用时
    解包 constexpr 实参、执行原函数、再把结果包回 `constexpr`。

    3.3.0 没有这个类，但它的 `compiler/code_generator.py::visit_Call` 收尾是：

        if fn in self.builtin_namespace.values():
            args = map(_unwrap_if_constexpr, args)
        ret = fn(*args, **kws)        # ← 非 JITFunction 的可调用对象被**直接调用**

    也就是任意可调用对象都会在 trace 期被真实调用。**但还有一个拦路的守卫**：
    `_define_name_lookup.global_lookup` 只放行「constexpr 全局变量」，否则抛
    `NameError: Cannot access global variable ... from within @jit'ed function`。

    于是正解是把包装对象做成真正的 `tl.constexpr`：

    * 守卫侧：`_is_constexpr_global` → `_is_constexpr` → `isinstance(val, constexpr)`
      ⇒ 放行；
    * 代码生成侧：`visit_Call` 先 `_unwrap_if_constexpr` 拿到 `.value`（我们的包装
      函数）再直接调用 ⇒ 走通；
    * 宿主侧：`constexpr.__getattr__` 把 `__call__` 转发到 `.value`
      ⇒ `dtype_tag(tl.float32)` 这类宿主调用照常返回裸值。

    比设置 `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1` 更可取：后者是全局放开守卫，
    会把**生成器产出的内核**里的非法全局引用也一并放过，等于削弱基准的静态约束；
    这里只给这一个 API 开绿灯。
    """
    if hasattr(tl, "constexpr_function"):
        return

    from triton.language.core import constexpr

    try:
        from triton.language.core import _unwrap_if_constexpr
    except ImportError:                                    # 兜底：老版本可能没导出
        def _unwrap_if_constexpr(x):
            return x.value if isinstance(x, constexpr) else x

    def constexpr_function(fn):
        @functools.wraps(fn)
        def _impl(*args, **kwargs):
            # 复刻 3.5.1：先解包 constexpr 实参，再交给原始 Python 函数
            args = [_unwrap_if_constexpr(a) for a in args]
            kwargs = {k: _unwrap_if_constexpr(v) for k, v in kwargs.items()}
            return fn(*args, **kwargs)

        # 必须包成 constexpr，见上面的守卫说明
        return constexpr(_impl)

    tl.constexpr_function = constexpr_function   # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# 3. triton.autotune(..., perf_debug=...)
# --------------------------------------------------------------------------- #
def _install_autotune_compat(triton) -> None:
    """让 `triton.autotune` 接受并忽略 3.5.1 才有的调优开关。

    3.5.1 的 autotune 有 `perf_debug` / `perf_profiling`（打印调优过程），3.3.0 的
    签名里没有。aiter 内核在**装饰器位置**写死 `perf_debug=True`，于是模块在被
    import 时就抛 `TypeError: autotune() got an unexpected keyword argument`，
    适配器根本加载不了。这里在包装层丢弃它们——只影响日志输出，不参与编译。
    """
    orig = getattr(triton, "autotune", None)
    if orig is None or getattr(orig, "_dsl_compat_wrapped", False):
        return

    @functools.wraps(orig)
    def autotune(*args, **kwargs):
        kwargs.pop("perf_debug", None)
        kwargs.pop("perf_profiling", None)
        return orig(*args, **kwargs)

    autotune._dsl_compat_wrapped = True   # type: ignore[attr-defined]
    triton.autotune = autotune


# --------------------------------------------------------------------------- #
# 4. triton.utils.hcutuner
# --------------------------------------------------------------------------- #
def _install_hcutuner(triton) -> None:
    """重建 `triton.utils` / `triton.utils.hcutuner`。

    3.3.0 里**整个 `triton.utils` 包都不存在**，而 aiter 有四处依赖它：

        from triton.utils.hcutuner import get_gpu_label          # 1026/1027/quant_per_block
        @triton.utils.hcutune(...)                               # gemm_w8a8
        functools.partial(triton.utils.run_saved_kernel, ...)     # extend_attention

    3.5.1 的 `triton/utils/hcutuner.py` 只有 1880 字节且几乎全是平凡实现，逐一对齐即可：
    `hcutune` 就是 `autotune`（只是多几个 3.3.0 不认识的调优开关），`hcutune_configured` /
    `graphtune` 是恒等装饰器，`get_gpu_label` 由「arch + CU 数」拼出。

    各名字同时挂到 `triton.utils` 与 `triton.utils.hcutuner` 两级——aiter 两种写法都有。
    """
    utils_name = "triton.utils"
    mod_name = "triton.utils.hcutuner"

    utils = sys.modules.get(utils_name)
    if utils is not None and getattr(utils, "hcutuner", None) is not None:
        return

    # 3.3.0 的 autotune：只保留它认识的形参
    base_autotune = getattr(triton, "autotune", None)
    try:
        from triton.runtime.autotuner import autotune as base_autotune   # noqa: F811
    except Exception:
        pass
    accepted = set(inspect.signature(base_autotune).parameters) if base_autotune else set()

    def get_gpu_label():
        """复刻 3.5.1：`f"{arch}:cu_{num_cu}"`。"""
        arch = "unknown"
        try:
            from triton.runtime.driver import driver as _driver
            arch = _driver.active.get_current_target().arch
        except Exception:
            try:
                import torch
                arch = torch.cuda.get_device_properties(
                    torch.cuda.current_device()).gcnArchName
            except Exception:
                pass
        num_cu = 0
        try:
            import torch
            num_cu = torch.cuda.get_device_properties(
                torch.cuda.current_device()).multi_processor_count
        except Exception:
            pass
        return f"{arch}:cu_{num_cu}"

    def hcutune(configs, key, **kwargs):
        # perf_debug / perf_profiling / always_tuning 在 3.3.0 的 autotune 里没有，
        # 它们只影响调优过程的日志，丢掉不影响结果
        for unknown in ("perf_debug", "perf_profiling", "always_tuning"):
            kwargs.pop(unknown, None)
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        return base_autotune(configs=configs, key=key, **kwargs)

    def hcutune_configured(get_config_fn=None, get_key_fn=None):
        if callable(get_config_fn) and get_key_fn is None:
            return get_config_fn

        def decorator(fn):
            return fn

        return decorator

    def graphtune(fn):
        return fn

    def run_saved_kernel(fn, path, *args, grid, **kwargs):
        del path
        return fn[grid](*args, **kwargs)

    class ConfigLoader:
        def __init__(self, config_dir=None, device=None):
            self.config_dir = config_dir
            self.device = device

        def get_tuned_cache(self, op_name, device_name):
            return None

    exports = {
        "get_gpu_label": get_gpu_label,
        "hcutune": hcutune,
        "hcutune_configured": hcutune_configured,
        "graphtune": graphtune,
        "run_saved_kernel": run_saved_kernel,
        "ConfigLoader": ConfigLoader,
    }

    if utils is None:
        utils = types.ModuleType(utils_name)
    utils.__path__ = []                      # 声明成包，供 `from triton.utils... import`
    sub = sys.modules.get(mod_name) or types.ModuleType(mod_name)
    for name, obj in exports.items():
        setattr(sub, name, obj)
        setattr(utils, name, obj)            # aiter 里也直接写 triton.utils.hcutune
    utils.hcutuner = sub
    sub.__all__ = sorted(exports)
    sub.__doc__ = "3.3.0 兼容垫片重建的 triton.utils.hcutuner（见 aiter/_triton_compat.py）"

    sys.modules[utils_name] = utils
    sys.modules[mod_name] = sub
    triton.utils = utils


# --------------------------------------------------------------------------- #
def install() -> None:
    """在 aiter 包被导入时应用全部垫片。任一垫片失败都不应影响其它。"""
    try:
        import triton
        import triton.language as tl
    except Exception:
        return

    for fn, target in (
        (_install_matrix_load, tl),
        (_install_constexpr_function, tl),
        (_install_autotune_compat, triton),
        (_install_hcutuner, triton),
    ):
        try:
            fn(target)
        except Exception:
            # 垫片只做兼容，不能因为补不上就让 import aiter 失败
            pass


install()
