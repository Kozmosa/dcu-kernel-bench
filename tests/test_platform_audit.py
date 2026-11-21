"""平台符号检查（kind=platform）回归：CUDA 专有符号要抓，ROCm 正确写法不能误伤。"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "benchmark" / "evaluator" / "static_audit.py"
TASK = "1001_paged_attention"


def run_audit(files, extra=()):
    return subprocess.run(
        [sys.executable, str(AUDIT), *extra, str(REPO / "benchmark" / "tasks" / TASK), *map(str, files)],
        capture_output=True, text=True,
    )


def kinds(out: str) -> list[str]:
    import json
    return json.loads(out)["kinds"]


def test_rocm_native_code_passes(tmp_path: Path):
    """hip 头 + at::cuda 命名空间 + torch.cuda 都是 ROCm 上的正确写法。"""
    ok = tmp_path / "ok.py"
    ok.write_text(
        "import torch\n"
        "HIP = r'''\n#include <hip/hip_runtime.h>\n#include <ATen/cuda/CUDAContext.h>\n"
        "__global__ void k(float* x) { x[threadIdx.x] = warpSize; }\n'''\n"
        "def sync():\n    torch.cuda.synchronize()\n",
        encoding="utf-8",
    )
    r = run_audit([ok])
    assert r.returncode == 0, r.stdout


def test_cuda_runtime_header_is_flagged(tmp_path: Path):
    bad = tmp_path / "bad.py"
    bad.write_text("HIP = r'''\n#include <cuda_runtime.h>\n__global__ void k(){}\n'''\n", encoding="utf-8")
    r = run_audit([bad])
    assert r.returncode == 1
    assert "platform" in kinds(r.stdout)


def test_cudart_constant_is_flagged(tmp_path: Path):
    bad = tmp_path / "bad.py"
    bad.write_text("HIP = r'''\n#include <math_constants.h>\n__global__ void k(){ float m = CUDART_INF_F; }\n'''\n",
                   encoding="utf-8")
    r = run_audit([bad])
    assert r.returncode == 1
    assert "CUDART_INF_F" in r.stdout


def test_cuda_runtime_api_and_types_are_flagged(tmp_path: Path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "HIP = r'''\n#include <hip/hip_runtime.h>\n"
        "void f(){ void* p; cudaMalloc(&p, 16); cudaStream_t s; cudaMemcpy(p, p, 1, 0); }\n'''\n",
        encoding="utf-8",
    )
    r = run_audit([bad])
    assert r.returncode == 1
    out = r.stdout
    assert "cudaMalloc" in out and "cudaStream_t" in out


def test_ptx_inline_asm_is_flagged(tmp_path: Path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "HIP = r'''\n#include <hip/hip_runtime.h>\n__global__ void k(float* o){\n"
        "  asm volatile(\"shfl.sync.bfly.b32 %0, %0, 1, 31, -1;\" : \"+f\"(o[0]));\n}\n'''\n",
        encoding="utf-8",
    )
    r = run_audit([bad])
    assert r.returncode == 1
    assert "platform" in kinds(r.stdout)


def test_commented_cuda_symbols_are_ignored(tmp_path: Path):
    ok = tmp_path / "ok.py"
    ok.write_text(
        "HIP = r'''\n#include <hip/hip_runtime.h>\n// 以前用 #include <cuda_runtime.h>，现已换 HIP\n"
        "/* cudaMalloc 已删除 */\n__global__ void k(){}\n'''\n",
        encoding="utf-8",
    )
    r = run_audit([ok])
    assert r.returncode == 0, r.stdout


def test_docstring_mentioning_cuda_header_is_ignored(tmp_path: Path):
    ok = tmp_path / "ok.py"
    ok.write_text(
        '"""task 说明：device code must include <hip/hip_runtime.h> -- '
        '<cuda_runtime.h> is not available."""\n'
        "def f():\n    return 1\n",
        encoding="utf-8",
    )
    r = run_audit([ok])
    assert r.returncode == 0, r.stdout


def test_cuda_only_intrinsics_that_hip_supports_are_not_flagged(tmp_path: Path):
    """__launch_bounds__ / warpSize 在 hipcc 下都能编过（实测 48/108 次），不能误伤。"""
    ok = tmp_path / "ok.py"
    ok.write_text(
        "HIP = r'''\n#include <hip/hip_runtime.h>\n"
        "__global__ void __launch_bounds__(256) k(float* x){ x[threadIdx.x] = warpSize; }\n'''\n",
        encoding="utf-8",
    )
    r = run_audit([ok])
    assert r.returncode == 0, r.stdout


def test_triton_nvidia_libdevice_namespace_is_flagged(tmp_path: Path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import triton\nimport triton.language as tl\n"
        "from triton.language.extra import cuda as cu\n"
        "@triton.jit\ndef k(x):\n    return cu.exp(x)\n",
        encoding="utf-8",
    )
    r = run_audit([bad])
    assert r.returncode == 1
    assert "platform" in kinds(r.stdout)


def test_triton_generic_libdevice_is_not_flagged(tmp_path: Path):
    ok = tmp_path / "ok.py"
    ok.write_text(
        "import triton\nimport triton.language as tl\n"
        "import triton.language.extra.libdevice as libdevice\n"
        "@triton.jit\ndef k(x):\n    return libdevice.exp(x)\n",
        encoding="utf-8",
    )
    r = run_audit([ok])
    assert r.returncode == 0, r.stdout


def test_no_platform_flag_disables_the_check(tmp_path: Path):
    bad = tmp_path / "bad.py"
    bad.write_text("HIP = r'''\n#include <cuda_runtime.h>\n__global__ void k(){}\n'''\n", encoding="utf-8")
    assert run_audit([bad]).returncode == 1
    assert run_audit([bad], extra=("--no-platform",)).returncode == 0
