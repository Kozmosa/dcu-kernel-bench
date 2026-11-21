#!/usr/bin/env python3
"""machine_fingerprint.py —— 一条命令测出这台 DCU 机器的指纹（在目标机上跑）。

为什么要这个脚本：换机器 / 换组之后，必须先判断"旧基线、旧 wiki 硬件页还能不能用"。
判据不是看平台标签，而是看下面这些实测值是否与锁定的基线来源一致
（现有基线来自 BW / gfx936:sramecc+:xnack- / 80 CU / 68.7 GB）。

用法（需要 hipcc 与 DTK 环境，CPU-only 机器上会跳过 HIP 探针）：

    source /opt/dtk/env.sh
    python3 benchmark/tools/machine_fingerprint.py            # 人读表格
    python3 benchmark/tools/machine_fingerprint.py --json     # 机器可读
    python3 benchmark/tools/machine_fingerprint.py --expect environment.yaml

`--expect` 会读 environment.yaml 的 `hardware.benchmark_target_arch` 与
`hardware.measured_machine`，逐项比对并给出结论：
  · 架构/设备名一致  → 只重采基线即可
  · 架构不同         → 基线重采 + wiki 硬件页要重测（架构口径要先决策）
"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ATTR = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
  int v = 0;
  hipDeviceProp_t p{};
  hipGetDeviceProperties(&p, 0);
  printf("device\t%s\n", p.name);
  printf("gcn_arch\t%s\n", p.gcnArchName);
  printf("compute_units\t%d\n", p.multiProcessorCount);
  printf("vram_mib\t%.0f\n", p.totalGlobalMem / 1024.0 / 1024.0);
  printf("wavefront\t%d\n", p.warpSize);
  printf("max_threads_per_block\t%d\n", p.maxThreadsPerBlock);
  printf("regs_per_block\t%d\n", p.regsPerBlock);
  printf("l2_kib\t%d\n", p.l2CacheSize / 1024);
  printf("clock_mhz\t%d\n", p.clockRate / 1000);
  return 0;
}
"""

SMEM64 = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void k64(float* o) { __shared__ float s[16384]; s[threadIdx.x] = 1.f; o[threadIdx.x] = s[threadIdx.x]; }
int main() { float* o; hipMalloc(&o, 4096); k64<<<1,256>>>(o);
             printf("smem_64k\t%s", hipGetErrorString(hipGetLastError())); return 0; }
"""

# 96 KiB 静态 shared 在这类卡上是**编译期**错误（local memory exceeds limit），所以单独编、按结果记录。
SMEM96 = r"""
#include <hip/hip_runtime.h>
__global__ void k96(float* o) { __shared__ float s[24576]; s[threadIdx.x] = 1.f; o[threadIdx.x] = s[threadIdx.x]; }
int main() { float* o; hipMalloc(&o, 4096); k96<<<1,256>>>(o); return 0; }
"""

BW = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void cp4(const float4* a, float4* b, int n) { int i = blockIdx.x*blockDim.x+threadIdx.x; if (i < n) b[i] = a[i]; }
int main() { size_t bytes = 512ull<<20; size_t n = bytes/sizeof(float4); float4 *a, *b;
  hipMalloc(&a, bytes); hipMalloc(&b, bytes); hipMemset(a, 0, bytes);
  hipEvent_t s, e; hipEventCreate(&s); hipEventCreate(&e);
  for (int i = 0; i < 3; i++) cp4<<<(n+255)/256,256>>>(a,b,n);
  hipDeviceSynchronize(); hipEventRecord(s);
  for (int i = 0; i < 10; i++) cp4<<<(n+255)/256,256>>>(a,b,n);
  hipEventRecord(e); hipEventSynchronize(e);
  float ms; hipEventElapsedTime(&ms, s, e);
  printf("copy_GBps\t%.1f\n", 2.0*bytes*10/1e9/(ms/1000.0)); return 0; }
"""


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace", **kw)


def probe(name, source, device_id=None):
    if not shutil.which("hipcc"):
        return {"error": "hipcc not found (source /opt/dtk/env.sh ?)"}
    with tempfile.TemporaryDirectory(prefix="fp_") as tmp:
        src = pathlib.Path(tmp) / f"{name}.cpp"
        exe = pathlib.Path(tmp) / name
        src.write_text(source, encoding="utf-8")
        flags = ["--offload-arch=" + device_id] if device_id else []
        build = run(["hipcc", "-O3", *flags, str(src), "-o", str(exe)])
        if build.returncode != 0:
            return {"error": "compile failed", "log": (build.stderr or build.stdout)[-400:]}
        out = run([str(exe)])
        values = {}
        for line in out.stdout.splitlines():
            if "\t" in line:
                key, _, value = line.partition("\t")
                values[key] = value
        if out.returncode != 0:
            values["error"] = (out.stderr or "")[-200:]
        return values


def torch_info():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"torch_error": f"{type(exc).__name__}: {exc}"}
    info = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
            "available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()}
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception as exc:  # noqa: BLE001
        info["triton_error"] = f"{type(exc).__name__}: {exc}"
    return info


def host_info():
    info = {"host": os.uname().nodename, "nproc": os.cpu_count()}
    for key, path in (("cgroup_memory_max", "/sys/fs/cgroup/memory.max"),
                      ("cgroup_cpu_max", "/sys/fs/cgroup/cpu.max")):
        try:
            info[key] = pathlib.Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            info[key] = None
    return info


def compare(expect_path, fingerprint):
    try:
        import yaml
    except Exception:  # noqa: BLE001
        return ["（本机无 pyyaml，跳过 --expect 比对）"]
    spec = yaml.safe_load(pathlib.Path(expect_path).read_text(encoding="utf-8")) or {}
    hw = spec.get("hardware") or {}
    target = hw.get("benchmark_target_arch") or []
    machine = hw.get("measured_machine") or {}
    lines = []
    arch = fingerprint.get("gcn_arch") or ""
    if target and arch:
        base = arch.split(":")[0]
        if base in target:
            lines.append(f"架构一致（{base} ∈ target_arch）→ 只需重采基线")
        else:
            lines.append(f"⚠ 架构不一致：实测 {base or '?'}，题面包声明 {target} "
                         f"→ 先决定架构口径（换机器 / 迁移题面 / 双架构），再谈基线")
    for key, expect_value in (("device", machine.get("device_name")),
                              ("compute_units", machine.get("compute_units")),
                              ("vram_mib", machine.get("vram_mib"))):
        got = fingerprint.get(key)
        if expect_value is not None and got is not None and str(got) != str(expect_value):
            lines.append(f"⚠ {key}: 实测 {got}，environment.yaml 记录 {expect_value}")
    if not lines:
        lines.append("与 environment.yaml 记录一致")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="只输出 JSON")
    parser.add_argument("--expect", type=pathlib.Path, help="比对 environment.yaml")
    args = parser.parse_args()

    fingerprint = {}
    host = host_info()
    fingerprint.update(host)
    fingerprint.update(torch_info())
    arch = None
    attrs = probe("attr", ATTR)
    if "error" not in attrs:
        arch = attrs.get("gcn_arch", "").split(":")[0] or None
        fingerprint.update(attrs)
    else:
        fingerprint["attr_error"] = attrs
    smem64 = probe("smem64", SMEM64, device_id=arch)
    fingerprint.update(smem64 if "error" not in smem64 else {"smem64_error": smem64})
    smem96 = probe("smem96", SMEM96, device_id=arch)
    if "error" in smem96:
        log = str(smem96.get("log") or smem96.get("error"))
        first = next((ln for ln in log.splitlines() if "error:" in ln), log[-120:])
        fingerprint["smem_96k"] = "compile_failed (预期：超出 64 KiB 上限) ← " + first.strip()
    else:
        fingerprint["smem_96k"] = smem96.get("error") or "compiled (超出预期，需复核上限)"
    bw = probe("bw", BW, device_id=arch)
    if "error" not in bw:
        fingerprint.update(bw)
    else:
        fingerprint["bw_error"] = bw

    if args.json:
        print(json.dumps(fingerprint, ensure_ascii=False, indent=1))
        return 0

    width = max(len(k) for k in fingerprint)
    for key, value in fingerprint.items():
        print(f"{key:<{width}}  {value}")
    if args.expect:
        print("\n=== 与期望比对 ===")
        for line in compare(args.expect, fingerprint):
            print(" ", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
