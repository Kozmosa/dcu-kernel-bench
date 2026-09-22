// 0001_vecadd — Starter（HIP track）
//
// 评测约定：
//   - 补全 vecadd_kernel 与 vecadd_launch；函数签名不可改。
//   - 只允许使用 HIP runtime / 设备代码；task.yaml 的 forbidden 列表会被静态审计。
//   - 由 evaluator 用 hipcc 与本文件连同 harness 一起编译，目标 gfx936/gfx938。

#include <hip/hip_runtime.h>

// TODO(Agent): 实现设备 kernel。
template <typename T>
__global__ void vecadd_kernel(const T* __restrict__ a,
                              const T* __restrict__ b,
                              T* __restrict__ out,
                              long long n) {
    // long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    // if (i < n) out[i] = a[i] + b[i];
}

// Host 启动入口。输入输出均为已分配好的设备指针。
// TODO(Agent): 选择 grid/block 并启动 kernel；返回 hipGetLastError()。
extern "C" hipError_t vecadd_launch_f32(const float* a, const float* b,
                                        float* out, long long n,
                                        hipStream_t stream) {
    return hipErrorNotReady;  // 替换为真实实现
}

extern "C" hipError_t vecadd_launch_f16(const __half* a, const __half* b,
                                        __half* out, long long n,
                                        hipStream_t stream) {
    return hipErrorNotReady;  // 替换为真实实现
}
