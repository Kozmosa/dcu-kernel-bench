# benchmark/sources/ — 来源审核层（仅维护者）

每个准入任务对应一份 `<id>.yaml` 审核记录，从 aiter（或其他题源）提取时填写：

```yaml
task_id: "0001_vecadd"
source:
  repo: OpenDAS/aiter
  remote: <克隆地址>
  commit: <40位哈希>
  license: <LICENSE 文件名与类型>
  files:                       # 完整记录，不只是一个 kernel 文件
    - path: csrc/...
      sha256: <证据哈希>
      role: device_kernel | host_launch | test | header
admission:
  core_compute_in_source: true   # 核心计算确实在可审查的 HIP/Triton 源码中
  calls_closed_libs: false       # 不依赖 MIOpen/rocBLAS/hipBLASLt/hipDNN 完成核心计算
  builds_standalone: true        # 能在 environment.yaml 锁定环境中独立编译
  has_reference: true            # 能写出独立 PyTorch/CPU reference
  decision: admitted             # admitted | rejected
  reviewer: ""
  date: ""
  note: ""
```

准入标准（参照 BANG C 指南）：核心计算在 Device 端可审查源码中完成；能独立编译运行；I/O、dtype、shape、layout、属性可明确描述；能构造正常/边界/性能输入；能写出独立 reference。排除：只有 Host 包装、核心计算走闭源库、缺依赖无法复现、与已有题高度重复。
