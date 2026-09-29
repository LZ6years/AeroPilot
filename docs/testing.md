# 测试与实验分层

## 产品回归

默认测试只验证离线产品契约，不启动真实 GPU、网络或模拟器：

```powershell
& 'F:\conda_env\flyagent\python.exe' -m pytest -q
```

共享 fake 和 builder 位于 `tests/support/`。产品测试不应直接依赖
`tests/manual/`。

## Diagnostics

`tests/diagnostics/` 包含 profiler、GPU、network、simulator 和长实验检查，不进入
默认 pytest：

```powershell
& 'F:\conda_env\flyagent\python.exe' -m pytest -q tests/diagnostics
```

可使用 `diagnostic`、`gpu`、`simulator`、`network` markers 做进一步选择。

## Manual 与真实实验

`tests/manual/` 是显式运行的脚本，清单见该目录 README。输出必须写到
`runs/experiments/<probe>/<timestamp>/`。实验只记录 commit、dirty diff hash、配置
和必要 artifact，不复制完整源码树。真实感知/GPU 实验必须记录解释器与 CUDA
诊断，CUDA 不可用时不得标为 GPU 推理。
