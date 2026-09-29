# Simulator control

生产边界是：

```text
TaskRuntime / SmokeRuntime
        → ToolDispatcher / EnvClient
        → SimulationServer
        → per-scene Backend
        → AirSim or UnrealCV adapter
```

- `TaskRuntime` 执行模型驱动任务，公共入口是 `run_task()`。
- `SmokeRuntime` 只验证 scene 生命周期、基础动作和 observation，不调用模型。
- 两者都必须由 `RuntimeFactory` 创建；CLI smoke 使用
  `RuntimeFactory.create_smoke_runtime()`。
- Runtime 和 Benchmark 不直接导入 AirSim、UnrealCV、SSH 或进程实现。
- SceneCapabilities 决定当前场景可暴露的动作；每个 SceneInstance 拥有独立 backend。

本目录编号文档提供部署和适配器细节；出现历史入口时以本文和源码为准。
