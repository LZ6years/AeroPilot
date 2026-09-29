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

## 碰撞结果

暂停式 Multirotor 控制把结果区分为 `ok`、`collision` 和 `error`。只有动作期间
产生的新碰撞（或明确的 `collision_blocked`）、冻结成功且最终位姿可信时，才返回
可恢复的 `collision`。Runtime 会停止当前物理窗口的剩余调用，采集新 observation，
再交给模型重新规划。无碰撞的大幅位姿跳变、RPC/cancel/freeze 失败以及不可用位姿
仍是 `error`，并可要求重建 scene。成功动作中仅有轻微接触诊断时仍保持 `ok`。

本目录编号文档提供部署和适配器细节；出现历史入口时以本文和源码为准。
