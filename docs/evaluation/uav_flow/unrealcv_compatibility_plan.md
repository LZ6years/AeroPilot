# UAV-Flow UnrealCV 迁移状态

本文件原先记录 gym-based benchmark 适配计划。该方案已经被通用 Runtime
composition 取代；当前实现以
[`benchmark_runtime_migration.md`](../benchmark_runtime_migration.md) 为准。

已完成：

- benchmark core、固定 executor 生命周期和冻结轨迹游标；
- UAV-Flow case/scenario/evaluation payload 类型化；
- MapPackage、SimulationServer、SimulatorEnvClient、TaskRuntime 统一执行链；
- 初始位姿、capture Profile、capability 校验复用 Runtime；
- UAV-Flow world meter → 初始局部 cm 坐标转换；
- nDTW、SR、plot、HTML 报告保留；
- gym scene lifecycle、benchmark 私有飞行控制、camera capture 和私有轨迹缓存
  删除；
- 显式注入的 Fake Backend、scenario、truth leak 和 trace 自动化测试；
- 真实引擎验收统一使用正式 `UavFlowBenchmarkRunner`；
- 根据真实 Turn 结果修正 `blueprint_yaw` 的命令 offset 和物理 yaw readback。

未改变：

- EnvClient v1；
- SimulationServer public contract；
- UnrealCV Backend public API；
- TaskRuntime public API；
- UAV-Flow upstream repository。

真实引擎验证命令：

```powershell
python evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py `
  --config evaluation/benchmarks/uav_flow/configs/uav_flow_turn_3cases.local `
  --execution-mode managed `
  --classes Turn `
  --limit-per-class 3 `
  --parser-mode model `
  --no-sr-judge `
  --no-visual-diagnostics
```
