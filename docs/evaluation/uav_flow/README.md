# UAV-Flow Benchmark 文档索引

本目录记录 UAV-Flow 作为外部 benchmark 接入 UAV-Agent 的需求、边界和实现影响。

UAV-Flow 在这里不是 runtime integration。它的角色是提供任务集、UnrealCV 场景、参考轨迹和评估口径，用来评估 UAV-Agent 原生闭环在 Flow-style 指令上的效果。

## 文档

| 文件 | 内容 |
|---|---|
| `external_eval_requirements.md` | 需求和边界：为什么接 UAV-Flow、输入输出按谁、指标怎么解释、哪些信息不能进入 planner |
| `unrealcv_compatibility_plan.md` | 实现影响：代码放哪里、哪些模块新增、哪些主链路不改、测试和输出怎么组织 |

## 目录约定

建议实现代码放在：

```text
evaluation/benchmarks/uav_flow/
```

手动真实环境入口放在：

```text
tests/manual/uav_flow_eval/
```

评估输出放在：

```text
evaluation/benchmarks/uav_flow/outputs/
```

临时诊断输出可以放在：

```text
tests/manual_outputs/uav_flow_eval/
```

## 边界摘要

- 不增加 HTTP policy server。
- 不兼容 UAV-Flow 原始 `/predict` 作为目标。
- 不输出 UAV-Flow `action_tokens`。
- 不让 UAV-Flow runner 控制主 loop。
- 不让 `target_pos` 和 `reference_path_preprocessed` 进入 planner context。
- 使用 UAV-Agent 自己的 observation、context、planner、tool action 和停止条件。
