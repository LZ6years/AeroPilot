# Run 输出目录约定

本文档定义 `runs/<run_id>/` 的落盘结构。目标是把系统状态、上下文记忆、原始大文件、调试材料和 demo/eval 输出分开，避免同一份 RGB/depth、state 或事件在多个目录重复保存。

开发脚本放置规则见 [development_conventions.md](development_conventions.md)：底层诊断和真实引擎探针优先放 `tests/manual/`，`demos/` 只放用户可体验的完整流程。

## 目录结构

```text
runs/<run_id>/
  events.jsonl
  state/
    latest_state.json
    flight_memory.json
    semantic_plan.json
    scene_instance.json
  memory/
    instructions.jsonl
    actions.jsonl
    observations.jsonl
    poses.jsonl
    frames.jsonl
  artifacts/
    rgb/
    depth/
  contexts/
    latest_context.json
    latest_phase_transition_context.json
  debug/
    debug_events.jsonl
    system.log
    latest_observation.json
    phase_windows/*.json
    phase_transition/*.json
  config/
  demo/
  eval/
  scenes/
```

## 主要文件

| 路径 | 类型 | 用途 |
| --- | --- | --- |
| `events.jsonl` | 简版事件流 | 给用户和 CLI/TUI 展示关键流程：指令开始、阶段规划、工具完成、观察更新、阶段推进。 |
| `state/latest_state.json` | 当前状态快照 | 当前 runtime 控制状态，只保存最新状态和当前 phase 进度。 |
| `state/flight_memory.json` | 记忆 manifest | 保存 `memory/*.jsonl` 的 refs，供 context modules 查历史。 |
| `state/semantic_plan.json` | 语义解析结果 | 当前或最近一条自然语言指令解析出的阶段计划。 |
| `state/scene_instance.json` | 场景信息 | 当前 UE/AirSim scene 的实例信息。 |
| `memory/*.jsonl` | 上下文历史 | append-only 结构化历史。ContextModule 从这里读取，不保存 RGB/depth 大文件。 |
| `artifacts/rgb` / `artifacts/depth` | 原始感知文件 | 真实 RGB/depth 文件唯一保存位置。 |
| `contexts/latest_context.json` | 模型动作输入 | `ModelActionPlanner` 使用的最终 PlannerContext。 |
| `contexts/latest_phase_transition_context.json` | 阶段判断输入 | 执行动作和自动观察后，给 phase judge 的上下文。 |
| `debug/debug_events.jsonl` | 详细事件流 | 给开发者排查模型输出、安全拦截、稳定等待、repair、phase judge 细节。 |
| `debug/latest_observation.json` | 最新原始 observation | 仅用于开发查看，不作为 context module 的历史来源。 |
| `debug/phase_windows/*.json` | 阶段窗口详情 | 每个 phase window 的完整执行记录。 |
| `debug/phase_transition/*.json` | 阶段判断详情 | 每次 post-action phase judge 的完整输入。 |

## 已删除的旧输出

新 run 不再生成以下路径：

- `mission_state.json`
- 根目录 `latest_state.json`
- 根目录 `flight_memory.json`
- 根目录 `semantic_plan.json`
- 根目录 `scene_instance.json`
- `observations/latest_observation.json`
- `memory/artifacts/`

读取旧 run 的工具可以保留兼容 fallback，但新代码不应该继续写这些路径。

## Event 分层

`events.jsonl` 是简版事件流，面向用户和后续 TUI。它只记录大阶段摘要，不放完整 observation、完整模型 payload 或完整 phase judge context。

`debug/debug_events.jsonl` 是开发流。模型原始 plan、validated plan、SafetySupervisor intervention、repair 尝试、稳定等待轮询和 shutdown 细节都写在这里。

完整大对象优先写到 `debug/phase_windows/*.json` 或 `debug/phase_transition/*.json`，event 里只保留 `debug_ref` / `context_ref` 和摘要字段。

## Context 与 Memory 的关系

`memory/*.jsonl` 是结构化历史来源。`state/flight_memory.json` 只保存 refs，不复制历史正文。`contexts/latest_context.json` 是给模型的整理结果，允许包含历史摘要，但不直接暴露完整 raw jsonl。

RGB/depth 文件只保存在 `artifacts/`。`memory/frames.jsonl` 只保存 frame metadata 和 artifact path。
