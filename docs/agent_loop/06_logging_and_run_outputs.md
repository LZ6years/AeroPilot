# 06 日志与 Run Outputs

本文说明 Agent Recovery 的运行记录应该如何落盘，以及 CLI 应该展示到什么程度。

## 1. 设计原则

recovery 日志目标是帮助开发者回答：

- 哪个模型模块输出错了？
- 第几次 attempt 错了？
- selector 选择了哪个 `RecoveryTool`？
- validator/contract 给出的错误是什么？
- tool 重写后输出了什么？
- 最终是修复成功，还是 attempts exhausted？

但它不应该把大上下文或敏感内容塞进普通事件流。

## 2. Debug artifact

建议落盘：

```text
runs/<run_id>/
  debug/
    agent_recovery/
      task_parser_attempt_001.json
      action_planner_attempt_001.json
```

attempt 文件建议字段：

```json
{
  "source": "task_parser",
  "stage": "semantic_validation",
  "attempt": 1,
  "max_attempts": 2,
  "selected_tool": "semantic_plan_rewrite",
  "status": "failed",
  "error": {
    "type": "SemanticPlanValidationError",
    "message": "Semantic plan must not contain primitive action token: takeoff"
  },
  "invalid_output": {},
  "recovery_request": {},
  "tool_output": {}
}
```

## 3. `events.jsonl`

`events.jsonl` 只保存轻量摘要和 artifact ref。

建议事件类型：

```text
agent_recovery_requested
agent_recovery_tool_selected
agent_recovery_attempt_failed
agent_recovery_succeeded
agent_recovery_exhausted
```

事件 payload 示例：

```json
{
  "source": "task_parser",
  "stage": "semantic_validation",
  "attempt": 1,
  "max_attempts": 2,
  "selected_tool": "semantic_plan_rewrite",
  "status": "repaired",
  "error_type": "SemanticPlanValidationError",
  "error_message": "Semantic plan must not contain primitive action token: takeoff",
  "artifact_ref": "debug/agent_recovery/task_parser_attempt_001.json"
}
```

## 4. CLI 展示

普通 CLI 不需要展示完整 recovery 过程。

建议：

- recovery 成功：默认不刷屏，最多 debug 模式显示一行。
- recovery 失败：summary 显示 `task_parser recovery exhausted` 或 `action_planner recovery exhausted` 和最后 validator/contract error。
- `/context` 或 run debug artifact 用于查看详细 attempts。

这样 CLI 不会变成模型调试日志，但开发者仍能追溯。

## 5. `interactive/latest_summary.json`

recovery 成功时：

- 不需要把每次 recovery 写进最终 summary。
- 可以在 debug 字段中保留 `recovery_triggered=true`、`selected_tool` 和 `successful_attempt`。

recovery 失败时：

```json
{
  "ok": false,
  "error_type": "RecoveryExhaustedError",
  "error": "task_parser recovery exhausted",
  "recovery": {
    "source": "task_parser",
    "stage": "semantic_validation",
    "selected_tools": ["semantic_plan_rewrite"],
    "attempts": 2,
    "last_error_type": "SemanticPlanValidationError",
    "last_error_message": "Semantic plan must not contain primitive action token: takeoff"
  }
}
```

## 6. 禁止落盘内容

attempt、event、summary 都不应包含：

- API key。
- Authorization header。
- `.env` 原文。
- `data:image`。
- base64 图片内容。
- 完整 RGB/depth。
- 完整 `contexts/latest_context.json`。
- UAV-Flow evaluation-only truth，例如 `target_pos` 和 reference path。

## 7. 文档同步

实现时需要同步更新：

```text
docs/run_outputs.md
```

补充内容：

- `debug/agent_recovery/*.json` 目录说明。
- 新增 event 类型说明。
- 说明 recovery logs 是开发调试产物，不是 planner context。

## 8. 与 AgentMessage 的关系

Agent Recovery 不应该直接写 AgentMessage。

原因：

- AgentMessage 是面向用户/CLI 的任务进度或最终回答。
- recovery attempt 是模型输出调试信息。
- 如果 recovery 成功，用户不一定需要知道模型曾经输出错。
- 如果 recovery 失败，最终 summary 可以表达失败原因，但不需要把每次 attempt 作为 agent message。
