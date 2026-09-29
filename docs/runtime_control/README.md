# Runtime built-in controls

生产接口使用单一原生工具循环：

- `runtime.update_plan`：独占控制回合，更新可选任务计划。
- `runtime.report_completed`：以当前 instruction 的环境证据报告完成。
- `runtime.report_blocked`：以证据和阻塞条件报告无法继续。
- `knowledge.load`：只读工具；正文以 ToolResult 进入同一 Transcript。

控制工具由 `TurnExecutor` 校验和执行，状态变化后的 checkpoint 由
`ContextCheckpointService` 发布。控制处理器不启动第二个模型，不执行物理动作。
所有正式依赖由 `RuntimeFactory.create_task_runtime()` 创建。

本目录旧编号设计已移入 `docs/archive/runtime_control/`。
