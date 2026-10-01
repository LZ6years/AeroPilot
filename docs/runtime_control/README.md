# Runtime built-in controls

生产接口使用单一原生工具循环：

- `runtime.update_plan`：更新可选任务计划；可单独调用，也可与独立的
  `parallel_safe` 只读调用出现在同一回合。
- `runtime.report_completed`：以当前 instruction 的环境证据报告完成。
- `runtime.report_blocked`：以证据和阻塞条件报告无法继续。
- `knowledge.load`：只读工具；正文以 ToolResult 进入同一 Transcript。

每个 plan step 都带 `evidence_refs`；`completed` 必须引用回合开始前已经存在的
observation、成功 ToolResult 或 action ref。Runtime 只验证引用存在性，不替模型判断
证据是否真正证明地标、路线关系或到达条件。同回合混合执行时，`TurnExecutor` 先完整
校验参数、配额、step ID 和证据引用；计划无效则所有调用都不执行。有效计划先提交，
随后执行只读 batch；单个只读失败不回滚计划，Transcript 仍按提交顺序记录结果。

控制工具由 `TurnExecutor` 校验和执行。计划与知识变化不发布 observation checkpoint；
控制处理器不启动第二个模型，不执行物理动作。
所有正式依赖由 `RuntimeFactory.create_task_runtime()` 创建。

本目录旧编号设计已移入 `docs/archive/runtime_control/`。
