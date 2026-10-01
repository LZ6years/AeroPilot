# Phase-free 任务闭环示例

用户输入：“向前移动，确认到达后停止。”

1. RuntimeFactory 创建共享 Plugin/Tool/Context/Planner 依赖。
2. TaskRuntime 获取场景并完成 preflight。
3. ContextCheckpointService 发布 `instruction_start`。
4. 模型返回一个 `uav.move_body` 原生 tool call。
5. TurnExecutor 校验回合；Safety 审查后交给 PhysicalWindowExecutor。
6. 物理窗口执行动作、稳定等待并自动采集一次 after observation。
7. Runtime 发布 `physical_window_completed`；下一请求读取新的 checkpoint，并在历史
   后附加一份当前计划与知识索引。
8. 模型调用 `runtime.report_completed` 并引用当前 observation 证据。
9. Runtime 采集终态证据，发布 `task_terminal`，执行 shutdown 与 cleanup。

只读工具可在任意非物理回合运行。一个 `runtime.update_plan` 可以和若干独立的
`parallel_safe` 只读调用同回合提交，但计划只能引用回合开始前的证据；同回合新得到的
只读结果要到下一次计划修订才能引用。计划先提交、只读调用后执行，ToolResult 仍按模型
提交顺序进入 Transcript。二者的最新状态在下一请求的 `CURRENT TASK STATE` 中统一
体现，不产生额外 observation checkpoint。
