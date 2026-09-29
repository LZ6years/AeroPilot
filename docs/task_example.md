# Phase-free 任务闭环示例

用户输入：“向前移动，确认到达后停止。”

1. RuntimeFactory 创建共享 Plugin/Tool/Context/Planner 依赖。
2. TaskRuntime 获取场景并完成 preflight。
3. ContextCheckpointService 发布 `instruction_start`。
4. 模型返回一个 `uav.move_body` 原生 tool call。
5. TurnExecutor 校验回合；Safety 审查后交给 PhysicalWindowExecutor。
6. 物理窗口执行动作、稳定等待并自动采集一次 after observation。
7. Runtime 发布 `physical_window_completed`；下一请求读取新的 checkpoint。
8. 模型调用 `runtime.report_completed` 并引用当前 observation 证据。
9. Runtime 采集终态证据，发布 `task_terminal`，执行 shutdown 与 cleanup。

只读工具可在任意非物理回合运行。加载知识导致模型状态变化时发布
`knowledge_loaded`；普通只读结果和协议拒绝只进入 Transcript。计划是可选的；
`runtime.update_plan` 成功后发布 `plan_updated`。
