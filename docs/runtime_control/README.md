# Runtime built-in controls

生产接口使用单一原生工具循环：

- `runtime.update_plan`：更新可选任务计划；可单独调用，也可与独立的
  `parallel_safe` 只读调用出现在同一回合。
- `runtime.report_completed`：以当前 instruction 的环境证据报告完成。
- `runtime.report_blocked`：以证据报告持续且无法由当前能力解决的阻塞；目标暂时
  不可见本身不是阻塞条件。
- `knowledge.load`：只读工具；正文以 ToolResult 进入同一 Transcript。

每个 plan step 只包含 `id`、`description`、`status`。`completed` 是模型记录的阶段
判断，不能作为任务完成的环境证据。计划接口不接受 `evidence_refs`；终止工具保留
自己的环境证据要求。同回合混合执行时，`TurnExecutor` 先完整
校验参数、配额、step ID 和最终合并后的状态；计划无效则所有调用都不执行。有效计划先提交，
随后执行只读 batch；单个只读失败不回滚计划，Transcript 仍按提交顺序记录结果。

控制工具由 `TurnExecutor` 校验和执行。计划与知识变化不发布 observation checkpoint；
控制处理器不启动第二个模型，不执行物理动作。
所有正式依赖由 `RuntimeFactory.create_task_runtime()` 创建。

物理回合固定为一个 `single` 调用，动作执行后自动取得新图片和反馈，再由模型决定下一动作。
例如转向和前进需要分为两个回合；多个物理调用整回合拒绝，诊断为 `not_started`，
不会自动拆分或执行前缀。单次移动可以同时包含水平和垂直分量。
历史 `ordered_steps`、`after_batch_observation` 字段保留，离线审计可以读取以前的多动作窗口；
历史兼容不会放宽当前回合校验。

## 计划输入与记录

接口使用 `operation=replace/patch` 与单个 `steps` 数组，并启用 `strict:true`。
顶层 `operation/explanation/steps` 和步骤 `id/description/status` 全部必填。
replace 创建或重排完整快照，description/status 不可为 null，保留已有 ID；不再需要的
步骤标记 cancelled。patch 数组非空，只引用已有 ID；description/status 为 null 时
保留原值，每项至少一个非 null 修改。explanation 为 null 保留原说明。

```json
{"operation":"replace","explanation":null,"steps":[{"id":"approach","description":"接近入口","status":"in_progress"},{"id":"pass","description":"穿过入口","status":"pending"}]}
```

```json
{"operation":"patch","explanation":null,"steps":[{"id":"approach","description":null,"status":"completed"},{"id":"pass","description":null,"status":"in_progress"}]}
```

所有更新先合并，再检查最多一个 `in_progress`，因此上例不依赖更新数组的顺序。
错误输入不改变计划或 revision；成功输入使 revision 加一。`MissionState.task_plan`
是计划校验的唯一状态输入，更新不再为证据检查复制完整历史。它
保存最新不可变快照，Transcript 保存真实参数及完整结果：`previous_revision`、`plan`、
`changed_step_ids`。后者按保存顺序列出新增、内容、状态或顺序变化的 ID。
历史成功回执发给模型时只保留 revision、变化 ID 和说明，真实原生调用参数仍保留；
最新完整计划由每轮唯一的 `CURRENT TASK STATE` 提供。

计划与飞行动作分回合。计划更新不发布新观测，也不重置非物理回合或拒绝次数限制。
模型在阶段完成、进度判断纠正或路线方案变化时更新；进展未变化时无需更新。

计划工具描述负责适用时机：较长、困难或有多个顺序阶段需要保留进度时按需使用。
主提示和周期提醒要求记录的阶段跟上实际进展；进展未改变时无需提交更新。
不要求首次物理动作前创建，也不要求每次动作前进行额外计划核对。
Runtime 只验证参数和保存状态，不判断任务复杂度或里程碑是否完成；
计划全部 completed 也不建立环境任务成功的证据。

共享诊断位于 `src/tools/diagnostics.py`。ToolResult 的可选 diagnostic 与协议反馈、
任务故障报告共用 code/phase；工具诊断必须包含 execution。执行前拒绝为 not_started，
执行中故障不回滚已有提交，RPC 超时为 unknown。LLM 错误由 Runtime 转换，模块只
提供错误事实。旧 data.rejection/error_type 仅由离线审计适配，原记录不改写。

## 离线验证

```powershell
& 'F:\conda_env\flyagent\python.exe' tests/manual/unified_tool_loop_replay_probe.py --output-dir runs/diagnostics/plan_context_replay --rgb tests/fixtures/visual_description_images/frame_000001.png
```

使用尚不存在的输出目录。该探针以脚本工具调用覆盖创建、飞行、增量推进、描述纠正及
再次推进，并生成 `replay_report.json` 与 HTML review；不会请求真实模型或评估导航成功。
旧日志的计划证据适配只存在于离线审计边界，不放宽生产工具接口。

本目录旧编号设计已移入 `docs/archive/runtime_control/`。
