> Status: Historical

# FlyAgent 原生 Tool-Call Transcript 执行计划

状态：核心迁移、HTML 审计和首个真实 UnrealCV 单例验证均已完成；第一版不设置应用级 token 预算。

## 1. 目标与边界

目标是让模型在每次决策时看到完整、按时间排列的主动任务历史，从而知道：它基于什么状态提出了什么动作、Runtime 实际执行了什么、结果成功或失败、动作后的新状态是什么。

第一版只迁移 `TaskRuntime` 的任务飞行循环。`TaskParser` 仍负责生成不可变语义阶段；暂不增加独立的通用视觉问答入口。图片继续使用现有 artifact 与 data URL 管线，不重做图片存储。

明确不做：

- 不实现 token 计数、滑动窗口、摘要或静默截断。
- 不保留 JSON action-plan provider fallback。
- 不通过 `uav.`、`skill.` 等名称前缀推断执行行为。
- 不把动作历史塞回 action context module。

## 2. 最终上下文形态

同一 active instruction 内，发送给模型的消息随时间增长：

```text
system: FLYAGENT / USER 稳定规则
system: native tool-loop protocol
user: UserInstruction
user: StateCheckpoint(obs_0001 + 当前图片/阶段/测量状态)
assistant: public intent + tool_call(call_detect)
tool: ToolResult(call_detect, detections...)
assistant: public intent + tool_call(commit_1) + tool_call(turn_1) + tool_call(move_1)
tool: ToolResult(turn_1, requested/executed/safety...)
tool: ToolResult(move_1, requested/executed/safety...)
tool: ToolResult(commit_1, obs_0001 -> obs_0002, observed_effect...)
user: StateCheckpoint(obs_0002 + 新图片/阶段/测量状态)
assistant: public intent + tool_call(commit_2)
tool: ToolResult(commit_2, phase complete)
```

StateCheckpoint 只表达“现在是什么状态”。历史意图、调用、失败与结果只由 transcript 表达。`allowed_action_constraints` 与 `runtime_control.available/contracts` 不进入 provider 状态 JSON，因为 provider `tools` 已经是唯一可调用能力契约；动态 control status 可继续作为当前状态。

## 3. ToolSpec 统一契约

每个工具显式声明：

```text
canonical name
provider-safe wire_name
description
input_schema
runtime.effect
runtime.result_timing
runtime.batching
runtime.model_exposed
backend/handler binding
```

其中：

- `effect`：`read_only | physical | runtime_control | lifecycle`
- `result_timing`：`immediate | after_batch_observation | async | user_reply`
- `batching`：`single | ordered | parallel_safe | sidecar`

这些属性决定内外循环、结果何时回写、能否组成动作批次。名称只用于身份和日志。

## 4. 内外两层循环

内层循环绑定同一个观察状态：

1. 模型收到完整 transcript 和当前 provider tools。
2. read-only tool 立即执行并追加 ToolResult。
3. 参数或组合非法时追加 rejected ToolResult 和 protocol feedback。
4. 模型可依据真实结果再次调用工具。
5. `runtime.commit_phase_turn` 结束本次内层决策。
6. `max_tool_rounds=10` 限制总轮次，连续协议拒绝默认最多 3 次；成功 read-only
   ToolResult 会清零连续拒绝计数。

外层循环绑定真实环境变化：

1. incomplete commit 后接受 1-3 个 ordered physical calls。
2. Safety 逐项审查，记录 requested args、executed args 与 intervention。
3. 按顺序执行整批动作并在每步后稳定。
4. 整批结束后只 observe 一次。
5. 新 observation 写入 memory，生成下一 StateCheckpoint。
6. complete commit 触发阶段推进或任务完成。

因此，调用检测模型不会增加 flight window；一批连续飞行动作只增加一个 window。

## 5. 图片与 Observation ID

图片仍由 image context module 选择、融合和编码，并由 frame artifacts 持久化。模型可见标签加入 `observation_id`、`frame_id`、`artifact_ref`：

- ID 能明确说明图片属于哪次观察，也能让检测工具拒绝过期引用。
- VLM 只有在消息中实际再次收到图片 bytes/URL 时才能看见图片；ID 不是视觉内容。
- active-task transcript 在内存中保留完整 provider media；每个 observation 图片只出现一次。
- 图片按时间排列，旧图标记 `HISTORY`，最新图标记 `CURRENT`；不重放 phase-start 或 before-action 图片。
- durable `agent_transcript.jsonl` 不复制 base64，只记录占位符与 artifact reference。

未来增加目标检测、VLM caption、OCR 或深度查询时，只需注册新的 read-only `ToolSpec`，返回带 `tool_call_id`、`observation_id`、`frame_id` 和 artifact refs 的 ToolResult，不需要修改 transcript 结构。

## 6. 实施清单

- [x] 增加 `ToolRuntimePolicy`、provider wire name 和统一 JSON schema。
- [x] 将 callable capability runtime 属性设为必填。
- [x] 建立 canonical `AgentTranscript` 和 provider message projection。
- [x] 迁移 `ModelActionPlanner` 到原生 `tools/tool_calls`。
- [x] 建立内层 read-tool/self-repair loop 与外层 physical-window loop。
- [x] 增加 `runtime.commit_phase_turn`。
- [x] 增加通用 `runtime.report_blocked` 不完整终止及证据引用校验。
- [x] 增加完整 observation 时间序列视觉上下文与 HISTORY/CURRENT 标签。
- [x] 历史阶段曾增加小 ROI 米制深度工具；现已由 JSON 驱动的 Visual Grounding
  capability 取代，旧结果只保留离线回放兼容。
- [x] 支持同一窗口 1-3 个有序物理动作、整批一次观察。
- [x] 从 action context module 移除动作 transcript 代理字段。
- [x] 为图片增加 observation/frame/artifact 身份。
- [x] 删除独立 action `ModelRepairPlanner` 与旧 coupled JSON prompts。
- [x] 增加 transcript JSONL，持久化时剥离 base64。
- [x] 增加 detector -> result -> physical batch -> observation -> complete 端到端测试。
- [x] Benchmark audit 冻结 `agent_transcript.jsonl`，HTML 按 ToolCall ID 和
  observation ID 展示调用因果与对应画面。
- [x] 真实 Qwen provider 探针验证多模态、read-only ToolCall、ToolResult 回放、
  protocol feedback 自修复和 commit + physical calls。
- [x] 真实 DowntownWest 单例验证两次动作窗口、逐窗口观察、完整配对和正常 shutdown。
- [ ] 后续版本：任务跨进程恢复时从 artifact 重建 provider media。
- [ ] 后续版本：在真实 provider context limit 到达后设计显式、可审计的压缩策略。
- [ ] 后续版本：增加独立通用视觉问答入口。

## 7. 验收条件

- provider 请求包含 `tools`，不包含 action-plan `response_format`。
- 每个 assistant tool call 都有唯一 `tool_call_id` 和后续 ToolResult。
- 模型能在下一轮读到检测失败、执行失败和 safety 实际修改。
- read-only round 不增加 phase flight-window count。
- 1-3 个 physical calls 只产生一次 post-batch observation。
- provider 状态 JSON 不重复工具 schema/available list。
- `agent_transcript.jsonl` 可审计完整调用因果，但不嵌入图片 base64。
- 旧确定性 planner 仅作为本地测试兼容入口；生产 `ModelActionPlanner` 没有 JSON fallback。
