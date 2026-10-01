# Context V3

`InstructionEnvelope` 在 instruction 开始时创建一次，包含 protocol、身份文档、
用户偏好、知识目录和静态模块说明；原始 instruction 仍由 `user_instruction` 事件
唯一拥有。

`StateCheckpoint` 只保存 checkpoint ID、发布原因、observation ID、动态环境
`ContextFragment`、`MediaRef` 和覆盖感知状态的 fingerprint。任务计划和知识索引不在
checkpoint 中；相同感知 fingerprint 不会重复发布。

`ContextFragment` 原生只含 `name`、最终 `content` 与有序 `media_refs`。来源、置信度、
warnings、artifact refs、metadata 与构建耗时位于独立 `FragmentAudit`，不进入模型
字段。

`MissionState.task_plan` 是当前计划的权威状态，`KnowledgeService` 是知识加载状态的
权威所有者，`MissionState.loaded_knowledge` 只是同步后的轻量索引。Runtime 在每次
`plan_turn()` 前临时构造一份 `CURRENT TASK STATE`，它位于完整历史之后、周期提醒
之前，并且既不写入 Transcript，也不复制到 checkpoint。历史压缩只能概括历史事件，
不能修改这一份当前状态；完整知识正文继续由成功的 `knowledge.load` ToolResult 保存。

`HISTORY CAPSULE` 只描述 `through_event_index` 及之前已经发生的进展、决策结果和
失败恢复。它不包含当前待办、下一步或剩余计划；这些内容只由请求末尾的
`CURRENT TASK STATE` 表达。Capsule 是 Provider 请求投影，不会替换或删改持久化的
完整 Transcript。

`TaskPlanStep` 原生包含 `id`、`description`、`status` 和 `evidence_refs`。所有引用必须
来自计划回合开始前已经存在的 observation 或成功 ToolResult/action ref，`completed`
状态必须至少包含一项引用。动作调用成功只直接证明该动作的执行结果，不自动证明地标、
路线关系或目标接近条件；这些语义判断仍由模型结合后续观察完成。

`MediaRef` 只保存路径/引用、SHA、MIME、尺寸和 observation/frame ID。Runtime 持久化
的 `ref` 固定为当前 run 目录下的 POSIX 相对路径，例如
`artifacts/scene_x/rgb/frame_000001.png`；禁止绝对路径、`..` 和依赖进程工作目录的
解析。选中的引用由 `MediaHydrator` 严格相对当前 `run_dir` 读取为 `PreparedMedia`；
instruction 级 SHA-LRU 上限为 32 项、32 MiB，结束后清空。独立诊断工具可以在不提供
`base_dir` 时使用绝对临时引用，但这种引用不得写入 Runtime 的 V3 artifact。

Provider 顺序固定为：身份和偏好；Planner/ToolUser；原始 instruction；protocol、
知识目录和静态说明；按时间排序的 checkpoint/assistant/tool/feedback；一份
`CURRENT TASK STATE`；可选进度提醒。

V3 artifact：

```text
contexts/instruction_envelope.json
contexts/latest_checkpoint.json
debug/context_checkpoints/<checkpoint_id>.json
memory/agent_transcript.jsonl
artifacts/model_media/<observation_id>/<sha>.png
```

V2 及更旧 run 不会自动迁移；读取时抛出 `UnsupportedRunSchema`。
