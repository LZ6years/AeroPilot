# Context V2

`InstructionEnvelope` 在 instruction 开始时创建一次，包含 protocol、身份文档、
用户偏好、知识目录和静态模块说明；原始 instruction 仍由 `user_instruction` 事件
唯一拥有。

`StateCheckpoint` 保存 checkpoint ID、发布原因、observation ID、任务计划、已加载
知识索引、动态 `ContextFragment`、`MediaRef` 和覆盖模型可见状态的 fingerprint。
相同 fingerprint 不会重复发布。

`ContextFragment` 只含最终 `model_content` 与有序 `model_inputs`。来源、置信度、
warnings、artifact refs、metadata 与构建耗时位于独立 `FragmentAudit`，不进入模型
字段。

`MediaRef` 只保存路径/引用、SHA、MIME、尺寸和 observation/frame ID。Runtime 持久化
的 `ref` 固定为当前 run 目录下的 POSIX 相对路径，例如
`artifacts/scene_x/rgb/frame_000001.png`；禁止绝对路径、`..` 和依赖进程工作目录的
解析。选中的引用由 `MediaHydrator` 严格相对当前 `run_dir` 读取为 `PreparedMedia`；
instruction 级 SHA-LRU 上限为 32 项、32 MiB，结束后清空。独立诊断工具可以在不提供
`base_dir` 时使用绝对临时引用，但这种引用不得写入 Runtime 的 V2 artifact。

Provider 顺序固定为：身份和偏好；Planner/ToolUser/案例说明；原始 instruction；
protocol、知识目录和静态说明；按时间排序的 checkpoint/assistant/tool/feedback；
可选进度提醒。

V2 artifact：

```text
contexts/instruction_envelope.json
contexts/latest_checkpoint.json
debug/context_checkpoints/<checkpoint_id>.json
memory/agent_transcript.jsonl
artifacts/model_media/<observation_id>/<sha>.png
```

V1 run 不会自动迁移；读取时抛出 `UnsupportedRunSchema`。
