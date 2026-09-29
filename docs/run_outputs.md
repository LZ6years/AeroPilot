# Run outputs

正式 run 使用 Context V2：

```text
<run>/
  contexts/instruction_envelope.json
  contexts/latest_checkpoint.json
  debug/context_checkpoints/<checkpoint_id>.json
  debug/tool_loop/<instruction>/<decision>.json
  memory/agent_transcript.jsonl
  memory/observations.jsonl
  memory/actions.jsonl
  artifacts/model_media/<observation_id>/<sha>.png
  state/latest_state.json
  state/flight_memory.json
  events.jsonl
```

Transcript 与 checkpoint 禁止内嵌 base64。Provider 请求中的临时媒体仅在请求组装
期间存在；需要审计的媒体按 SHA 写入 artifact。

`runs/experiments/` 属于诊断/实验输出，不是产品 run schema。脚本应使用新的时间戳
目录，保存 manifest、环境诊断和必要结果。已删除的 `tests/manual_outputs/` 不再是
合法输出位置。

V1 artifact 不被 V2 reader 兼容或静默解析。

物理窗口的 `status` 可以是 `ok`、`collision` 或 `error`；步骤还可以是
`skipped`。`collision` 表示可恢复的环境碰撞，不计作基础设施失败。碰撞步骤的
真实冻结位姿和窗口后 observation 仍进入 trajectory，但评测不会拿未完成的原请求
动作计算控制误差；报告使用独立的 collision step/window 计数。
