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
