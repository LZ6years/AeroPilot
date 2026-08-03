# UAV-Flow 人工测试

## 1. 本地 MapPackage

真实 UAV-Flow 使用本地 DowntownWest。不要把机器相关的绝对路径提交到公共示例
配置；创建被 git 忽略的 `.local` MapPackage，并确认：

- `scene_id=DowntownWest`；
- `control.protocol=unrealcv`；
- actor 为 `BP_Drone01_2`；
- `units_per_meter=100`；
- `yaw_offset_deg=-180`；
- `camera_follow=standard_pose`；
- `camera_validation.enabled=true`，`position_mode=max_distance`，
  camera/Actor 最大距离 `1.5m`、yaw 容差 `1°`；
- UnrealCV 监听 `127.0.0.1:9068`。

## 2. 正式真实评测

真实人工验收只调用正式 Runner，不使用静态 Parser、固定 Planner 或预设动作：

```powershell
python evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py `
  --config evaluation/benchmarks/uav_flow/configs/uav_flow_turn_3cases.local `
  --execution-mode managed `
  --case-ids 2025-03-30_11-49-14,2025-03-30_11-49-27,2025-03-30_11-52-45 `
  --eval-id uav_flow_turn_local_real_3cases `
  --parser-mode model `
  --planner-depth-input off `
  --stmr-mode off `
  --no-sr-judge `
  --no-visual-diagnostics
```

前三个 Turn case 是：

1. `2025-03-30_11-49-14`；
2. `2025-03-30_11-49-27`；
3. `2025-03-30_11-52-45`。

每个 case 使用独立 Runtime scene session。真实 HTML 运行自动启用 RGB artifact 和
Planner debug。检查：

```text
outputs/<eval_id>/review.html
outputs/<eval_id>/cases/<case_id>/case.json
outputs/<eval_id>/cases/<case_id>/review.html
outputs/<eval_id>/cases/<case_id>/audit.json
outputs/<eval_id>/cases/<case_id>/turn_diagnostics.json
```

Turn diagnostics 包含 reference/actual yaw delta、最终 yaw 误差、平移漂移和初始
姿态误差。HTML gallery 只展示 RGB；Depth 仍保存在 `audit.json` 和审计资产中。
不再生成 case 级 Markdown review；`case.json` 只保存 Core HTML 路径和紧凑的
`audit_summary`。

`case_ids` 是运行选择，不是环境校准。同一 DowntownWest 下更换案例只需替换 ID，
不修改 MapPackage。报告中的 `backend_action_tracking.json` 分别记录 yaw 和
translation primitive 的请求/观测误差；模型零动作结束会明确标为
`zero_action_completion`。

## 3. Dry 自动化测试

Dry 模式使用 in-memory UnrealCV Backend，只验证代码契约，不属于真实 benchmark：

```powershell
python evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py `
  --config evaluation/benchmarks/uav_flow/configs/uav_flow_eval.example.json `
  --execution-mode dry `
  --case-ids 2025-03-30_11-49-14 `
  --eval-id uav_flow_dry `
  --parser-mode rule_based `
  --no-sr-judge `
  --no-visual-diagnostics `
  --no-action-planner-debug
```

## 4. STMR

STMR Profile 必须包含：

```json
{
  "modalities": ["rgb", "depth", "segmentation", "pose"]
}
```

Profile 不满足要求时，Runner 会在 scene 启动前失败。
