# UAV-Flow 手工测试入口

UAV-Flow 的真实人工验收只使用正式链路：

```text
UavFlowBenchmarkRunner
→ TaskRuntime
→ ModelActionPlanner
→ Benchmark Core
```

不再提供固定转向或移动动作的 UAV-Flow real-engine probe。`dry` 模式只用于
自动化测试数据加载、生命周期、指标和报告结构，不能作为真实 benchmark 结果。

## 本地真实 Turn 三案例

本机 MapPackage 配置需要指向 DowntownWest 可执行文件，然后运行：

```powershell
python evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py `
  --config evaluation/benchmarks/uav_flow/configs/uav_flow_turn_3cases.local `
  --execution-mode managed `
  --classes Turn `
  --limit-per-class 3 `
  --eval-id uav_flow_turn_local_real_3cases `
  --parser-mode model `
  --planner-depth-input off `
  --stmr-mode off `
  --no-sr-judge `
  --no-visual-diagnostics
```

真实 HTML 运行会自动启用 `save_artifacts` 和
`enable_action_planner_debug`。运行完成后检查：

```text
evaluation/benchmarks/uav_flow/outputs/<eval_id>/review.html
evaluation/benchmarks/uav_flow/outputs/<eval_id>/cases/<case_id>/review.html
evaluation/benchmarks/uav_flow/outputs/<eval_id>/cases/<case_id>/audit.json
evaluation/benchmarks/uav_flow/outputs/<eval_id>/cases/<case_id>/turn_diagnostics.json
```

HTML gallery 默认只展示 RGB。Depth 仍保存在审计数据和资产目录中。

## Dry 自动化 smoke

```powershell
python evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py `
  --config evaluation/benchmarks/uav_flow/configs/uav_flow_eval.example.json `
  --execution-mode dry `
  --parser-mode rule_based `
  --case-ids 2025-03-30_11-49-14 `
  --no-sr-judge `
  --no-visual-diagnostics `
  --no-action-planner-debug
```

本目录只保留共享 STMR session 的 `run_stmr_canary.py`；普通 dry、attach 和
managed 评测都使用上面的正式入口。
