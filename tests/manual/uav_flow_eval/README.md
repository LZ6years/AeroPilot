# UAV-Flow 手工测试入口

UAV-Flow 的真实人工验收只使用正式链路：

```text
UavFlowBenchmarkRunner
→ TaskRuntime
→ ModelActionPlanner
→ Benchmark Core
```

不再提供固定动作 benchmark probe 或生产 dry 执行。
自动测试通过显式依赖注入验证生命周期、指标和报告，不代表真实 benchmark 结果。

## 本地真实 Turn 三案例

本机 MapPackage 配置需要指向 DowntownWest 可执行文件，然后运行：

```powershell
python evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py `
  --config evaluation/benchmarks/uav_flow/configs/uav_flow_turn_3cases.local `
  --execution-mode managed `
  --classes Turn `
  --limit-per-class 3 `
  --eval-id uav_flow_turn_local_real_3cases `
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

## 离线自动化测试

```powershell
python -m pytest tests/test_uav_flow_runner.py tests/test_benchmark_core.py -q
```

本目录保留共享 STMR session 的 `run_stmr_canary.py`；真实 attach/managed
评测使用正式入口，Fake 实现仅供测试注入。
