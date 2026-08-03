# AerialVLN Benchmark

AerialVLN v1 复用通用 benchmark executor 和现有 AirSim Multirotor Backend。
它兼容 official episode data、初始位姿、front RGB/depth/pose 和指标语义，但
动作协议是 UAV-Agent 原生连续 primitive：

```text
aerial_vln.uav_agent_native_actions.v1
```

配置示例：

```text
evaluation/benchmarks/aerial_vln/configs/aerial_vln_eval.example.json
```

必须显式配置 `scene_map`，例如：

```json
{
  "scene_map": {
    "1": "airvln_env_1"
  }
}
```

未知 scene 或缺失 MapPackage 会在启动模拟器前失败。official actions、
goal 和 reference path 只用于 evaluation/debug truth，不进入 Runtime
artifacts 或 Planner context。

## 可视化审计

AerialVLN 与 UAV-Flow 共用 `evaluation/benchmarks/core/` 的审计和 HTML
报告。每个 case 会生成 `audit.json`、`review.html` 和便携的
`review_assets/`，评测根目录生成统一 `review.html`。

`dry` 模式可用占位帧测试报告结构。真实 `attach/managed` 运行如果启用
`generate_html_review`，必须同时启用 `save_artifacts` 和
`enable_action_planner_debug`；正式 CLI 会自动设置。若只需要指标 JSON，
可显式设置 `generate_html_review=false`。

先运行自动测试：

```powershell
python -m pytest tests/test_aerial_vln_cases.py `
  tests/test_aerial_vln_adapter.py `
  tests/test_aerial_vln_metrics.py `
  tests/test_aerial_vln_runner_dry.py -q
```

真实 `airvln_env_1` 探针：

```powershell
python tests/manual/aerial_vln_real_engine_probe.py `
  --config evaluation/benchmarks/aerial_vln/configs/aerial_vln_eval.example.json `
  --execution-mode attach
```

正式 runner：

```powershell
python evaluation/benchmarks/aerial_vln/scripts/run_eval.py `
  --config evaluation/benchmarks/aerial_vln/configs/aerial_vln_eval.example.json `
  --execution-mode attach `
  --max-cases 1
```
