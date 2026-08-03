# UAV-Flow Benchmark

UAV-Flow 在本项目中是外部 benchmark，不拥有 UAV-Agent 的控制循环。
它提供 case、初始位姿、场景目标、reference path 和评测语义；执行统一走：

```text
SimulationServer → SimulatorEnvClient → TaskRuntime
```

当前协议标识：

```text
uav_flow.uav_agent_native_actions.v1
```

实现位于 `evaluation/benchmarks/uav_flow/`：

- `cases.py`：读取数据集；
- `adapter.py`：cm → canonical meter InitialPose，并隔离 truth；
- `scenario.py`：临时特权 UnrealCV 连接，只布置目标物体；
- `runner.py`：组合 Runtime 和通用 benchmark executor；
- `trajectory.py`：Runtime world pose → 初始局部 cm；
- `metric_suite.py` / `metrics.py`：轨迹导出和 nDTW；
- `reports.py`：UAV-Flow 指标解释和专属报告数据；

UAV 初始姿态保持旧 benchmark 契约：应用数据集 XYZ 和 yaw，pitch/roll 归零。
部分 Shift 数据的 slots 3/5 仍保留在 evaluation truth，但不发送给只支持
Blueprint yaw 的 DowntownWest 无人机。
- `evaluation/benchmarks/core/audit.py` / `review.py`：通用审计日志、图片
  资产和 HTML 输出。

旧 gym scene lifecycle 和 benchmark 私有飞行 EnvClient 已退出执行链路。

## 配置

示例：

```text
evaluation/benchmarks/uav_flow/configs/uav_flow_eval.example.json
```

场景配置使用：

```json
{
  "map_config_path": "configs/maps/unrealcv_actor_control.example.json",
  "map_id": "downtownwest_local_unrealcv",
  "execution_mode": "dry",
  "observation_profile_path": null,
  "headless": true,
  "save_artifacts": false,
  "generate_html_review": true
}
```

旧 `env_id`、gym reuse、time dilation、resolution 和 seed 配置会给出明确迁移
错误。

`dry` 模式允许不保存真实图片，用于验证报告结构。`attach/managed` 模式若
启用 `generate_html_review`，必须同时启用 `save_artifacts` 和
`enable_action_planner_debug`；仓库提供的真实评测脚本会自动设置这两个开关。

每个 case 输出：

```text
cases/<case_id>/case.json
cases/<case_id>/audit.json
cases/<case_id>/review.html
cases/<case_id>/review_assets/
```

评测根目录输出统一入口 `review.html`。SR judge、visual diagnostics 和 STMR
仍由 UAV-Flow 配置显式控制，不会因为启用通用审计报告而自动调用模型。
通用 HTML gallery 默认只显示 RGB；Depth preview/raw 继续保存在
`audit.json` 和 `review_assets`，但不会渲染到页面。真实人工验收统一使用正式
Runner，不再维护固定动作的 UAV-Flow real-engine probe。
`case.json` 中只保留指向 Core HTML 的 `review_html` 和紧凑
`audit_summary`；不再生成 case 级 Markdown review。

## 观测

- 普通运行：`rgb + depth + pose`；
- STMR：`rgb + depth + segmentation + pose`；
- `planner_depth_input` 只决定 depth 是否进入模型 context，不改变 Profile。

## 测试

自动测试：

```powershell
$files = Get-ChildItem tests/test_uav_flow_*.py
python -m pytest $files -q
```

真实 DowntownWest Gate：

```powershell
python evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py `
  --config evaluation/benchmarks/uav_flow/configs/uav_flow_turn_3cases.local `
  --execution-mode managed `
  --case-ids 2025-03-30_11-49-14,2025-03-30_11-49-27,2025-03-30_11-52-45 `
  --parser-mode model `
  --no-sr-judge `
  --no-visual-diagnostics
```

同一 DowntownWest 的其他案例只替换 `--case-ids`，不修改相机或 Actor
配置。`case_ids` 保持指定顺序并覆盖 class/limit/max selectors。MapPackage 中的
`camera_validation` 会在同步后读取 camera pose，阻止 UnrealCV 静默接受但未执行
相机命令。

完整架构见
[`benchmark_runtime_migration.md`](../benchmark_runtime_migration.md)。
