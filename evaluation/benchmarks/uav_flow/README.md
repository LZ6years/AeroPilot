# UAV-Flow Benchmark Harness

该模块通过 UAV-Agent 的通用 Benchmark Core 和 TaskRuntime 执行 UAV-Flow
任务；旧 Gym scene lifecycle 和 benchmark 私有飞行客户端已退出执行链路。

## 责任边界

- UAV-Flow Adapter 负责解析 case、初始位姿、参考路径和评测真值。
- UAV-Flow Scenario Controller 只负责布置或隐藏目标物体。
- TaskRuntime 负责连接、飞行、观测、Planner、稳定和 shutdown。
- Benchmark Core 负责正式 episode 审计、图片资产复制和通用 HTML。
- UAV-Flow Report Adapter 负责 nDTW、SR、STMR、动作位移和轨迹面板。
- `target_pos` 与 `reference_path_preprocessed` 只进入评测产物，不进入
  Runtime 目录或 Planner context。

## Dry Smoke

使用唯一正式入口精确运行一个案例：

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

Dry 模式用于检查数据、生命周期、指标和报告结构，不承诺输出真实模拟器图片。

## 真实 UnrealCV

通过正式 Runner 运行本地 DowntownWest：

```powershell
python evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py `
  --config evaluation/benchmarks/uav_flow/configs/uav_flow_turn_3cases.local `
  --case-ids 2025-03-30_11-49-14,2025-03-30_11-49-27,2025-03-30_11-52-45 `
  --execution-mode managed `
  --parser-mode model `
  --no-sr-judge `
  --no-visual-diagnostics
```

真实运行生成 HTML 审计时，runner 会要求 `save_artifacts=true` 和
`enable_action_planner_debug=true`；manual runner 会自动补齐这两个选项。
UAV-Flow 不再提供静态 Parser 或固定动作的真实 probe。

首次使用新的原生 ToolCall 循环前，先运行不连接模拟器的真实 provider 探针：

```powershell
python tests/manual/model_native_tool_loop_smoke.py --env-path .env
```

该探针会验证多模态输入、read-only ToolCall、ToolResult 回放，以及下一轮
`runtime.commit_phase_turn + UAV action`。它不会执行物理动作，完整的脱敏请求摘要
和 `agent_transcript.jsonl` 保存在打印出的 `runs/` 目录中。Provider 省略
assistant content 时，合法 ToolCall 会执行并标记为参数派生 intent；真正无法
规范化的响应才进入与 Runtime 相同的 transcript feedback 修正循环，并保留
`provider_turn_rejected` 供审计。

DowntownWest 的真实运行沿用旧 UAV-Flow Blueprint 视觉契约：

- controlled object 外观必须使用 `vbp <object> set_app <id>`，不回退到
  `set_mixamoapp`；
- 数据集 `target_pos[3:6]` 按 `[roll, yaw, pitch]` 解释，发送到 UnrealCV
  native rotation 时重排为 `pitch yaw roll`；
- 无人机 initial pose 与旧 Runner 一致，只应用数据集位置和 yaw；slots 3/5
  不作为 UAV pitch/roll，原值仅保留在 evaluation truth；
- 当前 DowntownWest 二进制使用 `camera_follow=standard_pose` 将 camera 0 的
  world pose 同步到受控 Actor。旧 batch 的 `vbp 0 set_cam` 把 camera id 错当成
  Actor id 且忽略了错误返回；当前实现不会复刻该无效成功。
- `activate_viewport=true` 在连接初始化时执行
  `vbp BP_Drone01_2 set_viewport`，让可见 UE 窗口与受控无人机一致；模型观测仍
  由同步后的 camera 0 提供，视口选择不会改变 Planner 输入来源。
- `camera_validation` 在同步后读取 camera pose；位置或 yaw 超出地图容差时，
  episode 在错误 RGB 进入 Planner 前失败。DowntownWest camera 0 使用地图级
  `position_mode=max_distance`：碰撞避让允许 camera 在 Actor 周围 `1.5m` 内
  改变安装偏移，同时 yaw 仍必须在 `1°` 内一致。
- startup normalization 隐藏其他默认 Pawn，并隐藏 camera-owning
  `BP_Drone01_2` 的网格以避免自机遮挡第一视角；隐藏网格不会改变 Actor 位姿
  控制、Blueprint camera 或 viewport 归属。

这些约束只作用于 UAV-Flow 的场景布置与 DowntownWest MapPackage；飞行、Turn
计算和观测生命周期仍由通用 TaskRuntime 与 UnrealCV Backend 负责。

## 输出

输出位于 `evaluation/benchmarks/uav_flow/outputs/<eval_id>/`：

- `review.html`：评测级概览；
- `cases/<case_id>/review.html`：图片、Agent Tool Transcript、实际动作、Planner
  窗口、phase 状态、动作诊断、STMR 和轨迹；Transcript 通过
  `tool_call_id` 配对调用与结果，并通过 `observation_id` 关联当时画面；
- `cases/<case_id>/audit.json`：冻结后的通用 episode 审计，包含向后兼容的
  `agent_transcript` 数组；旧 audit 没有该字段时仍可重新生成页面；
- `cases/<case_id>/case.json`：包含 `review_html` 和紧凑的
  `audit_summary`（RGB、动作和 Agent message 数量）；
- `cases/<case_id>/review_assets/`：可直接通过 `file://` 打开的便携图片；
- `cases/<case_id>/scenario_diagnostics.json`：目标实际位置、旋转和视觉契约回读；
- `cases/<case_id>/turn_diagnostics.json`：数据集 yaw 结果和独立的
  `backend_turn_tracking`；
- `cases/<case_id>/backend_action_tracking.json`：正式 yaw/translation primitive
  的请求值、观测值、执行误差和 `zero_action_completion`；
- `summary.json`、`summary.md`、case 指标和评测真值文件。

不再生成 case 级 Markdown review；Core HTML 是唯一的可视化审计页面。

可在真实评测结束后运行只读 Native ToolCall 审计器。它检查 ToolCall/ToolResult
闭合、两层循环协议、每批一次 observation、实际动作回放和 HTML 便携图片，
不会重新调用模型或模拟器：

```powershell
python tests/manual/native_tool_loop_real_audit.py `
  --eval-dir H:/UAV-Agent-Evaluation/uav_flow/outputs/<eval-id> `
  --output H:/UAV-Agent-Evaluation/uav_flow/outputs/<eval-id>/native_tool_audit.json `
  --strict
```

报告中的数量只表示测试覆盖；有效性由逐条 transcript 违规项决定。

HTML gallery 默认只渲染 RGB。Depth 仍按 ObservationProfile 采集，并保留在
`audit.json` 和便携审计资产中，不进入页面图片列表。

同一 DowntownWest 的不同数据集案例不需要重新配置相机或 Actor。使用
`--case-ids id1,id2` 精确选择并保持顺序；设置该参数后，class/limit/max selectors
不会再截断案例。只有更换 Unreal 地图或 Collection build 时才需要新建一次
MapPackage。
