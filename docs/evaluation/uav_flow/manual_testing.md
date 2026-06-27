# UAV-Flow 人工测试操作手册

这份文档用于你后续自己手动运行 UAV-Flow benchmark，观察 UAV-Agent 在真实 UnrealCV 环境中的动作、上下文、RGB、nDTW、SR judge 和 HTML 复盘结果。

## 1. 测试前检查

在 PowerShell 中进入项目根目录：

```powershell
cd C:\Users\HP\Desktop\UAV-Agent
$env:PYTHONIOENCODING = "utf-8"
```

确认 UAV-Flow-Eval 路径存在：

```powershell
Test-Path "C:\Users\HP\Desktop\UAV-Flow-main\UAV-Flow-main\UAV-Flow-Eval"
```

确认 `.env` 中模型配置可用。真实视觉任务建议使用具备视觉能力的模型，例如当前环境中的 `qwen3-vl-flash`。至少需要有 API key。

```powershell
Get-Content .env -Encoding utf8 | Select-String "MODEL|API|BASE"
```

运行真实 UnrealCV 前，先清理残留进程。只在你确认没有其他重要任务使用这些进程时执行：

```powershell
Get-Process | Where-Object {
  $_.ProcessName -match "python|Collection|UE4|Unreal"
} | Select-Object Id, ProcessName, Path
```

如确认是上一次 UAV-Flow 残留，可结束：

```powershell
Get-Process | Where-Object {
  $_.ProcessName -match "Collection|UE4|Unreal"
} | Stop-Process -Force
```

如果还有明确属于旧 UAV-Flow run 的 `python.exe`，再按 PID 单独结束：

```powershell
Stop-Process -Id <PID> -Force
```

真实 UE 测试前，可以先跑一次不启动 UE 的依赖检查：

```powershell
python evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py --preflight-only
```

如果 `real_unrealcv_preflight.status` 不是 `ok`，先修复依赖或 `uav_flow_repo_path`，不要继续跑真实 batch。

## 2. 快速 dry smoke

dry mode 不启动 UE，适合确认 Python harness、case 读取、报告生成是否正常。

```powershell
python evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py `
  --dry-run `
  --classes Turn `
  --limit-per-class 1 `
  --eval-id uav_flow_manual_dry_smoke `
  --parser-mode rule_based `
  --no-sr-judge `
  --no-visual-diagnostics `
  --no-action-planner-debug
```

如果 dry smoke 都失败，先不要跑真实 UE。优先检查 Python 依赖、配置路径和最近代码改动。

## 3. 真实 UnrealCV 小样本测试

推荐先跑小样本，例如 `Turn / Approach` 每类 1 个 case。主入口是：

```powershell
python evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py `
  --eval-id uav_flow_manual_turn_approach_1x_real_debug
```

输出目录：

```text
evaluation/benchmarks/uav_flow/outputs/uav_flow_manual_turn_approach_1x_real_debug
```

打开总览 HTML：

```powershell
Start-Process "evaluation/benchmarks/uav_flow/outputs/uav_flow_manual_turn_approach_1x_real_debug/review.html"
```

## 4. 中样本测试模板

当 1x smoke 正常后，再跑每类 3 个 case：

```powershell
python evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py `
  --classes Turn,Approach `
  --limit-per-class 3 `
  --eval-id uav_flow_manual_turn_approach_3x_real_debug
```

如果你要临时加入其他类别，改 `--classes` 即可：

```powershell
python evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py `
  --classes Turn,Approach,Rotate `
  --limit-per-class 1 `
  --eval-id uav_flow_manual_tar_1x_real_debug
```

当前不建议把 `Move` 当作主要判断对象，因为部分 Move 指令本身没有方向，语义成功与 nDTW 容易混在一起。需要调试 Move 时建议单独跑。

## 5. 推荐配置含义

| 配置 | 推荐值 | 作用 |
|---|---:|---|
| `real_unrealcv` | `True` | 启动真实 UnrealCV 环境 |
| `reuse_real_unrealcv_env` | `True` | 一个 batch 内尽量只启动一次 UE，case 间复用环境 |
| `classes` | `["Turn", "Approach"]` | 当前优先调试语义更明确的类别 |
| `limit_per_class` | `1` 或 `3` | smoke 用 1，正式小样本用 3 |
| `enable_sr_judge` | `True` | 生成 SR judge 结果，但注意它可能误判 |
| `enable_visual_diagnostics` | `True` | 额外生成视觉诊断结果 |
| `enable_action_planner_debug` | `True` | HTML 中展示每轮 action planner 输入/输出，适合调试 |
| `runtime_loop_guard_windows` | `100` | 避免小样本过早截断 |
| `parser_mode` | `"model"` | 使用模型做 semantic parser |

## 6. 每次运行后看什么

优先打开：

```text
evaluation/benchmarks/uav_flow/outputs/<eval_id>/review.html
```

总览页重点看：

- `case_id`
- `class`
- `instruction`
- `nDTW`
- `SR`
- `completed / stopped_reason`
- `turn_relative / move_body / hover`
- RGB 首帧、动作后帧、末帧
- action planner debug 卡片

每个 case 的详细目录：

```text
evaluation/benchmarks/uav_flow/outputs/<eval_id>/cases/<case_id>/
```

常看文件：

```text
review.md
review.html
case.json
ndtw.json
sr_judge.json
visual_diagnostics.json
plots/trajectory_2d.png
```

对应 runtime run 目录在 case 输出或 HTML 链接里可以找到，一般形如：

```text
runs/uav_flow_<case_id>_<timestamp>_<suffix>/
```

常看 runtime 文件：

```text
contexts/latest_context.json
contexts/latest_action_planner_model_context.json
memory/actions.jsonl
memory/agent_messages.jsonl
artifacts/rgb/
debug/action_planner/
```

## 7. 如何判断一次 case

不要只看 SR judge。建议按这个顺序看：

1. `review.html` 的 Decision Timeline：先看每个 window 的 coupled phase decision、reason，以及是否真的执行 UAV tool。
2. Tool Timeline：只看真实执行的 `turn_relative / move_body / hover`，不要用它判断无工具切阶段窗口。
3. Coupled Planner Debug：先看 Decision Summary、phase progress evidence、image input roles、action trend，再展开完整 Model-Facing Context。
4. `latest_context.json`：检查 planner 是否拿到了图片、skills、action trend、compact phase progress。
5. `latest_action_planner_model_context.json`：检查 coupled planner 是否拿到了最新图片、phase progress evidence 和 `phase_completion_evaluation`。
6. `trajectory_2d.png`：实际轨迹和 reference 是否方向一致。
7. `ndtw.json`：看 nDTW 数值和 metric mode。
8. `sr_judge.json`：作为参考，不作为唯一结论。

特别注意：

- `Turn / Rotate` 的 nDTW 更像旋转序列匹配，不能按平移路径解释。
- `Approach` 更适合看是否产生目标导向移动。
- `Move` 指令如果没有方向，语义本身可能不够明确，先不要用它判断整体 pipeline 好坏。
- SR judge 已发现有误判，人工复盘以 RGB、actions、trajectory 和 context 为主。

## 8. 检查 truth leak

每个 case 都应该确认：

```text
evaluation_truth_leak_check.status == "passed"
```

如果不是 `passed`，说明 benchmark 的 evaluation-only 信息可能进入了 planner context，这类结果不要用于结论。

你可以用 PowerShell 快速扫：

```powershell
$eval = "evaluation/benchmarks/uav_flow/outputs/<eval_id>"
Get-ChildItem "$eval/cases" -Directory | ForEach-Object {
  $caseJson = Join-Path $_.FullName "case.json"
  if (Test-Path $caseJson) {
    $data = Get-Content $caseJson -Encoding utf8 | ConvertFrom-Json
    [PSCustomObject]@{
      case_id = $_.Name
      truth_leak = $data.evaluation_truth_leak_check.status
    }
  }
}
```

## 9. 运行失败时怎么排查

### UE 没启动或启动后退出

检查：

- `uav_flow_repo_path` 是否存在。
- UAV-Flow-Eval 中 UnrealCV env 是否能注册。
- UAV-Flow 的 DowntownWest 配置是否指向本机 UnrealZoo/Collection 可执行文件。
- 是否有残留 `Collection / Unreal / python` 进程占用环境。

### batch 中途断连

如果 `reuse_real_unrealcv_env=True` 下某个 case 后环境断连，先不要继续在坏环境上解释结果。清理进程后重跑小样本。

### 模型没有看图

检查：

```text
runs/<run_id>/contexts/latest_context.json
runs/<run_id>/contexts/latest_action_planner_model_context.json
```

确认 image fragment 中：

- `content.visual_evidence.has_rgb == true`
- `content.visual_evidence.attached_image_count >= 1`
- `model_inputs` 中有 image input

### planner 输出反复左右微调

看：

```text
fragments[].name == "action"
content.action_trend
```

如果 `motion_pattern.type == "alternating_yaw"`，planner 应该减少机械 `+5/-5` 震荡。若仍出现，需要看 action planner debug 卡片中该轮模型是否读到了 action trend。

## 10. 跑完后清理

确认没有残留进程：

```powershell
Get-Process | Where-Object {
  $_.ProcessName -match "python|Collection|UE4|Unreal"
} | Select-Object Id, ProcessName, Path
```

如果确认是本次测试残留：

```powershell
Get-Process | Where-Object {
  $_.ProcessName -match "Collection|UE4|Unreal"
} | Stop-Process -Force
```

## 11. 建议的日常测试节奏

1. 代码改动后先跑 dry smoke。
2. 真实 UE 跑 `Turn / Approach` 每类 1 个 case。
3. 打开 `review.html` 看 action planner debug、RGB before/after、coupled phase decision context。
4. 如果行为合理，再跑每类 3 个 case。
5. 记录明显误判 SR、低 nDTW 但语义正确、视觉识别失败、coupled phase decision 过早完成等样本。
6. 每次分析时优先看 HTML，再回到 `latest_context.json` 和 JSONL 查细节。
