# AerialVLN-S Benchmark

AerialVLN-S 复用通用 Benchmark Core、TaskRuntime 和 AirSim Backend，控制协议
仍为 UAV-Agent 原生连续 primitive：

```text
aerial_vln.uav_agent_native_actions.v2
```

v2 使用通用 TaskParser 的地标 checkpoint 拆分处理长路线。AerialVLN 的
`land on/in/by X` 按 benchmark episode 协议解释为到达 X 对应的目标区域、
确认 instruction 指定的位置关系并 STOP；不要求物理接地、AirSim landed
状态或调用 `uav.land`。这段固定上下文只由 AerialVLN Runner 注入，原始
instruction 不改写，也不影响 UAV-Flow。

AerialVLN 的多地标长路线默认允许每个语义阶段最多 100 个 action/observation
window。该值只属于 AerialVLN Runner；TaskRuntime 和 UAV-Flow 的通用默认值仍为
40。视觉 yaw budget 按当前观察位置计算，动作后实际三维位移达到 0.5m 时开启
新的原地扫描段。

Benchmark 层负责 AerialVLN-S `val_seen` 和 `val_unseen` 的数据身份、难度
分类、案例选择、指标和报告。reference path、difficulty、goal 与 official
actions 都是
evaluation-only truth，不进入 Runtime artifacts 或 Planner context。

## 数据与难度

将官方 split 放在：

```text
.external_refs/AirVLN/DATA/data/aerialvln-s/val_seen.json
.external_refs/AirVLN/DATA/data/aerialvln-s/val_unseen.json
```

一次评测只选择一个 split。Runner 对所选 split 的完整 population 计算
reference path 三维长度，并使用该 split 自身的线性 1/3、2/3 分位点划分：

- `length <= q33`：low；
- `q33 < length <= q67`：medium；
- `length > q67`：high。

不设置 `limit_per_difficulty` 时运行所选难度的全部案例。设置限制后，选择器优先
覆盖尚未选中的 scene，并在每个难度桶的长度分布上确定性均匀取样。
`val_seen` 和 `val_unseen` 不共享阈值，也不会在同一评测目录中混合。

## 完全离线验证

Validation Seen 快速配置默认每类选择 3 个：

```powershell
python evaluation/benchmarks/aerial_vln/scripts/run_eval.py `
  --config evaluation/benchmarks/aerial_vln/configs/aerial_vln_eval.example.json `
  --validate-only
```

该命令只读取本地数据和 MapPackage 配置，不加载模型、不连接 SSH、不启动
AirSim。结果会打印到终端，并写入配置的 `output_dir`。输出包含：

- split SHA256、episode 数和全部 scene；
- q33/q67、三类总数和选中案例；
- `execution_readiness`，其中列出尚未配置的 scene。

首次验证后，将输出的 `split_sha256` 写入被 git 忽略的本机配置
`expected_split_sha256`。真实 attach/managed 运行会强制校验该值。
该 SHA 只属于当前选择的 split；切换 split 时必须同时切换为对应 SHA。

Validation Unseen 快速验证：

```powershell
python evaluation/benchmarks/aerial_vln/scripts/run_eval.py `
  --config evaluation/benchmarks/aerial_vln/configs/aerial_vln_s_val_unseen.example.json `
  --validate-only
```

也可以复用配置并通过 CLI 切换；若配置中已有 seen SHA，必须同时覆盖为 unseen
SHA：

```powershell
python evaluation/benchmarks/aerial_vln/scripts/run_eval.py `
  --config evaluation/benchmarks/aerial_vln/configs/aerial_vln_eval.example.json `
  --split val_unseen `
  --expected-split-sha256 <VAL_UNSEEN_SHA256> `
  --validate-only
```

Seen/Unseen 全量配置分别为：

```powershell
python evaluation/benchmarks/aerial_vln/scripts/run_eval.py `
  --config evaluation/benchmarks/aerial_vln/configs/aerial_vln_s_val_seen_full.example.json `
  --validate-only

python evaluation/benchmarks/aerial_vln/scripts/run_eval.py `
  --config evaluation/benchmarks/aerial_vln/configs/aerial_vln_s_val_unseen_full.example.json `
  --validate-only
```

也可以通过 CLI 选择难度和数量：

```powershell
# 仅测 medium/high，每类最多 5 个
python evaluation/benchmarks/aerial_vln/scripts/run_eval.py `
  --config evaluation/benchmarks/aerial_vln/configs/aerial_vln_eval.example.json `
  --difficulties medium,high `
  --limit-per-difficulty 5 `
  --validate-only

# low/medium/high 全部案例
python evaluation/benchmarks/aerial_vln/scripts/run_eval.py `
  --config evaluation/benchmarks/aerial_vln/configs/aerial_vln_eval.example.json `
  --all-per-difficulty `
  --validate-only

# 仅运行 Validation Seen 的 low 三案例
python evaluation/benchmarks/aerial_vln/scripts/run_eval.py `
  --config evaluation/benchmarks/aerial_vln/configs/aerial_vln_eval.example.json `
  --split val_seen `
  --difficulties low `
  --limit-per-difficulty 3
```

## STMR 历史压缩

AerialVLN-S 可选使用现有 STMR 将 episode 内的 RGB、metric depth、canonical
pose 和轨迹压缩成 UAV 居中的语义地图。配置项为：

```json
{
  "stmr_mode": "off",
  "stmr_prior": "default_uav_prior",
  "stmr_max_queries": 16,
  "stmr_device": "auto"
}
```

- `off`：保持原有行为，不加载 STMR 模型。
- `shadow`：episode 结束并关闭 Runtime logger 后离线生成 STMR；不修改
  TaskParser、Planner context 或动作。
- `planner`：在线增量生成 STMR，并通过现有 `map` context slot 向 Planner
  提供一张语义历史图和紧凑摘要。图像 observation 始终由 transcript 按时间
  各投影一次；STMR ready 时额外附加当前语义地图，失败时只省略地图。

Shadow/Planner 都要求 observation profile 至少包含 `rgb + depth + pose`。
STMR 空白格表示未知，不表示可通行。完整 voxel、性能数据和失败信息只写入
`stmr_summary.json`；HTML 展示复制到 case `review_assets/stmr/` 的最新地图。
Planner 输入始终只包含一份有界地图摘要，不包含 raw voxel、完整 frame history
或完整轨迹。建议先用同一 low case 完成 Shadow 验证，再切换 Planner 模式。

## 真实远端执行

每个被选中案例通过 `scene_id -> map_id` 查找 MapPackage。真实运行前会一次性
检查全部选中 scene；缺失映射、缺失 MapPackage 或 scene ID 不一致时，在建立
第一个 SSH 连接前失败。

仓库中的 `configs/maps/aerial_vln_s_remote.example.json` 显式覆盖官方 25 个
scene（`1-18, 20-26`）。第一版固定使用 GPU0 和远端 AirSim 端口 30001，
`max_scene_num=1`，所有 episode 串行。正式配置拒绝附着到已经存在的 AirSim
进程，以免把上一张地图误认为当前 scene。

默认 MapPackage 使用：

```text
mode=ComputerVision
control_strategy=pose_kinematic
```

这保持现有 AerialVLN 位姿式动作语义。通用 AirSim Backend 同时支持
`paused_multirotor`，但当前远端 AirVLN packaged build 在以 Multirotor 模式启动时
会在 AirSim RPC 就绪前崩溃，因此仓库不提供一份会误导为可运行的 AerialVLN
Multirotor MapPackage。若后续换用确认支持 Multirotor 的 AirVLN build，可按
TravelUAV 的 MapPackage 声明该策略，并先执行单 scene probe。物理变体能否阻挡
墙体仍取决于对应 Unreal 地图的 collision mesh；两种策略的评测目录和汇总不得
混合。

在当前 PowerShell 会话设置 SSH 私钥：

```powershell
$env:FLYAGENT_SSH_KEY="$env:USERPROFILE\.ssh\id_rsa_npu_wzq"
```

私钥路径不写入 `.env` 或 JSON。本机 Seen/Unseen 配置使用真正会被 git 忽略的
`.local` 后缀，并分别设置：

- 指向完整 25-scene MapPackage 配置；
- 设置 `dataset_version`；
- 设置已经离线验证的 `expected_split_sha256`；
- `execution_mode=managed`、`shutdown_policy=hover`；
- `save_artifacts=true`、`enable_action_planner_debug=true`。

连接模拟器前先执行只读检查。该命令不会创建 Runtime、SSH tunnel 或远端
进程：

```powershell
python tests/manual/aerial_vln_remote_readiness.py `
  --config evaluation/benchmarks/aerial_vln/configs/aerial_vln_s_val_seen_remote.local `
  --eval-id val_seen_readiness
```

随后用 scene 1、13、26 验证完整远端启动和清理链路：

```powershell
python tests/manual/remote_ssh_scene_probe.py `
  --config configs/maps/aerial_vln_s_remote.example.json `
  --map-id airvln_remote_env_1

python tests/manual/remote_ssh_scene_probe.py `
  --config configs/maps/aerial_vln_s_remote.example.json `
  --map-id airvln_remote_env_13

python tests/manual/remote_ssh_scene_probe.py `
  --config configs/maps/aerial_vln_s_remote.example.json `
  --map-id airvln_remote_env_26
```

数据集 initial pose 和 Benchmark Core HTML 使用 AerialVLN probe 检查：

```powershell
python tests/manual/aerial_vln_real_engine_probe.py `
  --config evaluation/benchmarks/aerial_vln/configs/aerial_vln_s_val_seen_remote.local `
  --scene-id 1
```

最后分别运行 Seen 和 Unseen 快速 9-case Gate：

```powershell
python evaluation/benchmarks/aerial_vln/scripts/run_eval.py `
  --config evaluation/benchmarks/aerial_vln/configs/aerial_vln_s_val_seen_remote.local `
  --eval-id aerial_vln_s_val_seen_quick

python evaluation/benchmarks/aerial_vln/scripts/run_eval.py `
  --config evaluation/benchmarks/aerial_vln/configs/aerial_vln_s_val_unseen_remote.local `
  --eval-id aerial_vln_s_val_unseen_quick
```

Runner 遇到 scene setup、cleanup 或 Provider 传输重试耗尽时，会在当前 case
完整落盘后写出 partial summary 并中止，防止后续案例继承未知远端或模型服务状态。
普通 Planner 协议/行为错误和指标失败仍作为真实 benchmark 结果保留并继续；
Provider 中止不会触发整例重跑。批次会正常生成 `batch_status=aborted` 的 manifest、
summary 和 HTML，不需要人工 Ctrl+C。本轮不运行 `--all-per-difficulty`，全量配置
只作为后续入口维护。

## 输出

评测根目录新增：

```text
dataset_provenance.json
difficulty_assignment.json
case_selection.json
execution_readiness.json
manifest.json
summary.json
summary.md
review.html
```

每个 case 继续生成 `case_result.json`、`metrics.json`、`trajectory.json`、
`audit.json`、`stmr_summary.json` 和 `review.html`。HTML 显示 dataset family、官方 split ID、
Validation Seen/Unseen 标签、difficulty、reference path length、RGB、Planner
debug 和轨迹。Depth 保留在 audit 资产中，不进入默认 gallery。

实机 AerialVLN probe 也生成评测入口 `review.html` 和独立 case HTML；纯
`--validate-only`、readiness 与通用 SSH probe 只生成 JSON，不标记为 benchmark
结果。

历史 `tests/manual/aerial_vln_remote_difficulty_batch.py` 只保留为正式 CLI 的
兼容别名，不再包含独立分类、Runner monkeypatch 或重复报告逻辑。

## 测试

### RGB 与 Phase 审查

默认模型图片由 `configs/plugins.json` 的
`context_options.image.include_current_depth=false` 控制。Depth 仍采集，按需供
Grounding 测距；显式设为 `true` 才使用融合图。20 张 observation 抽样与最近
3 张连续保留独立配置，不受此开关影响。

新 HTML 默认显示初始完整 SemanticPlan 和按 Phase/Decision 分组的时间线。
初始计划来自执行前的 `initial_semantic_plan` 事件；旧记录缺失时明确标记，
不从局部 strategy 回填。完整请求和详细事件在 Advanced audit 内。历史融合图
仍是历史真实输入，不能用现在的 RGB 配置重绘后冒充原请求。

只预览已有单例（不调用模型或模拟器，输出目录必须不存在）：

```powershell
python tests/manual/rgb_phase_review_probe.py --case-dir "<existing-case>" --output-dir "<new-preview>"
```

精确复验原抽样中的某一例时，不要改成 `--limit-per-difficulty 1`：该参数会
重新抽样，可能选择不同 episode。使用手动单例入口，保留原配置的抽样口径，
再按 ID 限定；该入口仅用于回归实验，不替代正式批测 CLI：

```powershell
python tests/manual/aerial_vln_selected_case_probe.py `
  --config evaluation/benchmarks/aerial_vln/configs/aerial_vln_s_val_seen_probe.local `
  --episode-id 3G0WWMR1U4IB7QGQVGGALABHGGXNQO `
  --eval-id "<new-eval-id>" --validate-only
```

确认 ID、远端进程/端口、Provider 和 RGB/Depth 后移除 `--validate-only` 才会执行。
该入口复用正常 Runner 生命周期、指标与输出，并在 selection 清单记录显式
子集筛选。completed 仍只是 Agent 自报终态，不等同 SR/OSR。

```powershell
python -m pytest tests/test_aerial_vln_cases.py `
  tests/test_aerial_vln_adapter.py `
  tests/test_aerial_vln_metrics.py `
  tests/test_aerial_vln_runner.py -q
```
# Scene startup and cleanup diagnostics

Managed scenes write `runs/<run>/scenes/<scene>/lifecycle.jsonl` and
`ssh-tunnel.log`. Lifecycle timings use a monotonic clock; timestamps are for
cross-log correlation only. Process launch, remote TCP readiness, forwarded RPC,
first modalities and independent cleanup attempts are separate evidence.
An empty benchmark cleanup-error list alone does not establish process exit.

The no-model engine probe accepts `--ready-timeout-sec 300` as an in-memory
diagnostic override. It does not rewrite map configuration. If the remote TCP
endpoint is reachable but primary RPC remains unavailable after 120 seconds, it
makes one independent tunnel comparison without restarting UE. A successful
comparison with a failed primary is an infrastructure failure, not a fallback.

Example (requires the configured SSH key environment variable):

```powershell
python tests/manual/aerial_vln_real_engine_probe.py `
  --config evaluation/benchmarks/aerial_vln/configs/aerial_vln_s_val_seen_probe.local `
  --scene-id 12 --execution-mode managed --ready-timeout-sec 300 `
  --output-dir tests/manual_outputs/airsim_lifecycle_scene12_unique
```

Before a fresh navigation batch, check scenes 12 and 8 twice and the other four
selected maps once. Stop on any timeout, cleanup residue or incomplete evidence.
Only verified normal slow startup justifies changing the selected maps' JSON
readiness ceiling. Provider timeouts, navigation windows and metrics are separate.
# Compact navigation review experiment

Default case HTML contains outcomes, actual/reference 2D and 3D trajectories,
the frozen initial plan, and Phase/Decision cards. Public intent is excerpted
to 180 characters with the original text expandable. Raw audit/trace/metrics
remain separate files; detailed requests and duplicate histories are opt-in.

Offline preview (a new output directory is required):

```powershell
python tests/manual/rgb_phase_review_probe.py --case-dir <existing-case> --output-dir <new-preview>
# Add --detailed-audit to also write cases/<id>/audit_review.html.
```

The preview also writes `grounding_threshold_preview.json`: a 0.4 filter of
historically rounded scores, explicitly NOT a new detector run or model input.
The live capability uses its JSON `box_threshold`; empty candidates express
uncertainty, not target absence. NED-to-positive-up conversion is display-only.

The current change does not propagate AerialVLN landing semantics beyond its
existing parser injection. Unrequested descent or landing reinterpretation
remains a known deferred issue and must be reported separately in comparisons.
