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
```

## 真实远端执行

每个被选中案例通过 `scene_id -> map_id` 查找 MapPackage。真实运行前会一次性
检查全部选中 scene；缺失映射、缺失 MapPackage 或 scene ID 不一致时，在建立
第一个 SSH 连接前失败。

仓库中的 `configs/maps/aerial_vln_s_remote.example.json` 显式覆盖官方 25 个
scene（`1-18, 20-26`）。第一版固定使用 GPU0 和远端 AirSim 端口 30001，
`max_scene_num=1`，所有 episode 串行。正式配置拒绝附着到已经存在的 AirSim
进程，以免把上一张地图误认为当前 scene。

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

Runner 遇到 scene setup 或 cleanup 错误时写出 partial summary 后中止，防止后续
案例继承未知远端状态。Planner 或指标失败仍作为真实 benchmark 结果保留并继续。
本轮不运行 `--all-per-difficulty`，全量配置只作为后续入口维护。

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
`audit.json` 和 `review.html`。HTML 显示 dataset family、官方 split ID、
Validation Seen/Unseen 标签、difficulty、reference path length、RGB、Planner
debug 和轨迹。Depth 保留在 audit 资产中，不进入默认 gallery。

实机 AerialVLN probe 也生成评测入口 `review.html` 和独立 case HTML；纯
`--validate-only`、readiness 与通用 SSH probe 只生成 JSON，不标记为 benchmark
结果。

历史 `tests/manual/aerial_vln_remote_difficulty_batch.py` 只保留为正式 CLI 的
兼容别名，不再包含独立分类、Runner monkeypatch 或重复报告逻辑。

## 测试

```powershell
python -m pytest tests/test_aerial_vln_cases.py `
  tests/test_aerial_vln_adapter.py `
  tests/test_aerial_vln_metrics.py `
  tests/test_aerial_vln_runner_dry.py -q
```
