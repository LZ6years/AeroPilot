# TravelUAV FlyAgent Transfer Evaluation

本目录对应的实现位于 `evaluation/benchmarks/travel_uav/`。首版协议固定为：

```text
travel_uav.flyagent_native_actions.front_view.no_assistant.v1
```

它使用 TravelUAV 数据集和远端 Multirotor 环境，但控制循环是 FlyAgent 当前
TaskParser、ModelActionPlanner、Safety 与 native UAV tools。观测只有
`FrontCamera` 的 RGB、depth 和 pose，并且不创建或模拟官方 L1/L2/L3
assistant。因此结果属于 TravelUAV 环境上的迁移评测，不能作为官方五视角
assistant-guided leaderboard 结果。

## 数据准备

官方 split 是逐帧记录，同一个 episode 会重复出现。先构建紧凑 index：

```powershell
python -m evaluation.benchmarks.travel_uav.scripts.build_index `
  --dataset-root D:/TravelUAV/dataset/replay_data `
  --seen-split D:/TravelUAV/data/uav_dataset/seen_valset.json `
  --unseen-split D:/TravelUAV/data/uav_dataset/unseen_valset.json `
  --spawn-areas D:/TravelUAV/data/meta/map_spawnarea_info.json `
  --output .external_refs/TravelUAV/data/travel_uav_index.json `
  --dataset-version traveluav-official
```

构建器会：

- 按 split 中的 `json` 路径去重 episode；
- 读取 episode 的 `merged_data.json` 和相邻 `mark.json`；
- 按官方最近 spawn-area 规则确定 target asset、pose 和 scale；
- 计算三维 reference path 长度；
- 将 Unseen 分为 `test_unseen_map` 与 `test_unseen_object`。

紧凑 index 只用于 evaluation。目标坐标、reference path 和 spawn pose 不会进入
Planner context 或 Runtime benchmark manifest。

## Split 与难度

支持三个互斥 split：

- `test_seen`
- `test_unseen_map`：官方 Unseen Map 集合；当前 3005-case index 的 958 条均来自
  `ModularPark`，因此当前执行配置不需要 `Carla_Town03`
- `test_unseen_object`：官方 Unseen 的其他地图

难度沿用官方规则：三维 reference path `<=250m` 为 Easy，`>250m` 为 Hard。
可以每类运行 N 个、每类全部，或通过 compact index 中的 `episode_id` 精确选择。

## 远端配置

复制 `configs/maps/travel_uav_remote.example.json` 为被 Git 忽略的 `.local`
配置，并将 `/path/to/TravelUAV/ENVs` 替换为远端解压根目录。当前 index 实际
使用 12 个地图；close-loop 脚本直接位于 `closeloop_envs/<Map>.sh`，CARLA
脚本位于 `carla_town_envs/TownXX/LinuxNoEditor/CarlaUE4.sh`。MapPackage 必须保持：

```text
mode=Multirotor
control_strategy=paused_multirotor
vehicle_name=Drone_1
camera_name=FrontCamera
control_port_start=35000
airsim.port=35000
launch.args includes -GraphicsAdapter=3
existing_process_policy=reject
max_scene_num=1
settings_template_source=local
settings_arg_style=single_dash_equals
```

`paused_multirotor` 不执行 Runtime 自动 takeoff。episode initial pose 通过完整
Kinematics（位置、姿态、零线速度和零角速度）写入；正式 move/turn 期间临时
恢复物理，结束后在实际终点清零速度并再次暂停。因此动作路径仍经过 AirSim
物理和地图碰撞，而 observation 之间不会继续漂移。地图缺少 collision mesh 时
仍不能由控制层凭空产生墙体碰撞。

仓库内的 `configs/airsim/travel_uav_official.settings.example.json` 是从官方
`AIRSIM_SETTINGS_TEMPLATE` 提取的只读源模板。SimulationServer 会为每个 scene
复制并渲染 `ApiServerPort=35000`，上传到该 scene 的独立远端目录，并通过官方
要求的 `-settings=<path>` 参数启动 Unreal；不会修改源模板。

`35000` 是 AirSim RPC 端口，必须同时出现在 port policy、MapPackage 和渲染后的
settings 中。`-GraphicsAdapter=3` 选择远端物理 GPU3；readiness
会按照 MapPackage 中声明的 adapter 检查对应 GPU，而不是固定检查 GPU0。

Hugging Face 使用 `local_dir` 下载时，`.cache/huggingface/download/` 只是断点下载
缓存，不是可执行环境目录。全部压缩分卷完成后还需要解压为官方目录结构，例如
`closeloop_envs/`、`carla_town_envs/`；MapPackage 的 `linux_executable` 应指向
这些解压目录中的 `.sh`，不能指向 `.incomplete` 缓存文件。`linux_bash` 由
远端 worker 以 `bash script.sh` 启动，因此脚本要求存在、可读且通过 `bash -n`，
不要求脚本自身有 executable bit；脚本引用的实际 Unreal binary 仍会在启动时
得到验证。

SSH 私钥只通过环境变量提供：

```powershell
$env:FLYAGENT_SSH_KEY = "C:/secure/path/id_ed25519"
```

v1 真实运行只支持 `managed`。动态目标依赖本次拥有的 Unreal 进程退出完成清理；
`attach` 不保证目标不会残留，所以配置会在启动前拒绝。

## 验证与运行

先做完全离线验证，不加载模型或连接远端：

```powershell
python -m evaluation.benchmarks.travel_uav.scripts.run_eval `
  --config evaluation/benchmarks/travel_uav/configs/travel_uav_test_seen_quick.example.json `
  --validate-only
```

将输出的 index SHA256 写入本机真实配置，然后执行只读 readiness：

```powershell
python tests/manual/travel_uav_remote_readiness.py --config path/to/travel_uav.local.json
```

readiness 是只读检查：验证所选地图脚本、bash 语法、本地 settings contract、
GPU3、35000 和残留进程，不上传 settings，也不启动 Unreal。随后分别对一个
CARLA 和一个 close-loop 地图运行通用基础 probe：

```powershell
python tests/manual/remote_ssh_scene_probe.py `
  --config path/to/travel_uav_maps.local.json `
  --map-id travel_uav_carla_town01 `
  --probe-pose-canonical x,y,z,yaw,pitch,roll
```

单案例真实 FlyAgent probe：

```powershell
python tests/manual/travel_uav_real_engine_probe.py `
  --config path/to/travel_uav.local.json `
  --case-id MapName__SequenceName `
  --eval-id travel_uav_probe
```

快速评测：

```powershell
python -m evaluation.benchmarks.travel_uav.scripts.run_eval `
  --config path/to/travel_uav.local.json `
  --difficulties easy,hard `
  --limit-per-difficulty 3
```

## 输出

每次正式运行生成：

```text
outputs/<eval_id>/
├── review.html
├── summary.json
├── manifest.json
├── dataset_provenance.json
├── case_selection.json
└── cases/<episode_id>/
    ├── review.html
    ├── audit.json
    ├── trajectory.json
    ├── metrics.json
    ├── case_result.json
    ├── evaluation_truth.json
    └── review_assets/
```

HTML gallery 默认只显示 RGB；depth 仍保留在 `audit.json` 和审计资产中。指标为
NE、Success、OracleSuccess、SPL 与 PathLength。Success 同时要求 Runtime
正常完成且三维目标距离不超过 20m；模型失败会保留为真实 benchmark 结果，
基础设施或 cleanup 失败则中止后续 batch 并生成 partial HTML。

官方参考：

- [TravelUAV GitHub](https://github.com/prince687028/TravelUAV)
- [TravelUAV paper](https://arxiv.org/abs/2410.07087)
