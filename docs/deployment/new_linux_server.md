# 新服务器部署：AerialVLN-S

新主机已通过 SSH 只读核验：Ubuntu 20.04.4、x86_64、4 张 RTX 3090。
连接为 `wangzequan@10.68.240.49`，端口 `10245`；Python 依赖和场景尚未安装验证。
先部署当前的飞行知识、像素测深和地点地图方案。PyTorch、检测模型权重和 CUDA Toolkit
不是这一方案的安装前提；UE 渲染仍需要可用的 NVIDIA 驱动和 Vulkan。
本流程不修改技能、提示词或导航逻辑，不自动启动评测。

## 1. 系统和代码

使用普通用户运行 UE，不以 root 启动。先检查机器：

```bash
cat /etc/os-release
uname -m
nvidia-smi
df -h
```

Ubuntu 安装基础工具和常见运行库：

```bash
sudo apt-get update
sudo apt-get install -y git rsync tmux curl ca-certificates build-essential \
  libvulkan1 vulkan-tools libgl1 libglib2.0-0 libx11-6 libxcursor1 \
  libxrandr2 libxi6 libxinerama1 libsm6 libice6
vulkaninfo --summary
```

`nvidia-smi` 失败时先由管理员配置匹配该服务器的 NVIDIA 驱动。
`nvidia-smi` 成功但 Vulkan 没有 NVIDIA 设备，可能只装了计算驱动，需补齐同版本图形组件。
不要在共享服务器上盲目安装另一版本驱动。以上通用库不保证覆盖所有场景二进制依赖；
遇到缺失库，对实际 `*-Linux-Shipping` 可执行文件运行 `ldd` 定位。

```bash
mkdir -p "$HOME/work"
cd "$HOME/work"
git clone git@github.com:LZ6years/FlyAgent.git
cd FlyAgent
git rev-parse HEAD
```

GitHub SSH 认证须在新机配置；也可使用有权访问该仓库的 HTTPS 认证。
本次新增脚本尚未推送时，先从本地上传 `tools/deployment/install_linux_env.sh` 到相同位置。

## 2. Python 环境

已有 Conda 可直接创建环境。没有 Conda 时先按
[官方 Linux 安装说明](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
安装并核对安装包 SHA-256，重新打开终端。

```bash
conda create -n flyagent python=3.10 pip -y
conda activate flyagent
PY="$CONDA_PREFIX/bin/python"
bash tools/deployment/install_linux_env.sh "$PY"
```

脚本依次安装核心依赖、AirSim、评测绘图和 psutil，并检查依赖与导入。
AirSim 必须在核心依赖后安装；脚本已使用 `--no-build-isolation`。
不安装 torch，不安装场景，不修改 `.env`。

## 3. 复制模拟器和数据

本次 AerialVLN-S 数据由本地上传，模拟器来自
[Kaggle](https://www.kaggle.com/datasets/shuboliu/aerialvln-simulators)。
上传解压后核对下面要求的目录结构，压缩包可能多套一层目录，不要直接照抄路径。
TravelUAV 使用本地已有精简索引，下载方法见 [国内下载与数据迁移](hf_dataset_download.md)。
以下旧机 rsync 命令仅作为旧机恢复连接后的可选方式。

`pip install airsim` 只安装 Python 客户端，不能得到场景。
需要 AerialVLN 的 **Linux 打包场景**、canonical 数据集及相机 settings。
已有打包场景时无需编译 Unreal Editor，参见
[AirSim 打包环境说明](https://github.com/microsoft/AirSim/blob/main/docs/use_precompiled.md)。

优先从旧服务器复制验证过的资源，保留目录结构、执行权限和符号链接。
在新服务器执行以下命令，先修改旧服务器的 SSH 地址和端口；私钥须由你安全配置，不能放进 Git。
旧服务器暂时连不上时可通过中转磁盘搬运相同目录；Git 仓库不包含这些资源。

```bash
ASSETS="$HOME/data/airvln"
OLD_HOST=wzq@10.68.240.49
OLD_PORT=10182
mkdir -p "$ASSETS/ENVs" "$ASSETS/DATA/data" "$ASSETS/settings"
rsync -a --partial --info=progress2 -e "ssh -p $OLD_PORT" \
  "$OLD_HOST:/home/Newdisk/wzq/data/airvln/ENVs/" "$ASSETS/ENVs/"
rsync -a --partial --info=progress2 -e "ssh -p $OLD_PORT" \
  "$OLD_HOST:/home/Newdisk/wzq/data/airvln/DATA/data/aerialvln-s/" "$ASSETS/DATA/data/aerialvln-s/"
rsync -a --partial --info=progress2 -e "ssh -p $OLD_PORT" \
  "$OLD_HOST:/home/Newdisk/wzq/data/airvln/AirVLN/airsim_plugin/settings/3/settings.json" "$ASSETS/settings/"
```

上面复制全部 ENVs，空间不足时只复制评测所需的 13 个场景：
`2 3 5 7 8 9 10 12 13 14 17 21 24`。
先在旧机用 `du -sh` 统计资源大小再分配空间，另留运行审计和图像结果空间。
需要这些布局：

```text
~/data/airvln/ENVs/env_3/env_3/LinuxNoEditor/AirVLN.sh
~/data/airvln/ENVs/...其他场景及各自完整 LinuxNoEditor 目录...
~/data/airvln/DATA/data/aerialvln-s/val_seen.json
~/data/airvln/DATA/data/aerialvln-s/val_unseen.json
~/data/airvln/settings/settings.json
```

场景 3 的相机模板已在旧机 13 个场景验证；新机仍须重新采集验证。
不要只复制 `.sh`：二进制、Engine、Content/Paks 等同目录资源都需要。

## 4. API 与新机 GPU 配置

在仓库根目录创建私有 `.env`，已有文件则保留：

```bash
if [ ! -e .env ]; then (umask 077; cp .env.example .env); fi
chmod 600 .env
nano .env
```

填入自己的 API Key、完整 base URL，并沿用以下设置：

```dotenv
FLYAGENT_MODEL_PROVIDER=openai_compatible
FLYAGENT_MODEL_NAME=gpt-6.1-sol
FLYAGENT_MODEL_API_PROTOCOL=chat_completions
FLYAGENT_MODEL_STREAM=true
FLYAGENT_MODEL_TIMEOUT_SEC=90
FLYAGENT_MODEL_TIMEOUT_RETRIES=1
FLYAGENT_MODEL_CONNECTION_RETRIES=8
FLYAGENT_MODEL_RETRY_DEADLINE_SEC=600
```

下面设置 **新服务器** 的物理 GPU 编号并生成新 execution 配置。
不要复制旧机的 UUID，也不要默认沿用旧机 GPU 0 的附带图形上下文例外。

```bash
GPU=0  # 改成新机允许使用且空闲的物理卡
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.used --format=csv
export DATADIR="${DATADIR:-/home/Newdisk/wangzequan}"
DEPLOY="$DATADIR/flyagent/aerial_vln/deploy-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEPLOY"
"$PY" - "$GPU" "$DEPLOY/execution.json" <<'PY'
import json, subprocess, sys
from pathlib import Path
gpu = int(sys.argv[1])
uuid = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
    '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
if not uuid.startswith('GPU-') or '\n' in uuid:
    raise SystemExit('Expected exactly one physical GPU UUID')
config = dict(mode='local', gpu_indices=[gpu], gpu_uuids={str(gpu): uuid},
              graphics_context_limits_mib={}, require_cuda=False, report_profile='audit')
with Path(sys.argv[2]).open('x', encoding='utf-8') as stream:
    json.dump(config, stream, indent=2)
print(json.dumps(config, indent=2))
PY
```

## 5. 离线检查、冻结配置与场景验证

以下命令在同一终端、仓库根目录执行，使用前面的 `$PY`、`$ASSETS`、`$GPU`、`$DEPLOY`。

```bash
"$PY" -m pytest -q tests/diagnostics/test_local_process_control.py \
  tests/diagnostics/test_linux_process_lifecycle.py \
  tests/test_sim_server.py tests/test_remote_scene_control.py
OUT="$DEPLOY/stability"
"$PY" -m evaluation.benchmarks.aerial_vln.scene_stability \
  --output "$OUT" --dataset-root "$ASSETS/DATA/data/aerialvln-s" \
  --env-root "$ASSETS/ENVs" --settings-template "$ASSETS/settings/settings.json" \
  --execution-config "$DEPLOY/execution.json"
"$PY" tests/manual/linux_local_preflight.py --gpu "$GPU" \
  --config "$OUT/inputs/val_seen.json" --config "$OUT/inputs/val_unseen.json"
"$PY" -m evaluation.benchmarks.core.campaign.parallel bind --output "$OUT"
"$PY" tests/manual/linux_scene_capture.py --output "$OUT/capture" \
  --maps "$OUT/parallel/maps_A.json" --gpu "$GPU" \
  --execution-config "$DEPLOY/execution.json"
```

逐条检查退出码，失败不要继续。prepare 只冻结 26 条清单，不调用模型。
capture 会实际启动各个 UE 场景、检查 RGB/深度、GPU 归属及清理，不调用模型 API。
这一步成功才说明模拟环境部署可用。固定端口 39131 必须空闲。
这里复用 campaign 的 Vulkan 枚举绑定，不直接把物理编号当作 `GraphicsAdapter`。
若出现额外 GPU 图形上下文，先检查报告，再决定例外，不能自动放宽。

## 6. 手动启动第一阶段（会产生 API 费用）

确认采集通过后再执行。默认仍是单卡、单 worker；不自动运行全量 864 条。

```bash
printf -v WATCH_CMD '%q -m evaluation.benchmarks.core.campaign.parallel watch --output %q --workers 1 >%q 2>&1' \
  "$PY" "$OUT" "$OUT/watch.log"
tmux new-session -d -s aerialvln-stability "$WATCH_CMD"
tail -f "$OUT/watch.log"
```

`Ctrl+C` 退出 tail 不停止 watchdog。状态和暂停：

```bash
cat "$OUT/parallel/watchdog/status.json"
touch "$OUT/PAUSE"
```

暂停用于停止继续派发，不代表立刻终止当前 episode。
恢复和报告说明见 [场景稳定性文档](aerial_vln_scene_stability.md)。
新 campaign 固定使用 `configs/plugins.pixel_map.json`：飞行知识、像素测深、地点地图，
不启用检测模型、flight_experience 或 STMR。输入和压缩参数从当前仓库冻结。

## TravelUAV 后续部署

同一 Python 环境可以复用；另需 TravelUAV 的 Linux ENVs、轨迹/标注和对应索引。
本地 `.external_refs/TravelUAV/data/travel_uav_index.json` 已核验可由当前 loader 完整加载，
与旧机配置预期 SHA-256 一致；它包含运行所需元数据，`source_json` 仅用于来源记录。
可以直接迁移索引，只需修改评测配置的 `index_path`，无需复制原始图像。
以 `configs/maps/travel_uav_linux_local.example.json` 和对应 smoke 配置为基础填写新路径，
保留 `FrontCamera`、`paused_multirotor`、`-settings=PATH` 和动态目标初始化要求。
先完成资源部署；实际场景路径和动态目标资产仍须在新机验证。
