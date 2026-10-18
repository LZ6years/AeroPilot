# 新服务器资源下载与测试数据迁移

2026-10-15 只读检查：`wangzequan@10.68.240.49:10245`，Ubuntu 20.04.4，
根分区可用约 290 GiB，`/home/Newdisk` 可用约 161 GiB；四张 GPU 均有任务。
没有启动下载、安装软件或操作 GPU。以下命令由用户在准备好目标目录后运行。

## TravelUAV 只需要哪些数据

官方来源：[数据](https://huggingface.co/datasets/wangxiangyu0814/TravelUAV/tree/main)、
[模拟器](https://huggingface.co/datasets/wangxiangyu0814/TravelUAV_env/tree/main)、
[代码和划分说明](https://github.com/buaa-colalab/TravelUAV/blob/main/Model/LLaMA-UAV/README.md#dataset)。
数据仓库约 483 GB，混合各场景数据；并不是一个独立的 test 下载包。
官方划分位于代码仓库 `data/uav_dataset/seen_valset.json` 和 `unseen_valset.json`。

**当前 FlyAgent 可只携带已有精简索引，不需要下载那 483 GB。**
本地 `.external_refs/TravelUAV/data/travel_uav_index.json` 为 94,703,465 bytes（约 90.3 MiB），
经当前 loader 验证含 test_seen 1418、test_unseen_object 629、test_unseen_map 958 条。
这是现有索引的统计，不代表本次重新审计了所有官方划分内容。
该文件包含指令、起点位姿、目标资产及位置、参考轨迹和来源哈希；runner 不读取原始图像。
运行画面由 UE 实时渲染，评分在评测层使用索引中的参考数据。

在本地 PowerShell 从仓库根目录执行：

```powershell
ssh -i C:/Users/HP/.ssh/id_rsa_npu_wzq -p 10245 wangzequan@10.68.240.49 'mkdir -p ~/data/traveluav/metadata ~/setup'
scp -i C:/Users/HP/.ssh/id_rsa_npu_wzq -P 10245 .external_refs/TravelUAV/data/travel_uav_index.json wangzequan@10.68.240.49:~/data/traveluav/metadata/
scp -i C:/Users/HP/.ssh/id_rsa_npu_wzq -P 10245 tools/deployment/download_hf_dataset.py tools/deployment/install_linux_env.sh wangzequan@10.68.240.49:~/setup/
```

`install_linux_env.sh` 使用仓库相对路径；下载器可在 `~/setup` 独立运行，
安装脚本应放回 `FlyAgent/tools/deployment/` 后运行。
新机校验索引：

```bash
sha256sum ~/data/traveluav/metadata/travel_uav_index.json
# 应为 8a2dbe59a182774f147869269a4a612546031498d5e40fac0697227412152e87
```

评测 JSON 填入 `/home/wangzequan/data/traveluav/metadata/travel_uav_index.json`，
同时保留上述 `expected_index_sha256`。
如以后重建索引，需要划分选中的 `merged_data.json`、`mark.json` 和
`data/meta/map_spawnarea_info.json`。不要直接重新运行官方生成器来覆盖旧元数据：
该生成器会检查图片文件存在性，并随机选择对象描述，可能改变已有指令。

## 国内 HF 下载脚本

### 2026-10-15 续传行为修正

新机实际安装 `huggingface_hub 1.33.0`。此版本使用随机临时文件，重启 CLI 时不复用
未完成文件；下面原脚本的跨进程续传说明不适用于该版本，已完成文件仍可复用。
需要跨进程续传时先 Ctrl+C 停止旧下载，然后使用已上传的新启动器：

```bash
bash /home/wangzequan/FlyAgent/tools/deployment/download_traveluav_env_resume.sh \
  /home/wangzequan/TravelUAV/ENVs
```

该启动器配合 `hf_resume_download.py`，仅在本次 Python 进程中适配 HF 1.33.0，
不修改 site-packages。固定版本、保留稳定的临时文件，并使用 HF 的 HTTP Range 实现；
LFS SHA256 标识存在时完成后校验内容。旧 CLI 的随机命名半成品不会自动迁移。
同一目录不得同时运行旧 CLI 和新启动器；新启动器之间有目录锁。
服务器已通过本地 HTTP 实测：下载中途终止进程，新进程从 10 MiB 处发起 Range 请求，
最终 SHA256 一致。外部镜像也必须支持 Range；这项测试不代表实测了镜像端中断恢复。

`tools/deployment/download_hf_dataset.py` 默认使用 `https://hf-mirror.com`，
以官方 huggingface_hub 下载，关闭 Xet 以兼容镜像 HTTP 传输，不发送登录 token。
镜像速度取决于网络和服务状态，不能保证固定带宽。
可指定官网 endpoint；切换 endpoint 后仍使用同一固定 commit。

下载工具建议单独环境，避免升级 FlyAgent 依赖：

```bash
conda create -n hf-download python=3.10 pip -y
conda activate hf-download
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 'huggingface_hub>=1.0,<2'
DL_PY="$CONDA_PREFIX/bin/python"
DL_SCRIPT="$HOME/setup/download_hf_dataset.py"
DOWNLOADS="$HOME/data/traveluav/downloads/envs"
```

先列出文件和总大小，不下载大文件：

```bash
"$DL_PY" "$DL_SCRIPT" --repo wangxiangyu0814/TravelUAV_env \
  --local-dir "$DOWNLOADS" --include 'closeloop_envs.*' --list-only
```

先下载闭环场景包，用于城市等场景部署：

```bash
"$DL_PY" "$DL_SCRIPT" --repo wangxiangyu0814/TravelUAV_env \
  --local-dir "$DOWNLOADS" --include 'closeloop_envs.*' --workers 4 --attempts 20
```

需要 Carla 场景时追加：

```bash
"$DL_PY" "$DL_SCRIPT" --repo wangxiangyu0814/TravelUAV_env \
  --local-dir "$DOWNLOADS" --include 'carla_town_envs.*' --workers 4 --attempts 20
```

不要只下载 `.zip`：同名前缀的 `.z01`、`.z02` 等分卷也需要。
闭环场景包约 18.7 GB，Carla 包约 44.7 GB（十进制估算）；解压还要额外空间。
其他环境是否需要，应以测试索引的地图清单和解压后的场景内容核对。
这里不把两个包直接认定为完整评测全部依赖。

脚本行为：

- 默认每次操作最多尝试 20 次，退避等待最多 60 秒；超时默认 120 秒。
- 保留 HF 的 `.cache` 临时下载状态。中断后用相同目录、相同命令重跑，复用已完成文件并续传可恢复的部分。
- `.flyagent-download.json` 固定仓库 commit，避免下载跨版本混合。
- 支持多个 `--include`；不支持压缩包内部文件筛选。
- 检查压缩文件下载所需空间并预留 1 GiB；不替代解压空间检查。
- 认证、文件不存在、权限或磁盘满错误不无限重试。
- 下载后检查文件大小；不把大小检查宣称为完整内容哈希校验。

若镜像失败，重跑同一命令追加：

```text
--endpoint https://huggingface.co
```

在新 tmux 窗口里运行下载可避免 SSH 断开后停止：

```bash
tmux new -s resource-download
# 在此窗口激活 hf-download 环境，并执行上面的下载命令。
# Ctrl-B 后按 D，退出查看；tmux attach -t resource-download 可返回。
```

## 校验、解压与 AerialVLN 上传

Ubuntu 20.04 可安装 `p7zip-full`，已有 `7z` 无需重复安装。
所有分卷放同一目录，先检查再解压：

```bash
7z t "$DOWNLOADS/closeloop_envs.zip"
mkdir -p "$HOME/data/traveluav/ENVs"
7z x "$DOWNLOADS/closeloop_envs.zip" -o"$HOME/data/traveluav/ENVs"
```

按同样方式处理 `carla_town_envs.zip`。解压完成后核对 `.sh` 和 Linux 二进制路径，
再填写 FlyAgent 地图配置，不自动删除压缩包或启动场景。

AerialVLN-S 的数据和 Kaggle 场景可以在本地下载后通过 MobaXterm SFTP 上传，
连接 `10.68.240.49:10245`，用户名 `wangzequan`。
数据放 `~/data/airvln/DATA/data/aerialvln-s/`，Linux 场景解压到 `~/data/airvln/ENVs/`。
前一轮评测使用 `val_seen.json` 和 `val_unseen.json`；如果你下载的是另一份 `test` 划分，
先核对文件名和标注，不能改名冒充这两份 canonical split。

参考：[HF 下载 API](https://huggingface.co/docs/huggingface_hub/package_reference/file_download)、
[HF-Mirror](https://hf-mirror.com/)。
