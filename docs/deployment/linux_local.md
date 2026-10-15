# Linux local 部署与单条验证

Agent 在 Linux 服务器运行时，仍使用 `connection.mode=local`、
`data_transport=direct`。LocalSceneControl 自动选择 `launch.linux`；Windows
选择 `launch.windows`。场景位置、GPU 和 settings 参数由部署配置提供。
不要把运行在 Windows 上的 Agent 配置为 local 来启动另一台 Linux 服务器。

## 部署代码和环境

使用独立目录检出明确的 Git commit，不覆盖正在执行的实验。`.env`、场景、
数据集和输出不通过 Git 分发。当前工作区如果存在未提交的运行时改动，单纯
拉取 HEAD 不会获得它们；先生成经过审查的部署快照并记录 commit。

服务器使用独立 Conda `flyagent` 环境，Python 3.10 与当前 Windows 环境一致。
以下路径是本次服务器的实际解释器；其他服务器应替换为其 flyagent 路径。

```bash
conda create -n flyagent python=3.10 -y
/home/wzq/miniconda3/envs/flyagent/bin/python -m pip install -r requirements.txt
/home/wzq/miniconda3/envs/flyagent/bin/python -m pip install --no-build-isolation -r requirements-airsim.txt
```

在检出目录创建权限为 600 的 `.env`，采用与原实验相同的模型配置。
凭证不写入命令行、Git 或报告。Linux local 不需要 Windows SSH 私钥路径。
可选 Python GPU 模型需另装与驱动匹配的依赖；未安装 torch 不能声称完成 CUDA
感知验证。UE 的 Vulkan 渲染与 Python CUDA 推理分别核验。

## 配置和 GPU

地图示例：

- `configs/maps/aerial_vln_linux_local.example.json`
- `configs/maps/travel_uav_linux_local.example.json`

两份示例引用本次服务器已有场景，实际部署前核对路径。启动方式为
`linux_bash`，以参数数组执行 `bash SCRIPT ...`，不经 shell 拼接。
`direct` 用于原生可执行文件；`wsl_bash` 只能在 Windows 使用。
`settings_arg_style=double_dash` 生成 `--settings PATH`，
`single_dash_equals` 生成 `-settings=PATH`。未指定时保留原有默认行为。
本地模板必须设置 `settings_template_source=local`。

测试仅使用物理 GPU 3，串行运行。先用 `nvidia-smi` 记录 UUID、PCI 地址、
显存和现有进程；0、2 号卡上的任务不操作。UE 适配器编号不一定等于物理卡号：
该服务器默认 Vulkan 枚举混入 llvmpipe 软件设备。先对本次 shell 限定 NVIDIA
ICD，再重新枚举 PCI 地址映射，确认 `-GraphicsAdapter=3` 指向物理 GPU 3：

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

这不会更改服务器全局驱动配置。场景启动后还需检查 UE 日志及 PID 的实际
GPU 归属。若不匹配，清理本次场景并停止测试，不能改用其他空闲 GPU。
Python 感知如需 CUDA，应使用物理卡 UUID 设置该测试进程的
`CUDA_VISIBLE_DEVICES`，隔离后进程内使用 `cuda:0`。

## 验证顺序

从仓库根目录执行。先运行不触及 UE 的测试：

```bash
/home/wzq/miniconda3/envs/flyagent/bin/python -m pytest -q tests/diagnostics/test_local_process_control.py tests/diagnostics/test_linux_process_lifecycle.py tests/test_sim_server.py tests/test_remote_scene_control.py
```

Linux 生命周期测试启动测试专用父子进程，验证父进程先退出、子进程忽略
TERM 时仍可清理，且另一独立进程组不受影响。Windows 上这些测试明确跳过。

预检不启动场景、不调用 API，只打印诊断：

```bash
/home/wzq/miniconda3/envs/flyagent/bin/python tests/manual/linux_local_preflight.py --gpu 3 --config evaluation/benchmarks/aerial_vln/configs/linux_local_smoke.example.json --config evaluation/benchmarks/travel_uav/configs/linux_local_smoke.example.json
```

两份 smoke 配置固定各一条真实 episode，独立端口为 39131、39141。
端口被占用时停止，不接管现有场景。使用 runner 的 `--validate-only` 核验
episode 选择和数据完整性，再分别执行：

```bash
/home/wzq/miniconda3/envs/flyagent/bin/python evaluation/benchmarks/aerial_vln/scripts/run_eval.py --config evaluation/benchmarks/aerial_vln/configs/linux_local_smoke.example.json --eval-id linux-local-aerial
# 确认上一条进程、端口均清理完成后，再运行下一条。
/home/wzq/miniconda3/envs/flyagent/bin/python evaluation/benchmarks/travel_uav/scripts/run_eval.py --config evaluation/benchmarks/travel_uav/configs/linux_local_smoke.example.json --eval-id linux-local-travel
```

真实任务会调用付费模型 API。不要重复使用已有输出的 eval-id。
输出位于 `runs/linux_local_smoke/`。检查评分、RGB/depth、模型调用日志、
`scenes/*/lifecycle.jsonl` 和场景进程日志；导航失败不等于部署失败。
异常退出验证可使用一次故意不可达的独立后端端口触发启动超时，确认本次
进程组和已分配端口均释放。不要用 `kill -9` 杀 Agent 来测试正常清理保证。

## 清理和结果边界

Linux 由 ProcessManager 记录每次创建的 session/process group，先 TERM、
超时后 KILL；即使脚本已退出，仍检查所属子进程。已退出的 zombie 不作为
活跃 UE 进程处理。未确认清理完成时保留所有权并抛错，不静默报告成功。
显式脱离 session 的自守护程序不在此本地启动契约内。

只清理本次拥有的进程，不使用全局 `pkill`，不按 GPU 批量杀任务。
记录 API、启动、观测、动作、清理的实际耗时；该 GPU 服务器的网络表现不能
用 AWS 回放结果代替。扩大评测前须确认两类单条任务完成且无本次资源残留。
