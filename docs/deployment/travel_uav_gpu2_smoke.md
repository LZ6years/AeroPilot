# 新服务器 TravelUAV 单例验证

服务器：`wangzequan@10.68.240.49:10245`；代码目录 `/home/wangzequan/FlyAgent`。
解释器：`/home/wangzequan/miniconda3/envs/flyagent/bin/python`。

## 部署配置

- 索引：`/home/wangzequan/TravelUAV/metadata/travel_uav_index.json`，SHA256
  `8a2dbe59a182774f147869269a4a612546031498d5e40fac0697227412152e87`。
- 本例场景根目录：`/home/wangzequan/TravelUAV/ENVs/closeloop_envs/closeloop_envs`。
- 启动脚本：`NYCEnvironmentMegapa.sh`，参数格式 `-settings=PATH`；修复了该脚本和对应二进制的 owner 执行权限，ldd 未发现缺失库。
- 使用现有 TravelUAV 官方 settings 模板：Multirotor、paused_multirotor、Drone_1、FrontCamera，RGB/深度 256×256、FOV 90。
- GPU 2 UUID：`GPU-f8a48181-a511-5895-e45f-455fd8537ff7`，实际 Vulkan adapter 为 3。
  允许共享，启动要求至少 4096 MiB 空闲；GPU 1 仅允许合计不超过 16 MiB 的附带 G 上下文。
- 控制端口 39141。模型与插件沿用当前配置，pixel_map 配置不启用视觉检测模型。
- 固定 `test_seen` case：`NYCEnvironmentMegapa__0386165f-ef66-44ba-aeab-8d0d220b5422`。

所有有效配置、私有 env 和源码冻结在：
`/home/wangzequan/traveluav-runs/20261016-gpu2-smoke-r2`。
只生成一条任务，不创建全量队列。API 技术失败由现有 campaign 最多尝试三次，正常导航失败不重跑。

## 修复与验证

共享 campaign 的 local 排序曾强制把场景 ID 转整数，导致 TravelUAV 命名场景无法 prepare。
`evaluation/benchmarks/core/campaign/prepare.py` 现保留数字排序，同时接受场景名称；
`tests/test_shared_campaign.py` 覆盖混合名称和数字排序。
Windows 回归 25 passed、1 skipped；Linux 对应回归 26 passed。
原始失败准备目录保留，不用于运行。

无模型采集预检通过：启动约 14.03 s，采集及清理合计约 20.31 s。
实测主渲染 GPU 2 使用 2637 MiB，GPU 1 纯 G 上下文 4 MiB；RGB/深度及车辆通过。
采集进程组清理确认成功，端口释放。

## 查看运行与报告

```bash
OUT=/home/wangzequan/traveluav-runs/20261016-gpu2-smoke-r2
cat "$OUT/phase.txt"
cat "$OUT/parallel/watchdog/status.json"
cat "$OUT/metrics.json"
```

专用 tmux 为 `traveluav-smoke-gpu2`。运行脚本先采集，再启动单 worker watchdog；
成功结束写 `phase.txt=complete` 和 `supervisor.exit=0`。报告入口为 `$OUT/report.html`。
实际导航结果以最终 progress、metrics 和逐例 review 为准。

## 运行中记录与后续目录规则

前两次真实导航尝试均因 provider 请求失败结束，且清理验证通过；
第三次尝试已由 watchdog 启动，最终评分尚待产生。不能将本轮记为完整通过。
本次导航始终使用直连，未在运行中切换代理。

用户随后要求新生成文件统一放到 `$DATADIR`。后续新任务使用
`/home/Newdisk/wangzequan/flyagent/travel_uav/<run-id>`，诊断使用
`/home/Newdisk/wangzequan/flyagent/diagnostics/<run-id>`。
当前活动目录保持原路径，避免破坏冻结快照和正在打开的日志；不要移动活动任务。
代理对照及启用方法见 [本机代理隧道](remote_pc_proxy.md)。
