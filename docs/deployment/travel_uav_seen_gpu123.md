# TravelUAV Seen 全量三路评测（2026-10-18）

本轮仅测试 Test Seen：1,418 条独立任务，783 Easy、635 Hard。完整审计 100 条，
其余 1,318 条保存精简执行回放；不加入 Unseen 或消融任务，不复用旧成绩。

## 运行位置与配置

- 服务器：`wangzequan@10.68.240.49:10245`。
- 输出：`/home/Newdisk/wangzequan/flyagent/travel_uav/20261018-seen30-gpu123-r2`。
- tmux：`traveluav-seen30-gpu123`；入口：输出目录下 `run_campaign.sh`。
- 导航源码提交：`808ab91cc7679ddcba436fee7741e7dd22396de1`。
- 清单指纹：`e257a3ce472f01ec47601af4af845ffc798c17352264092b9785d4fe5d49c49f`。
- 解释器：`/home/wangzequan/miniconda3/envs/flyagent/bin/python`。
- 模型沿用私有配置中的 `gpt-6.1-sol`；单前视、无辅助指导，停止协议 v3。
- 每次请求最多 30 张图，观察、地图、工具证据共同计数；历史压缩保持开启，
  沿用当前 48k 预取、64k 软触发、32k 目标与最近 4 个完整决策窗口设置。
- 启用飞行知识、像素测深和地点存档；不启用检测／分割推理模型。
- 每任务最多 300 个物理窗口。正常导航失败保留评分；技术故障最多三次尝试。
- 代理固定为 `http://127.0.0.1:17890`，本机模拟器连接直连；代理中断时暂停，
  不自动切换网络路线。

| 槽位 | 物理 GPU | UUID | Vulkan 编号 | 控制端口 |
|---|---:|---|---:|---:|
| A | 1 | GPU-cd30b132-6b2c-c6fc-b34e-d0ec28fccf20 | 0 | 39141 |
| B | 2 | GPU-f8a48181-a511-5895-e45f-455fd8537ff7 | 3 | 39142 |
| C | 3 | GPU-a57052c0-12fc-7234-f4cf-00fec1a49945 | 2 | 39143 |

GPU 编号与 Vulkan 编号经过本轮 UUID 枚举核验，不能将上表作为以后机器的固定映射。
允许共享，每张卡至少有 4GiB 可用显存才派发。其他任务及其进程不由本轮控制器终止。

远端 flyagent 未安装 PyTorch，环境诊断记录 `torch_version=null`、
`cuda_available=false`。本轮 GPU 用于 UE 场景渲染，MLLM 通过 API 调用；
不能把这次评测表述为本地 CUDA 模型推理。三个 UE 进程的实际 GPU 归属已核验。

## 数据来源与审计样本

索引 SHA256：`8a2dbe59a182774f147869269a4a612546031498d5e40fac0697227412152e87`。
[官方 Seen 划分](https://github.com/buaa-colalab/TravelUAV/blob/main/data/uav_dataset/seen_valset.json)
有 75,374 条逐帧记录，去重后为 1,418 个任务。
其 SHA256 `af5b7b660ed3b1dea95b53aa72a98f74ae7a37cd381aa0e818ee2a343cf54552`
与索引记录的源文件哈希一致，逐任务身份比对没有缺失或新增。
这排除了索引转换多加样本的情况；发布文件与论文中 1,410 的统计为何不同仍未定案。
论文应说明实际使用的发布文件、数量和哈希，不删去八条来凑原文数字。

完整审计按地图 × 难度比例分层，采用最大余数法分配 100 个名额，层内按
`SHA256("42:" + episode_id)` 排序。名单在运行前固定，不参考成功率。

| 地图 | 全部任务 | 完整审计 |
|---|---:|---:|
| Carla_Town01 | 165 | 12 |
| Carla_Town04 | 106 | 8 |
| Carla_Town05 | 73 | 5 |
| Carla_Town06 | 47 | 3 |
| Carla_Town07 | 60 | 4 |
| Carla_Town10HD | 141 | 10 |
| Carla_Town15 | 111 | 7 |
| ModernCityMap | 124 | 9 |
| NYCEnvironmentMegapa | 281 | 20 |
| NewYorkCity | 145 | 11 |
| TropicalIsland | 165 | 11 |

`inputs/audit_selection.json` 保存完整名单和六条技术试运行身份，
`inputs/source_split_audit.json` 保存原始划分核对。
100 条完整审计先执行，再进入精简组；六条试运行属于这 100 条，正式统计不重复计数。
精简记录在任务进行中保留全部配对深度，供历史测距读取；仅在评分、报告与清理确认后，
删除该次任务的临时深度文件。完整审计和失败诊断不按此规则清理。

## 已完成的验证

11 张地图的资源、目标资产列表、动态目标生成及位置回读、RGB／深度和清理均通过。
A 槽验证全部地图，B／C 各额外验证一张地图，共 13 次场景采集，无 LLM 调用。
只补齐本账户资产启动脚本缺失的 owner 执行位，记录内容哈希；未修改资产字节。
模型 API 探针通过后才启动导航，GPU 1／2／3 的三个任务均已产生模型决策与工具结果。

Windows 相关回归 127 passed、1 skipped；Linux 对应回归 128 passed。
Windows 全量默认测试 1,701 passed、1 failed；唯一失败仍为既有的默认测试导入
`tests.manual` 的边界检查，涉及三个原有审计测试，本轮未改其归属。

报告浏览器检查使用 Edge／Playwright（会话未提供 Browser 插件），桌面 1440×1000、
移动视口 430×932。使用旧录制输入验证当前报告代码：地图展开、地点跳转、动作展开，
以及独立 HTML 的 141 张内嵌图片均通过，没有破图或页面脚本错误。
这属于离线显示检查，不作为新版导航成绩。
证据位于本地 `G:/FlyAgent-Eval/travel_seen_prepare_20261018/browser/qa.json`。

## 查看状态与报告

```bash
export DATADIR=/home/Newdisk/wangzequan
OUT="$DATADIR/flyagent/travel_uav/20261018-seen30-gpu123-r2"
cat "$OUT/phase.txt"
cat "$OUT/guard_status.json"
cat "$OUT/parallel/watchdog/status.json"
cat "$OUT/parallel/status.json"
cat "$OUT/storage_check.json"
tail -n 40 "$OUT/watchdog.log"
```

六条任务分两批、每批三个槽运行。放行条件是有效执行、报告、GPU 和清理证明，
不要求导航成功。通过后自动进入剩余队列；技术检查失败时保留记录并停止扩展派发。
`parallel/trial_check.json` 是放行结果，`report.html`、`metrics.json`、
`summary.csv` 和 `per_case.csv` 是持续更新的汇总。NE、SR、OSR、SPL 的有效覆盖与
全量分母分开呈现，未完成的论文汇总值留空。

守护进程在 Linux 上运行，Windows 工作区仅用于开发和回归。原控制器曾将
`SceneRecoveryRequired` 的异常位移诊断留给人工处理，导致守护进程存活但不再派发。
共享评测恢复层现将适配器明确报告的 `move_body`／`turn_relative` 的
`motion_discontinuity` 纳入已知技术故障：只有旧场景清理独立验证通过后，才可在新场景
重新尝试。原场景仍禁止继续动作，位移检查边界保持原值。每任务最多三次，失败尝试及
部分轨迹保留为技术记录，不作为有效导航评分；未知错误、清理未确认和人工 `PAUSE`
仍暂停。此更新修复自动续跑，不表示已查明或消除了模拟器位姿跳变。

已存在的冻结任务通过新的不可变控制器目录加载恢复层更新；原 `source`、模型输入、
评测清单和已有成绩均保留。切换前须确认 controller／host 已退出，并备份进度、
执行覆盖和守护记录，再在 Linux 重启代理守护与三路 watchdog。活动任务不做此切换。
GPU 采样存档除最近 20 次记录外，还保留每一路最早的有效渲染绑定证明，供控制器
更新后复核已完成的技术试运行。旧存档若已丢失某一路证明，须从保留的原始测量恢复，
或重新采集真实场景；不能根据配置补造 GPU 观测。

历史外推 Seen 全量约 7.5 天，十天为目标预算；共享 GPU、API 和新版行为均会影响耗时，
应以本轮实测更新估计。磁盘按完整／精简组分别校准，容量不足会暂停，不删除旧产物。
初版 `20261018-seen30-gpu123` 仅做了离线准备，没有派发导航，保留作准备记录。

Git 同步前的远端改动备份位于
`$DATADIR/flyagent/diagnostics/git-sync-20261018-1738`，包含 tar、补丁和 stash 引用。
服务器 GitHub 主机校验尚未配置，这次使用校验过的 Git bundle 快进同步，未关闭 SSH 校验。
既有评测继续使用各自的冻结目录，未移动或改写活动路径。
