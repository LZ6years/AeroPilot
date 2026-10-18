# AerialVLN-S 全量：GPU 2 与本机代理

## 并行 unseen：GPU 3

用户另行授权531条 val_unseen 在物理GPU3运行，并明确允许本次产物存入主目录。
独立输出 `/home/wangzequan/aerialvln-runs/20261016-unseen-gpu3-proxy`，
tmux `aerialvln-unseen-gpu3`，端口39151；seen继续使用GPU2和39131。
GPU3 UUID `GPU-a57052c0-12fc-7234-f4cf-00fec1a49945` 对应Vulkan adapter 2。
指纹 `4350d4f4fbcfbad8a4f363f0276f8723e58eb3c960fe30243a5be3b5a8472f1b`。
unseen 5个场景7/9/13/21/24的采集、GPU和清理验证全部通过，已进入首条任务。
每个队列各1个worker，共享同一代理隧道，分别计分和守护，不复用任何先前结果。
本次主目录容量预检为可用94.52 GiB、两倍产物预算86.44 GiB，运行中继续检查。
这次主目录例外不改变其他新任务默认使用DATADIR的约定。

## 当前实际运行：仅 seen

用户随后明确要求先完成所有 seen。原864条队列未派发，已写 PAUSE 并停止等待守护，
不修改原冻结快照。新队列位于
`/home/Newdisk/wangzequan/flyagent/aerial_vln/20261016-seen-gpu2-proxy`，
仅有333条 val_seen，完成后停止，不会进入 unseen。
tmux 为 `aerialvln-seen-gpu2`；运行入口仍为新目录的 `run_guard.sh`。
新指纹 `3c018654b055a6dd760f2f94b5e91317b2b2e7406e5f3b16d4c43df34b04682c`。
两倍产物预算54.21 GiB，启动时空闲139.30 GiB，容量检查通过。
指标、GPU2单worker、代理及其他参数不变，不复用旧测试结果。

```bash
export DATADIR=/home/Newdisk/wangzequan
OUT="$DATADIR/flyagent/aerial_vln/20261016-seen-gpu2-proxy"
cat "$OUT/guard_status.json"
cat "$OUT/parallel/watchdog/status.json"
tail -n 50 "$OUT/watchdog.log"
# 暂停后续派发
touch "$OUT/PAUSE"
```

以下864条记录保留为先前准备过程，不能重启它与seen队列并行派发。

## 本次入口

输出：`/home/Newdisk/wangzequan/flyagent/aerial_vln/20261016-full-gpu2-proxy`。
`DATADIR=/home/Newdisk/wangzequan` 已在启动脚本显式设置。
tmux：`aerialvln-full-gpu2`；启动脚本：该目录的 `run_guard.sh`。
解释器：`/home/wangzequan/miniconda3/envs/flyagent/bin/python`。

冻结 333 条 val_seen、531 条 val_unseen，总计 864 条，全部为新队列，未复用 smoke 结果。
指纹：`c1d8ba786f838632ed45cde0435c17514db0c197a131a69f428d347796f7b1d1`。
独立外层 guard 脚本哈希见 `launch_preflight.json`，冻结源码哈希见 `source_hashes.json`。

物理 GPU 2，单 worker，允许共享且启动前至少 4 GiB 空闲。
UUID `GPU-f8a48181-a511-5895-e45f-455fd8537ff7` 对应 Vulkan adapter 3。
GPU 1 仅允许已授权的合计 16 MiB 纯 G 上下文。控制端口 39131。
模型 gpt-6.1-sol、流式 Chat Completions，当前模型和插件参数从之前批准的配置冻结。
像素测深、地点地图及飞行知识启用；当前深度图不直接附图；请求图片上限20、工具证据4、最近观察3。
每条最多3次技术尝试，单次7200秒，无进展900秒；正常导航失败不重跑。

## 评分与汇总

SR：正常完成且最终水平距离 ≤20m；SR@25：相同有效条件且严格 <25m。
OSR：轨迹曾进入水平20m；NE：最终水平距离；nDTW/SDTW 沿用既有评分。
临界任务为正常完成且 20m < 最终水平距离 <25m，附停止来源及回放路径。

`metrics.json` 的 groups 分别统计 seen/unseen，overall 汇集逐例值进行样本加权。
`full_denominator_rates` 用全部计划任务作分母，技术失败和未完成仍占分母。
`provisional` 仅为有效子集，`paper` 在覆盖不完整时为空；缺失 NE/nDTW 不补零。
`status_counts`、`unscored` 与首次/重试完成数量保留。
JSON、`summary.csv`、`per_case.csv`、`report.html` 使用同一汇总。
原始审计和失败尝试保留；现有逐例报告生成机制未在本轮改写。

## 预检与当前等待状态

本地汇总/调度回归 107 passed、1 skipped；Linux 汇总/调度/真实进程生命周期回归31 passed。
另外执行评分边界和代理 guard 离线测试。
10次相同请求交错网络对照（图片数量2、4、13、20）：直连5/5成功，代理5/5成功。
代理均值18.51秒，直连12.68秒；不能声称代理一定更快或长期无超时。
诊断位于 `$DATADIR/flyagent/diagnostics/20261016-aerial-full-preflight/api_routes.json`。

启动时所需容量为151015420224 bytes（140.64 GiB），
当时空闲149585461248 bytes（约139.32 GiB），因此 guard 已启动但等待磁盘空间，尚未派发。
预算基于之前 AerialVLN 最大单次保留量87393183 bytes ×864×2；未删除已有文件。
容量和代理同时通过后自动启动冻结 campaign watchdog，不需要重新准备队列。
长任务运行后，campaign 会按实际产物重新估算容量，不足时暂停。

## 查看和暂停

```bash
export DATADIR=/home/Newdisk/wangzequan
OUT="$DATADIR/flyagent/aerial_vln/20261016-full-gpu2-proxy"
cat "$OUT/guard_status.json"
cat "$OUT/progress.json"
tail -n 50 "$OUT/guard.log"
# 真正派发后可查看
tail -n 50 "$OUT/watchdog.log"
# 暂停后续派发，当前任务走正常收尾
touch "$OUT/PAUSE"
```

代理固定 `http://127.0.0.1:17890` 经 SSH 转发到 Windows Sakuracat 7890。
guard 每30秒检查 CONNECT；运行中失联写 PAUSE，绝不切回直连。
本机电脑、Sakuracat 和 SSH 隧道均须在线。guard 不主动删除运行时 PAUSE；
排查恢复后删除 PAUSE，并按原 watchdog 流程创建 `parallel/watchdog/RESUME`。
服务器重启后需手动以专用 tmux 运行上述启动脚本；不要重复启动。

## 两组后续任务全部精简（2026-10-17，最新设置）

用户确认 seen GPU 2 和 unseen GPU 3 均使用能够查看执行过程的精简 HTML。
两组各自已评分的 37 条保留原产物；seen 剩余 296 条、unseen 剩余 494 条均使用
`paper_process_metrics_v1`，不再等待 unseen 的前 200 条完整记录完成。
HTML 保留观察图片、公开动作意图、工具调用、轨迹和评分；模型请求统计、重试和错误记录保留，
不保存重复的完整请求正文与上下文调试快照。已有产物不删除，模型、导航和评分参数不变。

两组的 `parallel/execution_overlay.json` 记录有效策略。容量预算先按 10 GiB 门槛准入，
完成精简任务后按实测产物重新估算；该准入不是全量磁盘必然够用的保证。
迁移备份、验证回执和重启脚本位于
`/home/Newdisk/wangzequan/flyagent/diagnostics/20261017-compact-both/`。
本地与 Linux 的精简日志及保留策略回归各 9 项通过。

## Unseen GPU 3 分段保留（2026-10-16，已被上述设置替代）

当前 unseen 队列为 `/home/wangzequan/aerialvln-runs/20261016-unseen-gpu3-proxy`，
tmux 为 `aerialvln-unseen-gpu3`。按用户授权继续使用该目录，不移动已有结果。
本节记录最新的 unseen 保留规则；上方首次全量准备记录属于历史状态。

- 冻结清单的前 200 个 episode（含已完成的 10 个）保留完整日志，技术重试沿用同一规则。
- 其余 331 个使用 `paper_process_metrics_v1`：保留评分、轨迹、RGB、动作、模型调用统计、
  耗时、重试与错误记录及简版回放；省去完整请求正文和上下文调试快照。
- 精简任务仅在评分有效、清理确认且报告生成成功后删除本次临时深度文件。
  失败任务的诊断不按此规则删除；已有历史产物不删除。
- 先完成完整记录组，再进入精简组。容量按当前组的实测最大保留量乘剩余数量和两倍余量计算，
  仍保留至少 10 GiB 的门槛；精简组首次运行后用实际大小校准，不承诺剩余磁盘必然够用。
- 本次只更新外置调度器与日志适配器，原始 manifest、Runtime、模型、导航与评分冻结内容保持一致。
  seen GPU 2 的策略不变。保留规则和文件哈希记在 `parallel/execution_overlay.json`。

迁移检查记录及独立重启脚本位于
`/home/Newdisk/wangzequan/flyagent/diagnostics/20261016-unseen-retention/`。
仅在 tmux session 和旧守护程序均已退出后，使用其中的 `resume_unseen.sh` 重启；不要重复派发。
该脚本沿用已通过的场景采集结果，不重复执行 `run_guard.sh` 内的场景采集。

```bash
OUT=/home/wangzequan/aerialvln-runs/20261016-unseen-gpu3-proxy
cat "$OUT/parallel/watchdog/status.json"
cat "$OUT/storage_check.json"
tail -n 30 "$OUT/guard.log"
```

