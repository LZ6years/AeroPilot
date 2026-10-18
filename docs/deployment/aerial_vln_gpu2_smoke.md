# 新服务器 GPU 2 验证记录

## 固定范围

服务器 `wangzequan@10.68.240.49:10245`，代码目录 `/home/wangzequan/FlyAgent`，
解释器 `/home/wangzequan/miniconda3/envs/flyagent/bin/python`。
只验证 13 个场景和两条导航任务，不创建全量队列。
输出目录 `/home/wangzequan/aerialvln-runs/20261016-gpu2-smoke`。

`prepare_linux_aerial_smoke.py` 冻结全部场景地图，分别选择 seen 场景 2、unseen 场景 7
按 ID 排序第一条；构建后强制检查队列只有两条任务。
模型、上下文和插件由现有配置冻结，私有 `.env` 在 inputs 下权限为 600。

本次 `allow_shared_gpu=true`，`minimum_free_gpu_mib=4096`；只改变派发条件，
不授权清理其他进程。默认配置仍要求独占。派发前核对固定物理卡 UUID，
未知或不足的空闲显存不允许启动，采集和 campaign 均执行该检查。

## 已完成验证

- 本地离线回归 141 passed、5 skipped（Linux 专属）；Linux 回归 56 passed。
- 13 个场景执行权限已修复，Shipping 二进制 ldd 未发现缺失库。
- 两份 canonical 数据共 864 条，测试只选其中两条。
- 去重 `/usr/share` 与 `/etc` 下同一 NVIDIA 驱动 manifest 后，GPU UUID 绑定通过。
- 只读预检、campaign prepare/bind 通过；冻结源码及配置由 campaign 自身记录哈希。

## 2026-10-16 暂停情况

场景 2 启动成功后，主渲染在物理 GPU 2（1452 MiB，C+G），
同一 UE 进程在物理 GPU 1 另有 4 MiB 的纯 G 上下文。
由于本次 `graphics_context_limits_mib` 为空，检查按计划拒绝继续。
本次进程组 3040188 已清理、39131 端口释放；没有派发导航或调用模型。
不属于本次的既有场景及其他 GPU 任务未被操作。

`PAUSE` 标记已写入；`deployment_status.json` 记录暂停原因及部署文件哈希，
`capture/capture_validation.json` 保存实际进程/GPU/清理证据。
采集 tmux session 已随失败退出，当前没有本次守护程序继续运行。
尚未完成后续 12 场景、两条导航和受控中断恢复。

如果用户明确允许 GPU 1 的少量 G 上下文，需冻结新的执行配置及采集证据，
再继续验证；不能直接改旧快照后称其未变，也不能把失败采集改写为通过。
无论是否允许例外，都不自动启动全量评测。

## 获准继续：r2

用户随后明确允许物理 GPU 1 上合计不超过 16 MiB 的纯 G 上下文，主渲染仍固定 GPU 2。
因此另建 `/home/wangzequan/aerialvln-runs/20261016-gpu2-smoke-r2`，
旧失败记录不变；新执行配置 `graphics_context_limits_mib={"1":16}`。

13 场景重新采集全部通过；真实 UE 启动父进程被受控终止后的清理验证也通过，
独立 CPU sentinel 保持运行。具体证据在 `capture/capture_validation.json` 和
`interruption/interruption_validation.json`。

`aerialvln-smoke-gpu2` tmux session 中的 `run_validation.sh` 依次执行采集、
中断恢复、两条任务 watchdog。任一前置阶段失败则退出，不继续导航。
正常完成两条后 watchdog 退出，脚本记录 `phase.txt=complete` 和 `supervisor.exit=0`。
`validation_summary.json` 汇总场景、清理、任务指标、API 重试和阶段时间；
`report.html` 提供逐任务回放入口。当前完成情况以这些运行文件为准。

查看状态：

```bash
OUT=/home/wangzequan/aerialvln-runs/20261016-gpu2-smoke-r2
cat "$OUT/phase.txt"
cat "$OUT/parallel/watchdog/status.json"
cat "$OUT/metrics.json"
```

watchdog 运行阶段暂停派发用 `touch "$OUT/PAUSE"`；恢复需先排查原因，
再删除 PAUSE 并创建 `"$OUT/parallel/watchdog/RESUME"`。
它不是立即终止当前导航的命令。测试完成后不重启本次 watchdog 来发起全量。

## r2 最终结果

本轮已结束：`phase.txt=complete`、`supervisor.exit=0`，watchdog 状态为
`campaign_complete`，专用 tmux session 已退出。队列始终只有两条任务，未启动全量。

| split / 场景 | episode ID | 技术尝试 | 最终水平距离 | SR / SR@25 | nDTW |
|---|---|---:|---:|---|---:|
| seen / 2 | 308Q0PEVBHBD1QODHYSWU7PBK0L9IW | 1 | 101.37 m | 否 / 否 | 0.04277 |
| unseen / 7 | 31LVTDXBLG8LVRS5WIPVD8HX8E7RLE | 3 | 133.29 m | 否 / 否 | 0.09037 |

seen 首次正常完成并评分；unseen 前两次为 `provider_failed`，第三次正常完成并评分。
所有尝试均确认清理成功。导航失败没有重跑；unseen 重跑仅由 API 技术失败触发。
因此部署流程最终跑通，但不能称为首次运行全部稳定通过，API 稳定性仍需改进。

| 阶段 | seen | unseen（含三次尝试） |
|---|---:|---:|
| 场景 ready 累计 | 9.05 s | 27.06 s |
| 模型调用数 | 29 | 83 |
| API 调用累计耗时（包含请求内部重试） | 596.07 s | 2258.66 s |
| 观察采集累计 | 16.09 s | 42.92 s |
| 场景进程清理累计 | 1.13 s | 3.41 s |
| 启动至场景清理累计 | 628.46 s | 2350.30 s |

API 调用耗时分别约占上述场景生命周期的 94.8% 和 96.1%。各计时来源及范围不同，
不把表内各项简单相加；请求时间也不能直接拆解为纯网络或纯推理时间。
seen 有 1 次超时、1 次连接失败；unseen 累计 10 次超时、10 次连接失败。
成功恢复的调用分别为 2 和 11 次，另有 unseen 两次尝试因请求失败而终止。

13 个场景的 RGB、深度、车辆和 GPU 检查均通过；单场景启动、采集及清理合计
11.12–14.27 s。受控中断后清理通过，独立 sentinel 存活。
最终只读核验全部 18 个本次登记进程组均无残留，39131 端口关闭。
四个启动前记录的训练 PID 仍存在；原有无关 UE PID 3036198 已不在，
本次没有向它发送信号，不能仅凭最终快照判断其退出原因。

远端报告入口为 r2 目录的 `report.html`；逐任务 `review.html` 包含回放。
`validation_summary.json`、`final_cleanup_check.json`、`metrics.json` 和 `per_case.csv`
已复制到本地 `tests/diagnostics/artifacts/gpu2_smoke_20261016/`（Git 忽略）。
前两次失败日志保留在 unseen 的 `attempt_01`、`attempt_02`，不覆盖或混入成功尝试。
