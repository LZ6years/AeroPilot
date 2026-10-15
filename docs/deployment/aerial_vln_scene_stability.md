# AerialVLN-S Linux 场景稳定性测试

第一阶段从 canonical val_seen、val_unseen 中按每个场景的 episode ID 排序取两条。
准备命令冻结独立代码、插件、相机模板、上下文配置、数据哈希和任务清单。
此入口只生成第一阶段任务，不启动或复用结果到第二阶段。

## 准备与启动

在 Linux 部署目录使用 `/home/wzq/miniconda3/envs/flyagent/bin/python`，以下以 `$PY` 表示该解释器：

```bash
PY=/home/wzq/miniconda3/envs/flyagent/bin/python
OUT=/home/wzq/aerialvln-scene-stability-20261015
$PY -m evaluation.benchmarks.aerial_vln.scene_stability \
  --output "$OUT" \
  --dataset-root /home/Newdisk/wzq/data/airvln/DATA/data/aerialvln-s \
  --env-root /home/Newdisk/wzq/data/airvln/ENVs \
  --settings-template /home/Newdisk/wzq/data/airvln/AirVLN/airsim_plugin/settings/3/settings.json \
  --execution-config evaluation/benchmarks/aerial_vln/configs/linux_local_stability_execution.example.json
$PY tests/manual/linux_local_preflight.py --config "$OUT/inputs/val_seen.json" --config "$OUT/inputs/val_unseen.json" --gpu 3
tmux new-session -d -s aerialvln-stability "$PY -m evaluation.benchmarks.core.campaign.parallel watch --output $OUT --workers 1 >$OUT/watch.log 2>&1"
```

必须先验证实际 GPU UUID、Vulkan 枚举、空闲资源，以及数据与全部场景脚本。
示例 UUID 属于本次服务器，不能直接用于另一台机器。缺少 PyTorch 如实记录；像素测深与地点地图不要求 CUDA 模型。
UE 使用独立进程组、进程局部环境和持久归属凭据；异常恢复先核验归属，再 TERM/KILL 并检查端口释放。
拒绝清理其他实验或未经验证的进程组。

## 查看、暂停与恢复

```bash
cat "$OUT/parallel/watchdog/status.json"
cat "$OUT/metrics.json"
tmux attach -t aerialvln-stability
# Ctrl-B，然后 D，离开会话但继续运行。
touch "$OUT/PAUSE"
# 排查完成后才恢复：
rm "$OUT/PAUSE"
touch "$OUT/parallel/watchdog/RESUME"
```

SSH 断开不会停止 tmux。服务器重启后手动重启同一 watch 命令，继续冻结清单；不重新 prepare。
同一 campaign 有独立锁，重复 watchdog 不会同时派发。
每条最多三次技术尝试，正常导航失败不重跑；单条期限 7200 秒，无进展阈值 900 秒。
认证失败、归属不明或清理失败需人工处理，不能无界重试。

## 报告与第二阶段

`report.html` 是总索引，`per_case.csv` 是逐条数据，`metrics.json` 包含覆盖率、首次完成和重试后完成数量。
只有所有任务正常完成并清理成功，`stability_passed` 才为 true；导航成功率不是稳定性门槛。
保留完整审计和深度文件，以便复查输入图片、工具、API 重试和场景生命周期耗时。

官方 SR 仍为正常完成且最终水平距离 ≤20m；补充 SR@25 使用严格 <25m。
`marginal_cases.json` 单列 20m < 距离 <25m 的正常完成任务。
缺失结果保持覆盖率缺口，未齐全时 `paper` 指标为空，`provisional` 明确仅针对有效结果。
全量阶段需另行确定参数、建立新 campaign、重新运行全部 864 条，不复用本阶段结果。
