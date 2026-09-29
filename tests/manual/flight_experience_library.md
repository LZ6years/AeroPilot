# 飞行经验库 V1 手动验证

> 历史 V1 说明。当前 V2 已改为独立审核后自动发布、严格 Schema 和 180 秒经验调用超时。
> 新操作请使用 [V2 操作说明](flight_experience_v2.md)；下面的人工发布流程仅适用于历史 V1 提案。

使用独立数据库。以下为 PowerShell 示例，先在仓库根目录设置变量：

```powershell
$flightPython = 'F:\conda_env\flyagent\python.exe'
$flightLibrary = 'tests/manual/flight_experience_library.py'
$experienceDb = 'F:/UAV-Agent/tests/manual_outputs/experience_library_demo/experience.sqlite'
& $flightPython $flightLibrary --db $experienceDb init
& $flightPython $flightLibrary --db $experienceDb import tests/manual_outputs/flight_exp_5x2_20260927/experience/04/diagnostic
& $flightPython $flightLibrary --db $experienceDb list
& $flightPython $flightLibrary --db $experienceDb audit
```

导入只保存候选，不沿用原报告的采纳状态。实验来源不能发布到 production 范围。
同一实验重复导入不重复创建候选。原图缺失或来源变化被标记为 unavailable，不替换原始证据。

## 提交和后台执行

```powershell
& $flightPython $flightLibrary --db $experienceDb enqueue --all --request-key first_batch
& $flightPython $flightLibrary --db $experienceDb worker --once --output tests/manual_outputs/experience_library_demo/maintenance
& $flightPython $flightLibrary --db $experienceDb jobs
& $flightPython $flightLibrary --db $experienceDb proposals
```

`enqueue --ids <ID> ...` 可指定目标。指定 request-key 的同一次提交幂等；失败后重试应使用新的
key，或不传 key。不传 key 时只合并仍活跃的重复任务。模型调用发生在 worker，不发生在 enqueue/search。

持续 worker 去掉 `--once`；可在另一个终端前台运行并用 Ctrl+C 停止，或隐藏窗口启动：

```powershell
$workerArgs = @($flightLibrary, '--db', $experienceDb, 'worker', '--output', 'tests/manual_outputs/experience_library_demo/maintenance')
Start-Process -FilePath $flightPython -ArgumentList $workerArgs -WorkingDirectory 'F:/UAV-Agent' -WindowStyle Hidden
```

默认复用 `.env` 连接、超时与重试。可用 `--reflection-model`、`--review-model` 分别指定模型，
不修改正式配置。两个角色使用独立上下文；同模型不等于独立知识来源。
持续 worker 不随 Core 自动启动，也不随飞行结束自动提交。

## 人工检查与发布

每次尝试位于 `<output>/<job_id>/<attempt_id>/`：

- `before.json`：参与比较的旧内容、版本、适用条件、证据。
- `evidence.json`、`media/`：实际提供的原始事件与原图。
- `proposal.json`：模型提出的新增、修订、合并、拆分、冲突、归档或不变。
- `calls.json`：独立审核原文、调用耗时、返回 token、请求审计；超时的 usage 与费用未知。
- `result.json`：本次结果；失败不改动旧发布版本。

审核通过后仅标记 awaiting_human。人工确认后操作：

```powershell
& $flightPython $flightLibrary --db $experienceDb show <experience_id>
& $flightPython $flightLibrary --db $experienceDb publish <proposal_id> --actor reviewer --reason '已核对原始证据和适用范围'
& $flightPython $flightLibrary --db $experienceDb search 'invalid distance' --scope experiment
& $flightPython $flightLibrary --db $experienceDb withdraw <experience_id> --actor reviewer --reason '确认建议有误，等待修订'
& $flightPython $flightLibrary --db $experienceDb archive <experience_id> --actor reviewer --reason '被更具体的经验替代'
```

发布需独立审核 approve，且证据未变化、基础版本仍一致；否则重新维护。**不要为让搜索非空而发布这批候选。**
独立模型审核不通过不能通过 publish 强行放行。归档、撤回保留来源、正文和历史版本。

## 搜索插件配置

`configs/plugins.json` 默认显式禁用 `capabilities.flight_experience`。独立实验需要启用时，
在实验使用的配置中将它加入 enabled 并从 disabled 移除；在实验插件副本的
`search/capability.json` settings 中设置数据库绝对路径、scope=experiment。
正式配置保持默认不启用，数据库路径不进入 provider 工具参数。

插件只暴露 `skill.flight_experience.search(query, filters?, limit?)`。filter 为 decision、action、
observation、anomaly、platform、control、sensor 的字符串精确匹配；未知条件会返回提示，
不能当作已确认兼容。不会调用 reflection、更新数据库或返回历史图像。

## 复现 10 例隔离实验

```powershell
& $flightPython tests/manual/flight_experience_library_probe.py `
  --batch tests/manual_outputs/flight_exp_5x2_20260927 `
  --output tests/manual_outputs/experience_library_new_trial --run-models
```

输出目录必须不存在且不与历史批次重叠。导入 10 例的自主与诊断产物，核对证据，检查重复导入，
测试维护搜索与发布隔离，对实际存在的候选进行新模型维护。无候选的案例如实保留零结果。
不执行飞行、不自动发布，不把本实验当作闭环收益验证。

```powershell
& $flightPython -m pytest tests/test_flight_experience.py tests/test_flight_experience_library.py -q
```
