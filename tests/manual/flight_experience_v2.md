# 经验生成 V2 操作与验证

本轮不执行飞行，插件继续默认禁用。所有命令在仓库根目录使用 flyagent 环境。

```powershell
$flightPython = 'F:\conda_env\flyagent\python.exe'
& $flightPython -m pip install -r plugins/research/flight_experience/requirements.txt
& $flightPython tests/manual/flight_experience_v2_probe.py --output tests/manual_outputs/experience_v2_new --mode capability
& $flightPython tests/manual/flight_experience_v2_probe.py --output tests/manual_outputs/experience_v2_new --mode compare
& $flightPython tests/manual/flight_experience_v2_probe.py --output tests/manual_outputs/experience_v2_new --mode segments
```

capability 使用原图检验当前连接支持严格 Schema；失败时后续命令停止，不更换模型或降级。
compare 将四条旧候选作为一个固定比较组，旧/新提示使用相同事件、六张图、模型、180 秒超时
和输出 Schema，审核提示也相同。该对照只评估提示，不将旧提示批准结果发布到库。
segments 在调用前冻结三个片段及来源/原图散列，然后新生成、独立审核及自动发布到隔离库。
初始选片是指定片段实验；独立 reviewer 可以从原运行中选择反证，不能冒充自主初始选片效果。

查看提示：`plugins/research/flight_experience/prompts/v2/`。
每次调用记录 prompt_version、prompt_sha256、schema_sha256、实际请求、原图、公开响应、usage
及耗时。检查点复用记录标记 reused，不能计作新模型调用或再次计 token。

首次生成：

```powershell
& $flightPython tests/manual/flight_experience_probe.py run --source <原运行目录> --output <新实验目录> --observations <观察ID列表> --database <隔离库路径> --checkpoints <检查点目录>
```

发生审核超时后，使用新的 output 和相同 checkpoints 重跑；输入、模型、提示或证据变化将
使检查点失效。没有通过审核时允许零经验，不用减少校验获取发布。

维护旧经验：

```powershell
& $flightPython tests/manual/flight_experience_library.py --db <库路径> init
& $flightPython tests/manual/flight_experience_library.py --db <库路径> import <旧实验目录>
& $flightPython tests/manual/flight_experience_library.py --db <库路径> enqueue --all
& $flightPython tests/manual/flight_experience_library.py --db <库路径> worker --once --output <维护审计目录>
& $flightPython tests/manual/flight_experience_library.py --db <库路径> proposals
& $flightPython tests/manual/flight_experience_library.py --db <库路径> search 'motion feedback' --scope experiment
```

失败后重新 enqueue，保持相同维护输出根目录以复用阶段检查点。已有待人工提案不自动采用，
只有 V2 独立审核 approve 才走自动发布。人工撤回/归档继续要求记录身份和理由。

审核要求修订时保留原提案、修改要求、修订稿及复审。原图缺失、引用失效、版本冲突、格式失败、
超时分别保留，不计作语义 discard。没有可靠账单时费用未知，超时无 usage 不计作零消耗。

生成总览（只读产物，不调用模型）：

```powershell
& $flightPython tests/manual/flight_experience_v2_report.py --output tests/manual_outputs/experience_v2_20260927
```

`index.html` 展示旧文、新文、审核、全部候选、证据引用及成本；`summary.json` 提供逐阶段
usage 和分组统计。原三片段与后续提示修正诊断单列，不用额外诊断替换失败实验。
