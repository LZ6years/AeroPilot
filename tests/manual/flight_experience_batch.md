# 5＋5 新飞行与离线经验实验

`flight_experience_batch_probe.py` 是手动实验调度器，复用已有单例执行、清理和报告。
不修改飞行策略，不注册经验工具，不把实验经验写入正式知识。

```powershell
python -m pytest -q tests/test_flight_experience_batch.py tests/test_flight_experience.py tests/test_benchmark_20_case_probe.py
python tests/manual/flight_experience_batch_probe.py prepare --root tests/manual_outputs/<新批次>
python tests/manual/flight_experience_batch_probe.py fly --root tests/manual_outputs/<新批次>
python tests/manual/flight_experience_batch_probe.py experience --root tests/manual_outputs/<新批次>
```

默认从 2026-09-23 的 20 例清单选取序号 1、4、5、7、10、12、13、17、18、19。
`--source` 可指定来源清单，`--indices` 接受十个互异的一基序号，必须各含五个 benchmark
案例。完整 case ID 保存在新清单，失败后不换例。

准备阶段保存模型脱敏配置、源码散列、源码副本、Git HEAD、工作区状态和差异。
飞行前检查场景配置与远程可执行文件。每例前后检查源码和模型配置；检测到变化暂停。
模型连接测试的请求与耗时应另存批次目录。不要把 `.env`、私钥写入公开报告。

`fly --resume` 跳过已完成的导航或模型错误结果，仅在清理已确认后允许新目录重试。
未分类目录、清理失败、源码变化会暂停；不能把这些情况算成导航失败。
原始记录和失败尝试始终保留。每例报告位于原案例目录，短路径导出版在相邻 `reviews`。

离线经验按新运行目录逐例处理，保持原模块的 8 图／6 候选默认设置。自主选帧实验失败
后额外运行一次均匀指定选帧诊断，二者分别保存，不能合并完成率或采纳数。
重复运行跳过已落盘实验，包括失败尝试，不会为了获得成功静默重试；处于 running 的中断产物需要先检查，不覆盖重写。
进一步诊断可用单例 `flight_experience_probe.py` 写入另一个新目录。
`experience_summary.json` 关联新飞行、候选数量、采纳数量、失败状态及指定观察范围。

模型审核采纳仅代表实验产物。还需核对原图、错误判断的复述、过度概括及跨场景反例，
不能据此宣称闭环飞行收益。缺少单价时记录 token 与耗时，费用保持未知。

## 本次执行隔离

2026-09-27 第 1 例启动后发现其他工作修改源码，主目录调度将按约定暂停。
批次冻结源码另外部署在 `F:/UAV-Exp-5x2-20260927`，模型配置和数据只通过本地引用使用。
之后从该目录运行同一入口，输出继续写入原批次的绝对路径；两边冻结源码散列一致。
第 1 例在共享目录启动，单独保留其来源限制，后续不覆盖其日志。
