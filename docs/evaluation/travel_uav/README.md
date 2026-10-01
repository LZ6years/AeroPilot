# TravelUAV evaluation

TravelUAV runner 通过 `RuntimeFactory.create_task_runtime()` 获得 TaskRuntime，并将
episode instruction 与 protocol context 传入同一原生工具循环。Benchmark 负责数据
解析、case 生命周期、指标和 review；Runtime 负责工具协议、安全、观测与终态。

远端 readiness 的纯逻辑契约保留在产品测试，真实 SSH/模拟器检查属于 diagnostics
或 manual。任务真值和 reference route 不得进入模型上下文。

产物遵循 [Context V3](../../context_v3.md) 与 [run outputs](../../run_outputs.md)。
