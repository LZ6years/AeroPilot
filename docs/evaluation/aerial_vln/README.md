# AerialVLN evaluation

`evaluation.benchmarks.aerial_vln.runner.AerialVlnBenchmarkRunner` 负责 case 加载、
场景配置、TaskRuntime 调用和结果冻结。Runtime 由 `RuntimeFactory` 创建，不允许
Benchmark 自行组装 PluginRegistry、ToolRegistry 或 ContextBuilder。

每个 case 使用 phase-free 原生工具循环。评测真值只在 evaluator/metric 边界读取，
不得进入模型上下文、EnvClient 或安全策略。离线产品 gate 位于
`tests/test_aerial_vln_case_gate.py`；真实引擎、网络与性能检查位于
`tests/diagnostics/` 或 `tests/manual/`。

产物遵循 [Context V2](../../context_v2.md) 与 [run outputs](../../run_outputs.md)。
