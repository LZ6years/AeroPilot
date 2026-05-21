# 开发文件放置约定

本文记录本项目中 `tests/`、`tests/manual/`、`demos/` 和 `tools/` 的边界，避免后续把底层诊断脚本和用户可体验流程混在一起。

## tests

`tests/` 用于验证系统行为是否正确，包括：

- 单元测试、集成测试、回归测试。
- schema、contract、context、state、memory、evaluator、run output 等稳定性检查。
- 编码检查、中文 UTF-8 检查、日志和 artifact 结构检查。
- 可以由 pytest 自动执行的真实测试或模拟测试。

如果脚本主要目的是“检查某个底层行为是否正常”，即使它需要真实 UE 或 AirSim，也优先放在 `tests/`，而不是 `demos/`。

## tests/manual

`tests/manual/` 用于手动触发的开发诊断脚本，适合放：

- 真实 UE / AirSim 的手动探针。
- ToolDispatcher、adapter、stabilization、坐标投影、速度等待等底层实验。
- 不适合默认 pytest 自动跑、但需要保存在 repo 中反复使用的调试脚本。

这类脚本应当提供清楚的 CLI 参数、输出 run 目录，并在文件头说明它验证什么、如何看结果。文件名不要以 `test_` 开头，除非它确实应该被 pytest 收集。

示例：

```text
tests/manual/tool_dispatcher_forward_stability_probe.py
tests/manual/body_frame_projection_calibration_probe.py
```

它们用于直接调用 ToolDispatcher 或真实 UE/AirSim 检查 UAV 前进、平移、转向后的速度、姿态稳定性和坐标投影；这是底层诊断，不是用户 demo。

## demos

`demos/` 用于用户可以直接体验的流程。一个 demo 应该展示一个完整功能或工作流，例如：

- 从自然语言指令到任务解析、上下文生成、模型规划、安全审查、执行和自动观察的一轮闭环。
- 让用户看到某个功能如何使用、输出文件在哪里、应该看哪些结果。
- 可以帮助开发者手动调试，但它的第一目标仍然是“可体验的流程”，不是底层探针。

不应放入 `demos/` 的内容：

- 只验证 ToolDispatcher 单个动作的脚本。
- 只验证 adapter、坐标变换、速度阈值、日志格式、评估器解析的脚本。
- 只为某个 bug 做低层复现的临时诊断脚本。

这些内容应放到 `tests/` 或 `tests/manual/`。

当前保留的用户向 demo 命名统一为“主题在前，类型在后”：

```text
demos/memory_system_demo.py
demos/model_action_planner_context_demo.py
demos/real_engine_instruction_demo.py
```

## tools

`tools/` 只用于可复用的工程工具，例如批量迁移、格式检查、报告生成、离线转换等。它不是 demo，也不是测试目录。

当前真实 run 离线评估入口是：

```text
tools/real_run_evaluator.py
```

## 选择规则

判断一个新脚本放哪里时，优先问：

- 用户是否可以通过它体验一个完整流程？如果是，放 `demos/`。
- 它是否主要验证底层行为、稳定性、日志、坐标、评估、contract？如果是，放 `tests/` 或 `tests/manual/`。
- 它是否是可复用的项目维护工具？如果是，放 `tools/`。

所有中文文档、报告和 run 分析文件都必须用 UTF-8 写入，并在提交前检查是否出现问号替换符或 mojibake。
