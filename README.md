# FlyAgent

FlyAgent 是面向 UAV 仿真环境的原生工具循环 Agent。模型按时间顺序读取
Context V3 感知 checkpoint、当前任务状态、历史工具结果与图像引用，并选择只读工具、1–3 个物理
工具或 Runtime 控制工具；Runtime 负责安全审查、执行、自动观测和持久化。

## 开始使用

本仓库的 Python 命令必须使用项目指定的 Conda 环境：

```powershell
& 'F:\conda_env\flyagent\python.exe' -m src.cli
& 'F:\conda_env\flyagent\python.exe' -m pytest -q
```

安装可编辑 CLI 后也可以运行 `flyagent` 或 `flyagent start`。正式任务、交互任务
和 smoke 检查都通过 `RuntimeFactory` 组合依赖；`TaskRuntime.run_task()` 是稳定的
任务执行门面。

## 当前文档

- [文档索引](docs/README.md)
- [架构与职责边界](docs/architecture.md)
- [Context V3](docs/context_v3.md)
- [插件系统](docs/plugin_system.md)
- [测试与诊断](docs/testing.md)
- [运行产物](docs/run_outputs.md)
- [目录结构](docs/directory_structure.md)

历史设计与完成的迁移计划位于 `docs/archive/`，不代表当前 API。
