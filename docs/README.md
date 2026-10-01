# FlyAgent 文档索引

## Current

- [架构](architecture.md)：生产依赖方向、Runtime 服务链和状态所有权。
- [Context V3](context_v3.md)：Envelope、感知 Checkpoint、当前任务状态、Projection、MediaRef。
- [插件系统](plugin_system.md)：发现、贡献与 Factory 注入边界。
- [目录结构](directory_structure.md)：当前源码目录及职责。
- [任务闭环示例](task_example.md)：phase-free 原生工具循环。
- [运行产物](run_outputs.md)：V3 artifact 与诊断输出。
- [测试](testing.md)：产品回归、diagnostics、manual 和真实实验。
- [Runtime controls](runtime_control/README.md)
- [Simulator control](simulator_control/README.md)
- [Custom modules](custom_modules/README.md)

## Experiment and research

- `docs/experiments/`：可重复实验说明。
- `docs/research/`：论文材料和研究叙述，不作为生产 API。
- `docs/evaluation/`：各 benchmark 的当前运行入口及外部协议。

## Historical

`docs/archive/` 保存退役架构、已完成计划和旧实验记录。归档文件均标记
`Status: Historical`，其中出现的类型、工具或路径不能用于判断当前实现。
