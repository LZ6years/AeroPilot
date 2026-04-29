# FlyAgent

FlyAgent 是一个面向 UAV/UE 仿真环境的轻量级 Agent 项目。目标是让用户用自然语言描述任务，系统通过“整体语义规划 + 短期滚动执行 + 环境反馈更新”的方式控制无人机完成目标。

典型指令：

```text
前面右转会有一个黑色的路灯，右转面向路灯后停下
```

这类指令不能一次性翻译成一长串确定动作。更合理的方式是：

1. 先生成整体语义目标：右转后寻找黑色路灯，并最终让 UAV 朝向它后悬停。
2. 每次只预测未来几步短期动作。
3. 执行后读取观测，再更新计划。
4. 用户语言不确定、目标未确认或环境状态变化时，保守悬停或重新观察。

## 文档

- [需求文档](C:/Users/HP/Desktop/UAV-Agent/docs/requirements.md)
- [架构说明](C:/Users/HP/Desktop/UAV-Agent/docs/architecture.md)
- [C/S 架构职责划分](C:/Users/HP/Desktop/UAV-Agent/docs/cs_architecture.md)
- [MVP 计划](C:/Users/HP/Desktop/UAV-Agent/docs/mvp_plan.md)
- [示例任务闭环](C:/Users/HP/Desktop/UAV-Agent/docs/task_example.md)
- [MVP 流程图](C:/Users/HP/Desktop/UAV-Agent/docs/mvp_flow_diagrams.md)
- [仿真环境与地图资源策略](C:/Users/HP/Desktop/UAV-Agent/docs/simulation_environment_strategy.md)
- [前端预留策略](C:/Users/HP/Desktop/UAV-Agent/docs/frontend_strategy.md)
- [目录结构说明](C:/Users/HP/Desktop/UAV-Agent/docs/directory_structure.md)
- [参考项目说明](C:/Users/HP/Desktop/UAV-Agent/docs/reference_notes.md)

## 参考来源分工

- `C:\Users\HP\Desktop\AirNavigation\AirNavigation`
  - 只抽取 UAV 基础操作，例如连接、起飞、降落、悬停、速度控制、相机控制、图像/深度获取、位姿/轨迹记录。
  - 不参考它的整体架构。

- [AirVLN/AirVLN](https://github.com/AirVLN/AirVLN)
  - 参考 C/S 架构、环境 server、场景进程管理、多 scene / 多机器 / 多 UAV 扩展方式。
  - 第一阶段仍只做单 UAV。

- [buaa-colalab/UAV-Flow](https://github.com/buaa-colalab/UAV-Flow)
  - 后续参考 `UAV-Flow-Eval` 的 UnrealZoo Gym / `gym_unrealcv` 对象级环境操作。
  - 不作为第一阶段地图资源主模型。

- `F:\openclaw-main\wojiazaiyugang.QiClaw\QiClaw`
  - 参考 Agent runtime、planner、run store、records、Electron 前端预留。
  - Electron 前端等 CLI 在 UE 中完整跑通后再做。

## 第一阶段目标

- CLI 方式跑通完整任务。
- 支持 AirVLN 风格 scene 资源管理和本地 UE/AirSim 环境。
- 基础 UAV 控制只暴露原子能力。
- 多模态感知与决策先统一由 VLM 处理，不在第一阶段拆分视觉语义模块。
- 地图资源先参考 AirVLN 的 scene 管理方式：`MapPackage + SceneInstance + SimulationServer`。

## 建议目录

```text
UAV-Agent/                         # 项目根目录
  README.md                        # 项目入口说明
  docs/                            # 需求、架构、策略文档
  configs/                         # 地图、环境、模型、本地运行配置
  src/                             # 主要源码
  ui/                              # 未来 Electron 前端预留
  tests/                           # 单元测试和集成测试
```
