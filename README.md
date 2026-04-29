# UAV Agent

一个面向无人机自主导航任务的轻量级 Agent 项目。目标是让用户可以用自然语言描述空间动作和视觉目标，Agent 结合飞控、图像、深度、轨迹和可选手机端参数 API，完成“理解任务 -> 规划动作 -> 调用工具 -> 观察反馈 -> 安全停止”的闭环。

示例指令：

> 前面右转会有一个黑色的路灯，右转面向路灯后停下

Agent 应将其理解为：

1. 向右转。
2. 搜索黑色路灯。
3. 调整机头和相机，使 UAV 面向目标。
4. 确认目标后悬停停止。

## 文档

- [需求文档](C:/Users/HP/Desktop/UAV-Agent/docs/requirements.md)
- [架构说明](C:/Users/HP/Desktop/UAV-Agent/docs/architecture.md)
- [MVP 计划](C:/Users/HP/Desktop/UAV-Agent/docs/mvp_plan.md)
- [示例任务闭环](C:/Users/HP/Desktop/UAV-Agent/docs/task_example.md)
- [参考项目说明](C:/Users/HP/Desktop/UAV-Agent/docs/reference_notes.md)

## 参考来源

本项目主要参考两个本地项目：

- `C:\Users\HP\Desktop\AirNavigation\AirNavigation`
  - 复用飞控、图像、深度、轨迹、检测/跟踪等能力。
  - 重点参考 `utils/UAV_controller.py`、`utils/UAV_navigater.py`、`ai/ai_vision.py`、`utils/CommandDecorator.py`。

- `F:\openclaw-main\openclaw-main`
  - 轻量参考工具注册、能力清单、统一 dispatcher、手机参数 API 接入模式。
  - 不建议照搬完整 openclaw 架构。

## 第一阶段目标

先做一个能跑通的单机闭环：

- 包装 AirNavigation 的 UAV 控制 API。
- 支持获取 Scene、Depth、Segmentation 等视觉信息。
- 支持简单自然语言任务解析。
- 支持“右转 -> 搜索目标 -> 面向目标 -> 悬停”的任务。
- 所有动作经过安全限制：限速、超时、找不到目标则悬停。

## 建议目录

```text
UAV-Agent/
  README.md
  docs/
    requirements.md
    architecture.md
    mvp_plan.md
    task_example.md
    reference_notes.md
  src/
    agent/
      task_parser.py
      planner.py
      mission_state.py
      safety.py
    tools/
      registry.py
      dispatcher.py
      schemas.py
    adapters/
      airnavigation_uav.py
      airnavigation_perception.py
      phone_gateway.py
    runtime/
      main.py
      config.py
  tests/
    test_task_parser.py
    test_planner_lamp_task.py
    test_tool_dispatcher.py
```

