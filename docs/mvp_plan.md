# UAV Agent MVP 计划

## 1. MVP 范围

MVP 只做单机 UAV 视觉导航闭环，不做复杂多机协同，不做完整移动端平台。

优先支持任务：

```text
前面右转会有一个黑色的路灯，右转面向路灯后停下
```

## 2. 阶段划分

### M1：飞控工具封装

目标：

- 将 AirNavigation 的 `UAVController` 包装成 Agent 工具。
- 所有飞控调用通过 Adapter，不直接暴露 AirSim client。

交付：

- `src/adapters/airnavigation_uav.py`
- `src/tools/registry.py`
- `src/tools/dispatcher.py`

工具：

- `uav.connect`
- `uav.takeoff`
- `uav.hover`
- `uav.land`
- `uav.turn_relative`
- `uav.move_body`
- `uav.get_pose`
- `uav.get_frames`

验收：

- 可以连接 UAV。
- 可以起飞、右转、悬停、降落。
- 可以获取当前位姿和 RGB 图像。

### M2：视觉目标检测与面向

目标：

- 接入 AirNavigation 视觉能力。
- 根据目标描述返回 bbox。
- 根据 bbox 中心偏移调整 UAV yaw 和相机 pitch。

交付：

- `src/adapters/airnavigation_perception.py`
- `src/agent/planner.py` 的目标搜索与面向逻辑。

工具：

- `vision.detect`
- `vision.estimate_target_offset`
- `depth.estimate_bbox_distance`

验收：

- 给定“黑色路灯”描述，可以尝试返回目标 bbox。
- 目标不在中心时，UAV 可以慢速转向修正。
- 目标居中后，UAV 悬停。

### M3：任务解析与规则 Planner

目标：

- 将自然语言指令转换成结构化任务。
- 使用规则 Planner 执行步骤。

交付：

- `src/agent/task_parser.py`
- `src/agent/mission_state.py`
- `src/agent/safety.py`
- `src/runtime/main.py`

验收：

- 输入“右转面向黑色路灯后停下”，能生成步骤：
  - 右转
  - 搜索黑色路灯
  - 面向目标
  - 悬停

### M4：手机参数 API 辅助

目标：

- 轻量接入手机端参数 API。
- 不依赖手机完成飞控主闭环。

交付：

- `src/adapters/phone_gateway.py`

工具：

- `phone.device.status`
- `phone.device.permissions`
- `phone.location.get`
- `phone.motion.activity`
- `phone.camera.snap`

验收：

- 能查询手机状态。
- 能查询定位或运动状态。
- 手机 API 异常不影响 UAV 安全悬停。

## 3. 优先级

最高优先级：

- 飞控安全封装。
- 悬停和异常处理。
- RGB/Depth 获取。
- 目标搜索和面向。

中优先级：

- 自然语言结构化解析。
- 工具调用日志。
- 轨迹记录。

低优先级：

- 手机 API。
- 多机协同。
- 复杂记忆系统。
- Web UI。

## 4. 风险

- AirNavigation 中视觉模型和类别可能不包含“路灯”，需要使用多模态视觉模型兜底。
- 深度图格式需要确认，不能只依赖 `DepthVis` 可视化图做精确距离。
- LLM 输出不稳定，第一版应使用规则解析和 schema 校验。
- 飞控动作必须有限速和超时保护。
- `ai_vision.py` 中硬编码 API key 需要移除。

## 5. 推荐第一条 Demo

文件：

```text
src/runtime/lamp_right_turn_demo.py
```

流程：

1. 连接 UAV。
2. 起飞或确认已起飞。
3. 右转 90 度。
4. 获取 RGB + Depth。
5. 检测“黑色路灯”。
6. 若未找到，小角度扫描。
7. 找到后调整 yaw，使目标居中。
8. 深度安全时悬停。
9. 输出任务结果。

