# UAV Agent 需求文档

## 1. 项目目标

构建一个轻量级 UAV Agent，使用户可以通过自然语言指令控制无人机完成视觉导航任务。

核心目标不是做一个庞大的多智能体平台，而是先做一个可用、可调试、可扩展的单机闭环系统。

## 2. 典型用户指令

示例：

```text
前面右转会有一个黑色的路灯，右转面向路灯后停下
```

Agent 需要理解：

- “前面右转”：先执行右转，默认可解释为相对当前朝向右转 90 度。
- “黑色的路灯”：视觉目标，包含颜色和类别。
- “面向路灯”：调整 UAV yaw 和相机，使目标位于视野中心。
- “停下”：满足终止条件后悬停，而不是继续靠近。

## 3. 功能需求

### 3.1 自然语言任务理解

Agent 需要从用户指令中提取：

- 任务类型：视觉导航、坐标飞行、目标搜索、目标跟踪、悬停/降落。
- 动作序列：前进、后退、左转、右转、上升、下降、靠近、面向、停下。
- 目标描述：颜色、类别、位置关系、方位词。
- 终止条件：看到目标、面向目标、距离安全、超时、用户停止。
- 安全约束：最大速度、最小距离、最大执行时长。

建议结构化输出：

```json
{
  "intent": "navigate_to_visual_target",
  "steps": [
    {"type": "turn", "direction": "right", "angle_deg": 90},
    {"type": "search_target", "target": {"color": "black", "class": "street_lamp"}},
    {"type": "face_target", "target_ref": "street_lamp"},
    {"type": "hover", "condition": "target_centered"}
  ],
  "safety": {
    "max_speed_mps": 1.5,
    "timeout_sec": 60,
    "min_depth_m": 2.0
  }
}
```

### 3.2 UAV 控制能力

第一阶段需要支持：

- 连接 UAV。
- 起飞。
- 降落。
- 悬停。
- 相对转向。
- 机体系速度移动。
- 飞到指定坐标。
- 相机俯仰角调整。
- 获取 UAV 位姿、速度、角速度。
- 记录轨迹。

可复用 AirNavigation 中的 `UAVController`：

- `connect()`
- `take_off()`
- `land()`
- `hover()`
- `move_by_velocity_face_direction(...)`
- `move_by_velocity_with_same_direction(...)`
- `move_by_velocity_new(...)`
- `fly_to_position(...)`
- `rotate_camera(...)`
- `get_body_position()`
- `get_body_eularian_angle()`
- `get_velocity()`
- `start_logging()` / `stop_logging()`

### 3.3 感知能力

第一阶段需要支持：

- 获取当前 RGB 画面。
- 获取深度图。
- 获取语义分割图。
- 根据文字描述检测目标。
- 根据 bbox 估计目标在画面中的方向偏差。
- 根据深度估计目标距离。
- 对目标进行短时间跟踪。

可复用 AirNavigation 中的能力：

- `get_origin_frame()`
- `set_capture_type(...)`
- `get_all_frame()`
- `GenimiUAVVision.analyze_scene_for_target(...)`
- `get_offset_eularian_angle_to_screen_center(...)`
- YOLO/BoT-SORT 相关检测和跟踪逻辑。

### 3.4 Agent 状态管理

Agent 需要维护短期任务状态：

- 当前任务 ID。
- 用户原始指令。
- 结构化任务步骤。
- 当前执行阶段。
- UAV 当前位置、姿态、速度。
- 相机角度。
- 最近观察帧摘要。
- 目标候选列表。
- 目标 bbox、距离、最后出现时间。
- 已执行动作历史。
- 轨迹点。
- 失败次数与重试次数。

### 3.5 手机参数 API

手机端不是第一阶段的主控系统，而是可选辅助上下文。

建议接入能力：

- `phone.device.status`：电量、网络、存储、运行时间。
- `phone.device.permissions`：相机、定位、运动权限。
- `phone.location.get`：手机定位。
- `phone.motion.activity`：运动状态。
- `phone.camera.snap`：手机拍照作为旁路图像。

这些能力可参考 openclaw 的 Android node handler 设计，但第一版只需要轻量 RPC 接口。

## 4. 非功能需求

### 4.1 安全

- 默认最大速度不超过 1.5 m/s。
- 找不到目标时悬停。
- 深度小于安全阈值时停止前进。
- 工具调用超时后悬停。
- 连续视觉结果不一致时重新扫描。
- 总任务超时后悬停并返回失败原因。
- 禁止将 API key 写死在代码中。

### 4.2 可调试

- 每次工具调用需要记录输入、输出、耗时和失败原因。
- 每个任务保存轨迹。
- 关键视觉结果保存 bbox、目标描述、置信度和帧时间。
- Planner 的每一步决策要能回放。

### 4.3 可扩展

- UAV 控制、视觉、手机 API 都通过工具接口接入。
- Planner 不直接依赖 AirSim 原始 client。
- LLM 只输出结构化计划或工具调用，不直接操作底层飞控对象。

## 5. 第一阶段验收标准

完成以下能力即可认为 MVP 初版成立：

- 能连接 AirNavigation/AirSim UAV。
- 能起飞、右转、悬停、降落。
- 能获取 RGB 和深度图。
- 能识别用户描述中的目标。
- 能让目标进入画面中心。
- 能执行“右转后搜索黑色路灯，面向后停下”的任务闭环。
- 找不到目标时不会继续盲飞。

