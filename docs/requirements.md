# FlyAgent 需求文档

## 1. 项目目标

构建一个轻量级 UAV Agent，让用户可以通过自然语言控制无人机在 UE 仿真环境中完成导航、观察、目标搜索和停靠类任务。

第一阶段目标是 CLI 跑通完整闭环，不做复杂前端，不做多机协同，不做独立手机端能力。
但架构需要提前预留：

- 一张地图多个 UAV。
- 多种地图资源来源。
- AirSim / UnrealCV Gym 等多后端环境。
- 多模态感知和决策先由 VLM 统一处理。
- 多模态大模型作为第一阶段临时认知核心。
- 困惑时主动向用户提问。
- 未来 Electron 前端和飞书入口。

## 2. 用户指令理解

示例：

```text
前面右转会有一个黑色的路灯，右转面向路灯后停下
```

Agent 需要理解的是语义任务，而不是一次性生成完整动作序列。

语义理解结果应包含：

- 目标：黑色路灯。
- 空间线索：先向右转，目标可能在右转后的视野中。
- 终止条件：UAV 朝向路灯后悬停。
- 不确定性：目标是否存在、右转角度是否正好 90 度、距离是否安全都需要通过观测确认。

## 3. 动作与任务概念区别

### 3.1 原子动作

原子动作是底层 UAV 可以直接执行的控制动作：

- 前进
- 后退
- 左移
- 右移
- 左转
- 右转
- 上升
- 下降
- 悬停
- 降落
- 相机上仰/下俯

这些动作必须对应具体控制量：

- 速度
- 角速度
- 持续时间
- 目标 yaw
- 目标高度

### 3.2 语义子任务

“靠近”“面向”“寻找”“避开”“跟随”不是原子动作，而是语义子任务。

它们应由 Planner 在运行过程中转化成多个短期原子动作。

例如 `面向路灯`：

- 获取当前图像。
- 判断目标是否在视野内。
- 如果目标偏右，右转一点。
- 如果目标偏左，左转一点。
- 如果目标居中，悬停。

例如 `靠近目标`：

- 判断目标方向。
- 判断深度是否安全。
- 小步前进。
- 重新观察。
- 必要时停止。

因此，文档和代码都不能把“靠近、面向”写成与“前进、右转”同级的动作序列。

## 4. 规划方式

### 4.1 整体语义规划

整体语义规划负责回答：

- 任务大概要完成什么？
- 有哪些阶段？
- 成功条件是什么？
- 哪些地方不确定？

示例：

```json
{
  "intent": "navigate_by_visual_landmark",
  "goal": {
    "target": {
      "description": "黑色路灯",
      "attributes": {
        "color": "black",
        "category": "street_lamp"
      }
    },
    "final_state": "uav_facing_target_and_hovering"
  },
  "semantic_plan": [
    {"phase": "orient_right", "meaning": "根据用户线索，先向右侧观察"},
    {"phase": "search_landmark", "meaning": "寻找黑色路灯"},
    {"phase": "align_to_landmark", "meaning": "让 UAV 朝向路灯"},
    {"phase": "stop", "meaning": "悬停并结束任务"}
  ],
  "uncertainties": [
    "right_turn_angle_may_not_be_exact",
    "target_may_not_be_visible_after_first_turn",
    "target_identity_requires_observation"
  ]
}
```

### 4.2 短期滚动执行

短期执行每次只生成未来几步动作，执行后根据新观测更新。

示例：

```json
{
  "phase": "orient_right",
  "horizon": 3,
  "next_actions": [
    {"type": "turn_yaw", "direction": "right", "angle_deg": 30},
    {"type": "hover", "duration_sec": 0.5},
    {"type": "observe", "modalities": ["rgb", "depth", "pose"]}
  ],
  "replan_after": "observe"
}
```

这样做的原因：

- 用户语言通常不精确。
- 地图和目标位置未知。
- `move_forward` 的距离不能靠语言一次确定。
- 动作执行后环境状态会变化。
- 视觉判断可能失败，需要重新观察或换角度。

## 5. UAV 基础控制能力

第一阶段只暴露底层基础能力，不暴露高级语义能力。

基础控制能力：

- `connect`
- `takeoff`
- `land`
- `hover`
- `move_velocity_body`
- `move_velocity_world`
- `turn_yaw`
- `rotate_camera`
- `get_pose`
- `get_velocity`
- `get_camera_pose`
- `get_frame`
- `get_depth`
- `get_segmentation`
- `start_log`
- `stop_log`

不放入基础控制能力：

- 靠近目标
- 面向目标
- 寻找目标
- 跟踪目标
- 根据文字描述检测目标

这些属于 Planner 或 VLM 层。

## 6. 感知能力边界

第一阶段基础感知只包括原始观测获取：

- RGB 图像
- depth 图像
- segmentation 图像
- pose
- velocity
- camera pose

以下视觉语义能力暂不模块化，第一阶段统一由 VLM 结合上下文处理：

- 根据文字描述检测目标。
- 根据视觉线索估计目标方向偏差。
- 根据深度估计目标距离。
- 对目标进行短时间跟踪。
- 场景描述。
- 目标重识别。

这样做的原因：

- 感知算法会不断替换。
- 不同地图/后端的观测格式不同。
- 现在过早拆分会增加工程复杂度。
- 等完整闭环跑通后，再评估是否需要模块化。

## 7. 第一阶段多模态理解怎么做

第一阶段使用多模态大模型处理图像、深度摘要、任务上下文和历史动作，让它承担视觉理解和策略建议。

VLM 输入：

- 用户原始任务。
- 当前语义阶段。
- 最近 1-5 次动作。
- 当前 RGB 图像。
- depth 统计或安全摘要。
- 当前 pose / yaw / 高度。
- 历史观察结论。

VLM 输出必须结构化：

```json
{
  "scene_summary": "右侧视野中有一根黑色细长杆状物，可能是路灯",
  "target_status": "candidate_visible",
  "confidence": 0.68,
  "suggested_next_actions": [
    {"tool": "uav.turn_relative", "args": {"yaw_deg": 5}},
    {"tool": "uav.hover", "args": {"duration_sec": 0.5}}
  ],
  "ask_user": null
}
```

关键限制：

- VLM 只能提出建议，不能直接控制 UAV。
- 建议动作必须经过 schema 校验。
- 建议动作必须经过 Safety Supervisor 裁剪。
- 低置信度时不能执行靠近、长距离前进等动作。
- VLM 判断要写入 run record，方便复盘。

后续等 UAV 在 UE 中完整跑通后，再考虑是否把稳定的视觉语义能力模块化。

## 8. Agent 状态管理

Agent 状态不应假设已经有高级感知结果。
第一阶段状态应围绕任务、短期动作、原始观测、VLM 判断、工具调用组织。

建议状态：

```json
{
  "run_id": "run_001",
  "user_instruction": "前面右转会有一个黑色的路灯，右转面向路灯后停下",
  "semantic_goal": {},
  "current_phase": "search_landmark",
  "short_horizon_plan": [],
  "uav_state": {
    "pose": null,
    "velocity": null,
    "camera_pose": null
  },
  "observations": {
    "last_rgb_frame_id": null,
    "last_depth_frame_id": null,
    "last_segmentation_frame_id": null,
    "timestamp": null
  },
  "vlm_scene_judgement": null,
  "tool_calls": [],
  "trajectory": [],
  "safety": {
    "last_warning": null,
    "hover_required": false,
    "confusion_reason": null
  }
}
```

当前状态不预留视觉语义子模块字段，避免第一阶段设计膨胀。

## 9. 困惑状态与主动提问

当 Agent 对当前状态十分困惑时，不应该继续猜测飞行，而应该：

1. 立即悬停。
2. 保存当前观测和困惑原因。
3. 生成一个简短澄清问题。
4. 等待用户回复。
5. 将用户回复写入 Mission State 后重新规划。

困惑触发条件：

- VLM 置信度低于阈值。
- 连续多轮没有找到目标。
- 观测结果与用户描述冲突。
- 深度信息显示前方可能不安全。
- Planner 无法判断下一步应该左转、右转、前进还是停止。
- 用户目标描述存在多义性，例如视野中有多个黑色路灯。

提问原则：

- 每次只问一个问题。
- 问题必须能直接帮助下一步行动。
- 提问前必须 hover。
- 问题中尽量给出当前观察到的候选，而不是泛泛地说“不知道怎么办”。

示例：

```text
我现在右转后看到了两个疑似黑色路灯：一个在画面左侧较近，一个在画面中间偏远。
你希望我面向哪一个？
```

初步设计：

```text
Mission State
  -> confusion_detector
  -> hover
  -> clarification_question
  -> wait_user_reply
  -> update semantic constraints
  -> replan
```

## 10. 手机端与飞书

手机端不作为第一阶段重点。未来固定接入飞书即可，先预留接口。

预留方式：

- 飞书作为消息入口和状态通知入口。
- 手机端信息通过飞书或后续服务转发给 Agent。
- 第一阶段不实现手机传感器 API。

预留能力：

- 任务通知。
- 任务确认。
- 异常提醒。
- 人工接管确认。

## 11. 地图资源管理

地图资源优先参考 `gym_unrealcv` 的 JSON 配置方式。

要求：

- 地图资源通过配置加载，不写死在代码中。
- 支持本地 UE `.sh` 或 `.bat` 启动脚本。
- 支持 UnrealCV Gym 风格的 `env_bin_win`、`env_map`、`agents`、`safe_start`、`reset_area`。
- 支持删除物体、增加物体、隐藏物体、改变物体状态等环境操作。
- 使用工厂模式适配不同地图来源。

地图来源初期：

- `gym_unrealcv` / UnrealZoo Gym。

未来适配：

- AirSim UE 项目。
- AirVLN 场景。
- 自定义 UE 地图。
- 其他第三方仿真地图。

## 12. 第一阶段验收标准

- CLI 可以启动一次任务。
- 可以连接或启动一个本地仿真环境。
- 可以读取地图配置。
- 可以执行原子 UAV 控制动作。
- 可以获取 RGB/depth/pose 等原始观测。
- 可以生成整体语义规划。
- 可以滚动生成 1-3 个短期动作。
- 可以调用 VLM 生成结构化场景判断和下一步建议。
- VLM 建议不能绕过 Safety Supervisor。
- 每次短期动作执行后会重新观察并更新计划。
- 不确定、失败或困惑时悬停。
- 困惑达到阈值时可以主动向用户提出澄清问题。
