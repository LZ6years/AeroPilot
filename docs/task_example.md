# 示例任务闭环：右转面向黑色路灯后停下

## 1. 用户指令

```text
前面右转会有一个黑色的路灯，右转面向路灯后停下
```

## 2. 不推荐的一次性 Planner

不推荐直接生成：

```json
{
  "steps": [
    {"type": "turn", "direction": "right", "angle_deg": 90},
    {"type": "search_target", "target": "black street lamp"},
    {"type": "face_target", "target": "black street lamp"},
    {"type": "hover"}
  ]
}
```

原因：

- 用户说的“右转”不一定是精确 90 度。
- 目标不一定转一次就能看到。
- `面向` 不是一次动作，而是多个小角度调整和观测。
- `靠近/前进` 的距离不能提前确定。
- 视觉判断可能失败或目标描述有歧义。

## 3. 推荐的整体语义规划

```json
{
  "intent": "navigate_by_visual_landmark",
  "goal": {
    "target_description": "黑色路灯",
    "final_condition": "UAV 朝向目标并悬停"
  },
  "semantic_phases": [
    {"phase": "orient_right", "goal": "根据用户线索向右侧观察"},
    {"phase": "search_landmark", "goal": "寻找可能是黑色路灯的目标"},
    {"phase": "align_to_landmark", "goal": "通过小角度调整让 UAV 朝向目标"},
    {"phase": "stop", "goal": "悬停结束"}
  ],
  "uncertainties": [
    "目标可能不在首次右转后的视野中",
    "目标类别需要 VLM 确认",
    "右转角度需要根据观测滚动修正"
  ]
}
```

## 4. 短期滚动执行示例

### Round 1：先向右观察

```json
{
  "phase": "orient_right",
  "next_actions": [
    {"type": "turn_yaw", "direction": "right", "angle_deg": 30},
    {"type": "hover", "duration_sec": 0.3},
    {"type": "observe", "modalities": ["rgb", "depth", "pose"]}
  ],
  "replan_after": "observe"
}
```

### Round 2：用 VLM 临时理解观测

第一阶段将当前图像、深度摘要、任务目标和最近动作交给多模态大模型。

```json
{
  "vlm_input": {
    "user_instruction": "前面右转会有一个黑色的路灯，右转面向路灯后停下",
    "phase": "search_landmark",
    "last_actions": ["turn_yaw_right_30", "hover", "observe"],
    "rgb_frame_id": "frame_001",
    "depth_summary": {
      "min_forward_depth_m": 3.2,
      "has_close_obstacle": false
    },
    "pose": {
      "yaw_deg": 32.0,
      "height_m": 3.0
    }
  }
}
```

VLM 输出：

```json
{
  "scene_summary": "画面中间偏右有一个黑色细长杆状物，可能是路灯",
  "target_status": "candidate_visible",
  "confidence": 0.70,
  "suggested_next_actions": [
    {"tool": "uav.turn_relative", "args": {"yaw_deg": 5}},
    {"tool": "uav.hover", "args": {"duration_sec": 0.5}},
    {"tool": "uav.get_frames", "args": {"modalities": ["rgb", "depth", "pose"]}}
  ],
  "ask_user": null
}
```

注意：

- VLM 输出只是建议。
- `uav.turn_relative` 等动作必须经过 Safety Supervisor。
- 如果 VLM 置信度不足，不允许直接前进或靠近。

### Round 3：目标候选明确，执行小角度对齐

```json
{
  "phase": "align_to_landmark",
  "next_actions": [
    {"type": "turn_yaw", "direction": "right", "angle_deg": 5},
    {"type": "hover", "duration_sec": 0.3},
    {"type": "observe", "modalities": ["rgb", "depth", "pose"]}
  ],
  "replan_after": "observe"
}
```

### Round 4：困惑时主动提问

如果 VLM 返回多个候选：

```json
{
  "scene_summary": "画面左侧和中间偏远处都有黑色杆状物，二者都可能是路灯",
  "target_status": "ambiguous_candidates",
  "confidence": 0.42,
  "suggested_next_actions": [
    {"tool": "uav.hover", "args": {"duration_sec": 1.0}}
  ],
  "ask_user": "我看到了两个疑似黑色路灯：左侧较近一个，中间偏远一个。你希望我面向哪一个？"
}
```

系统行为：

1. 强制 hover。
2. 将困惑原因写入 run record。
3. 向用户提问。
4. 等待用户回复。
5. 根据用户回复更新 Mission State。
6. 重新规划。

### Round 5：目标已居中，停止

```json
{
  "phase": "stop",
  "next_actions": [
    {"type": "hover", "duration_sec": 1.0}
  ],
  "finish_reason": "target_aligned"
}
```

## 5. 后续能力拆分

当前 MVP 不拆分视觉语义模块。
先让 VLM 统一完成图像理解、目标候选判断、下一步建议和困惑提问。

等 UAV 在 UE 中完整跑通后，再评估是否需要把某些稳定能力拆出来。

## 6. 安全策略

- 每次只执行短动作。
- 每次动作后重新观察。
- 目标不确定时不前进。
- VLM 低置信度时不执行靠近。
- 无观测时悬停。
- 困惑时主动提问。
- 用户说“停下”解释为 `hover`，不是 `land`。
