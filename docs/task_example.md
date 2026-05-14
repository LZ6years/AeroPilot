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
- “面向目标”不是一个原子动作，而是多轮小角度调整、自动观察和重新规划。
- 视觉判断可能失败或出现多个候选。

## 3. 推荐的语义规划

```json
{
  "intent": "navigate_by_visual_landmark",
  "goal": {
    "target_description": "黑色路灯",
    "final_condition": "UAV 面向目标并悬停"
  },
  "semantic_phases": [
    {
      "index": 1,
      "total": 4,
      "name": "orient_right",
      "meaning": "根据用户线索向右侧观察"
    },
    {
      "index": 2,
      "total": 4,
      "name": "search_landmark",
      "meaning": "寻找可能是黑色路灯的目标"
    },
    {
      "index": 3,
      "total": 4,
      "name": "align_to_landmark",
      "meaning": "通过小角度调整让 UAV 面向目标"
    },
    {
      "index": 4,
      "total": 4,
      "name": "stop",
      "meaning": "悬停结束"
    }
  ]
}
```

## 4. Round 1: 生成 PlannerContext

runtime 把当前 phase、记忆 refs、最近 observation、动作历史、规则、用户偏好和安全约束整理成：

```text
runs/<run_id>/contexts/latest_context.json
```

模型接收的是这个 JSON context，不直接读取：

- `flight_memory.json`
- 完整 `memory/*.jsonl`
- 原始 RGB/depth 文件
- UAV tool registry 内部对象

`latest_context.json` 会包含：

- `context_schema`
- `active_instruction`
- `current_phase`
- `semantic_goal`
- `fragments`
- `allowed_action_constraints`
- `observation_policy`
- `phase_transition_policy`
- `safety_constraints`

## 5. Round 1: 模型动作计划

模型只输出 tool-only action plan：

```json
{
  "phase": "orient_right",
  "steps": [
    {
      "kind": "tool",
      "tool": "uav.turn_relative",
      "args": {"yaw_deg": 20},
      "reason": "turn right slightly to inspect the likely target area"
    }
  ],
  "phase_transition": {
    "decision": "continue_current_phase",
    "reason": "need automatic RGB/depth/pose observation after this turn"
  }
}
```

模型不能输出：

```json
{
  "kind": "observe",
  "modalities": ["rgb", "depth", "pose"]
}
```

观察由 runtime 自动执行。

## 6. Round 1: Contract Validation

`ToolContractValidator` 校验：

- `phase` 是否等于当前 phase。
- step 数是否为 1-3。
- step 是否只使用 `kind="tool"`。
- tool 是否在 `planner_tools` 中。
- tool 是否不在 `forbidden_tools` 中。
- `uav.turn_relative` 是否只有 `yaw_deg`。
- `yaw_deg` 是否在 context limits 内。

如果模型输出旧格式，例如：

```json
{
  "kind": "tool",
  "tool": "uav.move_body",
  "args": {"x_m": 4.0, "duration_sec": 2.0}
}
```

validator 会在进入 safety 前拒绝。

## 7. Round 1: Safety Review 和自动观察

通过 contract 后，`SafetySupervisor` 做最后安全处理：

- 单步 yaw 限幅。
- move_body 速度和 duration 限幅。
- yaw scan 累计预算检查。
- 安全介入时替换为 hover。

动作执行后，runtime 自动采集：

```json
{
  "mode": "automatic",
  "modalities": ["rgb", "depth", "pose"]
}
```

这次 observation 写入 `memory/*.jsonl` 和 run artifacts，并进入下一轮 context。

## 8. Round 2: 搜索目标

下一轮 context 会包含右转后的图像 metadata、pose delta 和刚才动作结果。模型可能输出：

```json
{
  "phase": "search_landmark",
  "steps": [
    {
      "kind": "tool",
      "tool": "uav.turn_relative",
      "args": {"yaw_deg": 10},
      "reason": "continue a small scan because the target is not confirmed"
    }
  ],
  "phase_transition": {
    "decision": "continue_current_phase",
    "reason": "target still needs confirmation"
  }
}
```

如果最新 image fragment 表示目标已经可疑地出现在画面中央，模型可以改为小幅 hover 或进入下一 phase。

## 9. 困惑时的行为

如果上下文显示多个候选目标或证据冲突，模型应保守输出 hover 或小角度安全扫描。后续主动提问模块可以把困惑原因转成用户问题：

```text
我看到两个疑似黑色路灯：左侧较近一个，中间偏远一个。你希望我面向哪一个？
```

当前阶段先保证：

- 不直接靠近低置信度目标。
- 不绕过 validator 和 safety。
- safety intervention 会压制 `phase_transition` 推进建议。

## 10. 任务完成

当上下文支持“目标已居中且 UAV 已面向目标”时，模型输出：

```json
{
  "phase": "stop",
  "steps": [
    {
      "kind": "tool",
      "tool": "uav.hover",
      "args": {},
      "reason": "final state is satisfied"
    }
  ],
  "phase_transition": {
    "decision": "complete_task",
    "reason": "UAV is facing the target and holding position"
  }
}
```

runtime 执行 hover、自动观察、写入 run record，然后标记任务完成。
