> Status: Historical

# FlyAgent 需求文档

## 1. 项目目标

FlyAgent 是一个轻量级 UAV Agent，让用户用自然语言控制无人机在 UE / AirSim 仿真环境中完成导航、观察、目标搜索和悬停类任务。

第一阶段目标是跑通 CLI 闭环：

- 不做复杂前端。
- 不做多机协同。
- 不做独立手机端能力。
- 先让模型基于 `PlannerContext` 生成短期动作计划。
- 动作必须经过 `ToolContractValidator` 和 `SafetySupervisor`。

未来预留：

- 多地图资源。
- 多 UAV。
- AirSim / UnrealCV 等多后端。
- 视觉能力模块化。
- Electron 和飞书入口。

## 2. 用户指令理解

示例：

```text
前面右转会有一个黑色的路灯，右转面向路灯后停下
```

Agent 需要理解的是语义任务，而不是一次性生成完整动作序列。

语义理解结果应包含：

- 目标：黑色路灯。
- 空间线索：先向右转，目标可能在右转后的视野中。
- 终止条件：UAV 面向路灯后悬停。
- 不确定性：右转角度、目标是否存在、目标是否可见、距离是否安全。

## 3. 动作与任务概念

原子动作是底层 UAV 可直接执行的动作，例如：

- `uav.hover`
- `uav.turn_relative`
- `uav.move_body`

语义子任务不是原子动作，例如：

- 靠近目标
- 面向目标
- 寻找目标
- 避开障碍
- 根据文字描述检测目标

语义子任务应由 planner 在运行过程中转化为多个短期原子动作。

## 4. 规划方式

### 4.1 整体语义规划

`TaskParser` 负责回答：

- 任务大概要完成什么。
- 有哪些语义阶段。
- 成功条件是什么。
- 哪些地方不确定。

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
  "phases": [
    {
      "index": 1,
      "total": 4,
      "name": "orient_right",
      "meaning": "根据用户线索向右调整视角"
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
      "meaning": "让 UAV 面向目标"
    },
    {
      "index": 4,
      "total": 4,
      "name": "stop",
      "meaning": "悬停并结束任务"
    }
  ]
}
```

### 4.2 短期滚动执行

短期执行每次只生成未来 1-3 个动作，执行后由 runtime 自动观察并更新上下文。

当前模型输出格式：

```json
{
  "phase": "orient_right",
  "steps": [
    {
      "kind": "tool",
      "tool": "uav.turn_relative",
      "args": {"yaw_deg": 20},
      "reason": "turn right slightly to search the likely area"
    }
  ],
  "phase_transition": {
    "decision": "continue_current_phase",
    "reason": "need the next automatic observation before deciding visibility"
  }
}
```

不再允许模型输出：

```json
{"kind": "observe"}
```

也不允许旧的“观察后再规划”控制字段。观察由 runtime 自动采集，不消耗模型动作 step。

## 5. UAV 控制能力边界

runtime 可以使用生命周期和诊断能力：

- `uav.connect`
- `uav.takeoff`
- `uav.land`
- `uav.get_pose`
- `uav.get_frames`
- `env.observe`

模型 planner 只能使用：

- `uav.hover`
- `uav.turn_relative`
- `uav.move_body`

生产模型通过 provider 原生 `tools/tool_calls` 输出动作。Runtime 根据 `ToolSpec.input_schema` 与显式 runtime policy 校验调用组合；非法调用以 rejected ToolResult 回到同一 transcript。`SafetySupervisor` 随后对 physical batch 做最后限幅和安全替换。

## 6. 感知能力边界

第一阶段基础感知只包含原始观察：

- RGB 图像。
- depth 图像。
- pose。

runtime 根据 `observation_policy` 自动采集这些数据。模型通过
`PlannerContext.fragments`阅读整理后的图像输入、可选 AGL 和相位累计动作历史；
绝对世界 pose 保留在运行审计中，不作为目标相对关系证据发送给模型。

以下能力暂不在第一阶段模块化：

- 文字描述目标检测。
- 目标方向偏差估计。
- 目标距离估计。
- 目标跟踪。
- 场景高级语义理解。

后续如要加入检测或视觉 skills，应作为新的 context module 或 tool 能力接入，而不是让 `FlightMemory` 或 `ContextComposer` 硬编码。

## 7. Agent 状态管理

`MissionState` 是 runtime 执行状态快照，包含：

- 当前 phase。
- connected / airborne。
- 最近 observation。
- 最近 action。
- safety counters。
- phase window 计数。
- yaw scan 累计预算。

`FlightMemory` 是飞行记忆 manifest，保存 refs 和少量索引。planner 不直接读取 `FlightMemory`，只读取 `PlannerContext`。

## 8. 困惑状态与主动提问

当 Agent 对当前状态非常困惑时，不应继续猜测飞行，而应：

1. hover。
2. 保存当前观察和困惑原因。
3. 生成一个简短澄清问题。
4. 等待用户回复。
5. 将回复写入状态后重新规划。

第一阶段先保证安全 hover 和 run record，主动提问可以后续补齐。

## 9. 第一阶段验收标准

- CLI 可以启动一次任务。
- 可以连接或启动一个本地仿真环境。
- 可以读取地图配置。
- 可以执行 UAV 原子动作。
- 可以自动采集 RGB/depth/pose。
- 可以生成 `SemanticPlan`。
- 可以生成 `PlannerContext`。
- 模型可以输出 tool-only `ShortHorizonPlan`。
- `ToolContractValidator` 能拒绝非法工具、非法参数和 observe step。
- `SafetySupervisor` 能做最终限幅和安全替换。
- 每轮动作后自动观察并写入 run 目录。
- 低置信度或安全介入时不会继续推进 phase。

## 10. Semantic Phase 边界

`SemanticPlan.phases` 是开放语义阶段，不是固定动作分发表。模型可以根据用户任务自由命名 phase，但每个 phase 必须包含：

```json
{
  "index": 1,
  "total": 4,
  "name": "turn_to_search_possible_lamp_area",
  "meaning": "根据用户的右转线索，调整视角去寻找可能出现黑色路灯的区域"
}
```

程序依赖 `index` / `total` 理解阶段位置，依赖 `meaning` 理解阶段目标，不应长期依赖 `if phase.name == ...` 的规则表。
