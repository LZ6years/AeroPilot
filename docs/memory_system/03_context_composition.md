# 03 上下文编排

上下文编排的目标是让模型看到“足够做下一步短期动作决策的信息”，而不是看到完整飞行历史。

## 1. 核心原则

```text
普通新指令只更新 active_instruction；
FlightMemory 保留历史指令、pose refs、observation refs、action refs；
reset pose / restart scene / switch map 才刷新飞行连续性；
MapMemory 可以长期保留，但召回后仍需要和当前观察复核。
```

`ContextComposer` 不负责读取完整历史，也不负责创造事实。它调用 `ContextModuleRegistry`，收集各 module 输出的 `ContextFragment`，再合并成稳定的 `PlannerContext`。

## 2. 完整链路

```mermaid
flowchart TD
    U["UserInstruction"] --> TP["TaskParser"]
    TP --> SP["SemanticPlan"]
    SP --> SI["StageContextInput"]

    FM["FlightMemory manifest<br/>recent refs"] --> REG["ContextModuleRegistry"]
    MR["MemoryRecordReader<br/>memory/*.jsonl"] --> REG
    MM["MapMemoryStore"] --> REG
    SI --> REG

    REG --> IM["InstructionModule"]
    REG --> IMG["ImageModule"]
    REG --> PM["PoseModule"]
    REG --> AM["ActionModule"]
    REG --> MAP["MapModule"]

    IM --> FR["ContextFragments"]
    IMG --> FR
    PM --> FR
    AM --> FR
    MAP --> FR

    FR --> CC["ContextComposer"]
    FLY["memory/FLYAGENT.md"] --> CC
    USER["memory/USER.md"] --> CC
    SAFE["Safety constraints"] --> CC
    CC --> PC["PlannerContext"]
    PC --> TR["AgentTranscript StateCheckpoint"]
    TR --> MAPL["ModelActionPlanner native tool calls"]
```

关键边界：

- `FlightMemory` 是 manifest，不生成 `ContextFragment`。
- `ContextModule` 只处理本类别信息。
- `ContextComposer` 只合并、排序、裁剪和加入规则。
- `ModelActionPlanner` 消费完整 active-task transcript 和 provider `tools`。
- provider tool schema 是模型动作契约的唯一来源；debug PlannerContext 可保留旧约束字段，但 provider projection 会移除它。
- 阶段推进由 `runtime.commit_phase_turn` 的 completion assessment 派生。

## 3. ContextFragment 外壳

每个 module 输出统一外壳：

```json
{
  "name": "image",
  "enabled": true,
  "source": "DefaultImageContextModule",
  "content": {},
  "facts": [],
  "inferences": [],
  "goals": [],
  "risks": [],
  "confidence": "medium",
  "artifact_refs": [],
  "budget": {
    "priority": 80,
    "tokens_estimate": 300
  },
  "warnings": [],
  "metadata": {}
}
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `name` | fragment 类型，例如 `instruction`、`image`、`pose`、`action`、`map` |
| `content` | module 自己维护的结构化内容 |
| `facts` | 当前可观察或可校验事实 |
| `inferences` | 推测、候选、地图线索或需要确认的判断 |
| `goals` | 当前 mission / phase 相关目标 |
| `risks` | 不确定性、安全风险或 reconfirmation 要求 |
| `artifact_refs` | 图像、depth、日志等 artifact 引用 |
| `budget` | priority 和 tokens 估计 |

## 4. Module 职责

- `InstructionModule`：当前指令、历史指令、semantic goal、phase goal。
- `ImageModule`：每个 checkpoint 的当前图像、可选 RGB/depth 融合、底部米制色条和 artifact refs。
  融合图标签区分原始 `scene_size`、含色条的 `display_size` 与 `scene_bounds`；
  测距 ROI 按显示图归一化后映射回场景，不能触及色条。
- `PoseModule`：模型侧可选 AGL；绝对 pose 和 delta 留在审计 metadata。
- `ActionModule`：phase-relative 自身运动、上一窗口位移、最近动作和必要字段语义。
- `MapModule`：当前 map/scene 相关线索、recall filter、requires_reconfirmation。

module 不做：

- phase 是否完成。
- 动作规划。
- safety 裁剪。
- UAV tool 调用。
- 跨类别事实合成。

## 5. PlannerContext 输出

`PlannerContext` 是传给 `ModelActionPlanner` 的结构化输入：

```json
{
  "active_instruction": "",
  "current_phase": {},
  "phase_strategy": {
    "stages": [],
    "active_stage": 0,
    "previous_reason": ""
  },
  "phase_neighborhood": {},
  "semantic_goal": {},
  "fragments": [],
  "disabled_fragments": [],
  "context_schema": {},
  "rules": {},
  "user_preferences": {},
  "allowed_action_constraints": {},
  "observation_policy": {},
  "phase_transition_policy": {},
  "phase_execution_summary": {},
  "safety_constraints": {},
  "metadata": {}
}
```

重要边界：

- 不包含完整 `FlightMemory`。
- 不包含完整 `memory/*.jsonl` 原始历史。
- 不让 planner 直接读 run artifacts。
- `context_schema` 解释顶层字段和各 module 输出字段。
- `allowed_action_constraints` 只为 debug/legacy planner 保留，不进入原生 tool-loop 的 provider 状态 JSON。
- `observation_policy` 告诉模型观察由 runtime 自动完成。
- `phase_transition_policy` 告诉模型 action plan 不输出 `phase_transition`，阶段推进在执行后判断。
- `phase_execution_summary` 保留为 runtime/debug 的原始 phase 进展来源，不直接发送给模型。
- planner 在 `current_phase` 后读取顶层 `phase_strategy`；首次进入 phase 时该字段
  省略。它只包含 `stages`、`active_stage` 和标记为历史理由的 `previous_reason`。
  Action Fragment 只包含 `motion_history_summary`：phase-start-relative UAV motion 和
  上一窗口 UAV motion。模型意图、调用和结果的完整历史由 AgentTranscript 提供。
- 每个 Image Module checkpoint 只含一张 observation 图；AgentTranscript 将所有
  observation 按时间投影一次，并在图前文本标记 `HISTORY/CURRENT`、ID 和场景尺寸。
- `last_window_uav_motion` 来自紧邻本轮规划前、已完成动作窗口的前后位姿差。规划前仅刷新
  累计进展时必须保留该快照；只有显式测得的零运动窗口才会把它更新为零。
- 模型先判断上一版 active stage 自身的可观察停止条件，再决定保持、推进或重规划；整个
  phase 的 `done_when` 尚未满足，不会阻止中间 stage 推进。

## 6. facts / inferences / goals / risks

这些字段在流水线中分阶段产生：

```text
TaskParser / SemanticPlan:
  产生 mission goals 和 phase goals

ContextModule:
  按类别整理 facts / inferences / risks

ContextComposer:
  合并、排序、裁剪，不创造新事实

ModelActionPlanner:
  在同一 transcript 上执行 read-tool rounds，并提交 completion assessment 与 physical batch

TaskRuntime:
  构造 StateCheckpoint，执行动作批次并在批次后自动观察一次

TaskRuntime:
  根据 completion assessment 派生 continue / advance / complete
```

当前不单独实现 `PhaseCompletionEvaluator`。阶段推进先复用 `ModelActionPlanner` 的模型调用能力和 `PhaseTransition` schema；后续如果判断复杂，可以把这部分独立成 evaluator。

## 7. Reset 语义

| reset 类型 | FlightMemory | MapMemory | active_instruction | reset_epoch |
| --- | --- | --- | --- | --- |
| 普通新指令 | 保留 | 保留 | 更新 | 不变 |
| soft reset instruction | 保留 | 保留 | 清空或替换 | 不变 |
| reset pose | 刷新飞行连续摘要 | 保留但需 reconfirm | 可保留或重设 | +1 |
| restart scene | 新建 | 当前 map 可保留 | 清空 | 重置 |
| switch map | 新建 | 只召回新 map | 清空 | 重置 |
| clear map memory | 不影响 | 删除或禁用指定记忆 | 不影响 | 不变 |

`reset pose` 后，最新 pose 从 `recent_pose_refs[-1]` 推导，不再维护单独 `latest_pose_ref` 字段。
