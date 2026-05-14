# 03 上下文编排

上下文编排的目标是让模型看到“足够做下一步决策的信息”，而不是看到所有历史。

## 1. 核心原则

```text
普通新指令只更新 active_instruction；
FlightMemory 继续保留历史指令、姿态摘要、图像摘要、动作效果摘要；
只有 reset pose / restart scene / switch map 才刷新飞行连续记忆；
MapMemory 继续保留。
```

`ContextComposer` 不实现具体压缩算法。它调用 `ContextModuleRegistry`，由各模块输出 context fragment，再合并为稳定的 `PlannerContext`。

## 2. 完整上下文链路

记忆系统不是从历史数据直接生成 planner context，而是从“当前任务阶段 + 历史数据”共同生成上下文。

主链路：

```text
用户输入
-> TaskParser
-> SemanticPlan
-> 当前 phase / 小阶段
-> 当前阶段数据 + FlightMemory / MapMemory / RunMemory
-> ContextModuleRegistry
-> Instruction/Image/Pose/Action/Map ContextModules
-> ContextFragments
-> FLYAGENT.md / USER.md / SafetyConstraints
-> ContextComposer
-> PlannerContext
-> ModelActionPlanner
```

```mermaid
flowchart TD
    U["UserInstruction<br/>用户输入"] --> TP["TaskParser<br/>指令解析"]
    TP --> SP["SemanticPlan<br/>语义任务计划"]
    SP --> CP["CurrentPhaseSelector<br/>当前 phase / 小阶段"]

    CP --> SCI["StageContextInput<br/>当前阶段数据"]
    SCI --> R["ContextModuleRegistry"]

    FM["FlightMemory<br/>历史指令/姿态/图像/动作"] --> R
    MM["MapMemory<br/>地图线索"] --> R
    RM["RunMemory<br/>运行经验"] --> R

    R --> I["InstructionContextModule"]
    R --> IMG["ImageContextModule"]
    R --> P["PoseContextModule"]
    R --> A["ActionContextModule"]
    R --> MAP["MapContextModule"]

    I --> F["ContextFragments"]
    IMG --> F
    P --> F
    A --> F
    MAP --> F

    F --> C["ContextComposer"]
    FA["FLYAGENT.md<br/>系统规则"] --> C
    US["USER.md<br/>用户偏好"] --> C
    SS["SafetyConstraints<br/>安全约束"] --> C
    C --> PC["PlannerContext<br/>contexts/latest_context.json"]
    PC --> PL["ModelActionPlanner"]
```

关键边界：

- `TaskParser` 和 `SemanticPlan` 决定任务语义边界。
- 当前 phase / 小阶段决定本轮 planner 需要关注什么。
- `FlightMemory`、`MapMemory`、`RunMemory` 是历史数据源，不是最终上下文。
- `ContextModule` 只整理自己负责的数据类型，输出 `ContextFragment`。
- `ContextComposer` 合并 fragments、规则、偏好和安全约束，生成最终 `PlannerContext`。
- `ModelActionPlanner` 只消费 `PlannerContext`，不直接消费 `FlightMemory` 或 `flight_memory.json`。

## 3. 连续指令流程

```mermaid
sequenceDiagram
    participant U as User
    participant TP as TaskParser
    participant PH as CurrentPhase
    participant FM as FlightMemory
    participant R as ContextModuleRegistry
    participant CC as ContextComposer
    participant PL as Planner

    U->>TP: instruction #1
    TP->>PH: SemanticPlan + current phase
    PH->>FM: update active_instruction
    FM->>R: history data + current phase data
    R->>CC: context fragments
    CC->>PL: PlannerContext
    PL->>FM: action/observation result appended

    U->>TP: instruction #2
    TP->>PH: new SemanticPlan + current phase
    FM->>FM: update active_instruction only
    FM->>R: keep flight history + new current phase data
    R->>CC: context fragments
    CC->>PL: PlannerContext from current UAV state
```

## 4. 软件架构

上下文编排使用：

```text
Ports & Adapters + Pipeline + Strategy/Registry
```

含义：

- 前置任务数据由 `TaskParser`、`SemanticPlan` 和当前 phase 提供。
- 固定历史数据源由 `FlightMemory`、`MapMemory`、`RunMemory` 提供。
- `ContextModuleRegistry` 加载默认模块和用户自定义模块。
- `instruction/image/pose/action/map` 各类模块分别生成 context fragment。
- `ContextComposer` 是固定外壳，只负责合并、裁剪和加入规则/安全约束。
- 具体历史压缩算法不写死在 `ContextComposer` 内。

```mermaid
flowchart TD
    CP["SemanticPlan / CurrentPhase"] --> R["ContextModuleRegistry"]
    A["FlightMemory / MapMemory / RunMemory"] --> R
    R --> I["InstructionContextModule"]
    R --> IMG["ImageContextModule"]
    R --> P["PoseContextModule"]
    R --> ACT["ActionContextModule"]
    R --> M["MapContextModule"]
    I --> C["ContextComposer"]
    IMG --> C
    P --> C
    ACT --> C
    M --> C
    F["FLYAGENT.md"] --> C
    U["USER.md"] --> C
    S["SafetyConstraints"] --> C
    C --> O["PlannerContext"]
```

## 5. ContextComposer 输入

`ContextComposer` 第一版输入：

```text
current_phase
semantic_plan / semantic_goal
context fragments from registered modules
FLYAGENT rules
USER preferences
current safety constraints
available planner tools
```

它不直接读取全部 run artifacts，也不直接实现所有压缩逻辑。它依赖各类 `ContextModule` 从 `FlightMemory`、`MapMemory`、`RunMemory` 和当前 phase 输入中提取和压缩信息。

`flight_memory.json` 是运行时快照和调试/审计 artifact，不是 `ModelActionPlanner` 的直接输入。最终给 planner 的结构化输入是 `PlannerContext`，通常落盘为 `contexts/latest_context.json`。

## 6. Context Fragment schema

每个 `ContextModule` 都必须输出统一外壳的 `ContextFragment`。这样五类模块可以共用同一种合并、裁剪和调试方式；用户关闭某个模块时，`PlannerContext` 结构也不会破坏。

统一外壳：

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

字段说明：

| 字段 | 说明 |
| --- | --- |
| `name` | fragment 类型，例如 `instruction`、`image`、`pose`、`action`、`map` |
| `enabled` | 当前模块是否启用 |
| `source` | 生成该 fragment 的模块名 |
| `content` | 模块自己的结构化内容 |
| `facts` | 已观测或可核验事实 |
| `inferences` | 推测、候选、模型判断和地图线索 |
| `goals` | 与当前 mission/phase 有关的目标 |
| `risks` | 不确定性、危险因素和需要重新确认的内容 |
| `confidence` | `low` / `medium` / `high` |
| `artifact_refs` | 图像、depth、日志等 artifact 引用 |
| `budget` | 当前 fragment 的优先级和 token 估计 |
| `warnings` | 模块输出警告 |
| `metadata` | 坐标系、map_id、modalities 等辅助信息 |

`content` 可以因模块类型不同而不同，但外层字段必须保持一致。第一版这是文档约束，后续实现可以再加入 schema validator。

## 7. facts / inferences / goals / risks 的生成阶段

这些字段不是一次性由某个模块生成，而是在上下文流水线中逐步形成：

```text
TaskParser / SemanticPlan:
  生成 mission goals 和 phase goals

InstructionContextModule:
  整理 active_instruction、历史指令和目标摘要

ImageContextModule:
  生成视觉 facts、视觉 inferences、视觉 risks

PoseContextModule:
  生成姿态 facts 和姿态变化 inferences

ActionContextModule:
  生成动作执行 facts 和动作效果 inferences

MapContextModule:
  生成地图线索 inferences 和 reconfirmation risks

ContextComposer:
  只合并、裁剪、排序，不创造新事实

PhaseCompletionEvaluator:
  后续独立使用 facts/inferences/goals/risks 判断 phase 是否完成
```

约束：

- `goals` 主要来自用户指令、`SemanticPlan` 和当前 phase。
- `ImageContextModule` 不能把“疑似目标”写成“已确认目标”。
- `MapMemory` 默认是 inference/risk，不是当前视觉事实。
- `ContextComposer` 不创造新事实，只合并模块输出。
- `ContextModule` 不负责跨模块 phase completion 判断；后续 `PhaseCompletionEvaluator` 独立消费 fragments。

## 8. Planner Context 输出

推荐输出结构：

```json
{
  "active_instruction": "向前飞行5米，然后悬停",
  "current_phase": {},
  "semantic_goal": {},
  "fragments": [],
  "disabled_fragments": [],
  "selected_run_memory": [],
  "rules": {},
  "user_preferences": {},
  "safety_constraints": {}
}
```

`fragments` 中只放 `enabled=true` 的模块输出。用户关闭某个模块时，例如关闭 `image` 或 `map`，`ContextComposer` 将模块名写入 `disabled_fragments`，PlannerContext 仍然保持兼容。

## 9. 信息流

```mermaid
flowchart TD
    U["UserInstruction"] --> TP["TaskParser"]
    TP --> SP["SemanticPlan"]
    SP --> CP["CurrentPhase"]
    CP --> R["ContextModuleRegistry"]

    A["Raw Observation<br/>RGB/depth/pose"] --> FM["FlightMemory"]
    D["Primitive Action Result"] --> FM
    E["Instruction"] --> FM
    G["MapMemory Manifest"] --> MM["MapMemory"]

    FM --> R
    MM --> R
    R --> I["Instruction Module"]
    R --> IMG["Image Module"]
    R --> P["Pose Module"]
    R --> ACT["Action Module"]
    R --> MAP["Map Module"]
    I --> C["ContextComposer"]
    IMG --> C
    P --> C
    ACT --> C
    MAP --> C
    C --> J["PlannerContext"]
```

## 10. Reset 语义

| reset 类型 | FlightMemory | MapMemory | active_instruction | reset_epoch |
| --- | --- | --- | --- | --- |
| 普通新指令 | 保留 | 保留 | 更新 | 不变 |
| soft reset instruction | 保留 | 保留 | 清空或替换 | 不变 |
| reset pose | 刷新飞行连续摘要 | 保留但需 reconfirm | 可保留或重设 | +1 |
| restart scene | 新建 | 当前 map 可保留 | 清空 | 重置 |
| switch map | 新建 | 只召回新 map | 清空 | 重置 |
| clear map memory | 不影响 | 删除或禁用指定记忆 | 不影响 | 不变 |

说明：

- `soft reset instruction` 只清理当前目标，不改变 UAV 物理连续状态。
- `reset pose` 表示 UAV 位置被外部工具或用户显式重设，飞行连续摘要不再可信。
- `restart scene` 和 `switch map` 都需要新建 `FlightMemory`。
- `MapMemory` 在 reset pose 后保留，但所有 pose-local 线索必须重新确认。
## 11. 使用 Manifest + JSONL 后的上下文链路

采用 JSONL store 后，主链路不变，但 `FlightMemory` 进入 module 的内容从“完整历史对象”变成“manifest / index + recent refs”。

```text
用户输入
-> TaskParser
-> SemanticPlan
-> CurrentPhase
-> StageContextInput
-> FlightMemory manifest refs + memory/*.jsonl store
-> ContextModuleRegistry
-> 各 ContextModule 按需读取对应 jsonl / artifact refs
-> ContextFragments
-> ContextComposer
-> PlannerContext
-> ModelActionPlanner
```

各模块推荐读取范围：

| Module | 主要读取 | 默认行为 |
| --- | --- | --- |
| `InstructionContextModule` | `flight_memory.json` 中的 instruction refs + `memory/instructions.jsonl` | 读取最近若干指令并生成 goals / history summary |
| `ImageContextModule` | observation refs + `memory/observations.jsonl` + `memory/frames.jsonl` + artifacts | 读取最近 RGB/depth metadata 和 artifact refs |
| `PoseContextModule` | latest pose ref + `memory/poses.jsonl` | 读取最近 pose window，计算本模块需要的 delta |
| `ActionContextModule` | action refs + `memory/actions.jsonl` | 读取最近 action window，生成 action effect fragment |
| `MapContextModule` | map memory + observation/pose refs | 召回地图线索，并标注需要重新确认 |

重要边界：

- `FlightMemory` 不生成 `ContextFragment`。
- `FlightMemory` 不压缩图像、pose、action。
- `ContextModule` 可以读取 raw refs，但输出必须是统一 `ContextFragment`。
- `ModelActionPlanner` 仍然只消费 `PlannerContext`，不直接读取 `memory/*.jsonl`。

### Manifest 示例

```json
{
  "run_id": "run_001",
  "map_id": "m1",
  "scene_instance_id": "scene_001",
  "agent_id": "uav_1",
  "active_instruction_id": "inst_0002",
  "instruction_history": [
    {"id": "inst_0001", "ref": "memory/instructions.jsonl#1"},
    {"id": "inst_0002", "ref": "memory/instructions.jsonl#2"}
  ],
  "recent_observation_refs": [
    {"id": "obs_0007", "ref": "memory/observations.jsonl#7"}
  ],
  "recent_action_refs": [
    {"id": "act_0012", "ref": "memory/actions.jsonl#12"}
  ],
  "latest_pose_ref": "memory/poses.jsonl#8",
  "reset_epoch": 0
}
```
