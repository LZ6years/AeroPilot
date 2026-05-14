# MVP 规划与流程图

本文只描述第一阶段 MVP 的可运行闭环，以及每张图对应的验收点。

## 图 1: MVP 总闭环

```mermaid
flowchart TD
    U["用户输入任务"] --> TP["TaskParser<br/>解析语义目标"]
    TP --> SP["SemanticPlan<br/>生成阶段化语义计划"]
    SP --> MS["MissionState<br/>保存阶段和运行状态"]
    MS --> CB["ObservationContextBuilder<br/>构建 PlannerContext"]
    CB --> MAP["ModelActionPlanner<br/>生成 1-3 个 tool steps"]
    MAP --> TCV["ToolContractValidator<br/>校验工具和参数契约"]
    TCV --> SS["SafetySupervisor<br/>限幅/拦截/替换"]
    SS -->|安全| UAV["UAV Primitive Tools<br/>执行原子飞控"]
    SS -->|风险| H["Safe Hover<br/>记录 safety intervention"]
    UAV --> AO["Automatic Observation<br/>RGB / Depth / Pose"]
    H --> AO
    AO --> MEM["FlightMemory + JSONL<br/>保存历史和 refs"]
    MEM --> CB
    MAP --> PT["phase_transition<br/>continue / advance / complete"]
    PT --> RT["TaskRuntime<br/>应用阶段推进"]
```

需要完成：

- CLI 能接收自然语言任务。
- 能生成语义阶段，而不是长动作序列。
- 每轮模型只规划短期 UAV tool steps。
- 模型输出经过 `ToolContractValidator` 和 `SafetySupervisor`。
- 动作后 runtime 自动采集 RGB/depth/pose。
- runtime 根据 `phase_transition` 推进、继续或完成任务。

## 图 2: MVP 阶段交付图

```mermaid
flowchart LR
    M1["M1<br/>UAV 原子飞控"] --> M2["M2<br/>地图和场景管理"]
    M2 --> M3["M3<br/>任务解析和语义规划"]
    M3 --> M4["M4<br/>短期滚动执行"]
    M4 --> M5["M5<br/>观察和记忆上下文"]
    M5 --> M6["M6<br/>模型动作规划"]
    M6 --> M6B["M6+<br/>ToolContractValidator"]
    M6B --> M7["M7<br/>困惑提问和安全中断"]
    M7 --> M8["M8<br/>运行记录和 CLI Demo"]
    M8 --> M9["M9<br/>Electron / 飞书预留"]
```

验收重点：

- `M1-M5` 跑通环境、观察和 context。
- `M6` 让模型基于 `PlannerContext` 生成动作和 `phase_transition`。
- `M6+` 把动作契约从 planner 内部硬编码拆成独立 validator。
- `M7` 以后再补主动提问和 repair。

## 图 3: 模型规划和契约校验

```mermaid
sequenceDiagram
    participant RT as TaskRuntime
    participant CC as ContextComposer
    participant MAP as ModelActionPlanner
    participant TCV as ToolContractValidator
    participant SS as SafetySupervisor
    participant TD as ToolDispatcher

    RT->>CC: fragments + rules + safety constraints
    CC-->>RT: PlannerContext
    RT->>MAP: plan_next(state, context)
    MAP->>MAP: call model with JSON context
    MAP->>MAP: parse ShortHorizonPlan
    MAP->>TCV: validate against allowed_action_constraints
    alt contract valid
        TCV-->>MAP: pass
        MAP-->>RT: validated plan
        RT->>SS: review plan
        SS-->>RT: safe plan
        RT->>TD: execute safe tool steps
    else contract invalid
        TCV-->>MAP: ToolContractViolation
        MAP-->>RT: fail before execution
    end
```

模型可用动作：

- `uav.hover`
- `uav.turn_relative`
- `uav.move_body`

模型不可用动作：

- `kind="observe"`
- `env.observe`
- 生命周期工具和诊断工具。

## 图 4: 自动观察策略

```mermaid
stateDiagram-v2
    [*] --> BuildContext
    BuildContext --> PlanActions
    PlanActions --> ValidateContract
    ValidateContract --> SafetyReview
    SafetyReview --> ExecuteTools
    SafetyReview --> SafeHover: intervention
    ExecuteTools --> AutomaticObservation
    SafeHover --> AutomaticObservation
    AutomaticObservation --> WriteMemory
    WriteMemory --> ApplyPhaseTransition
    ApplyPhaseTransition --> BuildContext: continue_current_phase
    ApplyPhaseTransition --> BuildContext: advance_to_next_phase
    ApplyPhaseTransition --> [*]: complete_task
```

说明：

- 观察不是模型动作。
- 即使模型只输出 hover，runtime 也会在 phase window 后采集一次 RGB/depth/pose。
- 下一轮 context 使用最新 observation 和 pose refs。

## 图 5: 首个 Demo 任务流程

```mermaid
sequenceDiagram
    participant User as 用户
    participant CLI as CLI Runtime
    participant Agent as Agent Runtime
    participant Model as ModelActionPlanner
    participant Guard as Validator + Safety
    participant UAV as UAV Tools
    participant Env as AirVLN-style Scene / UE

    User->>CLI: 前面右转会有一个黑色的路灯，右转面向路灯后停下
    CLI->>Agent: 创建 run + 解析语义阶段
    Agent->>Env: preflight connect / takeoff / hover
    Env-->>Agent: initial RGB / depth / pose
    Agent->>Model: PlannerContext JSON
    Model-->>Agent: tool steps + phase_transition
    Agent->>Guard: contract validation + safety review
    Guard-->>Agent: safe plan
    Agent->>UAV: turn_relative / move_body / hover
    UAV->>Env: 执行动作
    Env-->>Agent: automatic RGB / depth / pose
    Agent->>Agent: 写入 memory 和 run artifacts
    Agent->>Agent: 应用 phase_transition
    CLI-->>User: 输出 run 目录和结果摘要
```

验收效果：

- 能看到模型实际收到的 `latest_context.json`。
- 能看到模型原始输出和校验后的 `validated_plan.json`。
- 能看到 `safety_decision.json`。
- 能看到动作后自动生成的新 observation 进入下一轮 context。
