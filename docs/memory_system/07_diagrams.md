# 07 记忆系统图

## 1. 主链路

```mermaid
flowchart TD
    U["User Instruction"] --> TP["TaskParser"]
    TP --> SP["SemanticPlan"]
    SP --> SI["StageContextInput"]
    FM["FlightMemory manifest"] --> REG["ContextModuleRegistry"]
    JSONL["memory/*.jsonl"] --> REG
    MAPMEM["MapMemoryStore"] --> REG
    SI --> REG
    REG --> MOD["Context Modules"]
    MOD --> FR["ContextFragments"]
    FR --> CC["ContextComposer"]
    RULES["FLYAGENT.md / USER.md"] --> CC
    SAFETY["Action + Safety Constraints"] --> CC
    CC --> PC["PlannerContext"]
    PC --> MAP["ModelActionPlanner"]
    MAP --> TCV["ToolContractValidator"]
    TCV --> SS["SafetySupervisor"]
```

## 2. FlightMemory 与 PlannerContext

```mermaid
flowchart LR
    FM["flight_memory.json<br/>manifest refs"] --> MOD["Context Modules"]
    RAW["memory/*.jsonl<br/>append-only history"] --> MOD
    ART["artifacts<br/>rgb/depth/log files"] --> MOD
    MOD --> FR["ContextFragments"]
    FR --> PC["PlannerContext"]
    PC --> MODEL["ModelActionPlanner"]

    FM -. "debug artifact only" .-> DEV["Developer inspection"]
    RAW -. "not direct model input" .-> DEV
```

模型看 `PlannerContext`，不直接看完整 memory。

## 3. 自动观察

```mermaid
stateDiagram-v2
    [*] --> BuildContext
    BuildContext --> Plan
    Plan --> ValidateContract
    ValidateContract --> SafetyReview
    SafetyReview --> ExecuteTools
    SafetyReview --> SafeHover: intervention
    ExecuteTools --> AutomaticObservation
    SafeHover --> AutomaticObservation
    AutomaticObservation --> WriteMemory
    WriteMemory --> BuildContext: next window
```

`observe` 不是模型动作。runtime 每个 phase window 后采集 RGB/depth/pose。

## 4. Module 字段自解释

```mermaid
flowchart TD
    REG["ContextModuleRegistry"] --> IM["InstructionModule.describe_output"]
    REG --> IMG["ImageModule.describe_output"]
    REG --> PM["PoseModule.describe_output"]
    REG --> AM["ActionModule.describe_output"]
    REG --> MM["MapModule.describe_output"]
    IM --> SCHEMA["context_schema.module_content_fields"]
    IMG --> SCHEMA
    PM --> SCHEMA
    AM --> SCHEMA
    MM --> SCHEMA
```

谁生产字段，谁解释字段。`ContextComposer` 只解释 `PlannerContext` 外壳。

## 5. Phase Transition

```mermaid
sequenceDiagram
    participant MAP as ModelActionPlanner
    participant TCV as ToolContractValidator
    participant SS as SafetySupervisor
    participant RT as TaskRuntime

    MAP-->>TCV: ShortHorizonPlan + phase_transition
    TCV-->>SS: contract-valid plan
    SS-->>RT: safe plan + interventions
    alt no safety intervention
        RT->>RT: apply continue / advance / complete
    else safety intervention
        RT->>RT: suppress transition and continue current phase
    end
```

当前阶段推进先由 `ModelActionPlanner` 同步输出，未来复杂后可独立 evaluator。

## 6. Tool Contract 与 Safety 分层

```mermaid
flowchart TD
    OUT["Model JSON output"] --> PARSE["ShortHorizonPlan.from_dict"]
    PARSE --> TCV["ToolContractValidator<br/>schema / whitelist / limits from context"]
    TCV --> SS["SafetySupervisor<br/>hard clamp / replacement / intervention"]
    SS --> EXEC["ToolDispatcher execution"]
```

`ToolContractValidator` 拒绝明显不符合 `PlannerContext` 的输出。`SafetySupervisor` 负责最后一道安全门。
