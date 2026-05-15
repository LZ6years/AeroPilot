# UAV Agent MVP 计划

## 1. MVP 范围

MVP 只做单机、单 UAV、UE / AirSim 仿真闭环。第一阶段使用 CLI 跑通任务，不做完整 Electron 前端、多机协同或手机端能力。

优先支持任务示例：

```text
前面右转会有一个黑色的路灯，右转面向路灯后停下
向前飞行5米，然后悬停
```

这类任务不应一次性生成很长的固定动作序列，而应拆成：

- 语义目标：用户真正想让 UAV 达成的状态。
- 短期执行窗口：每轮只输出 1-3 个可校验 UAV 原子动作。
- 自动观察：runtime 在每个 phase window 后自动采集 RGB/depth/pose。
- 执行后阶段判断：基于新的 state、历史和本轮执行结果判断是否推进阶段。

## 2. 当前主链路

```mermaid
flowchart TD
    U["用户输入"] --> TP["TaskParser"]
    TP --> SP["SemanticPlan"]
    SP --> MS["MissionState"]
    MS --> CB["ObservationContextBuilder"]
    CB --> PC["PlannerContext"]
    PC --> MAP["ModelActionPlanner"]
    MAP --> TCV["ToolContractValidator"]
    TCV --> SS["SafetySupervisor"]
    SS --> TD["ToolDispatcher"]
    TD --> UAV["UAV primitive tools"]
    UAV --> OBS["Runtime automatic observation<br/>RGB / depth / pose"]
    OBS --> MEM["FlightMemory + memory/*.jsonl"]
    MEM --> CB
    OBS --> J["Post-action phase transition judge"]
    J --> MS
```

模型只输出 `kind="tool"` 的动作 step，不能输出 observe。观察由 runtime 的 `observation_policy` 管理。

## 3. 阶段划分

### M1: UAV 原子飞控封装

目标：

- 从 AirNavigation / AirSim 侧抽象 UAV 基础操作。
- 只封装原子飞控和原始观察，不继承外部项目的整体任务架构。

动作边界：

- 生命周期工具：`uav.connect`、`uav.takeoff`、`uav.land` 只允许 runtime 使用。
- 模型规划工具：`uav.hover`、`uav.turn_relative`、`uav.move_body`。
- 诊断/观察工具：`uav.get_pose`、`uav.get_frames`、`env.observe` 不暴露给 planner。

### M2: 地图资源与场景管理

目标：

- 使用 `MapPackage` 描述地图资源。
- 使用 `SceneInstance` 描述一次运行中的场景实例。
- 由 `SimulationServer` 查找 scene、分配端口、生成 AirSim settings、启动或复用场景。

### M3: 任务解析与语义规划

目标：

- 将用户自然语言指令解析为稳定的 `SemanticPlan`。
- 语义规划只描述 intent、goal、phase 和 uncertainty，不展开成长动作序列。
- phase 使用开放名称，但必须包含 `index`、`total`、`name`、`meaning`。
- movement-only 指令要保留 `direction`、`distance_m`、`final_state=hover_after_movement` 等信息。

### M4: 短期滚动执行器

目标：

- runtime 每轮构建 `PlannerContext`。
- `ModelActionPlanner` 输出短期 tool-only 动作计划。
- `ToolContractValidator` 拒绝不符合 context contract 的输出。
- `SafetySupervisor` 做最终安全审查。
- runtime 执行动作后自动 observation，记录当前 phase 的起点、窗口记录和累计执行事实，再调用模型判断 `PhaseTransition`。

当前事实：

- 规则版 short horizon planner 已移除。
- `run-task` 和交互式 CLI 使用 `ModelActionPlanner`。
- 模型不能输出 `kind="observe"`。
- `phase_transition` 不再由 action plan 输出，而是在执行后判断。
- `phase_execution_summary` 进入 planner 和执行后 judge 输入，用于减少重复转向/重复前进。

### M5: 原始观察与上下文管理

目标：

- 原始感知层只负责 RGB/depth/pose，不伪造高级视觉事实。
- `FlightMemory` 采用 manifest + `memory/*.jsonl` 保存可追溯历史。
- context modules 压缩出 `ContextFragment`。
- `ContextComposer` 输出 `PlannerContext`。

验收：

- 每轮写入 `observations/latest_observation.json`。
- 每轮写入 `contexts/latest_context.json`。
- 执行后阶段判断写入 `contexts/latest_phase_transition_context.json`。
- `mission_state.json` 包含 `phase_start_observations`、`phase_window_records` 和 `phase_execution_summaries`。
- planner 输入不暴露完整 `FlightMemory` 或完整 jsonl 历史。

### M6: 模型动作规划

模型动作输出格式：

```json
{
  "phase": "copy current_phase.name",
  "steps": [
    {
      "kind": "tool",
      "tool": "uav.hover",
      "args": {},
      "reason": "hold position"
    }
  ]
}
```

约束：

- 只允许 `kind="tool"`。
- 不允许 `env.observe` 或 `kind="observe"`。
- `uav.hover` 参数必须是 `{}`。
- `uav.turn_relative` 只允许 `{"yaw_deg": number}`。
- `uav.move_body` 只允许 `direction/duration` 或 `vx/vy/vz/duration` 两种格式。
- limits 来自 `PlannerContext.allowed_action_constraints`，最终安全限幅由 `SafetySupervisor` 执行。

### M6 后续增强: Plan-Execute-Repair

当前链路已经包含 contract repair v1：

```text
ModelActionPlanner
  -> ToolContractValidator
    -> if invalid: ModelRepairPlanner(max 3 attempts)
    -> SafetySupervisor
      -> ToolDispatcher
```

`ModelActionPlanner` 不再静默兼容旧参数。`uav.move_body` 的旧距离参数、`duration_sec`、方向别名等都不属于合法 contract；模型写错时先进入 validator violation，再由 `ModelRepairPlanner` 最多修正 3 次。

后续可以扩展执行失败 repair，但当前先不做。

### M7: 困惑提问与安全中断

目标：

- Agent 不确定继续行动是否正确或安全时，主动停下并询问用户。
- 触发困惑时先 hover。
- 用户回复可更新目标、方向、候选选择或停止指令。

### M8: 运行记录与 CLI

目标：

- CLI 是第一阶段主入口。
- 每次任务生成 run id。
- 每轮保存语义规划、context、模型计划、contract validation、动作结果、观察、安全介入和错误。
- 交互式 CLI 入口统一为 `python -m src.runtime.cli.main chat`。

### M9: Electron 与飞书预留

目标：

- Electron 等 UE CLI 完整跑通后再做。
- 飞书只作为未来消息入口和通知入口。
- 第一阶段不依赖手机传感器 API。

## 4. 优先级

最高优先级：

- UAV 原子飞控封装。
- EnvClient / SimulationServer 本地 C/S。
- RGB/depth/pose 原始观察。
- `PlannerContext` 构建。
- `ModelActionPlanner` 动作规划。
- `ToolContractValidator` contract 校验。
- `SafetySupervisor` 安全裁剪。
- 自动 observation 和执行后 phase transition。

中优先级：

- run store 和轨迹复盘。
- 主动提问。
- 执行失败 repair。
- 真实 UE / AirSim demo 验证。

低优先级：

- 视觉语义能力模块化。
- skills 系统。
- 多 UAV。
- Electron 前端。
- 飞书接入。

## 5. Demo

教学 demo：

```text
demos/demo_model_action_planner_context.py
```

真实引擎闭环 demo：

```text
demos/test_real_engine_basic_instruction_demo.py
```

CLI：

```powershell
$env:PYTHONIOENCODING='utf-8'
F:\conda_env\flyagent\python.exe -m src.runtime.cli.main chat --env-path .env --map-id cityenviron_win64 --save-artifacts
```

## 6. 风险

- map memory 和图像推测不能当作当前视觉事实。
- 模型动作必须先经过 contract 校验，再经过 `SafetySupervisor`。
- 自动观察目前是 phase-window 后采样，不是真实异步定时线程。
- RGB/depth 当前不会以像素形式直接进入 qwen，模型看到的是 metadata、artifact refs 和结构化 observation 信息。
- `PhaseTransition` 是执行后的模型判断，runtime 才能真正修改 `MissionState`。
- 如果阶段完成判断变复杂，再独立抽出 `PhaseCompletionEvaluator`。
