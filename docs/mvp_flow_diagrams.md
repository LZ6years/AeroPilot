# MVP 规划与流程图

这份文档只描述第一阶段 MVP 应该怎么跑通，以及每张图对应要完成什么、达到什么效果。

## 图 1：MVP 总闭环

```mermaid
flowchart TD
    U["用户输入任务"] --> P["Task Parser<br/>解析语义目标"]
    P --> S["Semantic Planner<br/>生成整体语义规划"]
    S --> M["Mission State<br/>保存阶段/观测/不确定性"]
    M --> H["Short Horizon Planner<br/>生成 1-3 个短期动作"]
    H --> G["Safety Guard<br/>限速/限距/超时/困惑检测"]
    G -->|安全| A["UAV Primitive Tools<br/>执行原子飞控"]
    G -->|困惑或风险高| Q["Clarification<br/>主动向用户提问"]
    Q --> M
    A --> O["Raw Observation<br/>RGB/Depth/Pose"]
    O --> V["VLM Decision Fallback<br/>多模态大模型临时理解"]
    V --> M
    M -->|未完成| H
    M -->|完成| R["Hover + Run Record<br/>悬停并保存记录"]
```

需要完成：

- CLI 能接收自然语言任务。
- 能生成整体语义规划，而不是长动作序列。
- 每轮只执行短期动作。
- 每轮动作后重新观察。
- 调用多模态大模型理解 RGB/depth/pose 和任务上下文。
- 困惑或风险高时先 hover，再向用户提问。

达到效果：

- 可以跑通“右转观察、寻找黑色路灯、面向后悬停”的最小闭环。
- 视觉能力先不拆模块，统一用 VLM 完成临时决策。
- 不会因为一次错误规划直接长距离乱飞。

## 图 2：MVP 阶段交付图

```mermaid
flowchart LR
    M1["M1<br/>UAV 原子飞控"] --> M2["M2<br/>AirVLN Scene 管理"]
    M2 --> M3["M3<br/>任务解析与语义规划"]
    M3 --> M4["M4<br/>短期滚动执行器"]
    M4 --> M5["M5<br/>原始观测与状态管理"]
    M5 --> M6["M6<br/>VLM 临时决策"]
    M6 --> M7["M7<br/>困惑提问与安全中断"]
    M7 --> M8["M8<br/>运行记录与 CLI Demo"]
    M8 --> M9["M9<br/>Electron / 飞书预留"]
```

需要完成：

- `M1`：封装起飞、悬停、移动、转向、降落、获取位姿和图像。
- `M2`：明确 EnvClient / SimulationServer 边界，用 AirVLN 风格管理 MapPackage、SceneInstance、端口和场景进程。
- `M3`：把用户输入解析成语义目标、目标描述、完成条件和不确定性。
- `M4`：每次只生成 1-3 个原子动作。
- `M5`：保存 RGB/depth/pose、动作日志、状态和轨迹。
- `M6`：用多模态大模型判断“看到了什么、下一步该怎么保守行动”。
- `M7`：困惑时进入 hover，并向用户提出一个简短问题。
- `M8`：CLI 跑通首个任务并保存 run 记录。
- `M9`：预留 Electron 和飞书接口。

达到效果：

- 每个阶段都有独立验收点。
- 先跑通可用闭环，暂不拆分视觉语义模块。

## 图 3：VLM 临时决策

```mermaid
flowchart TD
    O["Raw Observation<br/>RGB/Depth/Pose"] --> C["Context Builder<br/>拼接任务/阶段/最近动作/观测"]
    C --> V["Multimodal LLM<br/>理解场景并给出建议"]
    V --> J{"输出类型"}
    J -->|目标明确| N["建议短期动作<br/>如小角度右转/悬停观察"]
    J -->|目标不明确| E["建议继续观察<br/>换角度/拉远/悬停"]
    J -->|不安全或困惑| Q["请求用户确认"]
    N --> S["Safety Guard<br/>动作裁剪与校验"]
    E --> S
    Q --> H["Hover<br/>等待回复"]
    S --> A["执行原子动作"]
```

需要完成：

- 定义 VLM 输入：用户任务、语义阶段、最近动作、RGB 图、深度摘要、位姿。
- 定义 VLM 输出 schema：`confidence`、`scene_summary`、`target_status`、`suggested_next_actions`、`ask_user`。
- VLM 只能建议动作，不能直接绕过 Safety Guard 控制 UAV。
- 当 VLM 信心低、目标冲突、深度不安全时，必须返回 `ask_user` 或 `hover`。

达到效果：

- 第一阶段通过大模型完成视觉语义判断。
- 视觉语义能力模块化等当前闭环完成后再考虑。

## 图 4：困惑时主动提问状态机

```mermaid
stateDiagram-v2
    [*] --> Running
    Running --> Observing: 执行动作后观察
    Observing --> Planning: 观测可解释
    Observing --> Confused: 目标/方向/安全不确定
    Planning --> Running: 生成短期动作
    Confused --> Hovering: 立即悬停
    Hovering --> AskUser: 生成澄清问题
    AskUser --> WaitingReply: 等待用户回复
    WaitingReply --> Planning: 用户给出澄清
    WaitingReply --> SafeStop: 超时或用户要求停止
    SafeStop --> [*]
```

需要完成：

- 定义困惑触发条件。
- 进入困惑状态后先 hover，再提问。
- 每次只问一个对下一步最有帮助的问题。
- 用户回复进入 `Mission State`，重新规划。
- 如果用户长时间不回复，保持 hover 或安全降落，由配置决定。

达到效果：

- Agent 不会在看不懂场景时继续猜。
- 用户可以用自然语言纠正任务，例如“不是左边那个，是更远处黑色细杆的路灯”。

## 图 5：首个 Demo 任务流程

```mermaid
sequenceDiagram
    participant User as 用户
    participant CLI as CLI Runtime
    participant Agent as Agent Planner
    participant VLM as 多模态大模型
    participant UAV as UAV Tools
    participant Env as AirVLN-style Scene / UE

    User->>CLI: 前面右转会有一个黑色的路灯，右转面向路灯后停下
    CLI->>Agent: 创建 run + 解析语义目标
    Agent->>UAV: takeoff / hover
    Agent->>UAV: turn_relative(小角度或分段右转)
    UAV->>Env: 执行动作
    Env->>Agent: RGB + Depth + Pose
    Agent->>VLM: 任务上下文 + 图像/深度摘要
    VLM-->>Agent: 是否看到目标 + 下一步建议 + 置信度
    alt 置信度足够
        Agent->>UAV: 小角度调整 yaw / hover / observe
        Env->>Agent: 新观测
        Agent->>UAV: hover
        CLI-->>User: 任务完成，输出 run 记录
    else 置信度低或存在风险
        Agent->>UAV: hover
        CLI-->>User: 提问确认目标或方向
        User->>CLI: 回复澄清
        CLI->>Agent: 更新 Mission State
    end
```

需要完成：

- CLI demo：`lamp_right_turn_demo`。
- 分段右转，不直接假设 90 度一步到位。
- 每轮调用 VLM 或 mock VLM。
- VLM 输出必须经过 schema 校验和 Safety Guard。
- 低置信度时能主动提问。

达到效果：

- 第一阶段就能看到一个可交互、可记录、可安全中断的 UAV Agent 原型。
- 后续如果要拆分视觉能力，可以把 VLM 判断从“主决策参考”逐步降级为“兜底解释器”。
