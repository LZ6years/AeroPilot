# Runtime Control Actions 运行时控制动作

Runtime Control Actions 是 FlyAgent 的运行时控制面。它描述的是影响任务编排、用户交互、语义计划或等待状态的动作，不是直接驱动无人机飞行的动作。

这个专题最初按 JSON `steps/control_actions` 双平面设计。当前生产模型路径已经迁移到统一原生 tool calling：

- physical tools：真实影响无人机运动，例如 `uav.hover`、`uav.turn_relative`、`uav.move_body`。
- read-only tools：插件检测、VLM、OCR 等立即返回证据的能力。
- runtime-control tools：影响任务流程，例如 `runtime.ask_user`、`runtime.semantic_plan_update`、`runtime.commit_phase_turn`。

核心原则是：模型只能调用本轮 provider `tools` 中实际出现的动作。Runtime 配置与 `ToolSpec.runtime` 共同决定暴露集合，收到调用后仍会再次校验。动态 pending/status 可以进入 StateCheckpoint，但 availability 和 schema 不在 context 中重复。

## 文档结构

```text
docs/runtime_control/
  README.md
  01_architecture.md
  02_control_action_config.md
  03_model_contract_and_context.md
  04_mvp_implementation_plan.md
  actions/
    ask_user.md
    semantic_plan_update.md
```

目录职责：

- [原生 Tool-Call 执行计划](../agent_tool_loop_execution_plan.md)：当前生产架构与迁移状态。
- [01 架构](01_architecture.md)：旧 JSON 双平面设计记录，供迁移追溯。
- [02 控制动作配置](02_control_action_config.md)：说明如何显式开启、关闭、展示和覆盖 runtime control action。
- [03 模型契约与上下文](03_model_contract_and_context.md)：说明 capability contract、每轮动态 context、planner policy、validator、event/memory 之间的关系。
- [04 Runtime Control 全阶段实施路线](04_mvp_implementation_plan.md)：说明全阶段实现顺序、每阶段文件边界和验证方式。
- [ask_user](actions/ask_user.md)：说明主动询问用户这个控制动作的含义、触发条件、执行流程、事件，以及用户回复如何进入现有 instruction/conversation memory。
- [semantic_plan_update](actions/semantic_plan_update.md)：说明异步语义计划更新的含义、触发条件、后台流程、版本保护和应用边界。

## 第一版范围

当前配置可选择暴露两个控制动作：

- `ask_user`：阻塞式用户澄清。runtime 先让无人机悬停或确认稳定，再向用户提问并暂停当前任务循环。
- `semantic_plan_update`：异步语义计划更新。runtime 在当前窗口动作和自动 observation 之后提交后台更新，并只在后续 phase window 边界按版本校验后应用。

第一版不做：

- 不让运行时控制动作进入 UAV primitive dispatcher。
- 不让 `ToolDispatcher` 执行 `ask_user` 或 `semantic_plan_update`。
- 不新增专门的 `runtime_control.py` / `clarifications.py` context module；用户回复先复用现有 instruction/conversation memory。
- 不让 reusable skill 描述 runtime 私有字段或控制动作 schema。
- 不为了某个评测样例写专用触发规则。

## 总数据流

```text
provider tools 告诉模型“这些控制动作是什么意思、payload 怎么写、当前是否可用”
StateCheckpoint 只告诉模型“当前是否有 pending 等短状态”
planner policy 告诉模型“什么时候应该请求，什么时候不应该请求”
validator 保证“请求不等于执行”
event / memory 记录“执行后发生了什么”
AgentTranscript 把精确调用和结果交给下一轮模型
```

模型可见上下文应保持极简。不要在每轮 context 中重复输出动作默认语义、配置细节、环境变量、触发规则或 debug 状态。
