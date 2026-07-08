# Runtime Control Actions 运行时控制动作

Runtime Control Actions 是 FlyAgent 的运行时控制面。它描述的是影响任务编排、用户交互、语义计划或等待状态的动作，不是直接驱动无人机飞行的动作。

这个专题把系统分成两个平面：

- 飞行动作面：真实影响无人机运动或插件能力调用，例如 `uav.hover`、`uav.turn_relative`、`uav.move_body`、`skill.*`。
- 运行时控制面：影响任务流程，例如 `ask_user`、`semantic_plan_update`，后续也可以扩展为暂停任务、请求确认、整理记忆等。

核心原则是：模型不能凭空使用运行时控制动作。只有 runtime 配置开启、当前运行模式允许、并且当前 planner context 显式暴露在 `runtime_control.available` 中的动作，模型才能在 `control_actions` 中请求。runtime 收到请求后还要再次校验，不能因为模型输出了某个动作就默认执行。

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

- [01 架构](01_architecture.md)：说明 `steps` 与 `control_actions` 双平面、`RuntimeControlExecutor` 的边界，以及 phase window 主流程。
- [02 控制动作配置](02_control_action_config.md)：说明如何显式开启、关闭、展示和覆盖 runtime control action。
- [03 模型契约与上下文](03_model_contract_and_context.md)：说明 capability contract、每轮动态 context、planner policy、validator、event/memory 之间的关系。
- [04 Runtime Control 全阶段实施路线](04_mvp_implementation_plan.md)：说明全阶段实现顺序、每阶段文件边界和验证方式。
- [ask_user](actions/ask_user.md)：说明主动询问用户这个控制动作的含义、触发条件、执行流程、事件，以及用户回复如何进入现有 instruction/conversation memory。
- [semantic_plan_update](actions/semantic_plan_update.md)：说明异步语义计划更新的含义、触发条件、后台流程、版本保护和应用边界。

## 第一版范围

第一版建议只支持两个动作：

- `ask_user`：阻塞式用户澄清。runtime 先让无人机悬停或确认稳定，再向用户提问并暂停当前任务循环。
- `semantic_plan_update`：异步语义计划更新。runtime 在当前窗口动作和自动 observation 之后提交后台更新，并只在后续 phase window 边界按版本校验后应用。

第一版不做：

- 不把运行时控制动作做成 UAV tool。
- 不让 `ToolDispatcher` 执行 `ask_user` 或 `semantic_plan_update`。
- 不新增专门的 `runtime_control.py` / `clarifications.py` context module；用户回复先复用现有 instruction/conversation memory。
- 不让 reusable skill 描述 runtime 私有字段或控制动作 schema。
- 不为了某个评测样例写专用触发规则。

## 总数据流

```text
capability contract 告诉模型“这些控制动作是什么意思、payload 怎么写”
runtime_control context 告诉模型“当前这一轮哪些动作可用、是否有短状态”
planner policy 告诉模型“什么时候应该请求，什么时候不应该请求”
validator 保证“请求不等于执行”
event / memory 记录“执行后发生了什么”
context builder 把结果压缩给下一轮模型
```

模型可见上下文应保持极简。不要在每轮 context 中重复输出动作默认语义、配置细节、环境变量、触发规则或 debug 状态。
