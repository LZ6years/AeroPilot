> Status: Historical

# 03 模型契约与上下文（旧 JSON 设计记录）

> 当前生产模型路径通过 provider `tools/tool_calls` 暴露和调用控制动作，不再依赖 `control_actions` 或 context availability 列表。现行契约见 [原生 Tool-Call 执行计划](../agent_tool_loop_execution_plan.md)。

本文件定义 runtime control action 如何被模型理解和请求。核心目标是补上三件事之间的边界：

- capability contract：告诉模型控制动作是什么意思、payload 怎么写、runtime 默认怎么处理。
- runtime control context：告诉模型当前这一轮哪些控制动作可用、是否有短状态。
- planner policy：告诉模型什么时候应该请求，什么时候不应该请求。

这三层不能混在一个大 context 里。稳定语义放在契约和 prompt 中，动态状态放在每轮 context 中，执行权限由 runtime validator 最终决定。

## 1. 完整链路

```text
capability contract 告诉模型“这些控制动作是什么意思、payload 怎么写”
runtime_control context 告诉模型“当前这一轮哪些动作可用、是否有短状态”
planner policy 告诉模型“什么时候应该请求，什么时候不应该请求”
validator 保证“请求不等于执行”
event / memory 记录“执行后发生了什么”
context builder 把结果压缩给下一轮模型
```

如果只给 `available`，模型可能不知道 action 的含义。如果把全部含义、规则、执行细节都放进每轮 context，又会导致上下文膨胀。因此第一版采用：

- 稳定 contract 放在 planner system prompt 或 runtime control registry 生成的短契约中。
- 每轮 context 只暴露可用动作名和必要短状态。
- 每个 action 的详细触发和执行流程写在 `actions/*.md`。

## 2. Capability Contract

Capability contract 是稳定接口说明，不是每轮动态状态。它应该让模型知道：

- action 的名称。
- action 的用途。
- action 应该出现在 `control_actions` 中。
- payload 的最小结构。
- runtime 接受后默认会怎么处理。
- action 不应该被用来绕过什么边界。

第一版两个动作的短契约可以写入 planner prompt 或由 registry 注入 prompt：

```text
ask_user:
  用途：当继续飞行需要用户澄清才能安全推进时，请求用户回答一个短问题。
  输出位置：control_actions。
  payload：question, expected_answer_type。
  runtime 行为：提问前悬停，任务进入 waiting_for_user，收到回复后恢复。

semantic_plan_update:
  用途：当前 semantic phase 合约明显阻塞真实进展时，请求异步修复当前 phase 语义。
  输出位置：control_actions。
  payload：scope, update_goal。
  runtime 行为：不阻塞飞行，后台生成 patch，后续 phase/window 边界校验后应用。
```

这里可以说明 `ask_user` 是阻塞式、`semantic_plan_update` 是异步，但这些是 contract 内容，不要在 planner 每次输出的 JSON 里重复写成字段。

## 3. Runtime Control Context

Runtime control context 是每轮动态信息，只回答“现在能不能用”和“是否有短状态”。

最小示例：

```json
{
  "runtime_control": {
    "available": ["ask_user", "semantic_plan_update"]
  }
}
```

存在后台语义更新任务时：

```json
{
  "runtime_control": {
    "available": ["ask_user", "semantic_plan_update"],
    "status": {
      "semantic_plan_update": "pending"
    }
  }
}
```

用户回复后的澄清可以放在现有 instruction/conversation memory 的摘要区域，例如：

```text
Recent user clarification:
用户确认目标是左侧目标。后续目标选择应优先左侧目标。
```

也可以在已有结构化上下文中压缩为：

```json
{
  "active_clarifications": [
    {
      "answer_summary": "用户确认优先靠近左侧目标",
      "effect": "后续目标选择应优先左侧目标"
    }
  ]
}
```

第一版不要求新增专门的 clarification context module。只要现有 context builder 能在 instruction / dialogue summary / user memory 区域放入短澄清摘要即可。

## 4. 不放进每轮 Context 的内容

不要在 model-facing context 中放：

- `blocking`、`async`、`use_when`、`avoid_when` 这类动作说明字段。
- 完整配置、环境变量名、session override 内部结构。
- runtime control registry 的完整内容。
- pending job 的 run id、版本号、patch 内容、prompt 或完整日志。
- event log 全量。
- debug-only 状态。
- 已关闭动作的占位字段。

这些信息分别属于 prompt contract、runtime config、debug artifact 或 event log，不属于每轮模型决策上下文。

## 5. Planner Policy

planner policy 告诉模型什么时候应该请求 control action。它应该写在 planner prompt 和对应 action 文档中。

通用原则：

- 普通不确定性优先短动作、悬停或自动 observation。
- 只有继续执行可能选错目标、产生安全风险、改变用户任务含义，或陷入无意义循环时，才考虑 control action。
- `ask_user` 用于用户信息能显著降低风险的情况。
- `semantic_plan_update` 用于当前 semantic phase 合约和真实进展反复冲突的情况。
- control action 不能用于绕过 safety、phase decision、tool contract 或 runtime loop guard。

每个 action 的具体触发条件放在自己的文件：

- `actions/ask_user.md`
- `actions/semantic_plan_update.md`

## 6. Validator

模型请求不等于执行。runtime validator 至少检查：

- action 是否出现在 `runtime_control.available` 中。
- 当前 session 是否允许该 action，例如 batch 模式默认拒绝 `ask_user`。
- payload 是否合法。
- 是否已有 pending `semantic_plan_update`。
- 请求理由是否和当前 phase/context 相关。
- 是否试图绕过 safety、phase decision 或工具契约。

不合法时，runtime 应拒绝或进入 repair，而不是静默执行。

## 7. Event / Memory / Context Builder

事件用于记录发生了什么，供 UI、debug、回放和任务恢复使用。事件不是 planner 的完整输入。

用户回复类事件的处理链路：

```text
user_clarification_requested
  -> 用户回复
  -> user_clarification_received
  -> 写入现有 instruction/conversation memory，标记为 clarification
  -> context builder 在下一轮输出短澄清摘要
```

语义更新类事件的处理链路：

```text
semantic_plan_update_requested
  -> semantic_plan_update_submitted
  -> semantic_plan_update_ready
  -> phase/window 边界校验
  -> semantic_plan_update_applied 或 discarded
  -> 下一轮 context 只暴露必要短状态或更新后的 semantic plan
```

原则是：event/memory 可以详细，model-facing context 必须压缩。

## 8. 未来扩展

后续新增 runtime control action 时，应新增：

- registry contract 条目；
- action 自己的实现文件；
- `docs/runtime_control/actions/<action>.md`；
- 配置开关；
- validator 测试；
- prompt 边界测试。

不要把新 action 的长说明全部塞进每轮 context，也不要把 action 行为写进 reusable skill。
