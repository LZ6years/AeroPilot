# 01 问题与边界

本文说明为什么需要 Agent Recovery，以及它不应该解决哪些问题。

## 1. 触发问题

最近一次 CLI 任务中，用户输入：

```text
起飞20m，然后向前飞行50m，最后向右转100度
```

run 结果没有进入飞控执行阶段，而是在 semantic parser 阶段失败：

```text
SemanticPlanValidationError: Semantic plan must not contain primitive action token: takeoff
```

这说明模型输出了不合法 semantic plan：

- 用户自然语言里的“起飞20m”在任务语义上表示“向上调整高度 20m”。
- 模型把它写成了 primitive/lifecycle token `takeoff`。
- `validate_semantic_plan()` 正确拒绝了这个 token。
- 当前 `TaskParser` 没有 Agent Recovery，validator 一失败就直接抛错。

## 2. 根因判断

这不是以下问题：

- 不是 AirSim / UnrealCV / scene 启动失败。
- 不是 ToolDispatcher 没有执行动作。
- 不是 safety 把动作裁剪没了。
- 不是 runtime loop 提前完成任务。
- 不是 CLI 显示覆盖问题。

它发生在 `TaskRuntime.run_task()` 调用 `TaskParser.parse()` 的阶段，任务还没有进入 `run_parsed_task()`。这类错误应该进入 Agent Recovery，而不是让 CLI 用户看到一次不可恢复的任务失败。

## 3. 为什么不能放宽 validator

`takeoff` 不能被加入合法 semantic phase。

原因：

- semantic plan 是高层任务表达，不应该含有底层 tool/action token。
- simulator lifecycle 的 takeoff/land 和用户口语“起飞20m”不是同一层语义。
- 如果允许 `takeoff` 进入 semantic plan，后续 planner 可能把 lifecycle 命令当作普通飞行动作，破坏现有 contract 和 safety 边界。

正确方向是：

```text
非法输出：takeoff
修复后：adjust_altitude_up + translation/up/20m completion criteria
```

## 4. Agent Recovery 要解决什么

Agent Recovery 只处理“模型模块返回内容到 typed object / contract 通过之间”的错误：

- JSON 不是 object。
- JSON 可以解析，但 schema 字段缺失。
- typed object 可以构造，但 validator 拒绝。
- action planner 输出可以解析，但 tool contract 拒绝。

它的成功条件不是“模型说修好了”，而是原模块验收通过：

```text
parse 成功 + validate/contract 成功
```

## 5. RecoveryTool 和 UAV Tool 的边界

`RecoveryTool` 是修复模型输出的工具，例如：

```text
semantic_plan_rewrite
action_plan_contract_rewrite
json_object_rewrite
```

它不是飞行动作工具。它不能调用：

```text
uav.move_body
uav.turn_relative
uav.hover
uav.connect
uav.land
```

飞行动作仍然只能由 `TaskRuntime -> ModelActionPlanner -> SafetySupervisor -> ToolDispatcher` 链路执行。

## 6. Agent Recovery 不解决什么

第一版明确不处理：

- AirSim / UnrealCV 连接失败。
- UE scene 启动失败。
- ToolDispatcher 执行动作失败。
- safety 裁剪导致动作幅度变小。
- phase judge 判断目标未完成。
- 无人机实际飞偏、没看到目标、视觉模型识别错。
- runtime loop guard 触发 stopped/incomplete。

这些问题属于环境、执行、视觉事实或任务闭环，不属于模型输出契约错误。

## 7. 第一版范围

第一版接入两个入口：

```text
TaskParser 模型返回内容 -> SemanticPlan validator
ModelActionPlanner 模型返回内容 -> ShortHorizonPlan parser / ToolContractValidator
```

第一版不保留旧 planner repair 并行分支。`ModelRepairPlanner` 的能力会迁移到 `src/agent/recovery/tools/action_plan_contract_rewrite.py`，然后删除旧文件。

后续才接：

```text
Phase judge JSON repair
Visual caption schema repair
```

这样可以先修当前真实失败点，又不会扩大修改面。
