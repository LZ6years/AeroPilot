> Status: Historical

# FlyAgent Agent Recovery 文档集

本目录记录 FlyAgent 的通用错误恢复层设计。这里的 Agent Recovery 不是飞行动作循环，也不是 Safety；它是 parser、planner、judge 等模型模块在输出不符合 JSON/schema/validator/contract 时，共用的一层“根据错误选择修复工具并重新验收”的机制。

核心结论：

- `TaskRuntime phase loop` 负责真实任务闭环：观察、规划、执行、coupled phase decision、完成或停止。
- `Agent Recovery` 负责模型模块的错误恢复：把错误封装成 `RecoveryRequest`，让 selector 在候选 `RecoveryTool` 中选择修复方式。
- `Safety` 只做硬安全和动作裁剪，不判断任务语义完成，也不替模型修 schema。
- 第一版同时覆盖 `TaskParser` 和 `ModelActionPlanner`，不保留旧 `ModelRepairPlanner` 作为长期并行分支。
- `RecoveryTool` 不是 UAV tool；它只能重写模型输出，不能执行 `uav.move_body`、`uav.turn_relative` 等飞行动作。

## 文档目录

1. [问题与边界](./01_problem_and_boundaries.md)
2. [当前 Loop 盘点](./02_current_loop_inventory.md)
3. [Agent Recovery 架构](./03_agent_recovery_architecture.md)
4. [Recovery Tools 计划](./04_recovery_tools_plan.md)
5. [TaskParser 与 Planner 迁移计划](./05_task_parser_and_planner_migration.md)
6. [日志与 Run Outputs](./06_logging_and_run_outputs.md)
7. [测试与落地路线](./07_tests_and_rollout.md)

## 第一版目标

第一版做一个统一恢复层，覆盖两个真实入口：

```text
TaskParser
  -> SemanticPlan.from_dict / validate_semantic_plan failed
  -> RecoveryRequest
  -> RecoveryOrchestrator
  -> semantic_plan_rewrite or json_object_rewrite
  -> validate_semantic_plan passed

ModelActionPlanner
  -> ShortHorizonPlan.from_dict / ToolContractValidator failed
  -> RecoveryRequest
  -> RecoveryOrchestrator
  -> action_plan_contract_rewrite or json_object_rewrite
  -> ToolContractValidator passed
```

第一版不做：

- 不放宽 `src/agent/semantic/validator.py`。
- 不把 `takeoff` 加入合法 semantic phase。
- 不保留旧 `src/agent/planning/model_repair_planner.py` 作为并行 repair 分支。
- 不用 Agent Recovery 修 AirSim/UnrealCV 连接失败。
- 不用 Agent Recovery 修 ToolDispatcher 执行失败或 safety 裁剪。

## 建议代码落点

```text
src/agent/recovery/
  __init__.py
  orchestrator.py
  selector.py
  registry.py
  schemas.py
  errors.py
  json_utils.py
  tools/
    semantic_plan_rewrite.py
    action_plan_contract_rewrite.py
    json_object_rewrite.py
  prompts/
    recovery_selector_system.txt
    semantic_plan_rewrite_system.txt
    action_plan_contract_rewrite_system.txt
    json_object_rewrite_system.txt
```

## 关键验收

- CLI 输入 `起飞20m，然后向前飞行50m，最后向右转100度` 不再因为 `takeoff` token 直接失败。
- 如果模型第一次输出非法 semantic plan，repair 后应改成合法高度调整语义。
- planner 输出 forbidden tool / observe step / 参数 contract 错误时，走同一个 `RecoveryOrchestrator`。
- 如果 recovery 仍失败，summary 和 debug artifact 能说明 source、selected tool、attempts、最后 validator/contract error。
- 修复成功后不改变 runtime phase loop、Safety 和 ToolDispatcher 的现有边界。
