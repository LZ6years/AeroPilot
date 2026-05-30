# 05 TaskParser 与 Planner 迁移计划

本文说明如何把 `TaskParser` 和 `ModelActionPlanner` 接入同一套 `RecoveryOrchestrator`，并移除旧 `ModelRepairPlanner` 分支。

## 1. TaskParser 接入

目标文件：

```text
src/agent/semantic/task_parser.py
```

目标流程：

```text
TaskParser.parse(instruction)
  -> model call 得到 raw semantic JSON
  -> SemanticPlan.from_dict()
  -> validate_semantic_plan()
  -> if failed:
       RecoveryRequest(source="task_parser", stage="semantic_validation")
       RecoveryOrchestrator.run()
       semantic_plan_rewrite / json_object_rewrite
       validate_semantic_plan()
  -> TaskParseResult
```

新增状态：

```text
last_recovery_attempts
last_recovery_result
```

`rule_based` mode 不触发 recovery。rule-based 输出失败应视为本地开发错误。

## 2. ModelActionPlanner 接入

目标文件：

```text
src/agent/planning/model_action_planner.py
```

目标流程：

```text
ModelActionPlanner.plan_next()
  -> model call 得到 raw action plan JSON
  -> ShortHorizonPlan.from_dict()
  -> ToolContractValidator.validate()
  -> if failed:
       RecoveryRequest(source="action_planner", stage="tool_contract_validation")
       RecoveryOrchestrator.run()
       action_plan_contract_rewrite / json_object_rewrite
       ShortHorizonPlan.from_dict()
       ToolContractValidator.validate()
  -> ShortHorizonPlan
```

删除旧私有循环：

```text
_repair_contract_violation()
_record_repair_attempt()
_repair_error()
```

这些职责迁移到 `RecoveryOrchestrator` 和 `RecoveryAttempt`。

## 3. 删除旧 planner repair 文件

删除：

```text
src/agent/planning/model_repair_planner.py
```

更新：

```text
src/agent/planning/__init__.py
src/runtime/runners/task_runtime.py
tests/test_model_action_planner.py
tests/test_task_runtime.py
```

移除旧符号：

```text
ModelRepairPlanner
ModelRepairPlannerError
RepairContext
RepairAttemptRecord
```

替代符号：

```text
RecoveryOrchestrator
RecoveryRequest
RecoveryAttempt
RecoveryExhaustedError
```

## 4. Runtime 接入

目标文件：

```text
src/runtime/runners/task_runtime.py
```

runtime 不参与 recovery 选择，但需要处理统一错误：

```text
except RecoveryExhaustedError
```

记录统一事件：

```text
agent_recovery_requested
agent_recovery_tool_selected
agent_recovery_attempt_failed
agent_recovery_succeeded
agent_recovery_exhausted
```

旧事件：

```text
model_action_contract_repair_requested
model_action_contract_repair_failed
model_action_contract_repaired
```

应迁移或停止新增。

## 5. Prompt 迁移

旧 prompt：

```text
src/agent/planning/prompts/model_repair_system.txt
```

迁移到：

```text
src/agent/recovery/prompts/action_plan_contract_rewrite_system.txt
```

语义 parser 新增：

```text
src/agent/recovery/prompts/semantic_plan_rewrite_system.txt
```

selector 新增：

```text
src/agent/recovery/prompts/recovery_selector_system.txt
```

## 6. Import 迁移

当前 `_loads_json_object()` 在：

```text
src/agent/semantic/task_parser.py
```

迁移到：

```text
src/agent/recovery/json_utils.py
```

需要更新：

```text
src/agent/semantic/task_parser.py
src/agent/planning/model_action_planner.py
tests/manual/visual_instruction_eval.py
```

`task_parser._loads_json_object` 可保留短期 wrapper，但文档和新代码都使用 `src.agent.recovery.json_utils.loads_json_object`。

## 7. 兼容性决策

第一版不保留旧 `ModelRepairPlanner` 并行路线。

原因：

- 用户明确要求重新设计，不保留之前 model planner repair 部分。
- 并行两套 repair 会让 runtime event、错误类型、测试口径都变复杂。
- 统一到 recovery 后，parser 和 planner 能共享 attempts、selector、logging 和 exhausted error。

如需降低代码迁移风险，可以在实现中分小 PR/commit，但目标状态仍是删除旧 repair 分支。
