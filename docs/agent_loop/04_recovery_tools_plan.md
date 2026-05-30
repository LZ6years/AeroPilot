# 04 Recovery Tools 计划

本文定义第一版要提供的三个 `RecoveryTool`。这些 tool 只重写模型输出，不执行 UAV 动作。

## 1. `semantic_plan_rewrite`

位置：

```text
src/agent/recovery/tools/semantic_plan_rewrite.py
src/agent/recovery/prompts/semantic_plan_rewrite_system.txt
```

用途：

- 修复 `TaskParser` 输出的 semantic plan。
- 典型错误是 `SemanticPlanValidationError`。

输入：

```text
original_instruction
invalid_semantic_plan
error_type
error_message
forbidden_tokens
previous_errors
```

输出：

```text
完整 semantic plan JSON object
```

验收：

```text
SemanticPlan.from_dict()
validate_semantic_plan()
```

必须强调：

- `takeoff` 仍然是 forbidden token。
- “起飞20m / 升空20m / 起飞到20m”表示高度调整，不是 lifecycle takeoff。
- 不新增用户没要求的视觉搜索阶段。
- 不输出 UAV tool 名称。

## 2. `action_plan_contract_rewrite`

位置：

```text
src/agent/recovery/tools/action_plan_contract_rewrite.py
src/agent/recovery/prompts/action_plan_contract_rewrite_system.txt
```

用途：

- 替代旧 `ModelRepairPlanner`。
- 修复 `ModelActionPlanner` 输出的 action plan contract 错误。

输入：

```text
current_phase
invalid_action_plan
tool_contract_violation
allowed_action_constraints
observation_policy
phase_transition_policy
context_summary
previous_errors
```

输出：

```text
完整 ShortHorizonPlan JSON object
```

验收：

```text
ShortHorizonPlan.from_dict(
  require_phase_transition=False,
  require_action_window=True,
  require_agent_messages=True
)
ToolContractValidator.validate()
```

必须强调：

- 只能输出 `kind="tool"` steps。
- 只能使用 `allowed_action_constraints.planner_tools`。
- 不能输出 `kind="observe"`。
- observation 仍然由 runtime 自动管理。
- 修复后的 phase 必须等于当前 phase。

## 3. `json_object_rewrite`

位置：

```text
src/agent/recovery/tools/json_object_rewrite.py
src/agent/recovery/prompts/json_object_rewrite_system.txt
```

用途：

- 修 provider 输出不是 JSON object。
- 修 markdown/code fence 包裹导致的 parse failure。
- 修明显字段形状错误，例如顶层 list、字符串解释、缺失主对象。

输入：

```text
source
expected_output_contract
raw_output
parse_error
```

输出：

```text
一个 JSON object
```

验收：

- json object 解析成功只是第一层验收。
- 之后仍必须交给来源模块的 typed parser 和 validator。

## 4. Tool 选择策略

候选 tool 由 registry 过滤：

```text
json_parse error -> json_object_rewrite
semantic_validation from task_parser -> semantic_plan_rewrite
tool_contract_validation from action_planner -> action_plan_contract_rewrite
```

selector 仍然接收候选列表并输出选择结果。这样后续同一错误类型有多个工具时，不需要改 orchestrator。

## 5. Tool 输出不能包含什么

所有 recovery tools 都不能输出：

- API key。
- Authorization header。
- 图片 base64。
- ToolDispatcher 调用结果伪造。
- `target_pos` / reference path 等 evaluation-only truth。
- chain-of-thought。

## 6. 和 UAV Tool 的命名区别

文档和代码注释必须持续区分：

```text
RecoveryTool: semantic_plan_rewrite, action_plan_contract_rewrite
UAV tool: uav.hover, uav.move_body, uav.turn_relative
```

RecoveryTool 是模型修复能力；UAV tool 是飞行动作执行能力。
