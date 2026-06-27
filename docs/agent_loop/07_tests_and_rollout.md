# 07 测试与落地路线

本文给出 Agent Recovery 的测试安排和分阶段实现顺序。

## 1. 新增测试文件

建议新增：

```text
tests/test_recovery_orchestrator.py
tests/test_recovery_tools.py
```

建议更新：

```text
tests/test_task_parser.py
tests/test_model_action_planner.py
tests/test_task_runtime.py
tests/test_semantic_plan.py
```

建议删除或重写：

```text
tests/test_model_repair_planner.py
```

## 2. `test_recovery_orchestrator.py`

覆盖 orchestrator：

- registry 根据 `source/stage/error_type` 给出候选 tools。
- selector 选择正确 tool。
- selected tool 修复后通过调用方 validator。
- tool 输出仍非法时重试并最终 `RecoveryExhaustedError`。
- repeated invalid output 提前失败。
- attempts 摘要中不包含 `Authorization`、`api_key`、`data:image`、`base64`。

这些测试不需要真实模型，可以用 fake selector 和 fake tools。

## 3. `test_recovery_tools.py`

覆盖 v1 tools：

- `semantic_plan_rewrite` 修复包含 `takeoff` 的 semantic plan。
- `action_plan_contract_rewrite` 修复 forbidden tool。
- `action_plan_contract_rewrite` 修复 `kind="observe"` step。
- `json_object_rewrite` 修 markdown 包裹或非 object 输出。

真实模型调用不进入默认 pytest。需要真实 provider 的验证时，放到 `tests/manual/`。

## 4. 更新 parser 测试

`tests/test_task_parser.py` 增加：

- fake model 首次返回包含 `takeoff` 的非法 semantic plan。
- fake recovery 返回合法 `adjust_altitude_up` semantic plan。
- 最终 `TaskParseResult.plan` 通过 validator。
- 指令 `起飞20m，然后向前飞行50m，最后向右转100度` 形成 3 个 phase。
- recovery exhausted 时抛 `RecoveryExhaustedError`。

## 5. 更新 planner 测试

`tests/test_model_action_planner.py` 更新：

- 旧 `ModelRepairPlannerError` 断言改为 `RecoveryExhaustedError`。
- stub repair planner 改为 fake `RecoveryOrchestrator` / fake `RecoveryTool`。
- 保留现有 contract repair 行为验收：第一/第三次修复成功、重复非法输出失败、缺 `action_window` 或 `agent_messages` 失败。

## 6. 更新 runtime 测试

`tests/test_task_runtime.py` 更新：

- 旧 `model_action_contract_repair_*` event 改为统一 `agent_recovery_*` event。
- parser recovery 成功时，runtime 能继续进入 `run_parsed_task()`。
- planner recovery 成功时，runtime 能继续 safety 和 ToolDispatcher。
- recovery exhausted 时，runtime 写 `instruction_failed`，并 best-effort hover。
- recovery event 不包含完整 image/base64/context。

## 7. 实施顺序

建议按以下顺序实现：

1. 新增 `src/agent/recovery/json_utils.py`，迁移 JSON object 提取逻辑。
2. 新增 `schemas.py`、`errors.py`、`registry.py`。
3. 新增 `orchestrator.py` 和 fake-tool 单元测试。
4. 新增 `selector.py` 和 selector prompt。
5. 新增三个 v1 recovery tools 和 prompts。
6. 修改 `TaskParser` 接入 recovery。
7. 修改 `ModelActionPlanner` 接入 recovery。
8. 删除 `src/agent/planning/model_repair_planner.py`，更新 imports/tests。
9. 修改 `TaskRuntime` 捕获 `RecoveryExhaustedError` 和记录统一 events。
10. 更新 `docs/run_outputs.md`。
11. 跑定向测试和全量测试。

## 8. 验收标准

实现完成后必须满足：

- CLI 输入 `起飞20m，然后向前飞行50m，最后向右转100度` 不再因为 `takeoff` token 直接失败。
- 如果模型第一次输出 `takeoff`，recovery 后第一阶段应成为高度调整语义。
- validator 仍然能拒绝最终残留的 primitive token。
- planner 输出 forbidden tool / observe step 时，走 `RecoveryOrchestrator` 而不是旧 `ModelRepairPlanner`。
- recovery exhausted 时，错误中包含 source、stage、selected tools、attempts、最后 validator/contract error。
- parser/planner recovery 成功后，`TaskRuntime phase loop`、Safety、ToolDispatcher 行为不变。
- debug logs 不包含 API key、Authorization header、图片 base64。

## 9. 后续路线

第一版稳定后再接：

- coupled phase decision JSON/schema recovery。
- visual caption schema recovery。
- 更细粒度 deterministic fallback。
- selector 评估日志和不同 provider 的选择质量对比。
