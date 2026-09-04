# 06 记忆系统 MVP 计划

## 1. 当前目标

记忆系统 MVP 只服务一件事：为 `ModelActionPlanner` 生成稳定、可解释、可校验的 `PlannerContext`。

当前不做：

- 视觉 skills。
- 长期向量库。
- 自动地图构建。
- 独立 `PhaseCompletionEvaluator`。
- `ModelRepairPlanner`。

当前已形成的主链路：

```text
StageContextInput
-> ContextModuleRegistry
-> instruction / image / pose / action / map modules
-> ContextFragments
-> ContextComposer
-> PlannerContext
-> ModelActionPlanner
-> ToolContractValidator
-> SafetySupervisor
```

## 2. FlightMemory 角色

`FlightMemory` 是 manifest，不是 prompt。

它保存：

- `active_instruction_id`
- instruction refs。
- observation refs。
- pose refs。
- frame refs。
- action refs。
- map / scene / agent id。
- reset epoch。

最新 pose 由 `recent_pose_refs[-1]` 推导，不维护单独 `latest_pose_ref`。

## 3. 默认 Context Modules

第一版默认五类 module：

| Module | 职责 |
| --- | --- |
| `InstructionModule` | 当前指令、历史指令、当前 goal / phase goal |
| `ImageModule` | 当前与可选 phase-start 图像、可选当前 depth overlay、artifact refs |
| `PoseModule` | 模型侧可选 AGL；绝对 pose 和 delta 仅留在审计 metadata |
| `ActionModule` | phase-relative自身运动、最近动作与字段语义 |
| `MapModule` | 当前 map/scene 相关线索、requires_reconfirmation |

每个 module 必须实现字段自解释能力，输出自己的 `describe_output()`，由 registry 汇总进入 `context_schema.module_content_fields`。

## 4. ContextComposer 角色

`ContextComposer` 固定外壳：

- 收集 fragments。
- 按 priority 排序。
- 应用 budget v1 裁剪。
- 加入 `FLYAGENT.md`。
- 加入 `USER.md`。
- 加入 `allowed_action_constraints`。
- 加入 `observation_policy`。
- 加入 `phase_transition_policy`。
- 输出 `PlannerContext`。

它不做：

- 读取完整 `memory/*.jsonl`。
- 调用 UAV tool。
- 创造新 facts。
- 解释各 module 的业务字段。
- 动作规划或 safety 判断。

## 5. PlannerContext 必备字段

```json
{
  "context_schema": {},
  "active_instruction": "",
  "current_phase": {},
  "semantic_goal": {},
  "fragments": [],
  "disabled_fragments": [],
  "rules": {},
  "user_preferences": {},
  "allowed_action_constraints": {},
  "observation_policy": {},
  "phase_transition_policy": {},
  "safety_constraints": {},
  "metadata": {}
}
```

其中：

- `allowed_action_constraints` 是 `ToolContractValidator` 的主要输入。
- `observation_policy` 告诉模型不要输出 observe step。
- `phase_transition_policy` 告诉模型阶段推进建议如何被 runtime 应用。
- `context_schema` 解释顶层字段和 module 字段含义。

## 6. MapMemory

项目级 map memory 位于：

```text
memory/map/MEMORY.md
memory/map/map_memory.json
```

第一版只做 JSON manifest recall：

- map / scene 匹配时召回。
- 不匹配时不召回。
- 召回线索进入 `inferences` 和 `risks`。
- 不把 map clue 写进 `facts`。
- `requires_reconfirmation=true` 时必须提示风险。

## 7. 动作规划契约

模型只允许输出：

- `uav.hover`
- `uav.turn_relative`
- `uav.move_body`

模型不允许输出：

- `kind="observe"`
- `env.observe`
- 生命周期工具。
- 诊断工具。

`ToolContractValidator` 根据 `PlannerContext.allowed_action_constraints` 执行第一层契约校验；`SafetySupervisor` 继续做最终安全限幅和替换。

## 8. 验收标准

- 一次 run 后同时生成 `state/flight_memory.json` 和 `contexts/latest_context.json`。
- `latest_context.json` 包含五类 fragments。
- `latest_context.json.context_schema.module_content_fields` 包含启用 module 的字段说明。
- `PlannerContext` 不包含完整 `FlightMemory` 或完整 jsonl 历史。
- 模型输出 observe step 会在 contract validation 阶段失败。
- 动作执行后 runtime 自动采集 RGB/depth/pose，并进入下一轮 context。
