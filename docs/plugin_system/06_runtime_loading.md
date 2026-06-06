# 06 Runtime Loading

飞行时插件系统分两步工作：启动时发现和注册，每轮 context build 时选择和注入。

## 启动加载

```text
ObservationContextBuilder.__init__
  -> PluginRegistry.from_config(configs/plugins.json)
  -> PluginDiscovery scans plugin roots
  -> SkillCatalog indexes enabled skills
  -> SkillCapabilityCatalog indexes enabled callable skills
  -> ContextModuleRegistry receives plugin_registry
```

发现阶段不调用模型，不执行 handler。

## 配置形态

```json
{
  "plugins": {
    "roots": [
      "plugins/knowledge",
      "plugins/research"
    ],
    "enabled": [
      "knowledge.uav_flight"
    ],
    "disabled": []
  },
  "context_modules": {
    "instruction": "default",
    "skills": "default",
    "image": "default",
    "pose": "default",
    "action": "default",
    "map": "default"
  }
}
```

`enabled` 决定哪些插件包参与发现。`context_modules` 决定启用哪些 context slot。旧的 `knowledge` slot 可作为迁移期 alias，但默认使用 `skills`。

## 每轮注入

```text
ObservationContextBuilder.build(state, flight_memory)
  -> StageContextInput
  -> ContextModuleRegistry.build_fragments(...)
  -> DefaultSkillsContextModule builds skill listing
  -> model selects skill names
  -> runtime validates selected names
  -> SkillContentLoader reads selected skill content
  -> skills ContextFragment
  -> ContextComposer.compose(...)
  -> PlannerContext
```

`skills.facts` 始终为空。skill 是 guidance，不是观测事实。

## Callable Skill 工具合并

如果 enabled skill 声明了 `capability.json`：

```text
SkillCapabilityCatalog
  -> allowed_action_constraints.planner_tools += skill.* call names
  -> allowed_action_constraints.action_schemas += input_schema
  -> ToolRegistry registers kind=skill_capability
```

planner 可以像调用内置 tool 一样输出 `kind="tool"`，但只有出现在 `planner_tools` 中的 `skill.*` 才合法。

## 执行链路

```text
ModelActionPlanner
  -> ToolContractValidator
  -> SafetySupervisor
  -> ToolDispatcher
    -> uav.*: EnvClient
    -> skill.*: SkillCapabilityDispatcher
```

内置 UAV tools 不作为 skill。外接能力才作为 callable skill。

## 失败与降级

- skill frontmatter 无效：跳过该 skill 并记录 warning。
- `capability.json` 无效：跳过该 skill 并记录 warning。
- handler 不在 skill 目录内：拒绝加载。
- 模型选择不存在的 skill：拒绝并记录 warning。
- 模型调用未注册的 `skill.*`：validator 拒绝。
- handler 执行失败：dispatcher 返回失败 ActionResult，runtime 按普通 action failure 处理。
