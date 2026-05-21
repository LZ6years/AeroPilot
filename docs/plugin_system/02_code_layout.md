# 代码目录设计

未来建议新增 `src/plugins/`，作为统一插件声明和装配层。它不替代现有 `src/tools`、`src/memory/context`、`src/runtime`，而是把插件贡献分发给这些已有模块。

## 推荐目录结构

```text
src/
  agent/
    semantic/
    planning/
    response/

  runtime/
    runners/
    plugin_runtime.py

  plugins/
    core/
      manifest.py
      registry.py
      contribution.py
      loader.py

    discovery/
      skill_catalog.py
      skill_listing.py
      trigger_matcher.py

    injection/
      skill_content_injector.py
      planner_context_injector.py

    budget/
      skill_budget.py
      listing_formatter.py

    contracts/
      knowledge.py
      capability.py
      tool.py
      context.py
      response.py

    builtin/
      uav_flight/
        plugin.py
        SKILL.md
      uav_observation/
        plugin.py
        SKILL.md
      user_response/
        plugin.py
        SKILL.md
      safety/
        plugin.py
        SKILL.md

  tools/
    registry.py
    dispatcher.py

  memory/
    context/
      registry.py
      modules/
      custom_modules/
```

## 模块职责

`src/plugins/core` 负责插件元信息、注册表、加载和贡献聚合。

`src/plugins/discovery` 负责发现可用 skills，读取轻量元信息，并根据 UAV 场景触发信号筛选相关 skills。

`src/plugins/injection` 负责把轻量 `skill_listing` 或完整 skill 内容注入 planner context。

`src/plugins/budget` 负责控制目录列表和 skill 正文的上下文预算，避免把所有知识常驻 prompt。

`src/plugins/contracts` 定义插件能贡献什么，但不包含具体业务实现。

`src/plugins/builtin` 放第一批内置 skills。MVP 阶段只做内置 skills，不做外部包加载。

`src/memory/context/custom_modules` 是第一版开放给用户的主要扩展点。用户可以新增或替换 `ContextModule`，自定义已有 memory 如何被模型看到。

## 和现有代码的边界

现有模块继续保留原职责：

- `ToolDispatcher` 仍然负责工具调用和 EnvClient 路由。
- `ToolRegistry` 仍然负责工具 spec 查询和校验。
- `ContextModuleRegistry` 仍然负责构建上下文片段。
- `FlightMemory` 仍然作为 memory manifest/index，保存 refs 和窗口指针。
- `TaskRuntime` 仍然负责任务生命周期、动作执行、观察记录和状态落盘。
- `ModelActionPlanner` 仍然负责根据上下文生成短程动作计划。

PluginRegistry 不直接调用无人机、不直接执行动作、不直接决定 phase。它只回答一个问题：当前启用的插件给系统贡献了哪些知识、能力、工具、上下文和回复协议。

## 第一版配置形态

第一版可以只支持内置 skills：

```json
{
  "plugins": [
    "builtin.uav_flight",
    "builtin.uav_observation",
    "builtin.user_response",
    "builtin.safety"
  ]
}
```

上下文模块仍使用现有配置方式：

```json
{
  "modules": {
    "instruction": "default",
    "image": "custom_image_history",
    "pose": "default",
    "action": "default",
    "map": "default"
  },
  "custom_modules": {
    "custom_image_history": {
      "module_path": "src.memory.context.custom_modules.image_history",
      "class_name": "ImageHistoryContextModule"
    }
  }
}
```

后续再考虑外部 skills/plugin 包加载、版本约束、启用条件和记录层 memory schema 扩展。

