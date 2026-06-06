# 05 Context Module Plugins

Context module 插件化解决的是“模型看到什么样的记忆摘要”。它不负责动作规划，不调用 UAV tool，不判断任务完成。

## 旧机制不再作为长期机制

旧式目录：

```text
src/memory/context/custom_modules/
```

不再作为长期扩展入口。后续不支持旧式 `module_path/class_name` 自定义加载，不做 legacy wrapper。

新的 context module 必须通过插件目录贡献：

```text
plugins/{category}/{plugin_name}/context_modules/{slot}.py
```

文件内导出：

```python
PluginContextModule
```

## Slot 规则

文件名就是 slot：

```text
context_modules/image.py     -> image slot
context_modules/map.py       -> map slot
context_modules/skills.py    -> skills slot
context_modules/semantic_state.py -> semantic_state slot
```

slot 不应该在代码里写死成少数固定类别。插件发现可以发现任意 slot，真正启用哪些 slot 由 `configs/plugins.json` 决定。

默认配置可以先启用这些核心 slot：

```text
instruction
skills
image
pose
action
map
```

研究插件可以通过配置新增或替换 slot，例如：

```text
image
semantic_state
visual_diagnostics
```

通常不建议替换 `instruction`、`pose` 和 `action`，但这应是配置和文档约束，不是注册表硬编码限制。

## research.multimodal_image 迁移

当前已有的 multimodal image 研究模块应直接迁移到：

```text
plugins/research/multimodal_image/context_modules/image.py
```

迁移后它可以继续复用 `src` 中的公共代码：

- `src.memory.context.modules.image.DefaultImageContextModule`
- `src.memory.context.schemas.ContextFragment`
- `src.runtime.configs.model_config.ModelConfig`

但插件入口不再位于 `src/memory/context/custom_modules`。

## 配置选择

插件发现只说明“有哪些模块可用”，具体启用哪个模块由配置决定：

```json
{
  "context_modules": {
    "skills": "default",
    "image": "default",
    "pose": "default",
    "action": "default",
    "map": "default"
  }
}
```

启用 research image module：

```json
{
  "context_modules": {
    "skills": "default",
    "image": "research.multimodal_image",
    "semantic_state": "research.semantic_state"
  }
}
```

## 装配边界

```text
PluginRegistry
  -> ContextModuleContribution
  -> ContextModuleRegistry
  -> ContextComposer
  -> PlannerContext
```

`ContextModuleRegistry` 仍然是创建 context module 的地方。`PluginRegistry` 只是提供可用贡献。

## 模块职责

Context module 可以：

- 读取 `MemorySourceBundle`。
- 使用 `StageContextInput`。
- 输出 `ContextFragment`。
- 解释自己的 `describe_output()`。

Context module 不可以：

- 执行飞行动作。
- 直接推进 phase。
- 修改 `MissionState`。
- 绕过安全约束。
- 把 benchmark 私有字段暴露给 planner。
