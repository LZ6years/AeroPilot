# 当前插件系统

`RuntimeFactory` 是唯一发现和组合入口。它创建一个 `PluginRegistry`，再将同一实例
用于 ToolRegistry、callable skill dispatcher、知识目录和 ContextModule 注入。

贡献类型：

- `plugins/knowledge/<group>/<skill>/SKILL.md`：模型可按名称加载的知识正文。
- `plugins/capabilities/<group>/<skill>/capability.json`：注册为 `skill.*` 的可调用能力。
- `plugins/research/.../context_modules/`：实现 Memory ContextModule 契约的实验模块。

Memory 不发现插件，Tools 不发现插件，`TaskRuntime` 也不接收 PluginRegistry。
Context slot 的配置由 Factory 解析，再显式传给 `ContextModuleRegistry`。

扩展时应保证：Provider tool schema 完整、runtime effect 明确、read-only 结果只通过
ToolResult 返回、模块直接产出最终模型字段，审计字段不混入模型内容。
