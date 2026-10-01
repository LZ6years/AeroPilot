# 自定义模块接入

稳定入口有两类：

- callable skill：放在 `plugins/capabilities/`，声明 `capability.json`、Provider schema
  和 runtime effect；结果以原生 ToolResult 返回。
- ContextModule：实现 `src.memory.context` 定义的模块契约，由配置选择，再由
  RuntimeFactory 显式注入 `ContextModuleRegistry`。

模块应直接返回最终 `content` 和有序 `media_refs`，并通过 `FragmentAudit`
返回来源、置信度、warning 和 artifact refs。模块不能发现插件、调用 Runtime、改变
任务状态或自行编码 base64 媒体。

目录中的编号文件是较早的专题说明；若与本文、`context_v3.md` 或实际契约冲突，
以当前契约为准。
