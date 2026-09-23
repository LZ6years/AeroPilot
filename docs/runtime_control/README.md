# Runtime built-in controls

当前生产接口使用单一工具循环；不再提供 commit、Semantic Updater 或 ask_user。

- `runtime.update_plan`：独占控制回合，应用可选计划并增加 revision；不执行动作，不证明任务完成。
- `runtime.report_completed` / `runtime.report_blocked`：独占终态回合，校验当前指令的环境证据引用；Runtime 统一完成悬停、最终观察和清理。
- `knowledge.load`：只读工具，读取启用知识及声明内参考资料；增量正文进入 Transcript，不改变 Capability 权限。
- 物理工具直接提交 1–3 个有序调用，整批预检后执行，只有一次自动 after observation。

工具数据类统一位于 `src/tools/schemas.py`；内置控制工具实例位于
`src/tools/builtins/`，唯一注册入口为 ToolRegistry。ToolParameter 定义单个参数，
ToolSpec 从参数与对象级选项生成 input_schema，供 Provider 与原校验器共用。
插件 JSON Schema 在注册边界转换；嵌套结构、约束和内部扩展保留，示例在注册时校验。
模型只能看到 model_exposed=true 且满足场景允许名单的工具；允许名单不能开放隐藏工具。
hover 对模型隐藏，Runtime 稳定、安全替代与清理仍可使用。
ContextBuilder 只生产状态与环境证据；工具定义仅通过 Provider tools 发送。
HistoryProjector 清理旧状态中的重复工具字段，不删除原始历史或历史工具调用。
通用 ToolUser Prompt 解释参数、执行属性和结果；适用场景放在工具说明中，
复杂任务默认首轮计划、简单任务可省略属于软引导，不增加强制计划校验。
可见工具说明按“功能、适用需求、返回内容、能力边界”组织；参数说明只解释
含义、单位和行为影响。复杂任务计划是明确保留的引导，不要求其他工具遵循
固定选择顺序。图像说明保留传感器语义，知识保留领域方法与证据要求。
计划和终态处理位于 `src/runtime/control/actions/`；知识读取位于
`src/plugins/skills/knowledge_service.py`。控制处理器不负责启动第二个模型或执行循环。

限制为每指令 300 个物理窗口、10 轮非物理调用、6 次连续协议拒绝，保留配置覆盖。
成功计划更新不会重置限制。计划、知识和意图不能代替环境完成证据。

详见 [迁移与覆盖清单](../agent_loop/unified_tool_loop_migration.md)。

本目录其余编号文档及 actions 子目录是旧控制方案的历史设计，不能作为当前 API、
配置或可用工具清单。旧 Phase 兼容只存在于报告读取边界。
