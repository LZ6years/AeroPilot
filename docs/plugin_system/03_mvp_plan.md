# 插件系统 MVP 计划

插件系统应该在真实 Agent 闭环跑通后再抽象。当前最合适的顺序是先做回复消息，再做简单视觉信息接入，然后引入 SkillCatalog 和上下文层扩展，最后再做完整 PluginRegistry。

## 阶段 1：无人机回复消息系统

目标：让无人机能对人类说话，而不是只写事件日志。

最小能力：

- 过程回复：例如“我正在观察当前位置前方区域”。
- 澄清提问：例如“我看到两个疑似目标，需要你确认优先检查哪一个吗？”。
- 最终回答：例如“找到了，售货机在右前方约 8 米处，这是图片证据”。
- 状态落盘：写入 `events.jsonl`、`state/latest_agent_message.json` 和 `state/final_answer.json`。

这个阶段会自然沉淀出未来的 `ResponseProtocol` 和 `AgentMessage`。

阶段 1 只做 `AgentMessage` 协议和基础输出，不做完整通信插件。`report`、`ask_user`、`alert` 暂时不作为 `ToolDispatcher` 工具，而是沉淀为后续 `communication_protocol` 知识。

## 阶段 2：简单视觉信息接入

目标：让模型能基于图片上下文继续规划和回答。

第一版不做固定 `ObjectDetector`，也不写死目标搜索策略。默认 image module 保持轻量，但会把最近 1 张真实 RGB 作为 encoded image model input 交给 planner；custom image module 负责更复杂的图片处理，例如读取、融合、caption 或压缩。当前 `multimodal_image` 会结合真实 phase 生成 `visual_description`，再通过 `ContextFragment.model_inputs` 把文字摘要交给 planner。

视觉上下文只提供处理后的画面证据、轻量 metadata 和可选 encoded model inputs，不输出 `recommended_next_observation`、转向、扫描或移动建议。导航动作仍由 planner 结合 phase、pose/history 和安全约束决定。

最小能力：

- 观察动作能产生可引用图片证据。
- planner context 能看到最近图片 metadata、artifact refs、默认 image module 产出的最近 RGB image input，以及 custom module 产出的视觉摘要。
- 模型可以基于视觉摘要或显式 image model input 继续生成语义计划。
- 最终回答可以引用图片证据。

这个阶段会自然沉淀出未来的 `CapabilitySkill`、`KnowledgeSkill` 和 `ContextContribution`。

视觉 evidence 稳定后，`final`、`progress`、`question` 才能更可靠地附带图片、pose 和观察证据。完整通信智能层应在视觉 evidence 和 question 回流跑通后再抽象。

## 阶段 3：SkillCatalog + skill_listing

目标：先做 Claude Code 风格的轻量技能目录，而不是完整插件系统。

最小能力：

- 扫描内置 UAV skills。
- 从 frontmatter 读取 `name`、`description`、`when_to_use`、`triggers`。
- 根据用户任务、当前 phase、是否有图片、是否不确定等信号筛选相关 skills。
- 把轻量 `skill_listing` 注入 planner context。
- 控制 listing 预算，避免所有知识常驻 prompt。

这个阶段不展开完整 skill 正文，只让模型知道有哪些任务经验可用。

## 阶段 4：ContextModule 用户自定义记忆呈现

目标：让用户能自定义当前各类记忆如何被模型看到。

第一版只开放上下文层：

- 核心 memory 记录仍固定为 `instructions/actions/observations/poses/frames`。
- `FlightMemory` 仍作为 manifest/index，只保存 refs 和窗口指针。
- 用户可以新增或替换 `ContextModule`，把已有 memory 加工成不同 `ContextFragment`。
- 自定义 module 必须实现 `build()` 和 `describe_output()`。

这一阶段不开放任意自定义 memory writer。

## 阶段 5：SkillContentInjector 按需展开

目标：当模型或 runtime 判断某个 skill 真正相关时，再展开完整 `SKILL.md`。

最小能力：

- 记录本轮已展开的 skill，避免重复注入。
- 把完整 skill 正文作为 planner context 的 meta instruction。
- 写入 debug artifact，方便回放为什么模型看到某个 skill。
- 保留上下文预算控制。

这个阶段对应 Claude Code 中 `SkillTool` 展开完整 skill 的思想，但 UAV 第一版可以由 runtime 自动注入，不必先暴露模型可调用的 SkillTool。

`communication_protocol` 可以在这一阶段作为内置 `KnowledgeSkill` 进入 skills 注入系统，用来指导模型什么时候输出 `progress`、`question`、`warning` 和 `final`。

## 阶段 6：完整 PluginRegistry

只有当前面边界稳定后，再抽象完整插件贡献：

- `KnowledgeSkill`：给模型的任务经验和规则。
- `CapabilitySkill`：模型能力包装，例如看图理解、场景判断、结果总结。
- `ToolContribution`：工具声明，接入现有 `ToolRegistry`。
- `ContextContribution`：上下文片段，接入现有 `ContextModuleRegistry`。
- `ResponseProtocol`：过程回复、澄清提问、最终回答。

外部 skills/plugin 包加载、版本约束、启用/禁用和 schema 校验不属于当前 MVP。
