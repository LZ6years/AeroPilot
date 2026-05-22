# UAV Agent 插件系统设计

这个目录记录 UAV Agent 统一插件化架构的设计草案。当前目标不是立刻实现完整插件市场，而是先把插件系统要解决的问题、MVP 边界、代码目录、skills 发现与注入机制写清楚。

## 当前结论

插件系统第一版要解决的核心问题是：

> 模型如何按需获得任务知识、记忆摘要和回复规范。

因此第一版采用类似 Claude Code Skills 的方式：

- 先给模型一个轻量 skills 目录，而不是把所有知识常驻 prompt。
- 模型或 runtime 判断需要时，再展开完整 skill 内容。
- 用户第一版只在上下文层自定义记忆如何呈现，不直接改底层 memory writer 和索引生命周期。
- 回复协议未来先作为 communication knowledge/skill 注入，稳定后再抽成 `ResponseProtocol` 插件贡献。

## 为什么暂时不立即实现完整插件市场

当前项目还需要先跑通两条真实闭环：

1. 无人机可以向人类发送过程回复、澄清提问和最终回答。
2. 模型可以基于图片、观察结果和证据继续规划与回答。

这两条闭环跑通后，再抽象插件系统会更稳。否则容易提前设计出完整的 `KnowledgeSkill`、`CapabilitySkill`、`ResponseProtocol`，但实际接入 runtime 时发现边界不对。

## 和现有 registry 的关系

统一插件系统不推翻现有模块：

- `src/tools/registry.py` 继续负责工具名称和工具 schema 的注册。
- `src/memory/context/registry.py` 继续负责 planner context 片段的构建。
- `src/memory/flight_memory.py` 继续作为当前 run/scene/UAV 的 memory manifest。
- 未来的 `src/plugins/` 只作为统一装配层，把插件贡献分发给 tools、context、planner 和 response。

## 推荐阅读顺序

1. [05_requirements.md](05_requirements.md)：插件系统要解决什么问题、MVP 边界和记忆自定义策略。
2. [01_architecture.md](01_architecture.md)：整体架构和插件贡献类型。
3. [04_skill_discovery_and_injection.md](04_skill_discovery_and_injection.md)：参考 Claude Code 的轻量发现与按需注入模式。
4. [06_communication_layer.md](06_communication_layer.md)：通信智能层、`AgentMessage` 和 communication skill 的关系。
5. [02_code_layout.md](02_code_layout.md)：未来代码目录如何体现插件系统。
6. [03_mvp_plan.md](03_mvp_plan.md)：从当前项目状态到插件系统的 MVP 演进计划。
