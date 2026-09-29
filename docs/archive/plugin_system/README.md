> Status: Historical

# UAV-Agent Plugin System

这个目录记录 UAV-Agent 的统一插件系统设计。当前目标不是做完整插件市场，而是先搭起清晰的插件骨架，让飞行任务知识、上下文模块和未来能力可以用同一套方式被发现、注册、选择和装配。

第一版主要实现统一 `Skill`：普通 skill 提供 guidance；带 `capability.json` 的 callable skill 可作为外接能力路径，但默认不启用。

## 核心问题

UAV-Agent 在飞行时需要根据自然语言指令、当前图像、位姿、动作历史和语义计划持续规划。问题不只是“模型能不能看图”，还包括：

- 不同飞行任务有哪些通用经验和规则？
- 模型什么时候应该看到哪些规则？
- 图像、位姿、动作、地图等记忆应该如何被整理成 planner context？
- 未来研究模块、工具、通信协议如何进入系统，而不把 runtime 写乱？

插件系统回答的是：

> 当前启用的扩展给系统贡献了哪些知识、上下文模块、能力、工具声明、通信协议和策略规则？

插件系统不直接控制无人机，不直接调用 UAV tool，不直接决定 phase，也不绕过 validator 和 safety。

## 目录边界

`memory/` 不是插件目录。它继续保存运行记忆、用户偏好和地图记忆：

- `memory/FLYAGENT.md`：Agent 长期行为规则。
- `memory/USER.md`：用户偏好。
- `memory/map/`：地图记忆。
- `runs/*/memory/`：单次运行产生的 instruction、action、observation、frame、message 记录。

`plugins/` 是插件资产目录，保存可复用知识和实验扩展。用户如果要新增插件，也应按插件内容放到合理分类下，而不是放进一个笼统的 user 目录：

- `plugins/knowledge/`：稳定的飞行任务知识和语义规则，通常是纯 guidance。
- `plugins/capabilities/`：外接能力 skills，通常包含 `capability.json` 和受限 handler。
- `plugins/research/`：研究型上下文模块或能力。

`src/plugins/` 是插件系统代码，负责发现、注册、选择和装配。

`src/memory/context/` 保留为上下文运行接口，负责 `ContextModule`、`ContextFragment`、`ContextModuleRegistry` 和 `ContextComposer`。

## 关键决策

- 不使用 `plugin.json`。
- plugin id 从目录路径生成，例如 `plugins/knowledge/uav_flight` 对应 `knowledge.uav_flight`。
- skill metadata 写在 Markdown frontmatter 中。
- context module slot 从文件名生成，例如 `context_modules/image.py` 贡献 `image` slot；实际启用哪些 slot 由 `configs/plugins.json` 决定，不在代码里写死。
- 不保留旧式 `src/memory/context/custom_modules` 作为长期机制。
- 不做 legacy wrapper；研究模块要直接迁移到 `plugins/research/<name>/context_modules/`。
- 第一版主要实现普通 guidance skill；callable skill 作为外接能力扩展路径保留。
- 内置 UAV tools 不作为 skills；外接能力通过 callable skill 暴露为 `skill.*` 调用。

## 推荐阅读顺序

1. [01_architecture.md](01_architecture.md)：整体架构和飞行链路。
2. [02_directory_layout.md](02_directory_layout.md)：目录结构和命名规则。
3. [03_contribution_types.md](03_contribution_types.md)：插件贡献类型。
4. [04_skill_discovery_and_injection.md](04_skill_discovery_and_injection.md)：Skill 发现、选择和注入。
5. [05_context_module_plugins.md](05_context_module_plugins.md)：ContextModule 插件化。
6. [06_runtime_loading.md](06_runtime_loading.md)：飞行时如何加载和引入。
7. [07_migration_plan.md](07_migration_plan.md)：后续代码迁移路径。
