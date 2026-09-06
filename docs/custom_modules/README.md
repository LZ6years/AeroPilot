# 自定义模块接入指南

这个目录面向使用者和后续开发者，说明如何在不修改核心 runtime 的情况下，把自己的外接能力接入 UAV-Agent。

当前推荐入口是 **callable skill**：

- 内置工具仍然是系统 tool，例如 `uav.turn_relative / uav.move_body / uav.hover`。
- 自定义外接能力放在 `plugins/capabilities/` 下。
- 如果某个 skill 目录里有 `capability.json`，runtime 会把它注册成 planner 可调用的 `skill.*` 能力。
- `capability.json` 必须声明 provider `wire_name` 和完整 `runtime` 属性。
- planner 只能调用已启用且实际出现在 provider `tools` 数组中的能力。
- read-only 能力的结果会以原生 ToolResult 回到同一个 transcript，再由模型决定是否飞行。

先读：

- [02_knowledge_skill_guidance.md](02_knowledge_skill_guidance.md)
- [01_callable_skill_target_tracking.md](01_callable_skill_target_tracking.md)
- [03_custom_context_image_module.md](03_custom_context_image_module.md)
