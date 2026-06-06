# 自定义模块接入指南

这个目录面向使用者和后续开发者，说明如何在不修改核心 runtime 的情况下，把自己的外接能力接入 UAV-Agent。

当前推荐入口是 **callable skill**：

- 内置工具仍然是系统 tool，例如 `uav.turn_relative / uav.move_body / uav.hover`。
- 自定义外接能力放在 `plugins/capabilities/` 下。
- 如果某个 skill 目录里有 `capability.json`，runtime 会把它注册成 planner 可调用的 `skill.*` 能力。
- planner 只能调用已经 enabled 且出现在 `allowed_action_constraints.planner_tools` 里的 `skill.*`。

先读：

- [02_knowledge_skill_guidance.md](02_knowledge_skill_guidance.md)
- [01_callable_skill_target_tracking.md](01_callable_skill_target_tracking.md)
- [03_custom_context_image_module.md](03_custom_context_image_module.md)
