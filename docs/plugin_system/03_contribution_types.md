# 03 Contribution Types

插件包按贡献类型进入系统。一个插件可以贡献多种类型，但每种贡献有不同的装配位置和运行边界。

## 类型总览

| 类型 | 进入位置 | 第一版状态 | 是否直接影响飞控 |
| --- | --- | --- | --- |
| `Skill` | `skills` context fragment | 完整实现 | 否，间接影响 planner |
| `ContextModuleContribution` | `ContextModuleRegistry` | 框架支持 | 否，改变上下文呈现 |
| Callable skill capability | `SkillCapabilityCatalog` / `skill.*` tool | 框架支持，默认不启用 | 是，但必须过 validator/safety |
| `ResponseProtocol` | 未来 agent message/final answer 层 | 只定义接口 | 否 |
| `PolicyRule` | 未来约束合成层 | 只定义接口 | 间接约束动作 |

## Skill

`Skill` 是第一版的统一贡献类型。普通 skill 用 Markdown 维护，通过 frontmatter 提供发现信息，通过正文提供可读规则。

它适合表达：

- 飞行任务经验。
- 视觉对齐规则。
- 目标搜索策略。
- 低置信度时的保守行为。
- 通信和报告规范。

它不应该表达：

- 具体 UAV tool 调用。
- benchmark truth。
- 固定 case 坐标。
- 绕过 safety 的动作建议。

## ContextModuleContribution

`ContextModuleContribution` 允许插件贡献新的 context module，改变某一类记忆如何呈现给模型。

例如：

```text
plugins/research/multimodal_image/context_modules/image.py
```

贡献一个替代默认 image module 的 `PluginContextModule`。

第一版框架需要能识别和加载这种贡献，但默认不启用研究模块。

## Callable Skill Capability

如果 skill 目录额外声明 `capability.json`，它会成为可调用能力，例如：

- 目标检测。
- 场景关系分析。
- 轨迹诊断。

callable skill 不替代内置 UAV tool，也不能绕过既有工具链：

```text
Skill + capability.json
  -> SkillCapabilityCatalog
  -> ToolContractValidator
  -> SafetySupervisor
  -> ToolDispatcher
```

默认配置不启用 `plugins/capabilities`，避免调试知识库时暴露额外能力。

## ResponseProtocol

`ResponseProtocol` 表示未来插件可以贡献“如何与人沟通”的协议，例如 progress、question、warning、final 的结构和时机。

第一版不让 response protocol 接管消息系统。通信规范先以普通 `Skill` 进入 planner context。

## PolicyRule

`PolicyRule` 表示未来插件可以贡献约束规则，例如低置信度时限制速度、靠近人时限制距离、视觉证据不足时不要标记完成。

第一版只定义接口，不改 safety/planner constraints 合成逻辑。
