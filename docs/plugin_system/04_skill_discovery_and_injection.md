# Skills 发现与按需注入

UAV Agent 的 skills 系统建议参考 Claude Code 的核心模式：发现层轻量，执行/注入层按需展开。

## Claude Code 模式

Claude Code 的 skills 系统大致分成两段：

```text
SKILL.md
  -> 解析 frontmatter
  -> 生成轻量 Command
  -> skill_listing 只展示 name / description / when_to_use
  -> 模型选择 skill
  -> 展开完整 SKILL.md
  -> 作为 meta message 进入上下文
```

这个模式的关键收益是：

- 模型知道有哪些技能可用。
- 大量 skill 正文不会常驻上下文。
- 每条 skill 描述可以被预算控制和截断。
- 真正使用过的 skill 可以被记录，便于回放和压缩上下文。

## 映射到 UAV Agent

UAV Agent 不应直接照搬 Claude Code 的 `SkillTool`，因为 UAV 的核心链路是 `TaskRuntime -> Planner -> ToolDispatcher`，不是 CLI slash command。

推荐映射关系：

| Claude Code | UAV Agent |
| --- | --- |
| `Command` | `SkillCard` |
| `skill_listing` attachment | planner 可见的轻量技能目录 |
| `SkillTool` | 未来的 `SkillContentInjector` |
| `getPromptForCommand()` | 展开完整 `SKILL.md` |
| `sentSkillNames` | run/agent 级已注入 skill 记录 |
| `paths` 条件触发 | UAV 场景触发信号 |

## UAV 场景触发信号

UAV 不建议用文件路径触发 skills。第一版更适合使用这些信号：

- 用户意图：找、观察、确认、报告、售货机、饮水点、附近。
- 当前 phase：preflight、observe、plan、execute、answer。
- 可用证据：是否有 RGB 图片、pose、recent actions。
- 不确定性：低置信度、多候选、视野不足。
- 安全状态：高度、速度、距离、禁飞或安全限制。

示例：

```yaml
---
name: flight_search_object
description: 寻找周围目标物时的飞行与观察经验
when_to_use: 用户要求寻找、确认、定位周围物体或地点时使用
domains: [flight, search, observation]
triggers:
  intents: [search_object, find_place, thirst_help]
  phases: [observe, plan, execute, answer]
  evidence: [rgb_image, pose]
budget_priority: high
---
```

## MVP 数据形态

`SkillCard` 是模型第一眼看到的轻量目录项：

```python
@dataclass(frozen=True)
class SkillCard:
    name: str
    description: str
    when_to_use: str
    domains: list[str]
    triggers: dict[str, list[str]]
    budget_priority: str
    source_path: str
```

`SkillInjectionRecord` 记录本轮真正展开过的 skills：

```python
@dataclass(frozen=True)
class SkillInjectionRecord:
    name: str
    source_path: str
    reason: str
    trigger_signals: dict[str, list[str]]
    injected_at_phase: str
```

## MVP 不做什么

第一版不做语义检索排序算法，不做远程 skill search，不做外部插件市场。

第一版只做：

- 内置 skills 扫描。
- frontmatter 轻量索引。
- 规则过滤和固定预算。
- 相关 skills 的目录注入。
- 必要时按需展开完整正文。

