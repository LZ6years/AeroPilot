# 接入知识库内容：Knowledge Skill

Knowledge skill 是给无人机 planner 的软指导。它可以帮助模型理解任务、处理不确定性、保持安全和判断完成证据，但它不会直接执行代码，也不会产生 observed facts。

如果你需要目标检测、目标跟踪、OCR、分割等外接可调用能力，请看 callable skill 文档。普通知识库内容不要加 `capability.json`。

## 1. 放在哪里

知识库目录：

```text
plugins/knowledge/<plugin_name>/<skill_folder>/
  SKILL.md
  reference.md          # optional
  references/*.md       # optional
```

默认配置已经启用：

```json
{
  "plugins": {
    "roots": ["plugins/knowledge", "plugins/research"],
    "enabled": ["knowledge.uav_flight"]
  }
}
```

所以新增通用飞行知识时，通常放到：

```text
plugins/knowledge/uav_flight/<skill_folder>/SKILL.md
```

如果新增一个独立知识包，例如 `inspection`，再把 `knowledge.inspection` 加到 `configs/plugins.json.plugins.enabled`。

## 2. SKILL.md 写作格式

```markdown
---
name: keep_target_centered
description: Use when the UAV needs to keep a visible target near the center of the RGB image while turning or moving.
---

# Keep Target Centered

## Application Boundary

Describe when this knowledge applies, when it does not, and the boundary with adjacent skills.

## Evidence Model

Describe useful evidence and conclusions that the evidence cannot support.

## Decision Procedure

Give short state-based steps: assess evidence, choose a movement type, then re-observe.

## Completion And Stop

Define sufficient completion evidence and when further correction should stop.

## Recovery

Describe conservative recovery for missing, conflicting, or no-progress evidence.
```

`name` 是模型选择 skill 时使用的稳定名字，建议 snake_case。`description` 很重要，它决定模型什么时候会选择这个 skill。

这五段是 `knowledge.*` 的写作契约，不是新的 runtime schema。runtime 仍把正文作为普通 Markdown 整体加载，不解析标题，也不会按 planner/parser 分别裁剪章节。callable skill 不强制使用这套结构。

核心正文应控制在 2400 字符以内，并保留 parser criteria 生成完成条件所需的应用边界、证据模型和停止知识。`Decision Procedure` 应按状态教学，不写死一串 primitive UAV tools 或固定动作参数。

## 3. reference.md 什么时候用

核心完成语义必须留在 `SKILL.md`。公式、长例子、搜索尺度和非必要动作细节可以放：

```text
reference.md
references/*.md
```

当前加载边界：

- catalog 常驻只暴露轻量 metadata；
- planner 选中 skill 后读取完整 `SKILL.md`，因此正文必须主动遵守 2400 字符预算；
- planner 只有显式请求时才读取对应 reference，每份最多 1600 字符；
- parser criteria 只读取 bounded `SKILL.md`，不读取任何 reference；
- references 只使用一层目录，不建立深层引用链。

正文应直接说明何时需要某份 reference，例如“复杂侧向通过证据不足时请求 `references/route_progress_examples.md`”。不要在正文和 reference 中重复同一套规则。

## 4. 如何进入 planner context

运行时流程：

```text
PluginRegistry discovers enabled skills
  -> skills module exposes available_skills
  -> model selects skill names
  -> runtime validates selection
  -> SkillContentLoader loads selected skill text/reference
  -> ContextComposer adds skills fragment
  -> ModelActionPlanner reads skills fragment
```

planner context 中的形态：

```json
{
  "name": "skills",
  "content": {
    "available_skills": [
      {
        "plugin_id": "knowledge.uav_flight",
        "name": "keep_target_centered",
        "description": "Use when ..."
      }
    ],
    "selected_skills": [
      {
        "name": "keep_target_centered",
        "skill_text": "...",
        "requested_references": [{"path": "reference.md", "text": "..."}],
        "capability": null
      }
    ]
  },
  "facts": []
}
```

关键点：`facts == []`。知识库是建议，不是当前观察事实。

## 5. 如何验证接入成功

静态 demo：

```powershell
python demos/plugin_system_demo.py
```

真实模型检索 probe：

```powershell
python tests/manual/plugin_system_real_probe.py --cases tests/fixtures/uav_knowledge_skill_selection_cases.json
```

检查输出：

```text
runs/plugin_system_demo_<timestamp>/
  02_discovered_skills.json
  04_loaded_skill_summaries.json
  05_latest_context.json

runs/plugin_system_real_probe_<timestamp>/
  real_skill_selection.json
  real_skill_selection.md
```

你应该看到：

- 新 skill 出现在 discovered skills；
- 被选中时有 `skill_text`；
- context 中有 `name == "skills"`；
- `skills.facts == []`；
- real probe 中 invalid name 为 0，且不会对简单动作选择大量 skill。

## 6. Task Parser 中的 Knowledge Skill

同一份通用 knowledge skill 也可以参与 model task parser，但职责比 planner 注入更窄：它只帮助定义某个不可变 phase 的完成条件、继续条件、证据来源和合法 metric。

```text
structure：不加载 skill，只生成 phase name 和 goal
criteria：对每个 phase 单独选 skill，再生成 done_when / continue_when / metrics
planner：运行时重新独立选择，不复用 parser 的选择结果
```

Parser criteria 注入只包含 `name + description + bounded SKILL.md body`。它不读取 reference 文件，不暴露本地路径、脚本或 callable capability。因此应用边界、证据含义、合理完成精度和停止条件必须保留在核心正文；只有公式、长案例和可选执行细节可以下沉。skill 正文不应出现 parser schema 字段；criteria model 负责把通用飞行知识映射到 semantic schema。

没有 skill 数量硬上限。selector 应选择全部真正相关的知识，同时保持最小必要集合，不能为了保险加载整个 catalog。未知、重复或未启用名称由 registry 拒绝或去重，合法选择不会被 runtime 截断。
