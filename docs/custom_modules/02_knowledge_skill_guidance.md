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

## 2. SKILL.md 最小格式

```markdown
---
name: keep_target_centered
description: Use when the UAV needs to keep a visible target near the center of the RGB image while turning or moving.
---

# Keep Target Centered

Use this guidance when the task depends on continuously facing or following a visual target.

Rules:
- Prefer small corrections rather than large oscillating turns.
- If the target is lost, stop forward movement and reacquire visual evidence.

Do not:
- Do not claim the target is centered from image file paths alone.
- Do not move toward a target when visual evidence is uncertain.
```

`name` 是模型选择 skill 时使用的稳定名字，建议 snake_case。`description` 很重要，它决定模型什么时候会选择这个 skill。正文标题自由，runtime 不解析固定标题。

## 3. reference.md 什么时候用

如果正文短，全部写在 `SKILL.md` 里即可。

如果有补充例子、术语、判断标准，可以放：

```text
reference.md
references/*.md
```

当前 loader 会读取：

- `SKILL.md` 正文，默认最多 2400 字符；
- `reference.md`，默认最多 1600 字符；
- `references/*.md`，不递归更深目录。

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
        "references": [{"path": "reference.md", "text": "..."}],
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
criteria：对每个 phase 单独选 skill，再生成 done_when / continue_when / observe / metrics
planner：运行时重新独立选择，不复用 parser 的选择结果
```

Parser criteria 注入只包含 `name + description + bounded SKILL.md body`。它不读取 reference 文件，不暴露本地路径、脚本或 callable capability。skill 正文也不应出现 parser schema 字段；criteria model 负责把通用飞行知识映射到 semantic schema。

没有 skill 数量硬上限。selector 应选择全部真正相关的知识，同时保持最小必要集合，不能为了保险加载整个 catalog。未知、重复或未启用名称由 registry 拒绝或去重，合法选择不会被 runtime 截断。
