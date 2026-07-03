# 04 Skill Discovery And Injection

第一版 skill 选择采用 Claude/Codex 风格的渐进披露：模型常驻只看到 `name + description`，只有被模型选中的 skill 才会加载正文和参考资料。

## 目录发现

发现规则：

```text
plugins/{category}/{plugin_name}/{skill_folder}/SKILL.md
plugins/{category}/{plugin_name}/{skill_folder}/capability.json   # optional
```

当前默认启用的知识 skill 示例：

```text
plugins/knowledge/uav_flight/visual_target_task_reasoning/SKILL.md
plugins/knowledge/uav_flight/approach_and_standoff/SKILL.md
plugins/knowledge/uav_flight/route_traversal_and_passage/SKILL.md
plugins/knowledge/uav_flight/orbit_and_surround/SKILL.md
```

## SKILL.md

`SKILL.md` 只要求最小 frontmatter：

```yaml
---
name: visual_target_task_reasoning
description: Use when a UAV task depends on seeing, facing, observing, centering, or keeping attention on a visual target.
---
```

正文标题自由，runtime 不解析固定的 `## When To Use / Rules / Do Not`。这些标题可以写，但只是普通文本。

## 渐进加载流程

```text
PluginRegistry discovers enabled skills
  -> skills module exposes name + description listing
  -> model selects skill names
  -> runtime validates selected names
  -> SkillContentLoader loads SKILL.md body
  -> SkillResourceLoader loads bounded reference.md / references/*.md
  -> selected skill content enters skills fragment
```

planner 没有通用 `read_file` 能力。runtime 只允许读取被选中 skill 目录内的受限资源。

## Skills Fragment

`skills` fragment 包含：

- `available_skills`：轻量 listing。
- `selected_skills`：被选中后展开的正文、references、assets、capability metadata。
- `available_capabilities`：当前 enabled callable skills；默认知识库调试时为空。
- `facts`：始终为空。

skill 是 guidance，不是当前观察事实，也不是 benchmark truth。

## 检索质量验证

静态测试检查 catalog 是否只有当前通用 skill；真实模型检索使用 manual probe：

```powershell
python tests/manual/plugin_system_real_probe.py --cases tests/fixtures/uav_knowledge_skill_selection_cases.json
```

输出：

```text
runs/plugin_system_real_probe_<timestamp>/
  real_skill_selection.json
  real_skill_selection.md
```

重点看：

- invalid selected name 是否为 0；
- 平均每条选择 1-3 个 skill；
- all-select 比例是否过高；
- 简单无关动作是否选择了不必要 skill；
- expected skill 是否被命中。

## 原则

- 内置 UAV tools 不作为 skill。
- 普通知识 skill 不直接执行代码。
- 外接能力以后通过 callable skill 暴露为 `skill.*`，但当前默认关闭。
- 不暴露 `target_pos / reference_path_preprocessed / end_pos`。
- 不允许 skill 绕过 validator、dispatcher、safety。
