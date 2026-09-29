> Status: Historical

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

`knowledge.*` 正文统一使用 `Application Boundary`、`Evidence Model`、`Decision Procedure`、`Completion And Stop` 和 `Recovery` 五段式写作契约。它用于保持知识完整、紧凑和可审查，但不是 runtime schema：runtime 仍把标题和正文当作普通 Markdown，不解析章节，也不按消费者路由章节。callable skill 可以使用适合其能力的其他结构。

## 渐进加载流程

```text
PluginRegistry discovers enabled skills
  -> skills module exposes name + description listing
  -> model selects skill names
  -> runtime validates selected names
  -> SkillContentLoader loads SKILL.md body
  -> SkillResourceLoader loads only explicitly requested references
  -> selected skill content enters skills fragment
```

## 模型配置的唯一来源

模型配置只允许在应用入口通过 `--env-path` 加载一次，然后以同一个
`ModelConfig` 实例显式注入各个模型消费者：

```text
--env-path
  -> load_model_config()
  -> TaskParser
  -> ModelActionPlanner
  -> ModelSkillSelector
  -> SemanticPlanUpdater
```

`ModelSkillSelector` 和 `SemanticPlanUpdater` 不读取当前工作目录中的
`.env`。启用模型 Skill 选择时，如果没有显式的 `ModelConfig`，构造阶段
立即报错；模型 Runtime 缺少 API key 时会在场景启动前报错。关闭 Skill
选择时则使用无网络副作用的 disabled selector。这样从 Git worktree、CLI
或 Benchmark 启动时不会因 cwd 不同而得到不同的模型依赖。

Parser-time Skill 与 Planner-time Skill 是两条独立选择链：前者帮助生成
语义 criteria，后者为当前 phase 的动作规划加载知识。两者独立选择，但
共享入口加载的同一个 `ModelConfig`。

planner 没有通用 `read_file` 能力。runtime 只允许读取被选中 skill 目录内、且由 selector 显式请求的受限资源。核心 knowledge 正文遵守 2400 字符写作预算；reference 用于公式、长示例和可选动作细节，不能承载 parser criteria 必需的完成语义。

## Skills Fragment

`skills` fragment 包含：

- `available_skills`：轻量 listing。
- `selected_skills`：被选中后展开的正文，以及显式请求的 references 和可用 capability metadata。
- `available_capabilities`：当前 enabled callable skills；默认知识库调试时为空。
- `facts`：始终为空。

skill 是 guidance，不是当前观察事实，也不是 benchmark truth。

## 检索质量验证

静态测试检查 catalog 是否只有当前通用 skill；真实模型检索使用 manual probe：

```powershell
python tests/manual/plugin_system_real_probe.py --cases tests/fixtures/uav_knowledge_skill_selection_cases.json
```

从 worktree 运行时应显式指向主工作区的环境文件：

```powershell
python tests/manual/plugin_system_real_probe.py `
  --env-path F:\UAV-Agent\.env `
  --cases tests/fixtures/uav_knowledge_skill_selection_cases.json
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

## 语义计划 Criteria 注入

Model task parser 使用独立的两阶段流程：

```text
用户指令
  -> structure model 生成 intent / task_goal / phase index,total,name,goal
  -> 每个 phase 独立选择相关 knowledge skill
  -> criteria model 生成 done_when / continue_when
  -> 确定性合并并执行 semantic validator
```

Structure 调用不加载 skill。phase 数量、顺序、名称和 goal 只能来自用户指令，skill 不能修改这些字段。

Criteria selector 不设置数量硬上限，可以保留任意数量的合法选择，但 prompt 要求选择最小必要集合。Runtime 只校验名称、去重并保持顺序，不截断合法结果。selector 失败时，该 phase 仍使用 `FLYAGENT.md + USER.md + 通用 done_when 规则 + 动态证据能力`生成 criteria。

Parser 侧只加载有长度限制的 `SKILL.md` 正文，不向 selector 或 criteria model 提供 references、source path、scripts 或 callable capability。因此每份核心正文保留应用边界、证据模型、行动与停止知识，但不定义 Agent 身份、JSON Schema、`done_when`拆分规则或 Runtime 字段。Planner 仍按当前 runtime evidence 独立选择 skill；parser 的选择结果不会传给 planner 作为隐式决定。
