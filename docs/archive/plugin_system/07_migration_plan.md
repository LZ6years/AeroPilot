> Status: Historical

# 07 Migration Plan

这份文档记录当前 plugin system 的迁移状态和后续边界。当前实现已经不再使用旧 `custom_modules`，skill 也已经迁移为文件夹式结构。

## 已完成

- 删除旧式 `src/memory/context/custom_modules` 长期机制。
- 新增 `src/plugins`，负责 discovery、registry、catalog、selection、loader 和 contracts。
- 新增 `plugins/knowledge` 和 `plugins/research`。
- `Skill` 使用文件夹式结构：

```text
plugins/knowledge/<plugin_name>/<skill_folder>/
  SKILL.md
  reference.md
  references/*.md
  scripts/*
```

- `research.multimodal_image` 通过插件方式贡献 image context module：

```text
plugins/research/multimodal_image/context_modules/image.py
```

## 当前加载策略

- 常驻 context 只暴露 skill listing：`name + description`。
- 模型选择 skill name 后，runtime 校验 selected names。
- runtime 只读取被选中 skill folder 内的 `SKILL.md`、`reference.md`、`references/*.md`。
- `scripts/*` 只列出路径，不执行。
- `skills.facts` 保持为空，skill 是 guidance，不是 observation fact。

## 后续可做

- 为 `references/*.md` 增加模型按需选择机制，而不是选中 skill 后全部小文件自动加载。
- 设计 `SkillScriptRuntime`，用于目标检测、bbox 后处理等确定性脚本能力。
- 为脚本能力增加参数 schema、输出 schema、权限边界和执行审计。
- 增加 skill folder validator，检查 frontmatter、命名、资源大小和非法路径。

## 非目标

当前仍然不做：

- 通用 `read_file` tool。
- 直接执行 `scripts/`。
- 外部插件市场。
- `plugin.json`。
- 新 UAV tools。
- response protocol 接管消息系统。
- benchmark truth 注入。
