# 02 Directory Layout

UAV-Agent 的插件目录采用“目录约定式发现”，不使用 `plugin.json`。当前默认重点是知识库 skill：它们是给 planner 的软指导，不直接执行代码。

## 目标结构

```text
plugins/
  knowledge/
    uav_flight/
      visual_target_task_reasoning/
        SKILL.md
        reference.md
      search_under_uncertainty/
        SKILL.md
        references/
      approach_and_standoff/
        SKILL.md
        reference.md
      orbit_and_surround/
        SKILL.md
      route_traversal_and_passage/
        SKILL.md
      vertical_landing_and_altitude_control/
        SKILL.md

    communication/
      agent_message_protocol/
        SKILL.md

    safety/
      conservative_low_confidence_flight/
        SKILL.md

  research/
    multimodal_image/
      context_modules/
        image.py
```

## 路径职责

| 路径 | 用途 |
| --- | --- |
| `plugins/knowledge/` | 稳定知识库，放可复用飞行经验、任务规则、语义解释规则。 |
| `plugins/knowledge/<plugin_name>/<skill_folder>/SKILL.md` | 一个知识 skill 的主文件，包含 `name + description` frontmatter 和自由正文。 |
| `reference.md` | 可选的小型默认参考，skill 被选中后由 runtime 受限加载。 |
| `references/*.md` | 可选扩展参考资料，不递归深层目录。 |
| `scripts/*` | 可选脚本资产；普通知识 skill 不执行脚本。 |
| `plugins/research/` | 实验模块，例如 context module。 |
| `plugins/capabilities/` | 外接可调用能力目录；当前默认配置不启用。 |
| `src/plugins/` | 插件发现、注册、选择、资源加载和贡献类型定义。 |
| `src/memory/context/modules/skills.py` | 把已选择的 skill 内容注入 planner context。 |

## Plugin ID

plugin id 从路径生成：

```text
plugins/knowledge/uav_flight      -> knowledge.uav_flight
plugins/knowledge/communication   -> knowledge.communication
plugins/research/multimodal_image -> research.multimodal_image
```

skill name 来自 `SKILL.md` frontmatter 的 `name`。文件夹名称用于组织资源，建议与 skill name 保持一致。

## 当前默认配置

默认只启用 UAV 通用知识库：

```json
{
  "plugins": {
    "roots": ["plugins/knowledge", "plugins/research"],
    "enabled": ["knowledge.uav_flight"]
  }
}
```

`plugins/capabilities` 不在默认 roots 中，避免在知识库调试阶段暴露额外可调用能力。
