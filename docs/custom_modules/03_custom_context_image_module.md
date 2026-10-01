# 接入自定义 Image / 数据处理模块：Context Module 示例

这份文档说明：如果你想自定义 image、depth、pose、semantic state 等数据如何进入 planner context，应该如何写一个 context module 插件。

和 skill 不同：

- skill 是给 planner 的知识或外接能力说明。
- context module 是把 memory 中的数据整理成 `ContextFragment`。
- image 处理、caption、目标状态摘要、地图摘要，都适合做 context module。

## 1. 适合什么时候用

适合做 context module 的场景：

- 你想替换默认 image module
- 你想把 RGB 图像先转成视觉描述
- 你想把 detection/tracking 结果整理成 `semantic_state`
- 你想新增一个 planner 可读的数据 fragment，例如 `target_state`

不适合做 context module 的场景：

- 你只是想加入飞行经验规则：用 knowledge skill
- 你想让 planner 调用外部检测器：用 callable skill
- 你想新增 UAV 运动原语：那是内置 tool，不走 plugin skill

## 2. 目录结构

以自定义 image module 为例：

```text
plugins/research/my_image_processor/
  __init__.py
  context_modules/
    __init__.py
    image.py
```

`image.py` 对应 `image` slot。

如果你新增的是 semantic state：

```text
plugins/research/my_semantic_state/
  context_modules/
    semantic_state.py
```

则对应 `semantic_state` slot。

## 3. 最小 image.py 模板

文件必须导出 `PluginContextModule`，并继承 `ContextModule`。

最简单的做法是继承默认 image module，然后在默认结果上追加字段：

```python
from __future__ import annotations

from typing import Any

from src.memory.context.modules.image import DefaultImageContextModule
from src.memory.context.schemas import ContextFragment, MemorySourceBundle, StageContextInput


class PluginContextModule(DefaultImageContextModule):
    def build(self, sources: MemorySourceBundle, stage_context: StageContextInput) -> ContextFragment:
        fragment = super().build(sources, stage_context)
        content = dict(fragment.content)
        warnings = list(fragment.warnings)

        latest_rgb = content.get("latest_frame_metadata", {}).get("rgb")
        custom_summary = _summarize_rgb(latest_rgb, stage_context)
        content["custom_visual_summary"] = custom_summary

        return ContextFragment(
            name=fragment.name,
            enabled=fragment.enabled,
            source=self.__class__.__name__,
            content=content,
            media_refs=list(fragment.media_refs),
            facts=list(fragment.facts),
            inferences=[
                *fragment.inferences,
                {
                    "type": "custom_visual_summary",
                    "value": custom_summary,
                    "source": self.__class__.__name__,
                },
            ],
            goals=list(fragment.goals),
            risks=list(fragment.risks),
            confidence=fragment.confidence,
            artifact_refs=list(fragment.artifact_refs),
            budget=fragment.budget,
            warnings=warnings,
            metadata=dict(fragment.metadata),
        )

    def describe_output(self) -> dict[str, Any]:
        description = dict(super().describe_output())
        content_fields = dict(description.get("content_fields", {}))
        content_fields["custom_visual_summary"] = "Custom image-derived summary for planner use."
        description["content_fields"] = content_fields
        return description


def _summarize_rgb(latest_rgb: dict | None, stage_context: StageContextInput) -> dict:
    if not isinstance(latest_rgb, dict):
        return {"status": "no_rgb"}
    return {
        "status": "metadata_only",
        "path": latest_rgb.get("path"),
        "phase": (stage_context.current_phase or {}).get("name"),
    }
```

## 4. 配置启用

在 `configs/plugins.json` 中启用插件包，并把 `image` slot 指到插件：

```json
{
  "plugins": {
    "roots": [
      "plugins/knowledge",
      "plugins/capabilities",
      "plugins/research"
    ],
    "enabled": [
      "knowledge.uav_flight",
      "research.my_image_processor"
    ]
  },
  "context_modules": {
    "instruction": "default",
    "skills": "default",
    "image": "research.my_image_processor",
    "pose": "default",
    "action": "default",
    "map": "default"
  }
}
```

加载规则：

```text
plugins/research/my_image_processor/context_modules/image.py
  -> plugin_id = research.my_image_processor
  -> slot = image
  -> module_path = plugins.research.my_image_processor.context_modules.image
  -> class_name = PluginContextModule
```

## 5. ContextFragment 字段怎么放

`ContextFragment` 的字段含义：

| 字段 | 什么时候用 |
| --- | --- |
| `content` | 模块的主要结构化输出 |
| `media_refs` | 要延迟水合并发送给模型的图片引用 |
| `facts` | 可靠、可验证、来自传感器或明确执行结果的事实 |
| `inferences` | 模型 caption、启发式分析、可能需要确认的判断 |
| `risks` | 缺图、mock frame、caption 失败、不确定性 |
| `artifact_refs` | RGB、depth、plot 等文件路径 |
| `warnings` | 非致命问题 |
| `metadata` | 版本、策略、耗时等辅助信息 |

建议：

- 图像路径存在，只能说明“有图”，不能说明“看到了人”。
- 模型 caption 或检测器输出如果不是高可信传感器事实，先放 `inferences`。
- 如果模块调用视觉模型，最好在 `content.visual_processing.status` 中记录状态。
- 不要输出动作建议字段，例如 `turn_left / move_forward`。动作仍由 planner 根据完整 context 决定。

## 6. 如果要传图片或文本给模型

如果你希望 planner 直接收到图片，使用 `media_refs`；额外文本属于 `content`。

`media_refs` 是有序图片引用序列。通用组装器会保持 module 提供的图片顺序，
并在这些输入之后追加结构化 JSON Context；它不会理解或重新排列自定义角色。
因此，若一段文字用于约束紧随其后的图片，应把文字放在图片之前；若用于提醒
如何使用刚发送的图片，应放在图片之后。

文本输入示例：

```python
media_refs = [
    {
        "type": "text",
        "source": "image",
        "role": "visual_caption",
        "text": "The latest RGB appears to contain a person-like target near center.",
        "metadata": {
            "strategy": "custom_caption_v1"
        }
    }
]
```

图片输入建议优先复用默认 image module 的能力。默认 image module 生成 `MediaRef`，只在请求组装期间由 `MediaHydrator` 编码。

如果你自己构造图片输入，必须保证：

- 不把大段 base64 写进普通 debug JSON 文本
- `metadata.mock` 正确标注
- mock/missing frame 不要伪造成真实视觉证据

## 7. 当前默认实现

当前正式图像链由 `src/memory/context/modules/image/` 提供。默认模块读取
FlightMemory 中的 RGB frame 引用，生成 checkpoint 的 image model input；历史图片
选择继续由同一个模块的 `select_history_images()` 实现。自定义模块应复用通用
`ContextModule` 与 `ContextModuleResult` 契约，不依赖已删除的研究 caption 插件。

## 8. 如何验证

最小验证：

```powershell
& 'F:\conda_env\flyagent\python.exe' -m pytest tests/test_context_builder.py tests/test_model_context_v3.py -q
```

如果你新增了自己的插件，建议新增一个测试：

```python
def test_my_image_plugin_loads_when_configured(tmp_path):
    plugin_registry = PluginRegistry.discover({
        "plugins": {
            "roots": ["plugins/research"],
            "enabled": ["research.my_image_processor"],
        }
    })
    registry = ContextModuleRegistry.from_dict(
        {"context_modules": {"image": "research.my_image_processor"}},
        plugin_registry=plugin_registry,
    )
    assert registry.modules["image"].__class__.__name__ == "PluginContextModule"
```

手动查看：

```powershell
python demos/plugin_system_demo.py
```

或者构造一次真实 run 后查看：

```text
runs/<run_id>/contexts/latest_context.json
```

确认：

- `fragments[].name == "image"`
- `fragments[].source == "PluginContextModule"`
- `context_schema.module_content_fields.image` 中有你新增的字段说明

## 9. 常见错误

| 问题 | 原因 | 处理 |
| --- | --- | --- |
| 配置后加载失败 | 没有导出 `PluginContextModule` | 在 `image.py` 中定义该类 |
| 插件没被发现 | plugin package 没 enabled | 检查 `configs/plugins.json.plugins.enabled` |
| slot 不匹配 | 文件名不是 `image.py`，或 fragment.name 返回不一致 | 文件名和 `ContextFragment.name` 保持 slot 一致 |
| planner 看不到字段说明 | `describe_output()` 没补充 content_fields | 给新增字段写说明 |
| 视觉模型失败导致任务中断 | 没捕获异常 | context module 应保守 fallback，只写 warning |
| 模型把 caption 当硬事实 | 字段放错到 `facts` | 不确定视觉分析放 `inferences` 或 `content`，并标注 confidence |

## 10. 和 skill/capability 的关系

推荐组合：

```text
ContextModule
  负责整理“当前看到了什么/有什么证据”

Skill
  负责告诉 planner“遇到这种任务一般怎么做”

CallableSkill
  负责让 planner 调用外接能力，例如 detect/track/OCR
```

例如目标跟踪完整链路可以是：

```text
image context module
  -> 提供 latest RGB / caption / visual evidence

skill.target_tracker.track
  -> 外接跟踪能力返回 track result

semantic_state context module
  -> 后续把最近 tracking result 整理成 target_state
```

当前已支持前两类；长期稳定的 `semantic_state` 更新机制可以作为下一步扩展。
