# 接入目标跟踪模块：Callable Skill 示例

这份文档说明：如果你想接入一个“目标跟踪模块”，应该新增哪些文件、每个文件写什么、runtime 如何让 planner 调用它。

核心原则：

- 不改 `src/perception/context_builder.py`
- 不改 `src/agent/planning/tool_contract_validator.py`
- 不改 `src/tools/dispatcher.py`
- 只新增插件目录，并在配置中 enabled 对应 plugin package

## 1. 适合放在哪里

目标跟踪属于外接感知能力，建议放在：

```text
plugins/capabilities/perception/target_tracker/
  SKILL.md
  reference.md
  capability.json
  scripts/
    track_target.py
```

如果你启用了 `capabilities.perception`，新增 `target_tracker/` 后会被自动发现。当前默认配置为了调试 UAV-Flow knowledge，只启用 `knowledge.uav_flight`，所以 capability 需要手动打开：

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
      "capabilities.perception"
    ]
  }
}
```

如果你新建的是另一个 domain，例如：

```text
plugins/capabilities/navigation/path_tracker/
```

则需要把 `capabilities.navigation` 加入 `configs/plugins.json.plugins.enabled`。

## 2. SKILL.md：告诉模型什么时候用

`SKILL.md` 是给 planner 看的软说明。它不是执行代码，也不是检测结果。

```markdown
---
name: target_tracker
description: Use when the UAV needs to keep tracking a visible target across consecutive RGB observations.
---

# Target Tracker

Use this skill when the instruction requires following, keeping centered on, or maintaining visual contact with a target.

Call the tracking capability when:
- a target has been detected or visually suspected
- the UAV needs to keep the same target in view after movement
- the planner needs a structured target center before deciding turn or move

Do not treat this skill text as evidence. Only the returned capability result is runtime evidence.
```

模型常驻只会先看到 `name + description`。当它选择这个 skill 后，runtime 才会加载正文和 reference。

## 3. capability.json：声明可调用能力

`capability.json` 是“这个 skill 可以被调用”的开关。没有它，skill 只能被阅读，不能被 planner 调用。

```json
{
  "call_name": "skill.target_tracker.track",
  "description": "Track a target in the latest RGB frame and return its current image-space position.",
  "input_schema": {
    "type": "object",
    "required": ["image_source", "target_hint"],
    "additionalProperties": false,
    "properties": {
      "image_source": {
        "type": "string",
        "enum": ["latest_rgb"]
      },
      "target_hint": {
        "type": "string"
      },
      "previous_track_id": {
        "type": "string"
      },
      "min_confidence": {
        "type": "number",
        "minimum": 0,
        "maximum": 1
      }
    }
  },
  "handler": "scripts/track_target.py",
  "mock": true
}
```

规则：

- `call_name` 必须以 `skill.` 开头。
- `handler` 必须在当前 skill 目录内。
- `input_schema` 会自动进入 `allowed_action_constraints.action_schemas`。
- planner 只有在 `skill.target_tracker.track` 出现在 `planner_tools` 时才能调用。

## 4. handler：执行你的跟踪逻辑

handler 是 Python 文件，导出一个 `handle(payload: dict) -> dict` 函数。

```python
from __future__ import annotations


def handle(payload: dict) -> dict:
    args = dict(payload.get("args", {}))
    runtime_context = dict(payload.get("runtime_context", {}))
    latest_observation = dict(runtime_context.get("latest_observation", {}))
    frames = latest_observation.get("frames") if isinstance(latest_observation, dict) else {}
    rgb = frames.get("rgb") if isinstance(frames, dict) else None

    if not isinstance(rgb, dict):
        return {
            "mock": True,
            "status": "no_rgb",
            "track_found": False,
            "detections": [],
        }

    return {
        "mock": True,
        "status": "tracked",
        "track_found": True,
        "track_id": args.get("previous_track_id") or "track_person_001",
        "target_hint": args.get("target_hint"),
        "image_source": args.get("image_source"),
        "rgb_metadata": rgb,
        "target": {
            "center": [0.52, 0.48],
            "bbox": [0.44, 0.30, 0.60, 0.72],
            "confidence": max(float(args.get("min_confidence", 0.5)), 0.55),
        },
    }
```

runtime 会传入：

```json
{
  "scene_instance_id": "...",
  "agent_id": "...",
  "tool": "skill.target_tracker.track",
  "args": {...},
  "runtime_context": {
    "latest_observation": {...},
    "observation_before": {...},
    "phase": "...",
    "run_dir": "runs/..."
  },
  "skill": {
    "plugin_id": "capabilities.perception",
    "name": "target_tracker",
    "skill_dir": "plugins/capabilities/perception/target_tracker"
  },
  "capability": {...}
}
```

第一版建议 handler 保持保守：

- 不读 benchmark truth
- 不读 repo 任意文件
- 只使用传入的 `latest_observation`
- 如果需要缓存跟踪状态，只写到当前 `run_dir` 下的临时文件，不写 repo-tracked 文件

## 5. planner 会如何调用

插件启用后，planner context 中会自动出现：

```json
{
  "allowed_action_constraints": {
    "planner_tools": [
      "uav.hover",
      "uav.turn_relative",
      "uav.move_body",
      "skill.target_tracker.track"
    ],
    "action_schemas": {
      "skill.target_tracker.track": {
        "source": "skill_capability",
        "input_schema": {
          "type": "object",
          "required": ["image_source", "target_hint"]
        }
      }
    }
  }
}
```

planner 可以输出：

```json
{
  "kind": "tool",
  "tool": "skill.target_tracker.track",
  "args": {
    "image_source": "latest_rgb",
    "target_hint": "person",
    "min_confidence": 0.5
  },
  "reason": "Track the person before deciding the next movement."
}
```

之后流程是：

```text
ModelActionPlanner
  -> ToolContractValidator validates input_schema
  -> SafetySupervisor passes through skill.* tool
  -> ToolDispatcher routes to SkillCapabilityDispatcher
  -> handler returns result
  -> runtime writes result to memory/actions.jsonl
  -> next context can see the recent action result
```

## 6. 如何验证是否接入成功

先跑 discovery/context/demo：

```powershell
python demos/plugin_system_demo.py
```

检查输出：

```text
runs/plugin_system_demo_<timestamp>/
  02_discovered_skills.json
  05_latest_context.json
  06_capability_call_result.json
```

你应该看到：

- `02_discovered_skills.json` 中有 `target_tracker`
- `05_latest_context.json.allowed_action_constraints.planner_tools` 中有 `skill.target_tracker.track`
- `05_latest_context.json.allowed_action_constraints.action_schemas` 中有它的 schema

如果你写了专门测试，可以参考：

```powershell
python -m pytest tests/test_skill_tools.py -q
```

## 7. 常见错误

| 问题 | 原因 | 处理 |
| --- | --- | --- |
| skill 没出现 | plugin package 没 enabled，或目录层级不对 | 检查 `plugins/capabilities/<domain>/<skill>/SKILL.md` 和 `configs/plugins.json` |
| capability 没进 planner_tools | 缺少 `capability.json`，或 `call_name` 不以 `skill.` 开头 | 修正 `capability.json` |
| handler 加载失败 | `handler` 路径不存在，或不在 skill 目录内 | 放到当前 skill 的 `scripts/` 下 |
| planner 调用被拒绝 | args 不符合 `input_schema` | 看 `ToolContractValidator` 报错 |
| handler 有结果但下一步 planner 不用 | 当前还没有专门的 semantic state updater | 先通过 `actions.jsonl` 和 action context 复盘；后续可接 semantic state update |

## 8. 当前限制

- 现在 callable skill 已经可以被 planner 调用。
- 当前没有专门的长期目标跟踪状态管理器。
- handler 结果会作为 action result 写入 memory；如果你希望它成为稳定动态语义状态，需要后续新增 `semantic_state` context module 或 semantic state updater。
- 真实目标跟踪模型的依赖、GPU、模型文件路径和运行权限，需要在 handler 内部自己处理，并保持输出结构稳定。

换句话说：接入一个目标跟踪能力已经足够方便；如果要让无人机持续使用“同一个目标”的跨帧状态，还需要下一步补 `semantic_state` 更新机制。
