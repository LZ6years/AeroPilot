# 08 可插拔 Context Modules

## 1. 设计目标

Context module 是记忆系统的可替换单元。用户可以替换 image module、map module 或新增研究型 module，而不改变 `ContextComposer`、`ModelActionPlanner` 和 `SafetySupervisor` 的边界。

核心原则：

- module 只输出 `ContextFragment`。
- module 不做动作规划。
- module 不做 safety 裁剪。
- module 不调用 UAV tool。
- module 不直接修改 `MissionState`。
- module 必须解释自己输出字段。

## 2. 抽象接口

每个 module 至少需要两个能力：

```python
class ContextModule:
    name: str

    def build(self, input: StageContextInput, sources: MemorySourceBundle) -> ContextFragment:
        ...

    def describe_output(self) -> dict:
        ...
```

`build()` 生成本类别上下文。`describe_output()` 描述该 module 会在 `content`、`facts`、`inferences`、`goals`、`risks` 中放什么字段。

## 3. 默认模块

| Module | 默认职责 |
| --- | --- |
| `InstructionModule` | 当前指令、历史指令、semantic goal、phase goal |
| `ImageModule` | 最近 3 轮 RGB/depth metadata、artifact refs、图像可用性 |
| `PoseModule` | latest pose、最近 pose、pose delta |
| `ActionModule` | 最近 action、执行结果、失败记录 |
| `MapModule` | map/scene 相关线索、recall filter、reconfirmation risk |

默认 module 输出不包含完整 jsonl 历史，也不暴露原始大文件。

## 4. 字段自解释

`ContextModuleRegistry` 负责收集启用 module 的说明：

```json
{
  "module_content_fields": {
    "image": {
      "content.latest_rgb": "latest RGB frame metadata and artifact reference",
      "content.recent_rgb_refs": "RGB references from the last three observation rounds",
      "risks": "visual uncertainty and stale/mock-frame warnings"
    }
  }
}
```

禁用 module 不进入 `context_schema.module_content_fields`。这样模型不会看到不存在字段的说明。

### 4.1 Semantic Evidence 能力声明

启用的 module 可以在 `describe_output()` 中额外声明一组很短的
`semantic_evidence`：

```python
def describe_output(self) -> dict:
    return {
        "summary": "Optional UAV above-ground clearance; absolute pose remains debug-only.",
        "semantic_evidence": [
            "UAV ground clearance when calibrated; self-state, not target-relative geometry."
        ],
    }
```

该字段描述配置上可能提供的证据，不代表当前场景已经采集到数据，也不代表
其中的事实已经成立。`ContextModuleRegistry` 只汇总启用 module 的声明，并按
module 顺序去重和限制长度；它不会调用 `build()` 或启动场景。

汇总结果只用于生成每个 phase 的 semantic criteria，以及后续异步 semantic
update。Structure parser 不接收它，因为证据能力不能改变用户要求的 phase
数量、顺序、名称或 goal。该清单也不会写入 `SemanticPlan`，不会成为普通
planner context 字段，Validator 不解析其中的模态名称。

声明应采用“一项能力 + 一条关键限制”的形式。不要写当前 observation 的事实、
动作策略、benchmark 经验、内部 debug 字段或冗长使用教程。

## 5. 自定义 Image Module 示例

研究用户可以替换 image module：

```text
DefaultImageModule
  -> PaperImageHistoryModule
```

新 module 可以读取更长的 frame refs，使用自己的图像历史压缩方法，但输出仍必须是 `ContextFragment`。

它可以新增字段：

```json
{
  "content": {
    "visual_history_summary": "...",
    "selected_keyframes": [],
    "method": "paper_x_keyframe_selection"
  },
  "inferences": [],
  "risks": []
}
```

同时必须在 `describe_output()` 中解释这些字段。

## 6. 与 Planner 的边界

`ModelActionPlanner` 不知道具体 module 的内部实现。它只读取：

- `context_schema`
- `fragments`
- `allowed_action_constraints`
- `observation_policy`
- `phase_transition_policy`
- `safety_constraints`

动作是否合法由 `ToolContractValidator` 校验，不由 module 决定。

## 7. 与 Skills 的边界

后续视觉检测、目标识别等 skills 不在当前阶段实现。未来如果接入：

- skill 可以作为外部能力生成 artifact 或结构化结果。
- 对应 context module 再读取结果并输出 `ContextFragment`。
- 不应让 `ContextComposer` 为某个 skill 硬编码字段。
- 不应让 skill 返回值直接成为 planner action。

## 8. 验收标准

- 每个默认 module 都实现 `describe_output()`。
- `latest_context.json.context_schema.module_content_fields` 只包含启用 module。
- 自定义 module 能添加自己的字段说明。
- 禁用某 module 后，context 仍合法。
- module 输出不会绕过 `ToolContractValidator` 或 `SafetySupervisor`。
