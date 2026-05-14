# 04 历史压缩策略

历史压缩的目标是服务飞行决策：让模型知道当前状态、刚才做过什么、动作效果如何、有哪些风险，而不是保存完整流水账。

第一版压缩策略不写死在 `ContextComposer` 里，而是由可注册的上下文模块分别负责。这样科研用户可以替换某一类数据的压缩方法，例如只替换 RGB/depth 图像历史记忆方式。

压缩/编排的职责边界：

- `FlightMemory`、`MapMemory`、`RunMemory` 保存历史数据和线索。
- `ContextModule` 从历史数据和当前 phase 输入中抽取本类别信息。
- `ContextModule` 只生成本类别 `ContextFragment`，不做跨模块 phase completion 判断。
- `ContextComposer` 只合并、排序、裁剪 fragments，并加入规则/偏好/安全约束。
- 后续 `PhaseCompletionEvaluator` 可以独立消费 fragments 判断 phase 是否完成，但不应把判断逻辑硬编码在 `ActionContextModule` 内。

## 1. 三层历史

```mermaid
flowchart TD
    A["memory/*.jsonl + artifacts<br/>原始历史和大文件引用"] --> B["Module Selected Window<br/>模块按 refs 读取短窗口"]
    A --> C["Rolling Summary<br/>指令/姿态/动作效果/风险摘要"]
    A --> D["Long-term Memories<br/>MapMemory / RunMemory"]

    B --> E["Context Modules"]
    C --> E
    D --> F["Recall Selection"]
    F --> E
    E --> G["PlannerContext"]
```

对应模块：

```text
InstructionContextModule 处理指令历史压缩
ImageContextModule 处理 RGB/depth 历史压缩
PoseContextModule 处理姿态历史压缩
ActionContextModule 处理动作效果压缩
MapContextModule 处理地图记忆召回与摘要
```

### Module Selected Window

Manifest 版本中不再由 `FlightMemory` 保存原始短窗口。`FlightMemory` 只保存 refs；各 `ContextModule` 从 `memory/*.jsonl` 和 artifacts 读取自己需要的短窗口并输出 `ContextFragment`。

保留最近 2-3 次 observation/action metadata。

适合保存：

- 最新 pose / velocity / attitude。
- 最近 RGB/depth 的尺寸、时间戳、artifact ref。
- 最近动作 tool、args、ok/error。

不保存：

- 完整图像字节。
- 完整 depth 矩阵。
- 很久以前的动作流水。

短窗口的默认策略由各模块各自实现。自定义模块可以读取 refs 指向的 JSONL 记录，但输出仍必须是可审计的 `ContextFragment`，而不是直接输出动作计划。

### Rolling Summary

持续压缩历史信息，保留高价值结论。

建议字段：

```json
{
  "instruction_summary": "连续执行过两条向前飞行指令",
  "pose_summary": "最近一次 forward 后水平位移约 1.8m，yaw 基本稳定",
  "image_summary": "最近 RGB/depth 可用，未做明确目标识别",
  "action_effect_summary": "uav.move_body forward duration=2.0 执行成功，但实际位移小于 5m",
  "risk_summary": "hover 后仍可能存在短时残余速度"
}
```

默认模块策略：

- `DefaultInstructionContextModule`：保留 active instruction 和最近历史指令摘要。
- `DefaultImageContextModule`：保留最新 RGB/depth metadata、artifact refs 和简单图像历史摘要。
- `DefaultPoseContextModule`：保留 latest pose、最近 pose delta、velocity、reset_epoch。
- `DefaultActionContextModule`：保留最近动作执行事实、动作效果摘要和失败原因。
- `DefaultMapContextModule`：召回当前 map/scene 相关线索，并标注 confidence 和 reconfirmation。

可替换策略：

- 用户可以只替换 `ImageContextModule`，使用论文中的图片历史压缩方法。
- 替换 image 模块不影响 instruction、pose、action、map 的默认逻辑。
- 自定义模块输出必须保留来源、置信度或 metadata，便于复盘和安全审查。

### Artifacts

完整数据留在 run 目录：

```text
observations/
memory/instructions.jsonl
memory/actions.jsonl
memory/poses.jsonl
memory/observations.jsonl
memory/frames.jsonl
contexts/
events.jsonl
mission_state.json
memory/artifacts/
```

模型只看到引用和摘要，需要复盘时再读 artifact。

`flight_memory.json` 属于运行时快照 artifact。它可以帮助开发者确认 FlightMemory 的内容，但不是 planner 的直接输入；planner 应只消费 `PlannerContext`。

## 2. 信息优先级

进入 planner context 的信息按优先级排序：

1. 当前 instruction 与目标。
2. 最新 pose / velocity / attitude。
3. 最近动作效果，例如 “forward 2s 后水平位移约 1.8m”。
4. 最近图像/depth 是否可用、是否稳定。
5. 历史指令摘要。
6. map memory 相关地标线索。
7. 用户偏好和系统规则摘要。

## 3. Budget v1

第一版先使用固定默认 budget，后续可让用户在 `configs/context_modules.json` 中配置。当前文档只定义默认优先级和设计约束，不实现复杂动态 token 估算。

```json
{
  "budget_defaults": {
    "instruction": {"priority": 100, "max_items": 5},
    "pose": {"priority": 95, "max_items": 5},
    "action": {"priority": 90, "max_items": 8},
    "image": {"priority": 80, "max_items": 3},
    "map": {"priority": 60, "max_items": 5}
  }
}
```

默认原则：

- `instruction` / `pose` / `action` 优先级最高。
- `image` 默认只保留最近少量 metadata、artifact refs 和摘要。
- `map` 是线索，不优先于当前 observation。
- 自定义模块应声明 `budget.priority` 和 `budget.tokens_estimate`。
- 用户后续可以关闭某个模块，或调整模块 budget 来减少 token。

## 4. 压缩触发点

建议触发：

- 每次 action 执行后。
- 每次 observation 更新后。
- 每条 instruction 更新 active_instruction 时。
- reset pose / restart scene / switch map 时。
- continuous session 结束时。

## 5. 压缩输出规范

压缩输出应避免模糊词，优先写可核验内容：

```text
不好：刚才飞得还可以。
好：最近一次 uav.move_body forward duration=2.0 执行 ok，pose delta 约 dx=1.55, dy=0.89。
```

压缩输出必须区分：

- 观测事实。
- 模型推测。
- 地图线索。
- 用户目标。
- 安全风险。

插件模块输出也必须遵守这个规范。模块可以改变压缩算法，但不能降低可审计性。

## 6. 过期策略

第一版不做复杂 TTL，但要按范围处理：

| 信息 | 过期条件 |
| --- | --- |
| Module Selected Window | 由各模块按 refs 读取和裁剪 |
| Pose/action rolling summary | reset pose 后刷新 |
| Image summary | 新图像更新后替换或合并；自定义 image 模块可扩展 |
| MapMemory | 切换 map 后不召回；长期保留但需 reconfirm |
| RunMemory | 长期保留，主要用于复盘 |
## 7. Raw Store 与压缩职责

第一版需要明确区分“原始历史保存”和“上下文压缩”：

```text
memory/*.jsonl / artifacts:
  保存原始历史、索引和大文件引用。

flight_memory.json:
  保存当前 run 的 manifest / index / recent refs。

ContextModule:
  读取自己需要的 refs，执行本类别压缩，输出 ContextFragment。

ContextComposer:
  合并 fragments，不读取完整 raw store，不创造新事实。
```

因此，`FlightMemory` 层不应该做这些事：

- 不用 `pose_ref: same_as_previous_observation` 这类去重字段替换原始 pose。
- 不裁剪 action result 中的 `data.pose` 等原始结果。
- 不把 RGB/depth 历史预先总结成固定摘要。
- 不替 image/pose/action/map 模块决定保留几条。

默认窗口建议：

| 层 | 默认窗口 | 说明 |
| --- | --- | --- |
| `flight_memory.json` refs | 50 条左右 | 给 module 留足近期材料 |
| `memory/*.jsonl` | append-only | run 内完整记录，必要时后续做归档 |
| `InstructionContextModule` | 5 条左右 | 进入 planner context 的短摘要 |
| `ActionContextModule` | 5 条左右 | 进入 planner context 的动作效果摘要 |
| `PoseContextModule` | 3 条左右 | 进入 planner context 的姿态短窗口 |
| `ImageContextModule` | 3 次 observation 左右 | 进入 planner context 的图像/depth metadata |

科研用户如果要使用论文中的图像历史记忆方法，应该替换 `ImageContextModule`，并从 `observations.jsonl` / `frames.jsonl` / `artifacts` 读取更长窗口，而不是要求 `FlightMemory` 预先做图像压缩。
