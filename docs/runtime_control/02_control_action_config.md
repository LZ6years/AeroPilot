# 02 控制动作配置

Runtime control action 必须支持显式开启和关闭。关闭的控制动作不应该出现在 model-facing context 中，也不应该被 planner 输出后静默执行。

## 1. 配置目标

配置层需要回答：

- runtime control 总开关是否开启？
- 哪些 control action 可用？
- 哪些可用动作允许暴露给 planner？
- 当前 CLI/TUI session 是否有临时覆盖？
- 某个动作被关闭时，已有 pending work 如何处理？

配置属于 runtime orchestration，不属于 reusable skill。

## 2. 建议文件

```text
src/runtime/control/config.py
configs/runtime_control.json
```

配置合并顺序建议为：

```text
默认配置
  -> 配置文件
  -> 环境变量覆盖
  -> CLI / TUI session 覆盖
```

session 覆盖默认只保存在内存中。除非用户明确要求保存偏好，否则不应改写配置文件。

## 3. 配置示例

```json
{
  "runtime_control_enabled": true,
  "actions": {
    "ask_user": {
      "enabled": true,
      "mode": "hover_then_pause",
      "expose_to_model": true
    },
    "semantic_plan_update": {
      "enabled": true,
      "mode": "phase_boundary_background_update",
      "expose_to_model": true,
      "max_pending_per_phase": 1
    }
  }
}
```

| 字段 | 含义 |
| --- | --- |
| `runtime_control_enabled` | 运行时控制面总开关 |
| `actions.*.enabled` | 单个 control action 是否允许运行 |
| `actions.*.mode` | runtime 执行模式，属于 runtime 内部默认语义 |
| `actions.*.expose_to_model` | 是否把动作名暴露给 planner |
| `max_pending_per_phase` | 同一 phase 内允许的后台任务数量 |

`mode` 不进入 model-facing context。模型只需要知道动作当前是否可请求，不需要每轮读取动作默认执行方式。

## 4. 环境变量覆盖

建议支持：

```text
FLYAGENT_RUNTIME_CONTROL_ENABLED=0
FLYAGENT_ASK_USER_ENABLED=0
FLYAGENT_SEMANTIC_PLAN_UPDATE_ENABLED=0
```

环境变量适合自动测试、批量运行、诊断脚本等场景。比如没有用户回复通道的批量运行，应禁用阻塞式 `ask_user`。

## 5. CLI / TUI 显式开关

交互层应该能展示和切换当前 session 的控制动作状态。

建议 slash commands：

```text
/control
/control ask_user on
/control ask_user off
/control semantic_plan_update on
/control semantic_plan_update off
/control all on
/control all off
```

`/control` 输出示例：

```text
运行时控制动作:
- ask_user: 已开启，提问前悬停并暂停任务
- semantic_plan_update: 已开启，在窗口边界应用后台更新
```

TUI 状态栏可以显示：

```text
控制: ask_user=开, semantic_update=开
```

## 6. 配置到模型上下文的转换流程

只有满足以下条件的动作才能出现在 `runtime_control.available`：

1. runtime control 总开关开启。
2. 单个 action 的 `enabled=true`。
3. 单个 action 的 `expose_to_model=true`。
4. 当前运行模式支持该 action，例如 interactive session 支持 `ask_user`，无人值守 batch session 默认不支持。

转换流程：

```text
配置文件 / 环境变量 / session 覆盖
  -> runtime 合成已开启能力
  -> context builder 只输出可暴露的动作名
  -> planner 只能请求 context 中出现的 action
```

model-facing context 示例：

```json
{
  "runtime_control": {
    "available": ["ask_user", "semantic_plan_update"]
  }
}
```

如果存在尚未应用的语义更新任务，只暴露短状态：

```json
{
  "runtime_control": {
    "available": ["ask_user", "semantic_plan_update"],
    "status": {
      "semantic_plan_update": "pending"
    }
  }
}
```

如果 `ask_user` 被关闭，就完全不要输出 `ask_user`。不要输出 `enabled=false`，因为这仍然会把不可用能力教给模型。

不要暴露完整配置、环境变量名、debug 状态、pending job 细节、session override 内部结构、动作触发规则或动作默认执行模式。

## 7. 触发规则放在哪里

触发规则应该放在 planner prompt 契约和对应 action 文档中，不作为每轮 model-facing context 重复输出。

模型只需要从 context 里知道：

- 当前哪些 control action 可请求。
- 是否已有与当前决策有关的短状态。
- 是否有用户澄清摘要需要优先尊重。

模型不需要每轮看到：

- `ask_user` 的完整安全流程。
- `semantic_plan_update` 的后台执行细节。
- 何时应该使用某个动作的长规则列表。
- runtime 内部配置和版本保护字段。

## 8. 关闭后的行为

如果 planner 输出了已关闭的 control action：

1. validator 拒绝 planner 输出。
2. repair planner 可以要求模型改成普通 UAV action、悬停、继续当前 phase，或者不输出该 control action。
3. runtime 不能静默执行已关闭的动作。

如果某个 control action 在已有 pending work 后被关闭：

- `ask_user`：不再启动新的阻塞式询问。
- `semantic_plan_update`：不再提交新的后台任务。
- 已经完成但尚未应用的 semantic update 在应用前必须再次检查配置。如果此时已关闭，应丢弃并记录 reason 为 `disabled_before_apply`。

## 9. 默认策略

第一版建议默认：

```text
runtime_control_enabled: true
ask_user.enabled: 交互式 session 默认可开启，batch/eval 默认关闭
semantic_plan_update.enabled: 默认关闭，等实现稳定后再按 session 开启
```

原因：

- `ask_user` 需要用户回复通道；没有交互入口时不能阻塞等待。
- `semantic_plan_update` 会改变 runtime 语义状态；第一版应先通过显式开关控制风险。

## 10. 后续测试建议

建议覆盖：

- 配置文件开启两个动作。
- 环境变量关闭 runtime control 总开关。
- 环境变量只关闭 `ask_user`。
- session override 关闭动作但不写配置文件。
- 关闭动作不会出现在 model-facing context。
- planner 输出关闭动作会被拒绝。
- pending semantic update 在应用前发现关闭后会被丢弃。
- model-facing context 示例不包含完整配置、触发规则或动作执行模式。
