> Status: Historical

# 自然语言 CLI / TUI 设计计划

## 1. 当前目标

当前先做好自然语言 CLI，不急着做完整 TUI。目标是用户启动一次 UE/AirSim 后，可以在同一个会话里连续输入自然语言指令：

```text
flyagent> 向前飞行2米，然后悬停
flyagent> 向前飞行5米，然后悬停
flyagent> 向右转30度，然后悬停
flyagent> /status
flyagent> /context
flyagent> /quit
```

CLI 只负责交互外壳；飞行任务仍由 `TaskRuntime`、`ModelActionPlanner`、`SafetySupervisor` 和自动 observation 链路完成。

## 2. 当前入口

统一维护 runtime 下的 CLI：

```powershell
$env:PYTHONIOENCODING='utf-8'
flyagent start
```

开发兜底入口：

```powershell
F:\conda_env\flyagent\python.exe -m src.cli
```

CLI 正式实现位于 `src/cli/main.py`。`runtime` 只保留执行层，不再拥有用户命令行入口。

## 3. CLI v1 已有能力

- 启动或连接一个 scene，并在同一个 CLI 会话中复用该 scene。
- 每条自然语言输入调用 `TaskRuntime.run_task(..., scene_policy="reuse_running", stop_scene_on_finish=False)`。
- 每条指令结束后打印简短 summary：完成状态、执行工具、pose delta、RGB/depth artifact 数量、run 目录。
- 支持 slash commands：`/status`、`/context`、`/summary`、`/hover`、`/land`、`/quit`、`/help`。
- `/quit` 时统一执行 shutdown policy，并停止 scene。

核心代码：

- `src/cli/main.py`：命令行入口，包含 `start / chat / smoke-test / parse-task / run-task`。
- `src/runtime/runners/interactive_runtime.py`：连续自然语言会话层，负责 scene 复用、turn summary、best-effort hover 和 shutdown。
- `tests/test_cli_main.py`：CLI 和 slash command 测试。
- `tests/test_interactive_runtime.py`：不启动 UE 的 session 行为测试。

## 4. 当前规划链路

CLI 每收到一条自然语言指令后，链路是：

```text
用户指令
  -> TaskParser(mode="model")
  -> TaskRuntime 构造 PlannerContext
  -> ModelActionPlanner 只输出 UAV tool steps
  -> ToolContractValidator 校验工具契约
  -> SafetySupervisor 做最后安全审查
  -> runtime 执行动作
  -> runtime 自动采集 rgb/depth/pose
  -> 执行后模型判断 PhaseTransition
  -> runtime 应用阶段推进结果
```

注意：

- `ModelActionPlanner` 不再输出 `phase_transition`。
- 模型动作计划不能输出 `kind="observe"` 或 `env.observe`。
- observation 由 runtime 自动采集，当前策略是每个 phase window 执行后采样一次 `rgb/depth/pose`。
- 阶段是否完成由执行后的模型判断决定，输入包含新旧 pose、本轮执行结果、安全介入和最近历史。

## 5. 参考 Claude Code TUI 的边界

后续做 TUI 时可以参考 Claude Code 的结构，但不需要照搬 React/Ink：

```text
F:/科研/claude-code/claude-code-main/src/screens/REPL.tsx
F:/科研/claude-code/claude-code-main/src/components/PromptInput/PromptInput.tsx
F:/科研/claude-code/claude-code-main/src/components/Messages.tsx
F:/科研/claude-code/claude-code-main/src/components/StatusLine.tsx
F:/科研/claude-code/claude-code-main/src/components/CustomSelect/select.tsx
```

可借鉴的设计边界：

- REPL/screen 只编排状态、输入、消息列表和后台任务，不塞业务逻辑。
- PromptInput 独立管理输入缓冲、历史、快捷键、slash command。
- Messages 独立渲染用户指令、semantic plan、model action plan、safety decision、tool result、observation summary。
- StatusLine 独立显示 map、scene、connected、airborne、pose、phase、run id。
- Slash command 分成本地 UI 命令和真实任务命令；本地命令不进入模型。
- 长任务需要可取消的运行控制；UI 层只发出 cancel/hover/shutdown 请求，不直接改 planner 或 state。

## 6. 后续 TUI 分层

第一层：自然语言 session runner，已实现：

```text
src/runtime/runners/interactive_runtime.py
```

职责：

- 持有 `TaskRuntime`、scene lifecycle、last state、run dir。
- 提供 `submit_instruction(text)`、`status()`、`hover()`、`land()`、`shutdown()`。
- 返回结构化 `InstructionTurnSummary`，不负责 UI 渲染。

第二层：CLI shell，已统一到：

```text
src/cli/main.py
```

职责：

- 读取用户输入。
- 识别 slash commands。
- 调用 interactive runtime。
- 打印简短中文结果和关键 run 文件路径。

第三层：TUI shell，后续再做：

```text
src/tui/
```

建议组件：

- `AppScreen`：总编排。
- `PromptInput`：自然语言输入和 slash command。
- `MessageLog`：指令、计划、安全审查、动作结果、观察结果。
- `StatusPanel`：scene、airborne、pose、phase、run id、artifact count。
- `ArtifactPanel`：最新 RGB/depth 文件路径，后续可预览图像。
- `CommandPalette`：类似 Claude Code 的 picker，用于 `/map`、`/model`、`/shutdown-policy`。

## 7. CLI 验收标准

命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
flyagent start
```

输入：

```text
向前飞行5米，然后悬停
```

期望：

- 不需要用户手动传 `SceneInstance`。
- 模型输出合法 `uav.move_body`。
- run 中有 `state/semantic_plan.json`、`state/latest_state.json`、`state/flight_memory.json`、`contexts/latest_context.json`、`contexts/latest_action_planner_model_context.json`。
- run 中有 RGB/depth artifact。
- CLI 打印 pose delta 和 artifact count。
- `/quit` 能安全 hover 或 land，并停止 scene。

## 8. 暂不做

- 执行失败 repair。
- 视觉目标检测 skills。
- 多 UAV。
- Electron 或完整图形界面。
