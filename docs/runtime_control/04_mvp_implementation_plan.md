# 04 Runtime Control 全阶段实施路线

本文档是 Runtime Control Actions 的实现主路线图。它把能力从“只存在于文档里的设计”拆成四个可验收阶段：先建立契约和边界，再实现语义计划异步更新，再实现用户澄清交互，最后做边界加固和测试收敛。

总原则是：runtime control 是运行时控制面，不是飞行动作面。它可以影响任务编排、计划更新、用户交互和等待状态，但不能绕过飞行安全、不能进入 `ToolDispatcher`，也不能让 reusable skill 感知 runtime 私有结构。

## 1. 总目标

最终链路应当是：

```text
稳定契约说明控制动作的含义和输入形状
短上下文告诉模型当前能请求什么、是否已有短状态
planner 在 control_actions 中请求运行时控制动作
runtime validator 校验请求是否允许
executor 接受、拒绝、延后或提交后台任务
event / memory 记录执行结果
context builder 将必要结果压缩给下一轮模型
```

核心边界：

- `steps` 只放真实 UAV / skill action。
- `control_actions` 只放 runtime orchestration action。
- `runtime_control.available/status` 是短上下文，不放长规则、配置细节、后台任务细节或 debug 字段。
- capability contract 告诉模型 action 是什么、payload 怎么写、runtime 默认如何处理。
- planner prompt 告诉模型什么时候适合请求，什么时候不适合请求。
- validator 保证模型请求不等于执行。
- 用户回复优先复用现有 instruction / conversation memory，不新增专门的 clarification context module。

## 2. 阶段一：控制动作基础设施

阶段目标：只打通 contract / config / schema / context / validator，不改变真实飞行行为。

### 完成内容

- planner schema 能解析和 round-trip `control_actions`。
- prompt contract 说明 `control_actions` 的含义、可用动作、payload 位置和输出规则。
- runtime config 能显式开启、关闭、暴露 runtime control action。
- context 能暴露极简 `runtime_control.available/status`。
- runtime executor 能拒绝未知、关闭、未暴露、payload 非法或本阶段未实现的 action。
- event 能记录控制动作请求、拒绝、接受和原因。

### 修改文件

```text
src/agent/planning/schemas.py
src/runtime/configs/runtime_control_config.py
src/runtime/configs/__init__.py
src/runtime/control/registry.py
src/runtime/control/executor.py
src/runtime/control/__init__.py
src/runtime/context/phase_window_context.py
src/runtime/runners/task_runtime.py
src/agent/planning/prompts/coupled_phase_decision_system.txt
src/agent/planning/prompts/coupled_action_planner_system.txt
tests/test_runtime_control_config.py
tests/test_runtime_control_actions.py
tests/test_model_action_planner.py
tests/test_phase_window_context.py
tests/test_prompts.py
tests/test_boundaries.py
```

### 文件边界

- `schemas.py`：新增 `RuntimeControlAction`，给 `CoupledActionPhasePlan` 增加 `control_actions` 字段；保持 `steps` 与 `control_actions` 分离；不在 schema 中写具体执行策略。
- `runtime_control_config.py`：定义 runtime control 总开关、action 开关、暴露开关和环境覆盖；默认不暴露任何 action，避免行为突变。
- `registry.py`：定义 `ask_user` 和 `semantic_plan_update` 的稳定 contract，包括名称、简短含义和 payload 必需字段；不读取运行时状态。
- `executor.py`：校验 action 是否存在、是否开启、是否暴露、payload 是否合规；阶段一只返回 accepted / rejected / not_implemented 等结果，不执行真实控制动作。
- `phase_window_context.py`：向 planner payload 注入极简 `runtime_control`；没有可用 action 时可省略该字段或输出空 `available`。
- `task_runtime.py`：planner 返回后先校验 `control_actions` 并记录事件；不改变 UAV steps 的安全检查和执行路径。
- planner prompts：写入 capability contract、请求时机原则和 `control_actions` 输出规则；明确 control action 不能写进 `steps`。
- tests：覆盖配置默认关闭、上下文极简、schema round-trip、未知 action 拒绝、关闭 action 拒绝、prompt 边界。

### 不做什么

- 不真正执行 `ask_user` 等待用户回复。
- 不真正执行 `semantic_plan_update` patch。
- 不让 control action 进入 `ToolDispatcher`。
- 不新增 `src/memory/context/modules/runtime_control.py`。
- 不新增 `src/memory/context/modules/clarifications.py`。
- 不改 `src/plugins/skills/selection.py`。
- 不为单个任务类型写专用触发逻辑。

### 验证

```text
pytest tests/test_runtime_control_config.py
pytest tests/test_runtime_control_actions.py
pytest tests/test_model_action_planner.py
pytest tests/test_phase_window_context.py
pytest tests/test_prompts.py tests/test_boundaries.py
```

验收标准：

- 默认配置下没有真实行为变化。
- planner 输出空 `control_actions` 时，原有任务路径不变。
- planner 输出未知或关闭 action 时，runtime 拒绝并记录原因。
- model-facing context 只包含可用动作名和必要短状态。
- `control_actions` round-trip 后不会进入 `steps`。

## 3. 阶段二：semantic_plan_update 异步闭环

阶段目标：实现 `semantic_plan_update` 的异步更新闭环，同时不阻塞无人机飞行。

### 完成内容

- planner 可在 `control_actions` 中请求 `semantic_plan_update`。
- 当前 phase window 继续基于旧 semantic plan 输出并执行安全飞行动作。
- 当前窗口动作和自动 observation 完成后，runtime 提交后台 semantic update job。
- 后续 phase/window 边界轮询 job 状态。
- job 完成后校验 `run_id + phase_key + semantic_plan_version`。
- patch 合法且未过期时应用到当前 semantic plan，并递增版本。
- patch 过期、非法、配置关闭、任务结束或 phase 已推进时丢弃。

### 修改文件

```text
src/agent/semantic/plan_updater.py
src/agent/semantic/prompts/semantic_plan_update_system.txt
src/agent/semantic/validator.py
src/agent/mission_state.py
src/runtime/control/actions/semantic_plan_update.py
src/runtime/control/executor.py
src/runtime/runners/task_runtime.py
src/runtime/context/phase_window_context.py
tests/test_semantic_plan_update.py
tests/test_task_runtime.py
tests/test_model_action_planner.py
tests/test_prompts.py
tests/test_boundaries.py
```

### 文件边界

- `plan_updater.py`：根据当前用户指令、semantic plan、当前 phase、最近执行摘要、当前 observation 摘要和 planner reason 生成 semantic patch；不自由重写整份 plan。
- `semantic_plan_update_system.txt`：约束 updater 只输出当前 phase patch，可选追加一个通用 recovery phase；不写评测专用策略。
- `validator.py`：校验 patch 不包含 primitive tool、不改写用户原始指令、不越权修改整份 plan、不引入私有 runtime 字段。
- `mission_state.py`：维护 `semantic_plan_version`，用于过期更新防护。
- `actions/semantic_plan_update.py`：管理后台 job、pending 状态、结果、版本校验和应用前检查。
- `executor.py`：提交、轮询、应用或丢弃 semantic update；应用前再次检查配置和版本。
- `task_runtime.py`：当前窗口继续飞行；动作和自动 observation 后提交 job；后续边界应用合法 patch。
- `phase_window_context.py`：pending 期间只暴露短状态，例如 `runtime_control.status.semantic_plan_update = "pending"`，不暴露 job 细节。

### patch 范围

允许更新：

- 当前 phase 的 `goal`
- 当前 phase 的 `done_when`
- 当前 phase 的 `continue_when`
- 可选追加一个通用 recovery / fallback phase

不允许更新：

- 原始用户指令
- 评测专用字段或隐藏真值
- UAV primitive tool 名称
- 整份 plan 的自由重写
- 为单个样例定制的完成条件

### 不做什么

- 不阻塞当前飞行动作等待 semantic update。
- 不实现 `ask_user` 的 UI 等待、用户回复保存和任务恢复。
- 不绕过 phase decision、runtime loop guard 或 safety。
- 不把 updater 输出直接当作 planner action。

### 验证

```text
pytest tests/test_semantic_plan_update.py
pytest tests/test_task_runtime.py
pytest tests/test_model_action_planner.py
pytest tests/test_prompts.py tests/test_boundaries.py
```

验收标准：

- pending update 不阻塞飞行动作。
- stale update 不覆盖已经推进的 phase 或更新后的 semantic plan。
- 非法 patch 被拒绝并记录原因。
- 应用后的 semantic plan 仍能通过 validator。
- model-facing context 只暴露短 pending 状态。

## 4. 阶段三：ask_user 真实交互

阶段目标：实现 `ask_user` 的真实用户澄清闭环，只在 interactive session 可用。

### 完成内容

- interactive session 中允许 runtime 暴露 `ask_user`。
- batch / eval / non-interactive session 默认不暴露 `ask_user`。
- planner 可在目标选择、关键参数缺失、安全风险或长期无法确定下一步时请求 `ask_user`。
- runtime 接受后先执行安全 hover 或确认当前稳定状态。
- runtime 发出用户可见的澄清请求事件，并将任务状态置为等待用户。
- 用户回复后写入现有 instruction / conversation memory，标记为 clarification。
- 下一轮 context 只暴露 1-3 条简短 clarification 摘要，不回放完整对话。
- task loop 从等待状态恢复，重新进入 planner。

### 修改文件

```text
src/runtime/control/actions/ask_user.py
src/runtime/control/executor.py
src/runtime/runners/task_runtime.py
src/memory/flight_memory.py
src/memory/context/modules/instruction.py
src/memory/context/composer.py
src/runtime/session/interactive_runtime.py
src/runtime/events.py
tests/test_interactive_runtime.py
tests/test_runtime_control_actions.py
tests/test_task_runtime.py
tests/test_phase_window_context.py
tests/test_prompts.py
tests/test_boundaries.py
```

### 文件边界

- `actions/ask_user.py`：处理提问 payload、用户可见消息、等待状态和恢复信号；不负责飞行动作调度。
- `executor.py`：校验 `ask_user` 是否在当前 session 可用；拒绝 non-interactive 场景中的请求。
- `task_runtime.py`：在 accepted `ask_user` 后进入安全悬停和等待流程；恢复后重新构建 context。
- `flight_memory.py`：复用现有 instruction / conversation memory 保存 clarification；只需要能区分“用户澄清”与普通历史，不新增专门 context module。
- `instruction.py`：把有效 clarification 压缩到当前指令相关上下文中；只暴露短摘要和影响，不暴露完整对话。
- `composer.py`：保持上下文顺序稳定，避免 clarification 抢占核心飞行证据。
- `interactive_runtime.py`：提供用户回复入口和任务恢复入口；batch/eval 不实现等待用户通道。
- `events.py`：定义用户可见事件和 debug 事件的边界。

### 用户回复保存方式

用户回复不伪装成 tool result，也不直接覆盖原始指令。推荐保存为 instruction / conversation memory 中的一条 clarification 记录，至少包含：

```text
answer_summary: 用户回复的简短摘要
effect: 对后续规划的影响
source: user_clarification
created_at: 记录时间
```

下一轮 model-facing context 中只出现类似：

```json
{
  "active_clarifications": [
    {
      "answer_summary": "用户确认优先靠近左侧目标",
      "effect": "后续目标选择应优先左侧目标"
    }
  ]
}
```

### 不做什么

- 不在 batch/eval 默认启用用户提问。
- 不把用户回复当成外部 tool result。
- 不把完整对话历史塞回 planner。
- 不新增 `clarifications.py` context module，除非后续复杂度证明现有 instruction module 无法承载。
- 不允许模型因为懒得规划而提问。

### 验证

```text
pytest tests/test_interactive_runtime.py
pytest tests/test_runtime_control_actions.py
pytest tests/test_task_runtime.py
pytest tests/test_phase_window_context.py
pytest tests/test_prompts.py tests/test_boundaries.py
```

验收标准：

- interactive session 中可请求并进入等待用户状态。
- non-interactive session 中请求会被拒绝。
- 进入等待前无人机处于安全 hover 或稳定状态。
- 用户回复被压缩成 clarification 摘要，并在下一轮 context 中可见。
- 原始用户指令不被覆盖。

## 5. 阶段四：边界加固与收敛

阶段目标：收敛边界、测试、文档和 debug/report，使 runtime control 成为通用 Agent 能力，而不是个别任务补丁。

### 完成内容

- 梳理所有 prompt，确认 runtime control 规则只在 planner / updater contract 中出现。
- 梳理 reusable skill，确认 skill 不引用 runtime control schema、context 字段或评测经验。
- 梳理 event/report，区分用户可见事件和 debug 细节。
- 增加静态边界测试，防止评测专用词、隐藏真值、路径经验、指标经验进入 prompt / skill / runtime。
- 增加回归测试，确保关闭 runtime control 后行为等同于旧路径。
- 更新文档，保持实现文件和架构说明一致。

### 修改文件

```text
tests/test_boundaries.py
tests/test_prompts.py
tests/test_runtime_control_actions.py
tests/test_task_runtime.py
tests/test_interactive_runtime.py
docs/runtime_control/README.md
docs/runtime_control/01_architecture.md
docs/runtime_control/02_control_action_config.md
docs/runtime_control/03_model_contract_and_context.md
docs/runtime_control/actions/ask_user.md
docs/runtime_control/actions/semantic_plan_update.md
```

### 文件边界

- `tests/test_boundaries.py`：扫描 prompt、skill、runtime docs 和 reusable docs 的禁用词与越界引用。
- `tests/test_prompts.py`：确认 planner prompt 只写通用 contract，不写样例修复策略。
- `tests/test_runtime_control_actions.py`：覆盖 action 注册、关闭、暴露、拒绝、pending、完成和丢弃。
- `tests/test_task_runtime.py`：覆盖默认关闭不改变行为、semantic update 不阻塞、ask_user 等待恢复。
- docs：只解释架构和通用流程，不写具体评测样例经验。

### 不做什么

- 不扩大 `semantic_plan_update` 到整份 plan 自由重写。
- 不把 runtime control 逻辑下沉到 skill selection。
- 不把 debug job 细节暴露给 model-facing context。
- 不让 evaluation 代码影响正常 runtime。

### 验证

```text
pytest tests/test_boundaries.py
pytest tests/test_prompts.py
pytest tests/test_runtime_control_actions.py
pytest tests/test_task_runtime.py
pytest tests/test_interactive_runtime.py
```

验收标准：

- runtime control 默认关闭时，旧任务路径保持一致。
- enabled 但未 exposed 的 action 不进入 model-facing context。
- model-facing context 保持短小，不出现配置细节、长规则或后台 job 细节。
- prompt / skill / runtime 中没有评测专用修复策略。
- 用户可见事件不包含 schema/debug 细节。

## 6. 全局非目标

- 不新增 `runtime_control.py` / `clarifications.py` context module，除非后续复杂度证明需要拆分。
- 不改 `src/plugins/skills/selection.py`。
- 不让 `ToolDispatcher` 执行 runtime control action。
- 不让 reusable skill 感知 runtime control schema。
- 不把评测经验、样例名、参考路径或指标结果写进 prompt、skill、context 或 runtime。
- 不为单个任务类写专门控制逻辑。

## 7. 文档静态验证

每次修改本目录文档后，建议做静态检查：

```text
UTF-8 读取 docs/runtime_control/**/*.md
扫描常见中文乱码标记
扫描旧 capability 字段名
扫描冗长 runtime control JSON 字段
扫描评测专用词、指标词和路径词
```

本路线图只定义实施顺序和文件边界。真正代码实现应按阶段分批提交，每一阶段都要能独立回滚、独立验收。
