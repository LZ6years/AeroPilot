# Demos 使用说明

`demos/` 只放用户可以体验或阅读的完整流程。底层诊断脚本放在 `tests/manual/`，离线评估工具放在 `tools/`。

## llm_context_demo.py

理解“保存上下文、组织每次请求、发送、把回答写回历史”的最小两轮 Demo。
默认离线 MOCK，无费用，不启动模拟器；显式 `--real` 才调用真实模型。

```powershell
python demos/llm_context_demo.py
```

先读 [两轮教学说明](llm_context_demo.md)，再对照运行目录中的
`round_01_request.json`、`round_02_request.json` 和
`memory/agent_transcript.jsonl`。不新增配方，不改变框架或飞行策略。

## plugin_system_demo.py

插件系统教学 demo。它不启动 UE，默认用一个固定 selector 展示插件如何从目录加载、如何选择 skill、如何把 skill 正文注入 `skills` context fragment，以及 research image context module 如何通过插件方式替换默认 image module。注意：默认模式不调用模型来选择 skill；如果 `.env` 有 API key 且 demo RGB 可读，research image 插件仍可能调用视觉模型生成 caption。

运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
python demos/plugin_system_demo.py
```

如果要看真实模型选择 skill：

```powershell
$env:PYTHONIOENCODING='utf-8'
python demos/plugin_system_demo.py --selector real
```

重点输出：

```text
runs/plugin_system_demo_<timestamp>/
  plugin_system_walkthrough.md
  01_config_snapshot.json
  02_discovered_skills.json
  03_selection_result.json
  04_loaded_skill_summaries.json
  05_latest_checkpoint.json
```

先读 `plugin_system_walkthrough.md`。它会按代码链路解释：

- `configs/plugins.json` 如何进入 `PluginRegistry`
- `plugins/knowledge/<plugin_name>/<skill_folder>/SKILL.md` 如何变成 skill catalog
- selector 如何选择 skill
- `SkillContentLoader` 如何展开正文
- `DefaultSkillsContextModule` 如何生成 `skills` fragment
- `ContextModuleRegistry` 如何装配默认 Context Module 与启用的插件模块

---

如果你想看目前设计的完整执行 pipeline，优先读：

```text
demos/real_engine_instruction_demo.py
```

它能说明顶层流程如何串起来；但真正的每一步内部逻辑不全在 demo 文件里，而是在 `TaskRuntime` 和各个 agent/runtime/memory 模块中。

## 当前 Demo

### memory_system_demo.py

本地 deterministic 记忆系统 demo。

运行：

```powershell
python demos/memory_system_demo.py
```

它做什么：

- 不调用真实模型。
- 不启动真实 UE。
- 不创建 Parser、Planner、Runtime 或模拟 Backend。
- 写入两条静态 instruction 和 pose fixture，观察 memory refs 与 context 的更新。

适合看：

- `FlightMemory` 如何保存 refs。
- `memory/*.jsonl` 如何保存结构化历史。
- `contexts/latest_checkpoint.json` 里的动态 fragments 如何组成。
- `image / pose / motion / map / skills` module 输出是否稳定。

重点输出：

```text
runs/memory_system_demo_<timestamp>/
  state/flight_memory.json
  memory/instructions.jsonl
  memory/observations.jsonl
  memory/poses.jsonl
  contexts/instruction_envelope.json
  contexts/latest_checkpoint.json
```

这些是静态教学记录，不表示真实动作、飞行轨迹或任务完成。

### model_action_planner_context_demo.py

真实模型 action planner 上下文 demo。

运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
python demos/model_action_planner_context_demo.py --env-path .env
```

它做什么：

- 调用真实模型。
- 构造 demo memory。
- 生成 `PlannerContext`。
- 调用 `ModelActionPlanner`。
- 保存模型输入、原始模型输出、validated plan、repair 结果和 safety decision。
- 不执行真实 UAV 动作。

适合看：

- 发给模型的 V2 envelope、checkpoint 与按时间排列的 Transcript。
- `schema_version`、fragment 与 audit 如何解释字段。
- 状态上下文不含重复工具定义，工具参数只经 Provider `tools` 进入模型输入。
- 模型输出的 action plan 是否通过 contract validation。
- `ModelRepairPlanner` 是否触发。
- `SafetySupervisor` 最终如何审查 plan。

重点输出：

```text
runs/model_action_planner_demo_<timestamp>/
  state/semantic_plan.json
  state/latest_state.json
  state/flight_memory.json
  contexts/instruction_envelope.json
  contexts/latest_checkpoint.json
  debug/model_payload.json
  debug/raw_model_response.json
  debug/raw_model_content.json
  debug/validated_plan.json
  debug/repair_attempts.json
  debug/repair_result.json
  debug/safety_decision.json
  demo/demo_preview.md
```

它不适合看真实 UAV 执行后的 pose delta，也不适合看真实 phase 完成判断。

### real_engine_instruction_demo.py

真实 UE + model + runtime 的完整闭环 demo。

运行单条指令：

```powershell
$env:PYTHONIOENCODING='utf-8'
F:\conda_env\flyagent\python.exe demos/real_engine_instruction_demo.py --env-path .env --instruction "向前飞行5米后向右转40度，最后再向前走2米" --save-artifacts --only 1
```

运行标准任务集：

```powershell
$env:PYTHONIOENCODING='utf-8'
F:\conda_env\flyagent\python.exe demos/real_engine_instruction_demo.py --env-path .env --save-artifacts --task-suite configs/eval/standard_tasks.json --suite-group single
```

它做什么：

- 读取模型配置和 UAV 配置。
- 启动或连接真实 UE/AirSim。
- 接收自然语言 instruction。
- 由 runtime 创建单任务状态和 Context V2 checkpoint。
- 用 `ModelActionPlanner` 生成原生 assistant/tool-call turn。
- 用 `ToolContractValidator` 校验动作契约。
- 用 `SafetySupervisor` 做最终安全审查。
- 通过 `ToolDispatcher` 执行 UAV 动作。
- 每个 motion step 后执行 stabilization。
- 每个 physical window 后自动采集 RGB/depth/pose。
- assistant turn 和 tool result 按真实顺序写入 AgentTranscript，再进入下一轮规划。
- 写出 run 目录和 demo summary。

适合看：

- 整个上层 pipeline 如何连接。
- 每一轮模型输出了哪些 tool calls。
- action 是否执行成功。
- 自动 observation 是否采集成功。
- `latest_checkpoint` 的动态状态是否与 state 同源。
- runtime-control 是否正确完成或停止任务。
- memory 是否随着多轮 instruction 增长。

重点输出：

```text
runs/real_engine_basic_instruction_<timestamp>/
  state/latest_state.json
  state/flight_memory.json
  memory/instructions.jsonl
  memory/actions.jsonl
  memory/observations.jsonl
  memory/poses.jsonl
  memory/frames.jsonl
  contexts/instruction_envelope.json
  contexts/latest_checkpoint.json
  debug/context_checkpoints/*.json
  debug/debug_events.jsonl
  debug/action_planner/*.json
  artifacts/rgb/
  artifacts/depth/
  demo/summary.json
```

如果跑完后要离线评估：

```powershell
& 'F:\conda_env\flyagent\python.exe' tools/real_run_evaluator.py --run-dir runs/<run_id> --task-suite configs/eval/standard_tasks.json --write
```

评估输出：

```text
runs/<run_id>/eval/report.md
runs/<run_id>/eval/metrics.json
```

## 如何读完整执行 Pipeline

第一层读 `demos/real_engine_instruction_demo.py`：

- `main()`：读取配置、启动 server、创建 runtime、循环执行 instruction。
- `make_runtime()`：把 `ModelActionPlanner`、`SafetySupervisor` 和工具依赖注入 `TaskRuntime`。
- `build_state_summary()`：从 run 输出中提取 action、observation 和 artifact 摘要。
- `build_session_summary()`：生成本次 demo 的聚合 summary。

第二层读 `src/runtime/runners/task_runtime.py`：

- 这里是长期 Runtime facade；它创建单任务 session、准备 scene，并调用 task loop。
- assistant turn 的分类、read-only 调度和 physical window 执行由 task 子模块协作完成。
- 任务是否完成由 runtime-control tool call 更新 MissionState，不由 demo 文件推导。

第三层按问题继续读：

```text
src/memory/context/builder.py
src/runtime/context/checkpoints.py
src/agent/context/request_assembler.py
```

看 V2 checkpoint 如何构建、去重、持久化并投影为 Provider 请求。

```text
src/agent/planning/model_action_planner.py
```

看模型如何根据 context 输出 action plan。

```text
src/agent/planning/tool_contract_validator.py
```

看 action tool 和 args 如何按 contract 校验。

```text
src/agent/safety.py
```

看安全限幅和安全拦截。

```text
src/runtime/control/stabilization.py
```

看每个 motion step 后如何等待 UAV 稳定。

## 结论

只读 `real_engine_instruction_demo.py` 可以理解“顶层 pipeline 怎么连起来”。  
如果要理解“每一步内部怎么做”，需要继续顺着它读 `TaskRuntime` 和相关模块。  
如果要看模型输入和动作规划细节，读 `model_action_planner_context_demo.py` 更直接。  
如果只想看 memory/context 的结构，读 `memory_system_demo.py` 更轻。
