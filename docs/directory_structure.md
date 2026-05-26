# 目录结构说明

项目按职责分层：Agent 不直接接触 UE、AirSim 或 UnrealCV；所有环境请求都经由 `EnvClient -> SimulationServer -> backend -> adapter`。

```text
src/
  runtime/                    # CLI、配置加载、日志和任务运行流程
    cli/
    configs/
    logging/
    runners/

  env_client/                 # C/S 中的 Client 抽象
  sim_server/                 # C/S 中的 Server，管理 scene、端口、进程和 backend
  adapters/                   # 外部 SDK / 协议适配封装
  tools/                      # AgentRuntime 到 EnvClient 的工具分发层

  agent/
    __init__.py
    mission_state.py          # 任务运行状态、phase 进度、动作/观测历史
    safety.py                 # 短期动作进入工具层前的安全监督
    semantic/                 # 用户指令语义理解：只描述目标和语义阶段
      schemas.py
      validator.py
      task_parser.py
      prompts/
    planning/                 # 短期动作规划：输出 1-3 个 UAV primitive step
      schemas.py
      model_action_planner.py
      prompts/

  memory/
    __init__.py
    flight_memory.py          # 当前 run/scene 的轻量飞行记忆 manifest
    context/                  # 给 planner/VLM 组装模型上下文
      schemas.py
      composer.py
      registry.py
      modules/
      custom_modules/
    stores/                   # JSONL、地图记忆、run memory 等持久化读写

  perception/                 # 原始观测、上下文构建和 artifact 管线

configs/
  maps/                       # 地图资源配置
  context_modules.json        # context module 开关和自定义模块配置
  uav_action_config.json      # UAV primitive 动作尺度配置
  uav_safety_config.json      # UAV 安全限制配置

evaluation/
  benchmarks/                 # 外部 benchmark harness，不属于主 runtime
    uav_flow/                 # UAV-Flow task/UnrealCV/reference path 评估适配

memory/                       # 项目级规则、用户偏好、地图记忆数据；不是源码包
docs/                         # 架构和开发策略文档
tests/                        # 单元测试和集成测试
runs/                         # 本地运行记录，通常不提交
```

## Agent 边界

- `agent.semantic` 负责把自然语言变成 `SemanticPlan`，不包含 primitive action。
- `agent.planning` 负责把当前 phase、上下文和工具约束变成短期动作计划。
- `mission_state.py` 保持顶层，因为它连接 runtime loop、phase、观测和动作历史。
- `safety.py` 保持顶层，因为它是 planner 和工具执行之间的统一安全门。

## Memory 边界

- `flight_memory.py` 保持顶层，表示当前 run/scene 内连续飞行的 memory manifest。
- `memory.context` 负责 planner context 的 schema、组装、模块注册和 fragment 生成。
- `memory.stores` 只负责持久化读写，不参与上下文排序和压缩策略。
- 根目录 `memory/` 保存运行时数据和规则，例如 `FLYAGENT.md`、`USER.md`、`map/map_memory.json`。

## Evaluation 边界

- `evaluation/benchmarks/` 放外部 benchmark 的 case loader、runner、metric adapter 和报告生成。
- benchmark 代码默认不被 CLI/runtime 导入，不应该影响主运行链路。
- UAV-Flow 这类任务集评估放在 `evaluation/benchmarks/uav_flow/`，而不是 `src/integrations/`。
- 如果某个 benchmark 中沉淀出通用环境能力，再单独下沉到 `src/env_client/` 或 `src/sim_server/backends/`。
