# 目录结构说明

项目按职责分层：Agent 不直接操作 UE、AirSim 或 UnrealCV，环境请求统一经过 `EnvClient -> SimulationServer -> backend -> adapter`。插件资产也不放在 `memory/` 里，而是放在根目录 `plugins/`。

```text
src/
  runtime/                    # CLI、配置加载、日志和任务运行流程
  env_client/                 # Client 抽象，面向工具层提供环境操作入口
  sim_server/                 # Server，管理 scene、端口、进程和 backend
  adapters/                   # 外部 SDK / 协议适配封装
  tools/                      # AgentRuntime 到 EnvClient 的工具分发层

  agent/
    semantic/                 # 用户指令语义理解，生成 SemanticPlan
    planning/                 # 短期动作规划，输出 UAV primitive step
    mission_state.py          # 任务运行状态、phase 进度、动作和观察历史
    safety.py                 # planner 和工具执行之间的安全监督

  memory/
    flight_memory.py          # 当前 run/scene 的轻量 memory manifest
    context/                  # 给 planner/VLM 组装模型上下文
      schemas.py
      composer.py
      registry.py
      modules/                # 默认 context module：instruction/skills/image/pose/action/map
    stores/                   # JSONL、地图记忆、run memory 等持久化读写

  plugins/                    # 插件发现、注册、选择、加载和贡献类型定义
    contracts/
    core/
    context/
    skills/

  perception/                 # 原始观察、上下文构建和 artifact 管线

plugins/
  knowledge/                  # 稳定可复用飞行知识库
    uav_flight/<skill_folder>/SKILL.md
    communication/<skill_folder>/SKILL.md
    safety/<skill_folder>/SKILL.md
  research/                   # 实验性插件能力
    multimodal_image/context_modules/image.py

configs/
  plugins.json                # 插件根目录、启用插件、context module slot 配置
  maps/                       # 地图资源配置
  uav_action_config.json      # UAV primitive 动作尺度配置
  uav_safety_config.json      # UAV 安全限制配置

evaluation/
  benchmarks/
    uav_flow/                 # UAV-Flow task、UnrealCV、reference path 和报告适配

memory/                       # 项目级规则、用户偏好、地图记忆；不是插件目录
docs/                         # 架构和开发策略文档
tests/                        # 单元测试和集成测试
runs/                         # 本地运行记录，通常不提交
```

## Plugin 边界

- `plugins/knowledge/` 放可复用的 Markdown skills，由模型从 `name + description` listing 中选择是否加载。
- `plugins/research/` 放实验性 context module，例如 `research.multimodal_image`。
- `src/plugins/` 只负责发现、校验、注册和加载插件贡献，不存放具体飞行知识正文。
- `src/memory/context/` 只负责把插件贡献装配成 planner context fragment，不负责插件发现规则。
- 旧式 `src/memory/context/custom_modules` 不再保留。

## Memory 边界

- `memory/FLYAGENT.md` 是 Agent 长期行为规则。
- `memory/USER.md` 是用户偏好。
- `memory/map/` 是地图记忆。
- `runs/*/memory/` 是单次运行记录。
- 可复用飞行技能和任务知识统一放在 `plugins/knowledge/`。

## Evaluation 边界

- `evaluation/benchmarks/` 放外部 benchmark 的 case loader、runner、metric adapter 和报告生成。
- benchmark 代码默认不被主 runtime 导入，不应该影响主运行链路。
- 如果某个 benchmark 中沉淀出通用环境能力，再单独下沉到 `src/env_client/` 或 `src/sim_server/backends/`。
