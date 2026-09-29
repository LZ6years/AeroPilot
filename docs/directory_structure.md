# 当前目录结构

```text
src/
  agent/
    config/                    # Provider、Planner、ToolLoop 配置
    context/                   # history projection、request assembly、media hydration
    planning/                  # 模型 planner 与原生 tool-call 校验
  memory/
    context/
      contracts.py             # Context V2 与 MissionStateView
      builder.py               # Envelope / Checkpoint 构建
      documents.py
      registry.py
      modules/                 # skills、pose、motion、map、image
    stores/                    # append-only memory 与 artifact 索引
  plugins/                     # 插件契约、发现、Context 与 callable skills
  runtime/
    context/                   # checkpoint 发布与 model media store
    runners/                   # TaskRuntime、SmokeRuntime
    task/                      # session、loop、turn、physical window、journal
    factory.py                 # 唯一生产组合入口
  tools/                       # ToolSpec、registry、dispatcher、内建工具
  uav/config/                  # action 与 safety 配置
  env_client/                  # Runtime/Benchmark 使用的环境边界
  sim_server/                  # scene、backend 与进程管理
  adapters/                    # AirSim / UnrealCV 协议适配

tests/
  support/                     # 共享测试 fixture 与 fake
  diagnostics/                 # profiler/GPU/network/simulator 诊断
  manual/                      # 显式运行的实验脚本

docs/archive/                  # 历史文档，不是当前 API
runs/experiments/              # 手工实验默认输出（Git 忽略）
```

`plugins/knowledge` 保存可复用知识，`plugins/capabilities` 保存 callable skill，
`plugins/research` 保存实验性贡献。`memory/` 保存用户/地图记忆，不承担插件发现。
