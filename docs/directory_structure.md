# 目录结构说明

当前目录按职责分层：

```text
src/
  runtime/                  # CLI、AgentRuntime、run record
  env_client/               # C/S 中的 Client 抽象
  sim_server/               # C/S 中的 Server，管理 scene、端口、进程、backend
    process_manager.py      # 启动/停止 packaged UE 进程
    port_manager.py         # 分配本机端口
    scene_manager.py        # 管理 SceneInstance
    server.py               # SimulationServer 主入口
    backends/               # 按控制协议划分的 backend
      airvln_backend.py     # 当前 AirSim/UAV primitive backend，后续可改名 airsim_backend
      unrealcv_backend.py   # UnrealCV camera/object protocol backend
      local_ue_backend.py   # 本地 UE + AirSim 协议的兼容入口
  adapters/                 # 外部库薄封装，例如 AirSim UAV primitive
  tools/                    # AgentRuntime 到 EnvClient 的工具分发
  agent/                    # 后续 planner/safety/mission state
  cognition/                # 后续 VLM 上下文和输出 schema
  perception/               # 后续原始观测 artifact 管线

configs/
  maps/                     # MapPackage 配置

docs/                       # 架构和开发策略文档
tests/                      # 配置、EnvClient、SimulationServer、边界测试
runs/                       # 本地运行记录，通常不提交
```

## 命名原则

`provider` 描述地图资源形态：

- `packaged_unreal`: 已打包 UE 地图

`launch.method` 描述启动方式：

- `direct`: 本机直接启动 exe/sh
- `wsl_bash`: Windows 上通过 WSL 启动 Linux packaged scene
- `existing_process`: 地图已经在本机或远端机器运行
- `manual`: 人手动管理进程，测试时可 dry-run

`control.protocol` 描述通讯协议：

- `airsim`: AirSim UAV/ComputerVision RPC
- `unrealcv`: UnrealCV socket 命令

backend 应该按 `control.protocol` 扩展，而不是按每一类地图资源扩展。
