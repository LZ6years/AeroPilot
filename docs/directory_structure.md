# 目录结构说明

当前目录按职责分层。核心原则是：Agent 不直接碰 UE、AirSim 或 UnrealCV；所有环境请求都走 `EnvClient -> SimulationServer -> backend -> adapter`。

```text
src/
  runtime/                    # Agent 运行时、CLI、配置加载、日志和运行流程
    cli/
      main.py                 # 命令行入口：smoke-test、远程 AirSim 测试
    configs/
      map_config.py           # 加载 MapPackage / PortPolicy
      action_config.py        # 加载 UAV 原子动作速度、时长、角度尺度
    logging/
      async_run_logger.py     # 异步写 system.log、events.jsonl、artifact
      run_store.py            # 创建每次运行独立 run 目录
      record_writer.py        # 旧版同步记录器，保留兼容
    runners/
      agent_runtime.py        # 通过 EnvClient 执行 smoke/observation 流程

  env_client/                 # C/S 中的 Client 抽象
    base.py
    local_client.py           # 本地伪 C/S：直接调用 LocalSimulationServer
    remote_client.py          # 远程 Client 预留，后续接 RPC

  sim_server/                 # C/S 中的 Server，管理 scene、端口、进程、backend
    server.py                 # LocalSimulationServer 主入口
    port_manager.py           # 分配和释放端口
    process_manager.py        # 启动/停止 packaged UE 进程
    scene_manager.py          # 维护运行中的 SceneInstance
    backends/
      airvln_backend.py       # 当前 AirSim/UAV primitive backend，后续可改名 AirSimBackend
      unrealcv_backend.py     # UnrealCV camera/object backend
      local_ue_backend.py     # 本地 UE + AirSim 兼容入口

  adapters/                   # 外部库能力薄封装
    airsim_uav_adapter.py     # AirSim UAV primitive 封装
    null_uav_adapter.py       # dry-run / 测试用 UAV 模拟适配器

  tools/                      # AgentRuntime 到 EnvClient 的工具分发层
  agent/                      # 后续 planner / mission state
  cognition/                  # 后续 VLM 上下文、输出 schema
  perception/                 # 后续原始观测 artifact 管线

configs/
  maps/                       # 地图资源配置
  uav_action_config.json      # UAV 原子动作统一尺度配置

docs/                         # 架构和开发策略文档
tests/                        # 配置、EnvClient、SimulationServer、边界测试
runs/                         # 本地运行记录，通常不提交
```

## runtime 子目录职责

`runtime` 现在不再把所有东西平铺在一起，而是分成四类：

- `cli/`：只负责解析命令行参数、拼装运行对象、启动某个 runner。
- `configs/`：只负责把 JSON 配置变成 typed schema 或运行参数。
- `logging/`：只负责 run 目录、事件日志、系统日志和调试 artifact。
- `runners/`：只负责一段具体运行流程，例如 smoke test 和 observation test。

旧路径已经移除，新代码统一使用更清晰的新路径：

```python
from src.runtime.configs.action_config import UavActionConfig
```

## 命名原则

`provider` 描述地图资源形态：

- `packaged_unreal`：已经打包好的 UE 地图。

`launch.method` 描述启动方式：

- `direct`：本机直接启动 exe/sh。
- `wsl_bash`：Windows 上通过 WSL 启动 Linux packaged scene。
- `existing_process`：地图已经在本机或远程机器运行。
- `manual`：人工管理进程，测试时可 dry-run。

`control.protocol` 描述和 UE 进程通信的协议：

- `airsim`：AirSim UAV/ComputerVision RPC。
- `unrealcv`：UnrealCV socket 命令。

backend 应该按 `control.protocol` 扩展，而不是按每一种地图来源扩展。
