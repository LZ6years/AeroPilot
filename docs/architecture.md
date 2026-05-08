# FlyAgent 架构说明

## 1. 总体思路

FlyAgent 不让 Agent 直接操作 UE、AirSim 或 UnrealCV。Agent 只能通过 `EnvClient` 获取观测和执行原子动作。

```text
User / CLI
  -> AgentRuntime
    -> ToolDispatcher
      -> EnvClient
        -> SimulationServer
          -> launcher / process manager
          -> backend selected by control.protocol
          -> adapter for external SDK
```

这样做的好处是：后续无论地图在本机、WSL、远程 Linux，还是控制协议从 AirSim 换成 UnrealCV，Agent 上层都不用改。

## 2. 地图资源和控制协议分离

打包后的 UE 地图在启动形态上通常很像，所以不应该为每一种地图来源单独设计 backend。

新的边界是：

```text
provider: 地图资源形态
  packaged_unreal

launch.method: 进程启动方式
  direct
  wsl_bash
  existing_process
  manual

control.protocol: 与运行中 UE 通信的协议
  airsim
  unrealcv
```

例子：

- AirVLN packaged scene + AirSim：`provider=packaged_unreal`，`control.protocol=airsim`。
- UnrealZoo packaged scene + UnrealCV：`provider=packaged_unreal`，`control.protocol=unrealcv`。
- 远程已启动 UE + AirSim：`provider=packaged_unreal`，`launch.method=existing_process`，`control.protocol=airsim`。

## 3. 核心对象

`MapPackage` 是静态资源配置：

```text
map_id
provider
scene_id
root
launch
control
airsim / unrealcv
resources
```

`SceneInstance` 是一次运行中的 scene：

```text
scene_instance_id
map_id
scene_id
host
server_port
airsim_port
settings_path
process_id
status
```

`UavActionConfig` 是无人机原子动作尺度配置：

```text
linear_velocity_mps
vertical_velocity_mps
move_duration_sec
vertical_duration_sec
vertical_step_m
yaw_step_deg
```

## 4. Runtime 分层

`src/runtime` 现在按职责拆成子目录：

- `cli/`：命令行入口，只做参数解析和对象组装。
- `configs/`：加载地图配置、远程连接配置、UAV 动作尺度配置。
- `logging/`：创建 run 目录，异步写 `system.log`、`events.jsonl` 和 artifact。
- `runners/`：执行具体运行流程，例如 smoke test。

旧模块路径已经移除。后续代码统一使用子目录路径，例如 `src.runtime.cli.main`、`src.runtime.configs.action_config`。

## 5. Backend 定义

backend 不是“地图 backend”，而是“控制协议 backend”。

当前 backend：

- `AirVlnBackend`：当前承担 AirSim UAV primitive backend 的角色，后续可改名为 `AirSimBackend`。
- `UnrealCvBackend`：UnrealCV camera/object protocol backend。
- `LocalUeBackend`：本地 UE + AirSim 的兼容入口。

地图怎么启动由 `launch.method` 决定；启动后怎么通信由 `control.protocol` 决定。

## 6. 远程 UE 地图服务器

当前可以支持：

```text
本机 AgentRuntime
  -> 本机 LocalEnvClient
    -> 本机 LocalSimulationServer
      -> 远程 UE 进程的 AirSim/UnrealCV host:port
```

前提是远端 UE 已经启动，控制端口可以从本机访问。

当前还不支持：

```text
本机 RemoteEnvClient
  -> 远程 SimulationServer
    -> 远程启动/停止 UE 进程
```

后者需要下一阶段补真正的 RPC server。
