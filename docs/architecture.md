# FlyAgent 架构说明

## 1. 总体思路

FlyAgent 不让 Agent 直接操作 UE、AirSim 或 UnrealCV。Agent 只通过 `EnvClient` 获取观测和执行原子动作。

```text
User / CLI
  -> AgentRuntime
    -> ToolDispatcher
      -> EnvClient
        -> SimulationServer
          -> launcher / process manager
          -> backend selected by control.protocol
```

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

control.protocol: 与运行中 UE 通讯的协议
  airsim
  unrealcv
```

因此：

- AirVLN packaged scene + AirSim: `provider=packaged_unreal`, `control.protocol=airsim`
- UnrealZoo packaged scene + UnrealCV: `provider=packaged_unreal`, `control.protocol=unrealcv`
- 远端已启动 UE + UnrealCV: `provider=packaged_unreal`, `launch.method=existing_process`, `control.protocol=unrealcv`

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

## 4. Backend 定义

backend 不是“地图 backend”，而是“控制协议 backend”。

当前 backend：

- `AirVlnBackend`: 当前承担 AirSim UAV primitive backend 的角色，后续可改名为 `AirSimBackend`
- `UnrealCvBackend`: UnrealCV camera/object protocol backend
- `LocalUeBackend`: 本地 UE + AirSim 的兼容入口

## 5. 远端 UE 地图服务器

当前可以支持：

```text
本机 AgentRuntime
  -> 本机 LocalSimulationServer
    -> 远端 UE 进程的 AirSim/UnrealCV host:port
```

前提是远端 UE 已经启动，控制端口可访问。

当前还不支持：

```text
本机 RemoteEnvClient
  -> 远端 SimulationServer
    -> 远端启动/停止 UE 进程
```

后者需要补真正的 RPC server。
