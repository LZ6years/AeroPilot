# C/S 架构职责划分

## 1. 核心边界

FlyAgent 里的 Client 指 `EnvClient`，不是整个 Agent。

```text
User / CLI
  -> AgentRuntime
    -> ToolDispatcher
      -> EnvClient
        -> SimulationServer
          -> UE launcher / process manager
          -> protocol backend
```

`AgentRuntime` 负责任务流程和工具调用。

`EnvClient` 负责把环境请求发给环境服务。

`SimulationServer` 负责管理 scene、端口、进程和协议 backend。

## 2. AgentRuntime

负责：

- 接收用户任务或 CLI smoke-test。
- 调用 planner、VLM、safety、clarification 等后续模块。
- 维护任务状态。
- 通过 `ToolDispatcher -> EnvClient` 获取观测和执行 primitive action。
- 写 run record。

不负责：

- 启动 UE。
- 管理 AirSim/UnrealCV 端口。
- 直接调用 AirSim 或 UnrealCV。
- 判断某个地图应该如何启动。

## 3. EnvClient

负责统一环境接口：

- `start_scene`
- `stop_scene`
- `reset_scene`
- `get_observation`
- `execute_action`

当前实现：

- `LocalEnvClient`：本地伪 C/S，直接调用 `LocalSimulationServer`。
- `RemoteEnvClient`：预留，后续接真正 RPC。

## 4. SimulationServer

负责：

- 加载 `MapPackage`。
- 创建 `SceneInstance`。
- 启动本机 packaged UE scene，或连接已有进程。
- 等待 AirSim/UnrealCV 端口 ready。
- 按 `control.protocol` 选择 backend。
- 标准化返回 `Observation` 和 `ActionResult`。

资源形态和控制协议必须分开：

```text
provider = packaged_unreal
control.protocol = airsim | unrealcv
```

这样 WindowsNoEditor、LinuxNoEditor、AirVLN env、UnrealZoo env 都可以共用启动模型；差异只出现在控制协议 backend。

## 5. Runtime 代码组织

`runtime` 被拆成四个子目录，避免 CLI、配置、日志和执行流程混在一个目录：

```text
runtime/
  cli/        # 命令行入口
  configs/    # 配置加载
  logging/    # run 目录、异步日志、artifact
  runners/    # smoke test 等运行流程
```

CLI 统一使用新入口：

```bash
python -m src.runtime.cli.main smoke-test --map-id cityenviron_win64 --launch
```

## 6. 远程机器模式

当前已经可以支持“远端 UE 进程已经启动，本机直接连接控制端口”的模式：

```text
AgentRuntime on local PC
  -> LocalEnvClient
    -> LocalSimulationServer
      -> AirSim/UnrealCV backend
        -> remote UE process host:port
```

这个模式下，`SimulationServer` 不启动远端进程，只等待并连接配置里的 `host:port`。

当前还不支持“本机 Client 命令远端 Server 去启动 UE 进程”的完整 C/S：

```text
AgentRuntime on local PC
  -> RemoteEnvClient
    -> SimulationServer process on UE machine
      -> launch/stop UE locally
```

这需要下一阶段补 RPC server。
