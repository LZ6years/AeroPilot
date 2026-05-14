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

## 7. Semantic Phase 与模型动作规划链路

`SemanticPlan` 的 phase 是开放语义结构，不是固定枚举动作表。phase name 可以由模型根据当前任务自由生成；程序依赖 `index` / `total` 理解阶段位置，依赖 `meaning` 理解阶段目标。

推荐链路：

```text
User / CLI
  -> TaskParser
    -> SemanticPlan phases(index,total,name,meaning)
      -> TaskRuntime preflight(connect,takeoff,hover,initial observe)
        -> RawObservation
          -> ObservationContextBuilder
            -> ModelActionPlanner
              -> SafetySupervisor
                -> ToolDispatcher
                  -> EnvClient
                    -> SimulationServer
```

`ObservationContextBuilder` 将当前 phase、MissionState、当前 RGB/depth/pose、最近动作历史、可用 tool registry 和安全约束整理成稳定 JSON，作为 M6 `ModelActionPlanner` 的输入。`ModelActionPlanner` 输出短期动作建议，但不能绕过 SafetySupervisor，也不能直接调用底层 SDK。

`connect`、`takeoff`、起飞后 `hover` 和初始观测属于 Runtime preflight，不属于 phase 动作规划；模型动作候选工具不应包含 `uav.connect` / `uav.takeoff` / `uav.land` 这类生命周期动作。

当前已不保留规则版短期 planner。短期动作规划统一由 `ModelActionPlanner` 基于 `PlannerContext` 生成 1-3 个 `kind="tool"` 动作；RGB/depth/pose 由 runtime 按 `observation_policy` 自动采集，模型不再输出 `observe` step。

## 8. 后续 Plan-Execute-Repair 链路

当前阶段先把基础 pipeline 和真实引擎链路跑通。后续不应继续把模型输出错误都做成规则归一化补丁，而应引入结构化错误反馈和模型修复链路。

目标链路：

```text
ModelActionPlanner
  -> ToolContractValidator
    -> if invalid: ModelRepairPlanner
    -> SafetySupervisor
    -> ToolDispatcher
    -> if execution/observation failed: ModelRepairPlanner or safe hover
```

`ToolContractValidator` 负责检查模型输出是否符合工具契约，例如 `uav.move_body` 只能使用明确的 `direction/duration` 或 `vx/vy/vz/duration` schema。当前 `_normalize_move_body_args()` 只是临时兼容层，用于基础动作测试，不应扩展成长期的大型规则映射表。

错误反馈应写成结构化 `repair_context`：

```json
{
  "error_type": "tool_argument_schema_error",
  "phase": "reverse_translation",
  "invalid_plan": {},
  "validation_errors": [],
  "allowed_tools": [],
  "latest_context": {}
}
```

模型修复只能处理计划合法性和下一步规划问题；SafetySupervisor 仍然拥有最终裁剪和拒绝权。真实引擎执行失败时，Runtime 应先进入安全 hover，再决定是否请求模型 repair。第一版 repair 建议只覆盖 action plan schema 错误，等运行闭环稳定后再扩展到执行失败、观测异常、目标未确认和 Safety 拦截。
