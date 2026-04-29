# C/S 架构职责划分

## 1. 核心结论

参考 AirVLN 的 C/S 思路时，FlyAgent 里的 `Client` 不等于整个 Agent。

更准确的分层是：

```text
User / CLI / Future Feishu / Future Electron
  -> Agent Runtime
    -> Task Parser / Planner / VLM / Safety / Clarification
      -> EnvClient
        -> SimulationServer
          -> AirVLN-style Scene / Local UE / Future UnrealCV
```

其中：

- `Agent Runtime` 是智能体应用层。
- `EnvClient` 才是 C/S 里的 Client。
- `SimulationServer` 是 C/S 里的 Server。

一句话：

```text
Agent Runtime 负责想清楚要做什么。
EnvClient 负责把请求发给环境服务。
SimulationServer 负责管理仿真世界并执行原子能力。
```

## 2. 为什么不能把 Agent Runtime 当 Client

如果把 Client 定义成整个 Agent，那么 Client 会同时包含用户输入、任务解析、语义规划、VLM 图像理解、安全策略、用户提问和环境调用。

这样边界太大，后续很难扩展。

正确做法是拆成两层：

```text
Agent Runtime  # 大脑
EnvClient      # 环境服务调用封装
```

`EnvClient` 只是 Agent Runtime 的一个依赖，不是 Agent Runtime 本身。

## 3. Agent Runtime 职责

Agent Runtime 负责智能体任务闭环。

职责：

- 接收用户输入。
- 调用 `Task Parser`。
- 调用 `Semantic Planner`。
- 调用 `Short Horizon Planner`。
- 调用 VLM 理解图像和上下文。
- 维护 `Mission State`。
- 执行 `Safety Supervisor`。
- 判断是否困惑。
- 困惑时调用 `Clarification Manager` 向用户提问。
- 维护 `Run Store` 和任务记录。
- 调用 `EnvClient` 获取观测和执行原子动作。

不负责：

- 直接启动 UE 进程。
- 直接管理 UnrealCV / AirSim 端口。
- 直接处理多个 scene 的进程生命周期。
- 直接绑定某一种地图后端。

## 4. EnvClient 职责

EnvClient 是 C/S 架构里的 Client。

职责：

- 封装对 `SimulationServer` 的调用。
- 提供统一 Python API 给 Agent Runtime。
- 隐藏本地调用、RPC 调用、HTTP 调用之间的差异。
- 做请求参数校验和序列化。
- 做返回结果解析。
- 做连接状态管理。
- 做超时、重试、错误转换。

EnvClient 提供的接口分为四类：

```text
lifecycle:
  start_scene(map_id, scene_id)
  stop_scene(scene_id)
  reset_scene(scene_id)
  health_check()

observation:
  get_observation(scene_id, agent_id, modalities)
  get_rgb(scene_id, agent_id)
  get_depth(scene_id, agent_id)
  get_pose(scene_id, agent_id)
  get_collision(scene_id, agent_id)

control:
  takeoff(scene_id, agent_id)
  hover(scene_id, agent_id)
  land(scene_id, agent_id)
  move_body(scene_id, agent_id, vx, vy, vz, duration)
  turn_relative(scene_id, agent_id, yaw_deg)
  rotate_camera(scene_id, agent_id, pitch, yaw)

environment_edit:
  list_objects(scene_id)
  add_object(scene_id, object_spec)
  remove_object(scene_id, object_id)
  hide_object(scene_id, object_id)
  set_object_pose(scene_id, object_id, pose)
```

EnvClient 不负责：

- 用户交互。
- 任务理解。
- VLM 判断。
- 短期规划。
- 任务完成判断。
- 困惑提问。

## 5. SimulationServer 职责

SimulationServer 是 C/S 架构里的 Server。

职责：

- 管理 AirVLN 风格 scene、本地 UE / AirSim 和后续 UnrealCV 场景进程。
- 管理地图资源和启动脚本。
- 管理 `map_id`、`scene_id`、`agent_id`。
- 分配和记录端口，例如 AirSim port、UnrealCV port、server RPC port。
- 管理一个或多个 scene。
- 管理一个 scene 中的一个或多个 UAV。
- 把 EnvClient 请求翻译成具体后端调用。
- 返回标准化 observation 和 action result。
- 提供 reset、start、stop、health check。
- 提供对象级环境操作。

SimulationServer 不负责：

- 解析自然语言。
- 调用 VLM 做任务决策。
- 生成语义规划。
- 判断“应该飞到哪里”。
- 决定任务成功或失败。
- 向用户提问。

## 6. 第一阶段实现方式

第一阶段不需要立刻做真正的网络 RPC。

推荐先做“伪 C/S”：

```text
Agent Runtime
  -> LocalEnvClient
    -> LocalSimulationServer
      -> AirVLN-style Scene / Local UE
```

代码上保持 C/S 接口，但实现可以是本地类调用。

当本地闭环跑通后，再替换为真正远程调用：

```text
Agent Runtime
  -> RemoteEnvClient
    -> SimulationServer process
      -> AirVLN-style Scene / Local UE
```

## 7. 扩展方式

单机单 UAV：

```text
Agent Runtime
  -> LocalEnvClient
    -> LocalSimulationServer
      -> Scene A
        -> UAV 1
```

一张地图多个 UAV：

```text
Agent Runtime
  -> EnvClient
    -> SimulationServer
      -> Scene A
        -> UAV 1
        -> UAV 2
        -> UAV 3
```

多地图：

```text
Agent Runtime
  -> EnvClient
    -> SimulationServer
      -> Scene A
      -> Scene B
      -> Scene C
```

多机器：

```text
Agent Runtime
  -> EnvClient
    -> SimulationServer A
    -> SimulationServer B
    -> SimulationServer C
```

## 8. 建议代码位置

```text
src/
  runtime/
    main.py                       # CLI 入口，启动 Agent Runtime
    agent_runtime.py              # Agent Runtime 主循环

  env_client/
    base.py                       # EnvClient 抽象接口
    local_client.py               # 本地伪 C/S client
    remote_client.py              # 远程 RPC client 预留
    schemas.py                    # 请求/响应 schema

  sim_server/
    server.py                     # SimulationServer 入口
    scene_manager.py              # scene 生命周期管理
    process_manager.py            # UE 进程管理
    port_manager.py               # 端口管理
    backends/
      airvln_backend.py           # AirVLN 风格 scene 后端
      airsim_backend.py           # AirSim 后端
      local_ue_backend.py         # 本地 UE 脚本后端
```

## 9. 命名约定

统一使用：

- `Agent Runtime`：智能体主运行时。
- `EnvClient`：C/S 中的 Client。
- `SimulationServer`：C/S 中的 Server。
- `Backend`：Server 内部接入的具体仿真后端。
- `Scene`：一张已启动的地图实例。
- `Agent` / `UAV`：场景中的无人机实例。

避免使用：

- `Agent Client`：容易误解成 Agent 大脑也是 C/S Client。
- `Env Adapter`：如果它实际在做 client/server 通信，优先叫 `EnvClient`。
