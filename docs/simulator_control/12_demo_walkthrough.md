# 统一底层 Demo 指南

> Status: Implemented  
> Previous: [11_scenario_cookbook.md](11_scenario_cookbook.md)  
> Index: [README.md](README.md)

## Demo 负责什么

入口：

```text
demos/simulator_env_client_demo.py
```

它演示一条完整但不包含 Agent Planner 的底层流程：

```text
MapPackage
→ ObservationProfile
→ SimulationServer
→ SimulatorEnvClient
→ start/connect/observe
→ capability-aware optional motion
→ restore/reset
→ ownership-aware cleanup
```

如果需要自然语言 Agent 的完整闭环，使用
`demos/real_engine_instruction_demo.py`。不要把这两个 Demo 混为一层：

- `simulator_env_client_demo.py`解释底层配置和生命周期；
- `real_engine_instruction_demo.py`解释 TaskRuntime/Planner 如何消费 EnvClient。

## 第一步：列出路由

```powershell
python demos/simulator_env_client_demo.py --list-maps
```

只读取本地 JSON，不创建 run、不连接 SDK、不访问 SSH。

输出会列出：

```text
map_id
connection
protocol
control_mode
```

## 第二步：理解默认 managed

```powershell
python demos/simulator_env_client_demo.py
```

默认：

```text
config=configs/maps/simulator_control_demo.example.json
map_id=demo_local_airsim_cv
execution=managed
observation_profile=configs/observations/airsim_default.example.json
move_forward_m=0
```

该命令不会主动移动 UAV，但会连接已有 UE，或在配置允许时启动 UE。示例配置中的
路径是占位值，真实运行前必须替换为本机部署配置。

## 第三步：选择 attach 或 managed

```powershell
python demos/simulator_env_client_demo.py `
  --config <your-map-config.json> `
  --map-id <your-map-id> `
  --execution attach
```

`attach`只连接已有 UE；默认 `managed`会执行 attach-or-start。四种部署/协议
组合仍使用相同 Python 流程，区别由 MapPackage 路由。

## 第三步半：选择观测 Profile

观测选择与地图部署分离：

```powershell
python demos/simulator_env_client_demo.py `
  --config <your-map-config.json> `
  --map-id <your-map-id> `
  --observation-profile configs/observations/airsim_depth_vis.example.json
```

Profile 由 Demo、Benchmark runner 或 Runtime 上层选择。不要把
`depth_vis`写进 MapPackage，也不要在 Backend 中按地图名称判断。

UnrealCV 使用同一份 ObservationProfile 通用名称：

```json
{
  "modalities": ["rgb", "depth", "segmentation", "pose"],
  "capture_sources": {
    "rgb": "scene",
    "depth": "depth_perspective",
    "segmentation": "segmentation"
  }
}
```

MapPackage 中的 `control_target`只决定移动 camera 还是 actor，不决定采集
modality。

## 第四步：显式请求动作

连接真实测试场景后，可以显式请求一个可恢复的小动作：

```powershell
python demos/simulator_env_client_demo.py `
  --config <your-map-config.json> `
  --map-id <your-map-id> `
  --execution managed `
  --observation-profile configs/observations/airsim_depth_vis.example.json `
  --move-forward-m 0.5 `
  --speed-mps 0.5 `
  --soft-reset
```

Demo 会：

1. 保存初始 pose；
2. 根据 capability 判断是否 takeoff；
3. 执行 `move_body`；
4. 在支持时 hover；
5. 保存移动后 observation；
6. 使用 `set_pose`恢复初始 pose；
7. 记录恢复位置误差；
8. 根据 capability 执行或跳过 soft reset；
9. 在 finally 中 land/stop。

## 第五步：连接真实场景

Attach：

```powershell
python demos/simulator_env_client_demo.py `
  --config <your-map-config.json> `
  --map-id <your-map-id> `
  --execution attach `
  --save-artifacts
```

Managed：

```powershell
python demos/simulator_env_client_demo.py `
  --config <your-map-config.json> `
  --map-id <your-map-id> `
  --execution managed `
  --save-artifacts `
  --require-visual
```

真实动作默认仍为零。只有显式传入 `--move-forward-m`才移动。

## 输出目录

```text
runs/simulator_env_client_demo/<run-id>/
├── resolved_map_package.json
├── resolved_observation_profile.json
├── execution_summary.json
├── scene_instance.json
├── observation_before.json
├── observation_after.json
├── observation_restored.json
├── observation_after_reset.json
├── scenes/
└── artifacts/
```

未执行移动时不会生成 `observation_after.json`和
`observation_restored.json`。未启用真实 artifact 保存时 `artifacts/`也可以为空。

优先查看 `execution_summary.json`：

- `route`：三个配置轴的最终选择；
- `capabilities`：Backend 声明的动作和观测能力；
- `actions`：实际执行的统一 EnvClient 动作；
- `scene.process_ownership`：进程归属；
- `restore_position_error_m`：移动后恢复误差；
- `cleanup`：关闭了什么、保留了什么；
- `visual_quality`：RGB/Depth 的尺寸、范围、标准差和严格验收结果；
- `error`：失败类型和消息。

`--require-visual`必须与 `--save-artifacts`一起使用。它会在
`--visual-timeout-sec`内重试，拒绝纯色 RGB、恒定深度或缺失 artifact。该严格规则
只属于人工/实机验收，不会放入 Backend，从而避免把合法暗场景误判为协议失败。

## 常见错误

### SSH private key not found

Remote attach/managed 都需要：

```powershell
$env:FLYAGENT_SSH_KEY = "C:\path\to\private_key"
```

只运行 `--list-maps`不会读取 key 文件。

### Remote port is open but does not speak protocol

端口可连接，但 Backend handshake 失败。EnvClient 会报告冲突并拒绝启动第二个
UE。检查端口对应的是 AirSim、UnrealCV 还是其他进程。

### settings mismatch

检查：

- `SimMode`；
- vehicle name；
- camera name；
- VehicleType；
- settings template source。

渲染器不会静默改写这些结构字段。

### action is unsupported

读取 `scene_instance.json`中的 capabilities。ComputerVision/UnrealCV 不支持
takeoff/land，不应通过 Demo 或 Runtime 伪造 airborne。

### attach cannot find a running simulator

`attach`的职责是“只连接”。若希望 Remote 自动 attach-or-start，明确选择
`managed`并在 MapPackage 配置 executable。

## 真实 Remote AirSim 验收

已完成的 env_1 安全闭环、settings hash、RGB/Depth/Pose 和进程清理结果见：

[08_remote_airvln_test.md](08_remote_airvln_test.md)

## 真实 Local UnrealCV 验收

DowntownWest 使用独立的高注释 diagnostic probe，验证 Actor 转向、米制位移、位姿
恢复、批量 RGB/Depth/Segmentation 和 owned process 清理：

```powershell
python tests/manual/unrealcv_local_actor_probe.py `
  --config <your-unrealcv-map-config.json> `
  --map-id downtownwest_local_unrealcv `
  --execution managed `
  --yaw-deg 10 `
  --move-forward-m 0.25 `
  --speed-mps 0.5 `
  --require-visual
```

该 probe 不加载或修改 UAV-Flow case。完整说明见
[14_local_unrealcv_test.md](14_local_unrealcv_test.md)。
