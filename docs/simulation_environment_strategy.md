# 仿真环境与地图资源策略

## 1. 当前结论

地图资源管理先参考 **AirVLN 的场景管理方式**，不是 AirNavigation，也不是第一阶段直接接 UnrealCV Gym。

当前优先级：

```text
第一阶段：AirVLN 风格 scene 资源管理 + AirSim 场景进程启动
第二阶段：补充本地 UE 启动脚本适配
第三阶段：再接入 UnrealCV / gym_unrealcv 的对象级操作能力
```

原因：

- AirVLN 的 C/S 和场景进程管理更接近 FlyAgent 后续多地图、多 UAV 扩展目标。
- AirNavigation 的地图处理太薄，只是 `map.json + bat`，不适合作为主设计。
- UnrealCV/gym_unrealcv 的对象增删改能力有价值，但第一阶段先不让它主导架构。

## 2. AirVLN 地图处理方式拆解

AirVLN 的地图资源本质是 `scene_id -> env_N -> scene executable`。

关键流程：

```text
EnvClient 请求打开 scenes
  -> SimulationServer 接收 open_scenes
  -> 根据 scene_id 查找 ENVs/**/env_<scene_id>/LinuxNoEditor/AirVLN.sh
  -> 为每个 scene 分配 AirSim ApiServerPort
  -> 为每个 scene 写独立 AirSim settings.json
  -> 启动 scene process
  -> 等待 scene 可连接
  -> 返回 ip + ports
  -> EnvClient 建立 AirSim client
```

AirVLN 中值得参考的点：

- `machines_info` 描述多机器、多 scene。
- `open_scenes` 是当前要打开的 scene 列表。
- `reopen_scenes(ip, scen_ids)` 由 server 统一关闭旧 scene 并打开新 scene。
- server 根据 `scene_id` 搜索环境路径。
- server 统一分配 AirSim 端口。
- server 为每个 scene 写独立 `settings.json`。
- client 只拿返回的 `ip + ports` 建立连接。

不直接照搬的点：

- Linux-only 偏重，Windows 本地需要适配。
- scene 查找路径写死为 `ENVs/**/env_N/LinuxNoEditor/AirVLN.sh`，FlyAgent 要配置化。
- `settings.json` 目录按 index 写入，FlyAgent 要按 `scene_instance_id` 管理。
- 端口和进程管理要封装，不能散落在业务代码。

## 3. FlyAgent 地图资源模型

建议将地图资源拆成两层：

```text
MapPackage      # 静态地图资源包
SceneInstance   # 一次运行中的地图实例
```

### 3.1 MapPackage

`MapPackage` 描述一个可启动的地图包。

示例：

```json
{
  "map_id": "airvln_env_1",
  "provider": "airvln",
  "scene_id": 1,
  "display_name": "AirVLN Env 1",
  "root": "D:/AirVLN/ENVs/env_1",
  "launch": {
    "windows": null,
    "linux": "LinuxNoEditor/AirVLN.sh",
    "args": ["-RenderOffscreen", "-NoSound", "-NoVSync"]
  },
  "airsim": {
    "mode": "ComputerVision",
    "vehicle_name": "Drone_1",
    "camera_name": "front_0",
    "image_width": 224,
    "image_height": 224,
    "depth_width": 256,
    "depth_height": 256
  },
  "resources": {
    "supports_object_edit": false,
    "supports_weather": false
  }
}
```

### 3.2 SceneInstance

`SceneInstance` 描述一次正在运行的 scene。

示例：

```json
{
  "scene_instance_id": "scene_001",
  "map_id": "airvln_env_1",
  "scene_id": 1,
  "machine_id": "local",
  "host": "127.0.0.1",
  "server_port": 30000,
  "airsim_port": 30001,
  "settings_path": "runs/run_001/scenes/scene_001/settings.json",
  "process_id": 12345,
  "status": "running"
}
```

这样做的好处：

- 静态资源和运行实例分离。
- 同一张地图可以启动多个实例。
- 后续可扩展一张地图多个 UAV。
- run 记录能明确知道当时启动的是哪个 scene、哪个端口、哪个 settings。

## 4. SimulationServer 职责

地图资源相关能力放在 `SimulationServer`，不放进 Agent Runtime。

SimulationServer 负责：

- 读取 map package 配置。
- 查找 scene 可执行文件或启动脚本。
- 为 scene 分配端口。
- 生成 AirSim `settings.json`。
- 启动 scene process。
- 等待 scene ready。
- 维护 `SceneInstance` 状态。
- 关闭 scene process。
- reset scene。
- 返回标准化 observation。

接口：

```text
register_map(map_config)
list_maps()
start_scene(map_id, scene_id=None)
stop_scene(scene_instance_id)
reset_scene(scene_instance_id)
get_scene_status(scene_instance_id)
get_observation(scene_instance_id, agent_id, modalities)
execute_action(scene_instance_id, agent_id, action)
```

## 5. EnvClient 职责

EnvClient 不管理地图资源，只调用 SimulationServer。

EnvClient 负责：

- 请求 `start_scene`。
- 保存 `scene_instance_id`。
- 请求 observation。
- 请求 UAV 原子动作。
- 处理超时、重试、错误转换。

EnvClient 不负责：

- 查找地图路径。
- 写 AirSim settings。
- 分配端口。
- 启动 UE 进程。
- kill 进程。

## 6. 第一阶段配置建议

第一阶段用一个统一的 `configs/maps/airvln_maps.json`。

示例：

```json
{
  "provider": "airvln",
  "search_roots": [
    "D:/AirVLN/ENVs",
    "F:/AirVLN/ENVs"
  ],
  "maps": [
    {
      "map_id": "airvln_env_1",
      "scene_id": 1,
      "relative_executable": "LinuxNoEditor/AirVLN.sh",
      "windows_executable": null,
      "default_vehicle": "Drone_1",
      "default_camera": "front_0"
    }
  ],
  "port_policy": {
    "server_port": 30000,
    "airsim_port_start": 30001,
    "max_scene_num": 8
  }
}
```

Windows 本地如果不是 Linux packaged scene，则新增：

```json
{
  "map_id": "local_ue_city",
  "provider": "local_ue",
  "scene_id": "city_001",
  "windows_executable": "D:/UEMaps/City/City.exe",
  "windows_script": "D:/UEMaps/City/start_city.bat",
  "default_vehicle": "Drone_1",
  "default_camera": "front_0"
}
```

## 7. AirNavigation 只作为反例和兼容来源

AirNavigation 的地图处理方式：

```text
settings/map.json
  -> map
  -> start_map_batfile
  -> map_list

MapController
  -> set_map(name)
  -> 拼出 Shell/<map>.bat
  -> restart_UE(batfile)
```

它的问题：

- 配置只记录当前地图和 bat 路径，没有 scene instance 概念。
- 路径写死，跨机器不可迁移。
- 没有端口管理。
- 没有进程状态管理。
- 没有 scene ready 检测。
- 没有 map package / scene instance 分离。
- 不适合多 scene、多 UAV、多机器。

因此 FlyAgent 不继承它的地图架构，只保留两个兼容点：

- 可以读取旧式 `map_name + bat_path`，迁移成 `MapPackage`。
- 可以用本地 `.bat/.sh` 作为一种启动后端。

## 8. UnrealCV / gym_unrealcv 后续接入

UnrealCV / gym_unrealcv 的价值在于对象级环境操作：

- `list_objects`
- `add_object`
- `remove_object`
- `hide_object`
- `set_object_pose`
- `set_object_color`
- `get_object_pose`

但这些不作为第一阶段主地图资源模型。

后续接入方式：

```text
SimulationServer
  -> Backend
    -> AirVLNBackend
    -> LocalUEBackend
    -> UnrealCvGymBackend
```

也就是说，UnrealCV 是 `SimulationServer` 的一个 backend，而不是 Agent Runtime 的直接依赖。

## 9. 推荐验证顺序

1. 按 AirVLN 风格配置 `MapPackage`。
2. 实现 `SimulationServer.start_scene(map_id)`。
3. 自动生成 AirSim `settings.json`。
4. 分配 AirSim port。
5. 启动 scene process。
6. EnvClient 拿到 `scene_instance_id + airsim_port`。
7. 获取 RGB/depth/pose。
8. 执行 UAV 原子动作。
9. 保存 `SceneInstance` 到 run record。
10. 再考虑接入 UnrealCV 对象操作。
