# 仿真环境与地图资源策略

## 1. 目标

地图资源管理优先参考 `gym_unrealcv` 的方式：用 JSON 描述地图、可执行文件、agent、动作空间、起点、重置区域和目标对象。

本项目不把地图写死在代码里，而是通过工厂模式适配不同来源的地图资源。

## 2. 地图资源来源

第一阶段优先：

- `gym_unrealcv` / UnrealZoo Gym。

后续适配：

- AirSim UE 项目。
- AirVLN 场景。
- 自定义 UE 地图。
- 其他第三方地图包。

## 3. 工厂模式

```python
class MapResourceFactory:
    def create(self, config: dict):
        backend = config["backend"]
        if backend == "gym_unrealcv":
            return UnrealCvGymMap(config)
        if backend == "airsim":
            return AirSimMap(config)
        if backend == "airvln":
            return AirVlnMap(config)
        raise ValueError(f"Unsupported map backend: {backend}")
```

统一接口：

```python
class SimulationEnv:
    def start(self): ...
    def stop(self): ...
    def reset(self, agent_id: str): ...
    def step(self, agent_id: str, action: dict): ...
    def observe(self, agent_id: str, modalities: list[str]): ...
    def get_pose(self, agent_id: str): ...
```

## 4. gym_unrealcv 风格配置

建议地图配置：

```json
{
  "map_id": "suburb_day",
  "backend": "gym_unrealcv",
  "env": {
    "env_name": "SuburbNeighborhood_Day",
    "env_bin_win": "D:/UnrealZoo/UE4_ExampleScene/Binaries/Win64/UE4_ExampleScene.exe",
    "env_bin_linux": "UE4_ExampleScene_Linux/UE4_ExampleScene/Binaries/Linux/UE4_ExampleScene",
    "env_map": "SuburbNeighborhood_Day",
    "startup_script_win": null,
    "startup_script_linux": "./start_suburb_day.sh"
  },
  "agents": [
    {
      "agent_id": "uav_1",
      "object_name": "BP_Drone01_2",
      "class_name": "BP_drone01_C",
      "camera_id": 3,
      "type": "drone",
      "internal_nav": false,
      "start": [0, 0, 120],
      "action_space": "uav_velocity_4d"
    }
  ],
  "observations": ["rgb", "depth", "mask", "pose"],
  "safe_start": [[0, 0, 120]],
  "reset_area": [-1000, 1000, -1000, 1000, 0, 300],
  "targets": {
    "street_lamp": ["BP_StreetLamp", "SM_StreetLamp"]
  }
}
```

## 5. 本地 UE sh 文件启动

你本地已有 UE 的 `.sh` 文件时，不要把启动命令写死在代码中。

配置里保留：

```json
{
  "startup_script_linux": "F:/maps/start_map.sh",
  "startup_script_win": "F:/maps/start_map.bat",
  "args": ["-windowed", "-ResX=1280", "-ResY=720"]
}
```

启动流程：

1. 读取 map config。
2. 判断当前平台。
3. 优先使用平台对应启动脚本。
4. 如果没有启动脚本，再使用 `env_bin_win/env_bin_linux`。
5. 启动后连接 UnrealCV 或 AirSim。

## 6. 对象操作能力

`gym_unrealcv` / UnrealCV 方向适合预留对象级操作。

建议接口：

- `add_object(class_name, object_name, pose)`
- `remove_object(object_name)`
- `hide_object(object_name)`
- `show_object(object_name)`
- `set_object_pose(object_name, pose)`
- `set_object_color(object_name, color)`
- `set_object_material(object_name, material)`
- `list_objects()`
- `get_object_pose(object_name)`

这些能力不要放进 UAV 基础控制层，应归入环境工具或地图工具。

## 7. AirVLN C/S 预留

AirVLN 的启发是环境 server 负责重启和管理场景，Agent client 负责连接环境并执行任务。

后续可以扩展为：

```text
FlyAgent Runtime
  -> EnvClient
  -> EnvServer
  -> Map Process 1
  -> Map Process 2
  -> UAV 1 / UAV 2 / ...
```

第一阶段只做：

```text
FlyAgent CLI
  -> Local Env Adapter
  -> One Map
  -> One UAV
```

## 8. 验证顺序

1. 验证本地 UE 启动脚本可用。
2. 验证 UnrealCV 连接。
3. 验证 `reset/step/observe`。
4. 验证 RGB/depth/pose 获取。
5. 验证对象删除/隐藏/位置设置。
6. 验证 UAV 原子控制。
7. 再接 Agent 任务闭环。

