# MapPackage 配置参考

> Status: Implemented  
> 本文描述 `MapPackage.from_dict()` 当前支持的配置。  
> Previous: [06_backend_and_adapter_contracts.md](06_backend_and_adapter_contracts.md)  
> Next: [08_remote_airvln_test.md](08_remote_airvln_test.md)

## 三个配置轴

```text
connection.mode = local | remote
control.protocol = airsim | unrealcv
airsim.mode = ComputerVision | Multirotor
unrealcv.control_target = camera | actor
```

`connection.mode`决定进程位置，`control.protocol`决定 Backend，AirSim mode 决定动作策略。

图像 capture source 不属于 MapPackage。`scene`、`depth_vis`和
`depth_perspective`由上层运行时注入的 ObservationProfile 选择，参见
[10_envclient_usage.md](10_envclient_usage.md#observationprofile)。

执行方式不属于 MapPackage。每次构造 `SimulationServer`时单独选择：

```text
execution_mode = attach | managed
```

默认值是 `managed`。详见 [10_envclient_usage.md](10_envclient_usage.md)。
同一份地图部署配置可以连接已有 UE 或用于托管启停。只想离线检查配置时，应使用
`--list-maps`等不调用 `start_scene()`的入口。

## 通用字段

| 字段 | Owner | 必填 | 含义 |
|---|---|---:|---|
| `map_id` | Deployment | 是 | Runtime/Benchmark 使用的稳定逻辑 ID |
| `scene_id` | Deployment | 是 | 地图内部场景 ID |
| `provider` | Metadata | 是 | 地图来源，不参与 Backend 路由 |
| `connection.mode` | Deployment | 是 | `local`或 `remote` |
| `control.protocol` | Backend | 是 | BackendRegistry key |
| `launch` | Deployment | 托管进程时 | executable、参数、ready timeout |
| `airsim` | AirSim Backend | AirSim 时 | settings、vehicle、camera、port |
| `unrealcv` | UnrealCV Backend | UnrealCV 时 | endpoint、camera_id、受控目标 |

禁止在 MapPackage 中写入：

```text
episode_id
instruction
goal
reference_path
success truth
SR
nDTW
depth_image_type
```

## Local AirSim

```json
{
  "map_id": "cityenviron_win64",
  "provider": "packaged_unreal",
  "scene_id": "cityenviron",
  "root": "G:/ue-envs/airsim-win64/CityEnviron/WindowsNoEditor",

  "connection": {
    "mode": "local",
    "machine_id": "windows_local"
  },

  "launch": {
    "method": "direct",
    "windows": "CityEnviron/Binaries/Win64/CityEnviron.exe",
    "args": [
      "-windowed",
      "-ResX=1280",
      "-ResY=720",
      "-NoSound",
      "-NoVSync"
    ],
    "ready_timeout_sec": 90,
    "post_launch_wait_sec": 8
  },

  "control": {
    "protocol": "airsim"
  },

  "airsim": {
    "mode": "Multirotor",
    "settings_template_source": "local",
    "settings_template": "configs/airsim/cityenviron.settings.json",
    "vehicle_name": "Drone1",
    "camera_name": "front_center"
  }
}
```

本地 AirSim 端口由 control port policy 分配，settings 副本写入 run scene 目录。

## Remote AirSim

```json
{
  "map_id": "airsim_remote_env_1",
  "provider": "packaged_unreal",
  "scene_id": "1",
  "root": null,

  "connection": {
    "mode": "remote",
    "data_transport": "ssh_tunnel",
    "machine_id": "airvln_linux"
  },

  "ssh": {
    "host": "10.68.240.49",
    "port": 10182,
    "user": "wzq",
    "key_path": "${FLYAGENT_SSH_KEY}",
    "connect_timeout_sec": 10
  },

  "launch": {
    "method": "linux_bash",
    "linux_executable": "/home/Newdisk/wzq/data/airvln/ENVs/env_1/env_1/LinuxNoEditor/AirVLN.sh",
    "args": [
      "-RenderOffscreen",
      "-NoSound",
      "-NoVSync",
      "-GraphicsAdapter=0"
    ],
    "settings_arg_style": "double_dash",
    "gpu_id": 0,
    "remote_run_root": ".flyagent/scenes",
    "ready_timeout_sec": 120,
    "post_launch_wait_sec": 5
  },

  "control": {
    "protocol": "airsim"
  },

  "airsim": {
    "remote_host": "127.0.0.1",
    "port": 30001,
    "mode": "ComputerVision",
    "settings_template_source": "remote",
    "settings_template": "/home/Newdisk/wzq/data/airvln/AirVLN/airsim_plugin/settings/1/settings.json",
    "vehicle_name": "Drone_1",
    "camera_name": "front_0"
  }
}
```

Remote 规则：

- SSH key 通过环境变量提供；
- `remote_host`是从 Linux 主机视角访问的协议 endpoint；
- Tunnel 对 Runtime 暴露本机动态 `control_host/control_port`；
- settings 中的 SimMode 必须与 `airsim.mode`一致；
- attach 到外部 UE 时，只能记录 expected settings hash，不能声称实际 settings 已验证。

## Local UnrealCV

```json
{
  "map_id": "unrealzoo_collection",
  "provider": "packaged_unreal",
  "scene_id": "collection",
  "root": "G:/UnrealZoo-UE4/Collection_WinNoEditor_0424_25",

  "connection": {
    "mode": "local",
    "machine_id": "windows_local"
  },

  "launch": {
    "method": "direct",
    "windows": "Collection/Binaries/Win64/Collection.exe",
    "args": [
      "-windowed",
      "-ResX=1280",
      "-ResY=720"
    ],
    "ready_timeout_sec": 60,
    "post_launch_wait_sec": 5
  },

  "control": {
    "protocol": "unrealcv"
  },

  "unrealcv": {
    "host": "127.0.0.1",
    "port": 9068,
    "camera_id": 0,
    "width": 256,
    "height": 256,
    "fov": 90,
    "control_target": "actor",
    "startup": {
      "hide_objects": [
      "BP_Character_2",
      "BP_animal_2",
      "BP_BaseCar_2",
      "Sport_2"
      ]
    },
    "actor": {
      "name": "BP_Drone01_2",
      "scale": [0.1, 0.1, 0.1],
      "units_per_meter": 100.0,
      "yaw_offset_deg": -180.0,
      "rotation_method": "blueprint_yaw",
      "camera_follow": "standard_pose",
      "camera_validation": {
        "enabled": true,
        "position_mode": "max_distance",
        "max_distance_m": 1.5,
        "position_tolerance_m": 0.05,
        "yaw_tolerance_deg": 1.0
      },
      "disable_physics": true,
      "yaw_settle_timeout_sec": 5.0,
      "yaw_tolerance_deg": 1.0,
      "yaw_poll_interval_sec": 0.05
    }
  }
}
```

UnrealCV 不生成 AirSim settings，`settings_path`保持 null。完整便携示例见
[`configs/maps/unrealcv_actor_control.example.json`](../../configs/maps/unrealcv_actor_control.example.json)。

### UnrealCV 字段

| 字段 | 默认值 | 说明 |
|---|---|---|
| `control_target` | `camera` | 选择自由相机或主 Actor 控制 |
| `startup.hide_objects` | `[]` | Actor 模式连接时隐藏 mesh、但保留控制与 camera component 的对象 |
| `actor.name` | 无 | Actor 模式必填，不由 Backend 猜测 |
| `actor.scale` | 无 | 可选的三轴正数缩放；连接时设置一次 |
| `actor.units_per_meter` | `100.0` | 正数；UE native location 到米的比例 |
| `actor.yaw_offset_deg` | `0.0` | 标准 rotation 的 canonical/native 偏移；Blueprint yaw 的命令偏移 |
| `actor.rotation_method` | `standard_pose` | `standard_pose`或`blueprint_yaw` |
| `actor.camera_follow` | `none` | `none`、`standard_pose`或`blueprint_pose` |
| `actor.camera_validation.enabled` | `false` | 同步后读取 camera pose，拒绝只返回 `ok` 但未生效的命令 |
| `actor.camera_validation.position_mode` | `expected_body_offset` | `expected_body_offset` 用于固定安装相机；`max_distance` 用于带碰撞避让的相机 |
| `actor.camera_validation.expected_body_offset_m` | `[0,0,0]` | camera 相对 Actor 的 `[forward,right,up]` 安装偏移 |
| `actor.camera_validation.position_tolerance_m` | `0.05` | 实际与预期安装偏移的最大误差 |
| `actor.camera_validation.max_distance_m` | `1.5` | `max_distance` 模式下 camera 与 Actor 的最大距离 |
| `actor.camera_validation.yaw_tolerance_deg` | `1.0` | camera 与 Actor 的最大 wrapped-yaw 误差 |
| `actor.disable_physics` | `false` | 是否在连接后显式冻结主 Actor 物理 |
| `actor.yaw_settle_timeout_sec` | `5.0` | Blueprint yaw 插值的最长等待 |
| `actor.yaw_tolerance_deg` | `1.0` | readback 与目标 yaw 的允许误差 |
| `actor.yaw_poll_interval_sec` | `0.05` | yaw readback 轮询间隔 |

当 `control_target=actor`时，`actor.name`必须存在。非法 control target、非正
`units_per_meter`、未知 rotation/follow 策略和非布尔 `disable_physics`均在
`load_map_config()`阶段失败。

`blueprint_yaw`扩展在命令发送前应用 `yaw_offset_deg`。该 Blueprint 完成内部
mesh 朝向修正后，Actor rotation readback 已经是相机物理 yaw，因此直接作为
canonical yaw；标准 `standard_pose`仍在读写两侧使用 offset 换算。

`actor.scale`是地图级主控制目标初始化，不是 episode 或 Benchmark 场景布置。
例如 DowntownWest 的历史 UAV-Flow 初始化会把`BP_Drone01_2`从打包地图默认比例
调整为`[0.1, 0.1, 0.1]`，否则贴近 Actor 的主相机可能拍到机身部件。

`startup.hide_objects`同样属于显式地图归一化。Backend 只隐藏 MapPackage 中
列出的 mesh；对象的 pose、Blueprint 控制和 camera component 仍然保留，对象已
不存在时按已归一化处理。DowntownWest 使用该字段隐藏默认人物、动物和车辆 Pawn。
camera-owning `BP_Drone01_2` 不能整 Actor 隐藏，否则 Blueprint viewport/camera
跟随会失效；Schema 会直接拒绝将 controlled actor 写入该列表。Benchmark 创建的目标人物和
车辆不写入此列表。

相机验证是地图级契约，不是 episode 配置。scene prepare 和 Actor pose write
只负责同步，因为 Runtime preflight takeoff 也会经过 pose write；严格 readback
发生在 episode initial pose 后的首帧以及每次正式动作后的 capture sync。Backend
根据 `position_mode` 比较固定 body-frame 安装偏移，或检查碰撞相机仍处于 Actor
附近；两种模式都严格比较 yaw。因此 UnrealCV
命令静默无效时会在错误画面进入 Planner 前失败。

MapPackage 只描述主 Actor 如何控制。人物、车辆、目标位置、外观和 reference path
不能写入这些字段。

## Settings 处理

允许来源：

```text
local  → 读取本地模板
remote → SSH 只读下载模板
```

渲染器验证：

- SettingsVersion；
- SimMode；
- vehicle；
- camera；
- VehicleType 与模式一致性。

只允许覆盖运行字段，例如 `ApiServerPort`和显式配置的 `LocalHostIp`。不能静默修改 vehicle、camera、分辨率、FOV 或 SimMode。

## Port policy

目标配置使用通用命名：

```json
{
  "port_policy": {
    "control_port_start": 30001,
    "max_scene_num": 8
  }
}
```

远程固定端口从具体协议配置读取；本地托管场景才使用动态 control port 分配。
