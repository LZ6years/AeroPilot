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
    "actor": {
      "name": "BP_Drone01_2",
      "units_per_meter": 100.0,
      "yaw_offset_deg": -180.0,
      "rotation_method": "blueprint_yaw",
      "camera_follow": "standard_pose",
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
| `actor.name` | 无 | Actor 模式必填，不由 Backend 猜测 |
| `actor.units_per_meter` | `100.0` | 正数；UE native location 到米的比例 |
| `actor.yaw_offset_deg` | `0.0` | canonical 与 native Actor readback/标准 rotation 的偏移 |
| `actor.rotation_method` | `standard_pose` | `standard_pose`或`blueprint_yaw` |
| `actor.camera_follow` | `none` | `none`、`standard_pose`或`blueprint_pose` |
| `actor.disable_physics` | `false` | 是否在连接后显式冻结主 Actor 物理 |
| `actor.yaw_settle_timeout_sec` | `5.0` | Blueprint yaw 插值的最长等待 |
| `actor.yaw_tolerance_deg` | `1.0` | readback 与目标 yaw 的允许误差 |
| `actor.yaw_poll_interval_sec` | `0.05` | yaw readback 轮询间隔 |

当 `control_target=actor`时，`actor.name`必须存在。非法 control target、非正
`units_per_meter`、未知 rotation/follow 策略和非布尔 `disable_physics`均在
`load_map_config()`阶段失败。

`blueprint_yaw`扩展接收 canonical 目标 yaw，不再二次应用 native mesh offset；
`yaw_offset_deg`仍用于把 Actor rotation readback 转回 canonical yaw。

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
