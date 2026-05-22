# 仿真环境与地图资源策略

## 1. 当前结论

FlyAgent 的仿真层分成两件事：

```text
地图资源形态 / 启动方式
  packaged_unreal: 已打包 UE 场景，例如 WindowsNoEditor、LinuxNoEditor
  existing_process: 已经在本机或远端机器运行的 UE 进程

控制协议 / 观测协议
  airsim: AirSim RPC，适合真实 UAV primitive
  unrealcv: UnrealCV socket，适合相机、图像、对象级操作
```

因此 backend 不再按“地图来自哪里”来理解，而是按“怎么和运行中的 UE 通讯”来理解。

## 2. 推荐配置模型

`MapPackage` 描述静态地图资源，`SceneInstance` 描述一次运行中的 scene。

```json
{
  "map_id": "unrealzoo_collection",
  "provider": "packaged_unreal",
  "scene_id": "collection",
  "root": "G:/UnrealZoo-UE4/Collection_WinNoEditor_0424_25",
  "launch": {
    "method": "direct",
    "windows": "Collection/Binaries/Win64/Collection.exe",
    "args": ["-windowed", "-ResX=1280", "-ResY=720"],
    "ready_timeout_sec": 60
  },
  "control": {
    "protocol": "unrealcv"
  },
  "unrealcv": {
    "host": "127.0.0.1",
    "port": 9010,
    "camera_id": 0
  }
}
```

AirSim UAV 地图同样使用 `provider: "packaged_unreal"`，只把 `control.protocol` 换成 `airsim`。

## 3. 代码边界

```text
AgentRuntime
  -> EnvClient
    -> SimulationServer
      -> launcher/process manager
      -> backend selected by control.protocol
```

`SimulationServer` 负责：

- 读取 `MapPackage`
- 分配端口
- 写 AirSim settings
- 启动或连接 scene process
- 等待控制协议端口 ready
- 按 `control.protocol` 选择 backend

backend 负责：

- `airsim` backend: 调 AirSim UAV primitive，例如 connect、takeoff、hover、move_body、get_pose、get_frames
- `unrealcv` backend: 调 UnrealCV 命令，例如 camera location、rotation、lit/depth/object_mask

## 4. 远端 UE 资源服务器

如果 UE 地图已经在另一台电脑运行，本机有两种连接方式。

第一种是直接连控制协议，适合当前阶段：

```json
{
  "map_id": "remote_unrealzoo_collection",
  "provider": "packaged_unreal",
  "scene_id": "collection",
  "root": null,
  "launch": {
    "method": "existing_process",
    "ready_timeout_sec": 30
  },
  "control": {
    "protocol": "unrealcv"
  },
  "unrealcv": {
    "host": "192.168.1.20",
    "port": 9010,
    "camera_id": 0
  }
}
```

运行：

```powershell
flyagent smoke-test --map-id remote_unrealzoo_collection --config configs/maps/airvln_maps.json --connect-existing
```

要求：

- 远端 Windows 防火墙放行 UnrealCV/AirSim 端口
- UE 插件监听地址不能只绑定 localhost
- 两台电脑网络互通
- 本机配置里的 `host` 写远端机器 IP

第二种是真正的远程 SimulationServer：

```text
AgentRuntime on PC A
  -> RemoteEnvClient
    -> SimulationServer on PC B
      -> local UE process on PC B
```

这个模式可以让 PC B 负责启动、停止、重启地图进程，但需要实现网络 RPC server。当前代码只实现了本地伪 C/S 和直接连接已有控制端口，`RemoteEnvClient` 还只是预留。

远端 Linux AirSim 快速测试可以不写正式 map config，直接使用临时配置命令：

```powershell
flyagent remote-airsim-smoke-test --remote-host <LINUX_IP>
```

默认参数：

```text
port: 41451
vehicle_name: Drone_1
camera_name: front_0
```

如果 Linux 端使用了其他端口或名称，可以覆盖：

```powershell
flyagent remote-airsim-smoke-test --remote-host <LINUX_IP> --remote-port 41452 --vehicle-name Drone_1 --camera-name front_0
```

该命令会先做 TCP 连通性检查，并把临时 `map_package.json` 和 `remote_probe.json` 写入 run record。

如果远端 AirVLN 只能以 `ComputerVision` 模式启动，应使用观测 smoke test。这个命令不会调用 `enableApiControl`、`takeoff`、`hover` 或 `getMultirotorState`，只验证连接、pose、RGB 和 depth：

```powershell
flyagent remote-airsim-cv-smoke-test --remote-host <LINUX_IP> --remote-port 30001
```

## 5. Windows AirSim Packaged Map

Windows 打包的 AirSim 地图也可以使用每次 run 独立生成的 settings 文件。启动参数使用 AirSim 官方格式：

```powershell
CityEnviron.exe -settings=C:\path\to\settings.json
```

Linux AirVLN packaged scene 仍使用 AirVLN 脚本格式：

```bash
bash AirVLN.sh --settings /path/to/settings.json
```

`CityEnviron` 已经接入为：

```text
map_id: cityenviron_win64
provider: packaged_unreal
launch.method: direct
control.protocol: airsim
airsim.settings_profile: camera_defaults
```

`settings_profile: camera_defaults` 会把 RGB/depth 捕获配置写到 `CameraDefaults`，而不是写到 `Vehicles.<name>.Cameras`。这个 profile 是为了兼容部分 Windows AirSim 包对 per-vehicle camera 配置不稳定的问题。

验证命令：

```powershell
flyagent smoke-test --map-id cityenviron_win64 --config configs/maps/airvln_maps.json --launch
```

## 6. AirNavigation 和 AirVLN 的使用原则

AirNavigation 只参考基础飞控接口名称和行为，不使用它的地图结构、任务循环、UI 或硬编码配置。

AirVLN 只参考它的 C/S 思路和 scene 进程管理思路，不照搬路径组织、地图 JSON 或任务循环。

FlyAgent 自己的主模型是：

```text
MapPackage + SceneInstance + control.protocol backend
```
