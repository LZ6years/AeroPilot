# 参考项目说明

Status: Historical

> 历史参考资料。下列本机路径和旧架构说明不属于当前 Runtime 或 Context V2 契约。

## 1. AirNavigation

路径：

```text
C:\Users\HP\Desktop\AirNavigation\AirNavigation
```

定位：

- 只参考 UAV 基础操作和原始观测获取。
- 不参考整体架构、任务循环和 UI。
- 高级视觉逻辑暂不拆分，第一阶段统一交给 VLM 处理。

建议抽取：

`utils/UAV_controller.py`

- UAV 控制主封装。
- 包含连接、起飞、降落、悬停、速度控制、飞到坐标、相机旋转、状态记录。

`docs/uav_controller.md`

- 已整理 wrapper API。
- 可作为 Agent 工具层接口来源。

`utils/UAV_navigater.py`

- 只参考局部算法，例如视觉线索到角度偏差、相机调整。
- 不继承它的整体导航闭环。

`ai/ai_vision.py`

- 只作为“如何调用多模态模型理解图像”的参考。
- 不作为第一阶段基础感知能力直接继承。
- 其中硬编码 API key 必须移除，改为环境变量或配置。

注意：

- AirNavigation 整体架构较重且耦合，不适合直接继承。
- 基础层只保留原子飞控和 RGB/depth/pose 获取。
- “面向目标、靠近目标、搜索目标”应由 Planner + VLM + 安全策略组合完成。

## 2. AirVLN

链接：

```text
https://github.com/AirVLN/AirVLN
```

定位：

- 主要参考 C/S 仿真架构。
- 用于后续扩展“一张地图多个 UAV / 多 scene / 多机器”。
- 第一阶段只做单机单 UAV，不照搬完整训练和评测框架。

关键参考：

`airsim_plugin/AirVLNSimulatorServerTool.py`

- server 管理 UE/AirSim 场景进程。
- 通过 msgpack-rpc 暴露 `reopen_scenes`。
- 为每个场景写入独立 AirSim `settings.json`。
- 为每个场景分配 `ApiServerPort`。
- 通过 subprocess 启动场景。

`airsim_plugin/AirVLNSimulatorClientTool.py`

- client 根据 `machines_info` 连接多个 server。
- 每台 machine 可以包含多个 `open_scenes`。
- 每个 scene 对应一个 AirSim client。
- 统一获取 RGB/depth。

`utils/env_vector.py`

- 用 worker 管理多个环境实例。
- command 包含 `set_batch`、`get_obs`、`get_collision_sensor`。

对 FlyAgent 的启发：

```text
Agent Runtime
  -> EnvClient
    -> SimulationServer
      -> UE/AirSim Scene Process
      -> UAV Control / Observation
```

第一阶段可以退化为：

```text
Agent Runtime
  -> SimulatorEnvClient
    -> SimulationServer
      -> Local/Remote UE
```

注意：这里的 `EnvClient` 只是环境服务调用封装，不包含任务解析、规划、VLM 或用户交互。

但接口保留：

- `map_id`
- `scene_id`
- `agent_id`
- `control_host`
- `control_port`
- `capabilities`

## 3. UAV-Flow

链接：

```text
https://github.com/buaa-colalab/UAV-Flow
```

定位：

- 不作为第一阶段地图资源主模型。
- 作为后续外部 benchmark 来源，用于评估 UAV-Agent 在 Flow-style 指令上的闭环效果。
- 参考 UnrealCV / `gym_unrealcv` 的对象级环境操作能力，但不让 UAV-Flow runner 主导本项目 runtime。
- 当前地图资源主线改为 AirVLN 风格 scene 管理。
- 需求与实现影响文档见 `docs/evaluation/uav_flow/`。

关键参考：

`UAV-Flow-Eval/README.md`

- evaluation environment 基于 UnrealZoo Gym。
- Windows 使用 packaged UnrealZoo environment。
- 需要修改 JSON 配置中的 `env_bin_win` 指向本地 UE 环境可执行文件。

`UAV-Flow-Eval/load_env.py`

- 通过 modelscope 下载 UnrealZoo/UnrealCV 环境资源。
- 根据平台选择 binary。
- 解压后放入 UnrealCV 环境目录。

`gym_unrealcv/envs/setting/Navigation/*.json`

- 地图配置包含：
  - `env_name`
  - `env_bin`
  - `env_bin_win`
  - `env_map`
  - `third_cam`
  - `agents`
  - `safe_start`
  - `reset_area`
  - `targets`

`gym_unrealcv/envs/base_env.py`

- 读取 setting JSON。
- 选择平台对应的 env binary。
- 提供 `reset/step/close`。
- 通过 UnrealCV 执行移动、相机、动画、对象命令。
- 返回 pose、image、mask、depth 等观测。

`gym_unrealcv` 是否可用：

- 可以作为候选，先做最小 demo 验证。
- 优点是地图、agent、动作空间、safe_start、reset_area、targets 都可配置。
- 还可参考其对象操作能力，例如删除物体、增加物体、设置目标对象。
- 风险是依赖 UnrealZoo 环境包、UnrealCV 版本和地图资源路径。

建议：

- 第一阶段不要让 `gym_unrealcv` 主导地图资源模型。
- 先实现 AirVLN 风格的 `MapPackage + SceneInstance + SimulationServer`。
- 对 UAV-Flow 的任务集评估代码放在 `evaluation/benchmarks/uav_flow/`，不放在 `src/integrations/`。
- 等 AirVLN-style scene 启动和观测闭环跑通后，如果需要正式支持 UnrealCV 环境，再把通用能力沉淀为 `src/sim_server/backends/` 或 `src/env_client/` 的实现。

## 4. QiClaw

路径：

```text
F:\openclaw-main\wojiazaiyugang.QiClaw\QiClaw
```

定位：

- Electron + Agent runtime + task 编排系统。
- 主要参考 Agent 架构和前端预留方式。
- 第一阶段只做 CLI，Electron 等 UE 完整跑通后再接。

建议参考：

`src/core/assistant-service.js`

- 统一处理用户请求、planner、task runtime、records。

`src/core/planner.js`

- 规范 planner 输出。
- 支持分类、路由、execution hint。

`src/core/gateway/`

- run queue、run store、session manager、gateway control plane。

`src/hosts/electron/`

- Electron host，可作为最终前端参考。

对 FlyAgent 的启发：

- `Agent Runtime`：任务入口、任务状态、运行记录。
- `Planner`：语义计划与短期执行计划。
- `Tool Registry`：飞控、环境工具注册。
- `Tool Executor`：统一执行和错误处理。
- `Run Store`：保存任务状态、日志、轨迹、artifacts。
- `Electron UI`：最后只作为 control surface。

最小闭环：

```text
CLI 输入
  -> Task Parser
  -> Semantic Planner
  -> Short Horizon Planner
  -> Safety
  -> Tool Executor
  -> Env/UAV
  -> Raw Observation
  -> State Update
  -> Record
```

## 5. 参考分工总结

| 来源 | 只参考什么 | 不参考什么 |
| --- | --- | --- |
| AirNavigation | UAV 基础操作、RGB/depth/pose 获取、多模态调用方式参考 | 整体架构、UI、任务循环 |
| AirVLN | C/S、场景进程管理、多 scene/machine/client 扩展 | 训练框架、大模型 baseline |
| UAV-Flow | 外部 Flow benchmark、UnrealCV/gym_unrealcv 对象操作能力参考 | 第一阶段地图主模型、OpenVLA 训练主链路、主 runtime loop |
| QiClaw | Agent runtime、planner、gateway、records、Electron 预留 | 具体业务 Agent、微信/企微/抖音技能 |
