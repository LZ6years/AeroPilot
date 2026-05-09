# UAV Agent MVP 计划

## 1. MVP 范围

MVP 只做单机单 UAV 的 UE / AirSim 仿真闭环，先以 CLI 跑通任务，不做完整 Electron 前端，不做多机协同，不做独立手机端能力。

优先支持任务：

```text
前面右转会有一个黑色的路灯，右转面向路灯后停下
```

这个任务不应一次性生成很长的固定动作序列，而应拆成：

- 整体语义目标：右转后寻找黑色路灯，并让 UAV 朝向它悬停。
- 短期执行窗口：每轮只输出少量可验证的原子动作。
- 观测更新：每次动作后重新读取 RGB/depth/pose，再决定下一步。
- VLM 判断：用多模态大模型理解当前图像和任务上下文。
- 困惑提问：Agent 不确定且继续飞行有风险时，先 hover，再向用户提问。

## 2. 阶段划分

C/S 边界以 [C/S 架构职责划分](C:/Users/HP/Desktop/UAV-Agent/docs/cs_architecture.md) 为准。

```text
Agent Runtime
  -> LocalEnvClient
    -> LocalSimulationServer
      -> One AirVLN-style Scene
      -> One UAV
```

### M1：UAV 原子飞控封装

目标：

- 从 AirNavigation 中抽取 UAV 基础操作。
- 只封装原子飞控和原始观测，不继承它的整体架构和地图架构。

交付：

- `src/adapters/airnavigation_uav.py`
- `src/adapters/uav_primitives.py`
- `src/tools/registry.py`
- `src/tools/dispatcher.py`

原子动作：

- `uav.connect`
- `uav.takeoff`
- `uav.hover`
- `uav.land`
- `uav.move_body`
- `uav.turn_relative`
- `uav.ascend`
- `uav.descend`
- `uav.get_pose`
- `uav.get_frames`

验收：

- 可以连接 UAV。
- 可以起飞、悬停、转向、低速移动、降落。
- 可以获取 pose、RGB、depth。

### M2：AirVLN 风格地图资源与场景管理

目标：

- 参考 AirVLN 的 scene 管理方式管理地图资源。
- 使用 `MapPackage` 描述静态地图资源。
- 使用 `SceneInstance` 描述一次运行中的地图实例。
- 由 `SimulationServer` 负责查找 scene、生成 AirSim settings、分配端口、启动进程。
- UnrealCV / gym_unrealcv 放到后续 backend，不作为第一阶段主模型。

交付：

- `src/env_client/base.py`
- `src/env_client/local_client.py`
- `src/env_client/schemas.py`
- `src/sim_server/server.py`
- `src/sim_server/scene_manager.py`
- `src/sim_server/process_manager.py`
- `src/sim_server/port_manager.py`
- `src/sim_server/backends/airvln_backend.py`
- `src/sim_server/backends/local_ue_backend.py`
- `configs/maps/airvln_maps.json`

验收：

- 可以从配置加载 AirVLN 风格 `MapPackage`。
- 可以根据 `scene_id` 查找并启动 scene。
- 可以自动生成 AirSim `settings.json`。
- 可以分配 AirSim port 并记录到 `SceneInstance`。
- Agent Runtime 只通过 EnvClient 获取 RGB/depth/pose。
- Agent Runtime 只通过 EnvClient 请求 UAV 原子动作。
- 后续可以把 UnrealCV/gym_unrealcv 接成 SimulationServer backend。

### M3：任务解析与语义规划

目标：

- 将用户自然语言解析成稳定的语义任务。
- 语义任务只描述意图、约束和完成条件，不直接展开为完整动作列表。

交付：

- `src/agent/task_parser.py`
- `src/agent/semantic_plan.py`
- `src/agent/schemas.py`

验收：

- 输入中文任务后能生成结构化语义计划。
- schema 校验失败时拒绝执行。
- 计划不会把“靠近/面向”误写成原子动作。

### M4：短期滚动执行器

目标：

- 每轮根据当前状态和语义计划，只输出未来几步原子动作。
- 每次执行后重新观测并更新状态。
- `connect`、`takeoff`、起飞后稳定悬停和初始观测属于 Runtime preflight，不属于 phase planner 或模型动作规划。

交付：

- `src/agent/short_horizon_planner.py`
- `src/agent/mission_state.py`
- `src/agent/safety.py`
- `src/runtime/main.py`

验收：

- phase 短期窗口不包含 `uav.connect` / `uav.takeoff` / `uav.land` 等生命周期动作。
- run record 能区分 `control_connected`、`preflight_takeoff_completed`、`initial_observation_updated` 和后续 phase 事件。
- 不一次性执行长距离 `move_forward`。
- 动作有速度、角度、时间、距离上限。
- 每轮执行后重新读取观测。
- 异常、超时、低置信度时进入 hover。

### M5：原始观测与状态管理

目标：

- 基础感知层只负责原始数据，不做高级语义。
- 视觉语义判断暂不拆成独立模块，第一阶段统一交给 VLM 处理。

交付：

- `src/perception/raw_observation.py`
- `src/agent/mission_state.py`
- `src/runtime/record_writer.py`

验收：

- 能保存原始图像、深度、位姿、动作日志。
- 目标框、目标距离等视觉语义结果不作为必填状态字段。
- VLM 未确认目标时系统不会假装已经看见目标。

### M6：VLM 临时决策层

目标：

- 第一阶段使用多模态大模型处理图像和上下文。
- VLM 负责“看图理解 + 下一步建议”，但不直接执行控制。

交付：

- `src/cognition/vlm_client.py`
- `src/cognition/context_builder.py`
- `src/cognition/schemas.py`
- `configs/models/vlm.json`

验收：

- 能把当前观测和任务上下文发送给 VLM。
- VLM 输出必须通过 schema 校验。
- VLM 建议动作必须经过 Safety Supervisor。
- 低置信度时只允许观察、悬停、小角度扫描或提问。
- 后续等完整闭环跑通后，再考虑是否做视觉语义能力模块化。

### M7：困惑提问与安全中断

目标：

- 当 Agent 不确定继续行动是否正确或安全时，主动停下来问用户。

交付：

- `src/agent/confusion_detector.py`
- `src/agent/clarification.py`
- `src/runtime/user_reply_handler.py`

验收：

- 触发困惑时立即 hover。
- 每次只问一个问题。
- 用户回复可以更新目标、方向、候选选择或停止指令。
- 用户超时不回复时保持 hover 或按配置安全降落。

### M8：运行记录与 CLI

目标：

- 参考 QiClaw 的 run store / records。
- CLI 是第一阶段主入口。

交付：

- `src/runtime/run_store.py`
- `src/runtime/record_writer.py`
- `src/runtime/main.py`
- `runs/<run_id>/`

验收：

- 每次任务有 run id。
- 每轮保存输入、语义计划、短期计划、动作、观测、VLM 判断、困惑提问、用户回复、错误。
- 保存 `MapPackage`、`SceneInstance`、AirSim port、settings 路径和 scene process 信息。
- 失败时能看到失败阶段和原因。

### M9：Electron 与飞书预留

目标：

- Electron 等 UE CLI 完整跑通后再做。
- 手机端先不做独立 API，未来固定接入飞书。

交付：

- `ui/electron/README.md`
- `src/integrations/feishu_gateway.py`
- runtime event schema

验收：

- 文档明确 UI 只能调用 runtime/gateway。
- 飞书只作为未来消息入口和通知入口。
- 第一阶段不依赖手机传感器 API。

## 3. 优先级

最高优先级：

- UAV 原子飞控封装。
- AirVLN 风格 scene 资源配置和启动。
- EnvClient / SimulationServer 本地伪 C/S。
- RGB/depth/pose 原始观测。
- 短期滚动执行。
- VLM 临时决策。
- hover/timeout/confusion guard。

中优先级：

- 自然语言到语义计划。
- run store 和轨迹记录。
- 主动提问。
- AirVLN 风格 scene 启动 demo 验证。

低优先级：

- 视觉语义能力模块化。
- UnrealCV / gym_unrealcv 对象级操作。
- 多 UAV。
- AirVLN C/S 完整迁移。
- Electron 前端。
- 飞书接入。

## 4. 推荐第一条 Demo

文件：

```text
src/runtime/lamp_right_turn_demo.py
```

流程：

1. 读取 AirVLN 风格 `MapPackage`。
2. Agent Runtime 通过 LocalEnvClient 请求 LocalSimulationServer 启动 scene。
3. LocalSimulationServer 生成 AirSim `settings.json`、分配 port、启动 scene process。
4. EnvClient 获得 `SceneInstance`。
5. 连接 UAV。
6. 起飞或确认已起飞。
7. 解析用户指令为语义计划。
8. 执行短期窗口：先小角度右转或右转到约 90 度。
9. 获取 RGB/depth/pose。
10. 调用 VLM 判断是否有黑色路灯候选。
11. 若 VLM 置信度足够，则用小角度 yaw 调整让目标居中。
12. 若 VLM 困惑或发现多个候选，则 hover 并向用户提问。
13. 满足完成条件后 hover。
14. 输出 run 记录。

## 5. 风险

- AirNavigation 整体架构不适合继承，只能抽取 UAV 基础操作。
- AirVLN 场景包路径、AirSim settings、端口分配和进程 ready 检测需要先验证。
- Windows 本地启动方式可能和 AirVLN Linux packaged scene 不同，需要 `local_ue_backend` 兼容。
- LLM 直接生成动作序列不稳定，必须经过 schema 校验和短期执行器。
- VLM 视觉判断可能幻觉，必须被 Safety Supervisor 约束。
- VLM 视觉判断不稳定时，系统不应执行“靠近目标”等危险动作。

## 6. Phase 语义与 M4/M6 目标形态补充

M3 生成的 phase 不再被视为固定枚举。phase name 是模型对当前阶段的语义命名，主要用于后续模型动作规划和人工复盘，而不是让规则代码直接 `if/elif` 分发动作。

phase 结构应包含阶段位置：

```json
{
  "index": 1,
  "total": 4,
  "name": "turn_to_search_possible_lamp_area",
  "meaning": "根据用户的右转线索，调整视角去寻找可能出现黑色路灯的区域"
}
```

当前 `ShortHorizonPlanner` 是 M4 dry-run 骨架，用于验证 runtime 闭环、日志、Safety 和 ToolDispatcher 调用路径。它不是最终智能规划方式。

最终目标是引入 `ModelActionPlanner`：

```text
当前 phase（含 index/total/name/meaning）
  + MissionState
  + 当前 RGB/depth/pose
  + 最近动作历史
  + 可用 tools/skills
  + Safety 约束
  -> 模型生成 1-3 个短期动作建议
  -> SafetySupervisor 校验/裁剪
  -> ToolDispatcher 执行
```

因此 M4 当前阶段以“可运行、可复盘、安全兜底”为主；真正复杂场景下的下一步动作选择应交给 M6/ModelActionPlanner，而不是依赖固定 phase 名。
