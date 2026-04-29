# 参考项目说明

## 1. AirNavigation

路径：

```text
C:\Users\HP\Desktop\AirNavigation\AirNavigation
```

定位：

- Python/Qt + AirSim 的 UAV 自主导航仿真系统。
- 已包含飞控、图像获取、深度/分割、目标检测、目标跟踪、轨迹记录等能力。

### 1.1 建议复用文件

`utils/UAV_controller.py`

- UAV 控制主封装。
- 包含连接、起飞、降落、悬停、速度控制、飞到坐标、相机旋转、状态记录。

`docs/uav_controller.md`

- 已整理 wrapper API。
- 适合作为 Agent 工具层的接口来源。

`utils/UAV_navigater.py`

- 视觉导航逻辑。
- 包含目标检测、bbox 到角度偏差、相机调整、目标跟踪。

`ai/ai_vision.py`

- 使用多模态视觉模型根据图像和描述返回目标 bbox。
- 可以作为 `vision.detect(description)` 的早期实现。

`utils/CommandDecorator.py`

- 命令注册装饰器。
- 可借鉴为 Python 版工具注册表。

### 1.2 需要注意的问题

- 部分源码注释存在编码显示问题，但 API 名称和结构仍可读。
- `ai_vision.py` 中存在硬编码 API key，后续必须迁移到环境变量。
- 当前视觉导航偏目标追踪，尚未形成完整 Agent 任务管理层。
- 深度信息已有获取入口，但需要进一步确认精确距离计算方式。

## 2. OpenClaw

路径：

```text
F:\openclaw-main\openclaw-main
```

定位：

- 大型 Agent/移动端/工具网关项目。
- 本项目只轻量参考架构思想，不照搬完整系统。

### 2.1 建议参考文件

`apps/android/app/src/main/java/ai/openclaw/app/node/InvokeCommandRegistry.kt`

- 能力清单和命令注册。
- 可借鉴工具的名称、可用性、权限条件设计。

`apps/android/app/src/main/java/ai/openclaw/app/node/InvokeDispatcher.kt`

- 统一工具分发。
- 可借鉴未知命令、权限不可用、后台不可用、handler 调用等错误处理方式。

`apps/android/app/src/main/java/ai/openclaw/app/gateway/GatewaySession.kt`

- WebSocket RPC、事件、请求、响应和 invoke result。
- 可作为手机端 API 通信参考。

`LocationHandler.kt`、`MotionHandler.kt`、`CameraHandler.kt`、`DeviceHandler.kt`

- 可参考手机端定位、运动、相机、设备状态的工具能力。

### 2.2 适合借鉴的思想

- 工具能力清单。
- 工具可用性检查。
- 统一 dispatcher。
- 统一错误结构。
- 手机端能力作为 node 工具暴露。
- 权限、前台状态、超时等安全检查。

### 2.3 不建议照搬的内容

- 完整 Android/iOS App 架构。
- 复杂网关认证系统。
- 大量通信、UI、插件和多端平台代码。
- 与 UAV MVP 无关的日历、短信、联系人等能力。

## 3. 对本项目的结论

本项目应采用：

- AirNavigation 作为 UAV/视觉能力底座。
- OpenClaw 作为工具系统和手机 API 接入参考。
- 自己实现轻量 Agent 层。

最小闭环优先级：

```text
自然语言任务 -> 结构化步骤 -> 安全 Planner -> 工具调用 -> UAV 观察 -> 状态更新 -> 完成/失败
```

