> Status: Historical

# 前端预留策略

## 1. 结论

前端放到后期。

第一阶段只做 CLI，等 UAV Agent 能在 UE 中完整运行后，再参考 QiClaw 接 Electron 前端。

## 2. 为什么后做前端

当前最大风险不在 UI，而在：

- 仿真环境能否稳定启动。
- UAV 原子飞控是否稳定。
- RGB/depth/pose 是否能统一获取。
- Planner 是否能用短期滚动方式安全执行。
- VLM 多模态判断是否能稳定进入运行闭环。

如果前端过早介入，会把调试复杂度拉高。

## 3. QiClaw 可参考内容

路径：

```text
F:\openclaw-main\wojiazaiyugang.QiClaw\QiClaw
```

参考点：

- `src/hosts/electron/main.js`：Electron host。
- `src/renderer/`：前端界面。
- `src/core/assistant-service.js`：统一业务入口。
- `src/core/gateway/`：run/session/control plane。
- `src/core/records/`：记录和 artifact。
- `src/core/records/`：运行记录和 artifact。

## 4. FlyAgent 前端边界

未来 Electron 前端只能调用 runtime/gateway，不直接调用 UAV 底层。

允许：

- 提交任务。
- 查看任务状态。
- 查看 RGB/depth/pose 观测。
- 查看轨迹。
- 查看工具调用和 VLM 判断日志。
- 点击 hover/pause/stop。

不允许：

- 直接调用 AirSim client。
- 绕过 Safety Supervisor。
- 绕过 Planner 发送高危飞控命令。

## 5. 未来 UI 模块

```text
ui/
  electron/                         # Electron 前端预留目录
    README.md                       # 前端边界、启动方式、事件协议说明
    main.js                         # Electron 主进程
    preload.js                      # 安全桥接 runtime API
    renderer/                       # 页面渲染层
      index.html                    # 页面入口
      app.js                        # UI 逻辑
      styles.css                    # 样式
```

## 6. Runtime Event Schema

前端订阅 runtime 事件，不订阅底层 UAV client。

示例：

```json
{
  "run_id": "run_001",
  "event": "short_horizon_completed",
  "phase": "orienting",
  "timestamp": "2026-04-29T10:00:00Z",
  "payload": {
    "actions": [
      {"tool": "uav.turn_relative", "ok": true},
      {"tool": "uav.hover", "ok": true}
    ],
    "observation_id": "obs_001",
    "vlm_scene_judgement": {}
  }
}
```

常见事件：

- `run_started`
- `semantic_plan_created`
- `short_horizon_created`
- `tool_call_started`
- `tool_call_completed`
- `observation_updated`
- `vlm_judgement_updated`
- `trajectory_updated`
- `safety_warning`
- `run_completed`
- `run_failed`

## 7. 飞书预留

手机端暂不单独设计一套 API。

未来固定通过飞书作为入口：

- 飞书消息提交任务。
- 飞书卡片查看 run 状态。
- 飞书通知异常、完成、需要人工确认。

飞书 Gateway 和 Electron Gateway 都应接 runtime，不接 UAV 底层。
