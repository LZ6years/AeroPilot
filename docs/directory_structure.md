# 目录结构说明

建议目录如下，注释写在每个文件夹/文件后面。

```text
UAV-Agent/                              # 项目根目录
  README.md                             # 项目总入口，说明目标、参考来源和文档索引

  docs/                                 # 文档目录
    requirements.md                     # 需求文档：目标、能力边界、VLM 决策、困惑提问、验收标准
    architecture.md                     # 架构说明：语义规划、短期执行、VLM、Clarification、工具层、runtime
    mvp_plan.md                         # MVP 分阶段计划
    mvp_flow_diagrams.md                # MVP 流程图：总闭环、阶段交付、VLM、提问状态机、Demo
    task_example.md                     # 典型任务示例：右转面向黑色路灯后停下
    simulation_environment_strategy.md  # 仿真环境、地图资源、gym_unrealcv、工厂模式
    frontend_strategy.md                # Electron 和飞书预留策略
    reference_notes.md                  # AirNavigation、AirVLN、UAV-Flow、QiClaw 参考说明
    directory_structure.md              # 当前文件，解释目录用途

  configs/                              # 配置目录
    maps/                               # 地图资源配置，每个地图一个 JSON
      suburb_day.json                   # 示例地图配置
    envs/                               # 仿真环境后端配置
      airsim.json                       # AirSim 连接/启动配置
      unrealcv_gym.json                 # gym_unrealcv 环境配置
    models/                             # VLM 使用的模型配置
      vlm.json                          # 多模态大模型配置

  src/                                  # 核心源码
    agent/                              # Agent 决策逻辑
      task_parser.py                    # 自然语言解析为语义任务
      semantic_planner.py               # 生成整体语义规划
      horizon_planner.py                # 短期滚动规划，每次只产生少量动作
      mission_state.py                  # 任务状态管理
      safety.py                         # 安全检查、限速、限距、强制悬停
      confusion_detector.py             # 判断 Agent 是否困惑或风险过高
      clarification.py                  # 困惑时生成问题、等待用户澄清、更新状态

    cognition/                          # 第一阶段 VLM 临时认知层
      vlm_client.py                     # 调用多模态大模型
      context_builder.py                # 构建 VLM 输入：任务、阶段、图像、深度、pose、历史动作
      schemas.py                        # VLM 输出 schema：置信度、目标状态、建议动作、提问

    perception/                         # 原始观测管线
      raw_observation.py                # 保存 RGB/depth/segmentation/pose 引用，不做高级语义

    tools/                              # 工具系统
      registry.py                       # 工具注册表
      dispatcher.py                     # 工具调用分发器
      schemas.py                        # 工具输入输出 schema

    adapters/                           # 外部系统适配层
      uav_primitives.py                 # UAV 原子控制：起飞、悬停、转向、速度控制等
      airnavigation_uav.py              # AirNavigation UAV 控制封装适配
      sim_env_factory.py                # 仿真环境工厂，根据配置创建后端
      airsim_env.py                     # AirSim 环境后端
      unrealcv_gym_env.py               # gym_unrealcv 环境后端
      local_ue_env.py                   # 本地 UE .sh/.bat 启动方式适配
      airvln_env_client.py              # AirVLN C/S 客户端预留

    runtime/                            # CLI 运行时
      main.py                           # CLI 主入口
      config.py                         # 加载配置
      run_store.py                      # 保存 run 状态
      record_writer.py                  # 保存日志、轨迹、观测 artifact、VLM 判断、用户澄清
      user_reply_handler.py             # CLI/未来飞书回复处理

    integrations/                       # 外部消息入口预留
      feishu_gateway.py                 # 未来飞书接入，第一阶段只预留

  ui/                                   # 前端预留
    electron/                           # 未来 Electron 前端
      README.md                         # UI 边界：只调用 runtime/gateway，不直接控制 UAV

  tests/                                # 测试
    test_task_parser.py                 # 测试自然语言解析
    test_horizon_planner.py             # 测试短期滚动规划
    test_safety.py                      # 测试安全策略
    test_confusion_detector.py          # 测试困惑触发条件
    test_clarification.py               # 测试主动提问流程
    test_vlm_schema.py                  # 测试 VLM 输出 schema
    test_tool_dispatcher.py             # 测试工具分发
    test_sim_env_factory.py             # 测试地图/环境工厂

  runs/                                 # 本地运行记录，通常不提交 Git
    run_*/                              # 单次任务记录：计划、动作、观测、日志、提问

  .gitignore                            # Git 忽略规则
```

## 关键边界

- `src/adapters/uav_primitives.py` 只做原子飞控，不做“面向目标”“靠近目标”。
- `src/cognition/` 是第一阶段临时认知层，VLM 只能建议，不能直接执行飞控。
- `src/agent/safety.py` 是所有 UAV 动作的最后安全门。
- `src/agent/clarification.py` 负责困惑时先 hover 再提问。
- 视觉语义能力模块化不进入当前 MVP，等完整闭环跑通后再评估。
- `configs/maps/` 管理地图路径和对象信息，不允许把地图路径散落在代码里。
- `ui/electron/` 和 `integrations/feishu_gateway.py` 只在 CLI 完整跑通后实现。
