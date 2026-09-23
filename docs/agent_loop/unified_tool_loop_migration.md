# 单一工具循环迁移与离线验收

分支：`codex/unified-tool-loop`。最初迁移未调用真实 Provider、SSH 或模拟器；
后续工具契约改造进行了文末记录的真实模型离线选择评测；周期提醒改造另进行了
真实模型与仿真验证，结果见下方链接。历史实验与研究文档保留，不宣称导航行为改善。

## 当前架构

原始指令／适用协议 → MissionState 与 TaskContextInput → Context / Transcript /
HistoryProjector → ModelActionPlanner → 原生工具 → 结果与观察 → 下一轮。

- 正式 TaskRuntime 构造器与 run_task 是唯一入口；没有 Parser 或 parsed-task 模式。
- 计划由模型显式维护，runtime.update_plan 管理稳定 ID 和 revision；复杂任务在首次物理动作前受提示词引导建立计划，简单单步任务可直接行动；完成计划不自动结束任务。
- Knowledge 按需显式加载，正文只经 ToolResult 进入历史，当前状态只保存目录和加载索引。
- 1–3 个物理调用直接组成批次，无 commit；Safety 和 physical_window_v2 保留。
- 每指令 300 个物理窗口、10 轮非物理限制、6 次连续协议拒绝；计划更新不重置限制。
- CLI、Interactive、AerialVLN、UAV-Flow、Travel-UAV 使用原始指令入口。
- RGB、20/3 图片选择、通信层、动作符号、landing 协议和指标算法不改。

## 规划、知识调用与周期提醒（2026-09-22）

`agent_tool_loop_system.txt` 按任务特征引导规划：多个有序子目标、依赖地标的
动作，或需要跨观察跟踪进度时，首次物理动作前调用 `runtime.update_plan`。
计划记录结果与顺序，不预先编造未观察区域的飞行参数；新证据可以改变计划。
行动方法不明确且目录有相关知识时，先调用 `knowledge.load`。现场事实不足则
由观察和感知补充；不重复加载已获得的相同知识，不机械重写未变化的计划。

Runtime 将每条任务从 1 开始的 `decision_index` 传给 Planner。默认在第 6、11、16
次请求的完整历史之后追加 `task_progress_reminder.txt` 系统消息。动作、只读、
计划和协议拒绝共用决策时钟；HTTP 重试复用原请求，内部采图和悬停不计数。
提醒只属于当次请求，不写入 Transcript，也不在以后轮次回放；现有请求审计
保留真实发送内容。提醒是进度检查，不是周期性强制调用计划或知识工具。

配置 `FLYAGENT_AGENT_TASK_PROGRESS_REMINDER_INTERVAL=5`；设为 0 关闭周期提醒，
基础指导仍然生效。该值保存在 ModelConfig 的脱敏摘要中，不进入 ProviderSettings。
独立调用 `plan_turn` 时省略 `decision_index` 不触发提醒；正整数序号由调用者负责。
当前自定义任务提示文件仍会覆盖默认任务提示正文，周期提醒独立受上述配置控制。

两个开发验证入口均放在 `tests/manual/`：

- `task_progress_guidance_probe.py`：两个已有 run 的第 1、6 轮，三组、三次重复，
  共 36 次单响应请求；不执行工具。先用 `--validate-only` 检查所有图像和输入。
- `task_progress_simulation_probe.py`：`--benchmark aerial|travel --group baseline|guided
  --output-dir <新目录>`，沿用生产 runner 执行固定案例，保存轨迹、请求与 HTML。
  需要现有模型配置和 `FLYAGENT_SSH_KEY`；基线只恢复已保存的旧提示与工具描述，
  两组共享当前运行时、传感器、安全和评测规则。案例固定用于诊断，不代表整体成绩。

基线保存在 `tests/fixtures/task_progress_baseline.json`；历史提示词审计全文不覆盖。
工程测试通过与导航效果改善分别报告，工具调用次数增加不能替代任务结果验证。
本次全量回归 1093 passed；36 次模型请求及四次完整实飞均未出现计划/知识调用。
逐例结果、提醒审计及额外补跑说明见[实现及验证报告](task_progress_guidance_results_20260922.md)。

两个探针均支持可选 `--model qwen3.8-flash`，只覆盖实验进程中的模型配置，
不修改 `.env`。录制模型、请求模型和返回模型标识分别审计；缺少返回标识记为 null。
未指定时保持原模型及录制模型一致性检查。跨模型探针遇到不兼容原生工具响应时
保留该响应后停止，不自动切换端点、降级图片或修改提示词；`--validate-only` 不发请求。
Flash 与 VL-Plus 的对照结果、实飞中断及协议错误见[跨模型验证报告](qwen_flash_comparison_20260922.md)。

## 已删除的旧执行链

删除 src/agent/semantic、agent/phase.py、自动 skills/selection.py、
skills/semantic_criteria.py、旧控制 registry/recommendations/semantic_plan_update、
phase_window_context.py 和 runtime_control_config.py。删除旧纯解析及自动选 Skill 探针。
通用 JSON 辅助、PlannedStep、稳定控制和通信测试迁移保留。

旧 Phase 报告读取仍支持；不提供旧 Runtime 执行别名。
废弃 parser 配置和旧 Runtime-control 环境配置明确拒绝。

## 生命周期和审计

- 新场景获取后立即承担清理责任；初始化审计失败回滚新场景，不停止已有复用场景。
- 原始错误不被失败记录、最终观察、shutdown 或 stop 的次级错误覆盖。
- 终态刷新与 physical window 自动 after observation 分开核对。
- 计划 proposal / applied revision / checkpoint 一致性、Knowledge 增量回执和终态引用参与严格审计。
- UAV-Flow manifest 保存 loop_schema；Campaign 拒绝缺失或混合循环版本。
- 无计划是正常状态；主报告显示任务、可选计划与 Decision。
- 脚本回放醒目标注非真实模型；没有实际请求图片时，只显示明确标注的传感器 RGB，
  不把它当成模型曾收到的图片。完整原始审计文件保留。

## 覆盖迁移清单

| 旧测试职责 | 当前覆盖／退役理由 |
| --- | --- |
| Parser/Updater 的语义结构、Phase 划分、done_when 纠错 | 接口已删除，专用语义断言退役；通用 JSON 读取移至 test_model_support，通信覆盖由 test_llm_*、test_provider_* 保留 |
| 固定 Phase／strategy 推进 | test_task_plan、test_mission_state、test_builtin_control_tools；检查稳定 ID、版本、显式取消及证据边界 |
| 自动模型选 Skill、Phase 缓存 | 专用选择行为退役；test_knowledge_load_tool、test_plugins、test_skill_tools 检查发现、完整正文、路径安全与增量加载 |
| Context 与控制注册 | test_context_builder、test_agent_transcript、test_model_action_planner、test_control_routing、test_native_turn_validation；状态投影、唯一注册、合规组批、非法参数拒绝 |
| 旧 TaskRuntime 的阶段装配 | test_task_runtime 保留初始位姿、复用、modalities 和 preflight 故障；test_unified_tool_loop 使用正式构造器覆盖直接动作、知识、计划、终态、两条指令隔离、窗口限制 |
| Safety、物理失败和稳定等待 | test_safety、test_stabilization 与 unified-loop 中改参、首动作失败／后续 skipped、稳定失败、相机失败和 turn→move 顺序覆盖 |
| Provider 失败、空 content、协议闭合 | test_model_action_planner、test_agent_transcript、test_native_tool_loop_real_audit、共享 llm 测试 |
| 清理与原始错误保留 | unified-loop 中获取后记录失败、Provider+stop 双故障、记录失败、shutdown+stop 双故障、终态观察与场景释放 |
| Benchmark Parser fixtures | 三个 test_*_runner_dry 迁移为 native AssistantTurn + fake backend，通过正式入口落盘和清理 |
| 图像／历史／轨迹 | Context、history_projection、benchmark_core、native audit、campaign 原有模块测试保留并适配新字段 |
| 完整离线产物 | test_unified_offline_replay 检查 transcript/trace 严格审计、tracking、图片安全引用、计划/知识展示及非真实模型标记 |

测试数量减少包含专用旧接口测试退役，不能用总数直接推导功能覆盖相等。
不再测试已不存在的语义解析质量、自动 Skill 选择、阶段内策略或 commit 接受逻辑。
通用通信、安全、稳定等待、工具闭环、清理与轨迹职责继续各自测试。

## 离线验收产物

先前 checkpoint 的 1019 collected / 5 import errors 已解决。
完整测试结果以本轮最终命令记录为准（下方更新）；没有通过删除失败的通用职责规避异常。

2026-09-22 最终全量：`python -m pytest -q --tb=short`，**983 passed，10 warnings，27.14s**。
警告来自 Pillow 与 msgpack 的弃用接口，不是收集或运行错误。
独立冷启动发现并修复 planning 包的 eager export 循环导入；新增子进程回归，
`native_tool_transcript_probe.py` 返回 `probe_ok`。compileall 无错误。
Memory 双指令 demo 与插件目录 demo 均离线通过；前者场景 start/stop 为 1/1。
浏览器离线截图已核对 CURRENT 传感器图、非真实模型声明、计划与知识回执。

独立目录：
`tests/manual_outputs/unified_tool_loop_final_20260922_v4/`

入口 `review.html`；案例 `cases/replay/review.html`；严格结果 `replay_report.json`。
脚本依次加载知识、更新计划、执行双 move 批次并完成。重复使用录制 RGB，
位姿由 fake backend 确定性计算，不是新场景渲染，不评估 SR/OSR 或导航成功。
v1–v3 调试产物不覆盖。

自动测试另覆盖 blocked、窗口耗尽、Provider／物理／观察／清理故障，
这些故障覆盖不是实际外部服务可靠性的证明。
上述迁移验收未进行真实模型回放与原 AerialVLN Low 单例。

## 工具契约统一改造（2026-09-22）

ToolParameter、ToolRuntimePolicy、ToolSpec、ToolCall 统一定义于 `src/tools/schemas.py`。
参数是 Schema 的唯一维护源；ToolSpec 保留对象级选项与输入示例，生成 input_schema。
插件旧 JSON Schema 在注册边界转换，嵌套结构和扩展保留；注册时用现有校验器验证示例。
Provider 导出继续剔除内部 x-flyagent 扩展，执行校验仍使用完整生成 Schema。

工具定义仅经 Provider tools 发送。ContextBuilder 不生产工具清单或参数 Schema；
ToolLoopContext 及其旧测试已删除，有效检查迁入上下文、历史投影和模型请求测试。
HistoryProjector 仅清理请求投影中的旧重复字段，原始历史和工具记录不变。
模型工具集合取 model_exposed 与 Runtime 场景允许名单的交集；hover 始终隐藏，
稳定、安全替代及终态清理继续由 Runtime 调用。不新增 ToolAvailability 层。

任务 Prompt 保留目标、观察和反馈循环；ToolUser 解释通用字段和执行结果，
CaseStudy 单独演示参数与依赖。工具适用场景归工具描述，参数含义归 ToolParameter。
复杂任务首轮计划属于软引导，不增加强制计划门槛或固定工具路线。

自动回归覆盖参数转换、示例拒绝、隐藏工具、上下文去重、历史保留、安全与完整循环。
最终全量：`python -m pytest -q --tb=short`，**1008 passed，10 warnings，29.02s**；
警告来自既有 Pillow/msgpack 弃用接口。
新增 `tests/manual/tool_contract_behavior_probe.py` 使用生产请求装配进行六次真实模型请求，
不执行任何返回工具。qwen3-vl-plus 在三个复杂任务均首轮计划，三个简单任务均直接选择
对应转向或平移动作（6/6），所有调用通过原生参数/组批校验。
这是合成状态的首轮选择小样本，不证明导航成功率、真实感知质量或多轮策略改善。

## 工具描述规范化（2026-09-22，后续修订）

7 个可见工具说明统一为功能、适用需求、返回内容与能力边界；删除重复调用顺序、
定位参数中的使用时机和计划路线式示例，补齐移动速度单位。图像、身份与启用知识
同步清理重复引导及 hover 建议。10 个内部工具定义和安全/终态/组批校验不变。

全量回归：1026 passed，10 条既有弃用警告。探针现支持十场景的前后配对检查，
包含默认身份与偏好，不能与上面的旧六场景结果直接比较。本次共 20 次真实模型请求，
不执行工具；参数/组批检查全部通过，预设首轮选择从 9/10 变为 8/10，复杂任务首轮
计划从 2/3 变为 1/3。该回退如实保留，不宣称描述规范化消除了模型偏好。

新版全文见 [工具与 Prompt v2](tool_prompt_audit_20260922_v2.md)，
具体变化和复现入口见 [差异与验证](tool_prompt_cleanup_diff_20260922.md)。

## Flash 错误反馈与受监督诊断（2026-09-22）

本地测试模型现为 `qwen3.8-flash`。Planner 严格拒绝非法 JSON，反馈包含工具名、
出错位置和有限上下文，不修补数字或执行被拒绝调用。终态 evidence_refs 只接受
本任务观察 ID 或成功环境工具结果的调用 ID/结果引用；图片帧、路径、计划和知识
引用不属于环境证据。拒绝反馈列出无效项及合法示例，新鲜度与完成条件不变。

手工实飞探针增加父进程监督，保留原生崩溃现场并核验所属场景及隧道清理。
阶段追踪、注意力对照和编码器拆分只用于诊断，不改变生产感知默认配置。
实测结果、未解决的 SAM CPU 原生崩溃和重放入口见
[Flash 实飞错误调查](flash_flight_error_diagnosis_20260922.md)。
