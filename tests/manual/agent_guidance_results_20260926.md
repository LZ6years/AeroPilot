# UAV-Agent 分阶段实现与实测结果

代码实现已完成，基础契约测试通过；真实模型仍存在错误识别、误完成和工具描述遵循
问题。本报告不宣称导航问题已解决，也不宣称离线经验改善了飞行。

## 实施交付

| 阶段 | 主要文件 | 结果 |
|---|---|---|
| 基线 | `tests/fixtures/agent_guidance_baseline_20260926.json` | 原始 20 例索引、模型名、源码/记录散列；原日志未修改 |
| grounding 与终止 | `plugins/capabilities/visual_grounding/locate/capability.json`、`scripts/handler.py`、`src/runtime/control/actions/terminal.py`、`src/runtime/control/executor.py`、`src/tools/registry.py` | 新描述进入 provider；删除任务级结束前重测门槛；保留引用真实性、任务隔离和结果配对 |
| 五份知识 | `plugins/knowledge/uav_flight/*/SKILL.md`、路线 reference、`src/tools/builtins/knowledge.py` | 四项 description 字符串；稳定名称与自主加载；正文保持紧凑，原 reference 路径保留 |
| 动作与 Plan | `src/agent/planning/prompts/agent_tool_loop_system.txt`、`src/tools/builtins/planning.py` | 实际反馈估计窗口、观察位置与批量依赖；结果/条件式 Plan；允许修正已完成判断 |
| 任务介绍 | 两个 benchmark 的 `protocol.py` | 任务目标、指令解释、信息特点、结束条件；保留原 landing 协议含义 |
| 独立经验实验 | `plugins/research/flight_experience/`、`tests/manual/flight_experience_probe.py` | 只读适配、原图索引、复盘、独立审核、引用校验及隔离报告；未接入 Core 或正式知识 |
| 对照入口 | `tests/manual/agent_guidance_probe.py` | 恢复原图与真实历史请求，构造明确标记的新离线对照请求；不执行返回的工具 |

主模型仍为配置中的 `qwen3.8-flash`。未调整安全限值、深度一致性阈值和 1–3 动作
窗口限制，未实现 recall。工作区原有及并行产生的其他修改保留在原处，没有纳入本轮提交。

## 自动测试

- `python -m pytest -q --tb=short`：**1213 passed，10 个依赖弃用警告**。
- 最后补充 grounding 运行失败说明后：相关 27 项通过。
- knowledge description 加引号，保证含冒号的 frontmatter 字符串合法：相关 24 项通过；
  解析后的模型文字不变。
- 独立经验和固定上下文探针：相关 17 项通过，覆盖原图缺失、越界路径、无效引用、
  独立审核、修订校验、输出重跑保护、失败落盘、时序片段选择和非 Core 依赖边界。

原始 20 条 transcript 的 SHA256 再次核验全部一致。全量测试包含工作区已有的测试，
不能把全部 1213 项都描述成本轮新增。

## 固定历史上下文探测

6 个片段：brown query、beige query、出租车结束、道路尽头转弯、79° 初始 Plan、
前进窗口。每个发送 baseline/revised 两个新请求，共 12 次；另外有初始接入的 2 次请求。
全部保留原图和历史工具结果；不是重跑出来的“原始日志”，也不是新闭环轨迹。

真实观察到的结果：

- 新版 brown query 有缩短，但 beige 片段仍生成 `beige flat roof surface directly below`。
  description 修改没有完全解决不受观察支持的关系词。
- 79° 新版 Plan 区分了转向、可行空间中的搜索、接近确认；同一响应却混用了
  `knowledge.load` 与 `runtime.update_plan`，现有校验会拒绝 `mixed_effects`。
- 出租车新版输出了合法完成请求；baseline 输出多带了不允许的 `blocking_condition`。
  终止 gate 的修复另由执行器契约测试验证，单次离线回复不能验证其运行效果。
- 前进片段的请求从 3 m/s × 5 s 变为 4 m/s × 5 s，均为单动作。这里只能说明输出变化，
  不能证明 20 米动作更合理，也不能证明模型已学会批量执行。
- 12 个响应的离线参数/批次检查中，10 个合法、2 个拒绝；这不是导航成功率。

产物：

- `tests/manual_outputs/agent_guidance_contexts_20260926/index.json`
- `tests/manual_outputs/agent_guidance_context_responses_20260926/*/probe.json`
- 同目录 `contract_assessment.json`：独立参数/批次检查，未重建历史 read quota，未执行工具。

## 真实闭环

两个已知问题案例完成执行和资源清理，但都未达到 benchmark 成功条件：

| 案例 | 历史距离误差 | 本次距离误差 | 本次动作数 | 本次成功 |
|---|---:|---:|---:|---|
| AerialVLN `39GHHAVOMOP8ZV27ILH5P0TV5F6J4A` | 47.77 m | 56.87 m | 23 | 否 |
| TravelUAV 79° / `Carla_Town10HD__53e4e305-2f7f-46ef-ad07-554ba86b380f` | 32.61 m | 61.26 m | 31 | 否 |

两例都出现“模型自报完成，但 benchmark 未成功”。AerialVLN 本次 21 个实际尝试的
物理窗口，其中 19 个单动作、2 个双动作；grounding 成功执行 10 次，空候选 5 次，
另有 2 个调用被 observation quota 拒绝。TravelUAV 本次 31 个物理窗口均为单动作；
grounding 成功执行 18 次，空候选 12 次。没有按减少调用次数推断质量提升。

TravelUAV 会在转向后继续移动搜索，最终累计方向接近 79°，但仍误认了目标。该次
运行还包含 3 次转向控制器错误、1 次碰撞阻断结果及 9 次非法 JSON 参数拒绝。
动作执行错误可能伴随实际运动，统计没有把这些窗口当作“未发生移动”。
`oracle_success=true` 仅表示轨迹曾满足评测的位置半径，不证明模型当时看见了目标。

两例耗时约 479 s / 946 s；provider 报告 token 总量分别为 1,954,981 / 4,596,384。
未提供计费单价，货币成本保持未知。过冲、无效转向等需人工语义判断的指标保持未知，
没有把未核对的项目写成零。

这些比较还受到本轮之前的控制实现修改影响，不能把距离差异全部归因于新提示。
工作期间执行诊断脚本也有并行修改，各尝试保留自己的源码散列和实际请求。

### 未参与修改的案例与阻塞

第 18 例 `Carla_Town01__5a7c4254-ad4f-4fda-a3c7-569094b61fc1` 在飞行前因
`No simulator control ports available` 失败，模型调用和动作数均为零。
本机 35000 端口当时由非本实验拥有的 SSH 隧道占用，没有终止该进程。

旧监督汇总将空清理列表标为 `cleanup_failed`。额外核查确认这个尝试没有创建
`scene_instance.json` 或运行文件，属于分配前失败，没有该尝试遗留的场景需要停止。
原汇总保留，另写 `allocation_verification.json` 解释这一差异。

**完整 20 例基础改进版对照尚未执行完。** 不能把两个失败的已运行案例和未运行案例
合并成完整成功率。继续批次前需释放共享端口，并重新冻结当前执行源码；不能直接修改
旧 manifest 的散列来掩盖源码变化。

闭环产物：

- `tests/manual_outputs/agent_guidance_closed_loop_20260926/guidance_metrics.json`
- 同目录 `comparison.json`、`port_contention.json`、各案例原始 audit 与实际 provider 请求。
- `tests/manual_outputs/reviews/guidance_20260926/index.html`：短路径 HTML 审阅入口。

## 独立经验实验结果

适配器读取了历史批次的 21 个运行目录，包含一次重试，共 2605 个时序事件、504 张
原始 RGB；全部原图可读取。正式 20 例统计仍按基线索引选取，不重复计算重试。

真实模型实验共 7 次尝试，**1 次完成、6 次因超时或输出契约失败而失败，0 条经验采纳**。
这些是独立实验结果，不计入导航效果。

- 一个 TravelUAV 实验完整执行复盘与审核。3 个候选中，结构校验发现缺少时序引用、
  引用了本轮未提供的图像；审核指出另外一个候选把原飞行的错误视觉描述当成事实。
  修订回复又未提供约定的结构化 candidate，因此仍暂存。
- 出租车指定片段产生了“无效距离时照着图像位置开环前进”和“结束前总要重测”的
  候选，说明模型可能复述错误或过度概括旧协议。后一候选引用不完整被暂存，前者的
  独立审核请求超时。没有将它们写入采纳文档。
- hangar 指定片段实验返回顶层 JSON 数组 `[]`，不符合约定的对象结构，按契约失败
  处理，没有伪装成成功的空经验列表。
- 为处理长记录，增加紧凑时序索引、相邻 checkpoint 片段读取，以及可选
  `--observations` 显式选择原图。紧凑索引减少输入，但当前模型仍可能在选帧阶段超时。
  另外强化了“历史平台拒绝不等于通用飞行规则”的复盘/审核说明。

默认实验复用当前模型连接。部分独立重试将单次超时改为 180 秒、禁用超时重试，
只通过该实验子进程环境变量设置；没有修改主飞行配置。人工审核文件保持 pending，
未声称已完成真正的人审或跨场景迁移验证。

产物总表：`tests/manual_outputs/flight_experience_experiment_summary_20260926.json`。
每个实验目录均保留 `experiment.json`、`report.html`、原图、候选、审核/失败信息，
完整失败记录没有覆盖。使用方法见 `plugins/research/flight_experience/README.md`。

## 后续验收重点

先解决共享测试资源冲突并固定执行源码，再继续对照。真实结果已表明，当前主模型的
视觉身份确认、输出契约遵循和误完成仍需单独验证。经验模块目前适合分析失败机制，
尚无证据支持把其产物用于正式飞行；本轮继续保持与 Core、正式知识及飞行工具隔离。
