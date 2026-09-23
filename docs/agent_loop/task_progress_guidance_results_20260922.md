# 规划、知识指导与周期提醒：实现及验证

工程机制已通过验证；当前 qwen3-vl-plus 实验没有解决计划和知识工具漏调用。
提醒能促使部分文字回顾，但不能把文字回顾视为已调用 `runtime.update_plan`，也不能据此宣称导航改善。

## 实现

- `src/agent/planning/prompts/agent_tool_loop_system.txt`：按指令复杂度指导首次物理行动前规划，区分方法不明确与现场证据不足，允许简单单步直接执行。
- 同目录 `task_progress_reminder.txt`：回顾原始要求、按证据更新进度或修订路线、按需读取未加载知识。不要求每次提醒都调用工具。
- `ModelActionPlanner.plan_turn(..., decision_index=None)`：Runtime 传入每条任务从 1 开始的决策序号；在完整历史之后追加当次系统提醒。独立调用未传序号时不提醒。
- `ModelConfig.task_progress_reminder_interval`：默认 5，环境变量 `FLYAGENT_AGENT_TASK_PROGRESS_REMINDER_INTERVAL`；0 关闭，负数拒绝。进入脱敏摘要，不进入 ProviderSettings。配置示例已同步，实际 `.env` 未修改。
- `runtime.update_plan`、`knowledge.load` 的工具描述与基础指导一致。计划保留子目标顺序与稳定 ID，不编造未观察区域动作参数。
- 提醒不写入 Transcript；由原有请求审计保存。未增加飞行动作拦截，未修改通用 LLM 通信层、知识正文、终态判断和安全预算。

## 自动验证

最终执行 `python -m pytest -q`：**1093 passed，10 warnings，32.42s**。
警告为 Pillow/msgpack 弃用提示。

覆盖默认值与关闭、5/6/10/11 边界、自定义间隔、新任务归零、拒绝/读取/物理轮次计数、HTTP 重试请求一致、历史无污染、工具调用结果配对和请求快照。
专用探针还验证三组不改变捕获的工具参数/安全边界，首轮两种新方案请求相同。

中间回归曾受共享工作区中的感知代码缺失导入、实飞占用控制端口影响。未通过删除测试绕过；最终在仿真清理后完整通过。工作区原有和其他任务修改均保留。

## 36 次固定上下文模型对照

使用两类基准已有运行的第 1、6 轮，每个样本三组、每组三次，共 36 次逻辑请求。
恢复原请求与图像并核对哈希；不执行模型返回动作。三组为旧提示/描述、基础指导/新描述、基础指导/新描述加提醒。这里“基础指导”包含同步的工具描述变化，并非只改变一个文本文件。

| 组别 | 响应数 | 计划调用 | 知识调用 | 返回工具概况 |
| --- | ---: | ---: | ---: | --- |
| 基线 | 12 | 0 | 0 | 定位 7、移动 3、转向 2、blocked 1 |
| 基础指导 | 12 | 0 | 0 | 定位 9、移动 3、转向 1 |
| 基础指导＋提醒 | 12 | 0 | 0 | 定位 6、移动 5、转向 1 |

部分响应包含两个调用，工具次数可超过响应数。36 次均通过探针的名称、ID 和参数解析检查；这不替代完整 Runtime 安全/执行校验。
AerialVLN 首轮三组均直接移动，中途主要继续定位。TravelUAV 首轮为定位或转向，中途为定位/移动/转向或 blocked。
首轮两种新方案输入相同仍有输出差异，不能把重复请求差异全部归因于提醒。

[模型对照 HTML](../../tests/manual_outputs/task_progress_guidance_20260922/review.html) ·
[请求与图像引用清单](../../tests/manual_outputs/task_progress_guidance_20260922/manifest.json) ·
[逐次响应](../../tests/manual_outputs/task_progress_guidance_20260922/results.json)

## 四次完整实飞对照

模型均为 `qwen3-vl-plus`，使用原 benchmark runner、相同案例/数据哈希、相同场景设置、传感器、模型参数及预算。每次启动受管场景并由 runner 重置案例初始位姿。
最终四次运行的记录源码哈希完全相同，运行期间均无记录源码变化；同案例设置哈希相同。TravelUAV 为物理仿真，初始观测高度有约 0.9 毫米差异，不宣称逐像素确定性复现。

| 案例 / 组别 | 决策轮次 | 执行物理动作 | 计划 / 知识调用 | 结果 | 终点误差（米） | 总耗时（秒） |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| AerialVLN 基线（稳定补跑） | 18 | 12 | 0 / 0 | 证据不足，未成功 | 108.63 | 227.7 |
| AerialVLN 完整新方案 | 30 | 18 | 0 / 0 | 证据不足，未成功 | 109.73 | 365.5 |
| TravelUAV 基线 | 5 | 4 | 0 / 0 | runner 判定成功 | 6.96 | 86.1 |
| TravelUAV 完整新方案 | 12 | 5 | 0 / 0 | 证据不足，未成功 | 7.36 | 215.8 |

动作数为 runner 的 `uav_action_steps`，不含读取/规划、被拒绝动作或内部悬停。耗时包含场景准备、模型、感知、飞行及清理，不能解释为纯模型延迟。
AerialVLN 新方案实际提醒轮次为 6/11/16/21/26，TravelUAV 为 6/11；逐请求核对完全符合规则。基线无提醒。

### 逐例行为复核

**AerialVLN：`39GHHAVOMOP8ZV27ILH5P0TV5F6J4A`。** 多地标有序路线应先规划，但两组都直接开始飞行，未建立结构化计划。新方案第 6、11 等轮在公开意图中回顾了步骤，第 11 轮列出了后续棕色建筑、灰屋顶、拖拉机、大灰楼和脚手架要求。不过第 16、21 轮在承认棕色建筑/拖拉机未确认时继续推进，第 26 轮已经执行后续回头步骤。文字中保留顺序不等于执行时满足条件。第 28 轮试图完成，被现有规则以当前观察缺少成功定位结果拒绝，最终 blocked。没有计划修订记录，无法认定进度更新机制生效。

**TravelUAV：`c3f9fc36-7c55-4a48-8213-5adce9b35b12`。** 基线首轮转向，新方案首轮定位、随后转向，首次物理行动前均无计划。新方案多次定位紫色摩托车未确认，转查电话亭/垃圾桶并调整视点；第 8、10、11 轮公开意图表示重新考虑方向或搜索方式，仍未加载知识，最后以证据不足停止。基线以目视解释完成，runner 判成功；其公开描述同时承认电话亭颜色不符并归因于光照，不能把 runner 成功标记当成每个语义细节均经独立验证。

**知识调用判读。** 两类上下文都提供 `visual_target_task_reasoning` 等知识目录，AerialVLN 另有相关路线/高度指导可选。单纯“未检测到目标”属于现场证据不足，继续观察/定位本身合理，不能一律记为知识漏调用；在反复无进展、开始重新选择搜索/路线方法时则存在加载指导的机会。本次这些机会未产生知识调用，因此也没有“加载后应用到决策”的证据。不能断言加载知识就一定能解决定位问题。

### 产物和偏差记录

- [AerialVLN 稳定基线 HTML](../../tests/manual_outputs/task_progress_aerial_baseline_retry_20260922/evaluation/review.html)
- [AerialVLN 新方案 HTML](../../tests/manual_outputs/task_progress_aerial_guided_20260922/evaluation/review.html)
- [TravelUAV 基线 HTML](../../tests/manual_outputs/task_progress_travel_baseline_20260922/evaluation/review.html)
- [TravelUAV 新方案 HTML](../../tests/manual_outputs/task_progress_travel_guided_20260922/evaluation/review.html)
- [四次运行汇总、run 路径和轨迹指标](../../tests/manual_outputs/task_progress_guidance_20260922/simulation_summary.json)
- [远端进程组及端口清理核验](../../tests/manual_outputs/task_progress_guidance_20260922/cleanup_verification.json)

实际共启动 6 次任务：原定 4 次，另有 2 次基线补跑。初次 AerialVLN 基线遇到共享源码变化，仅作探索记录保留在 `task_progress_aerial_baseline_20260922`，不纳入上表。第一次补跑在第 4 轮后进程以代码 1 意外退出，无 traceback、无任务终态；原因未确定，不能归因于导航策略，也不能假称完成。该次独立[中断与清理记录](../../tests/manual_outputs/task_progress_aerial_baseline_stable_20260922/interruption.json)保留，按场景所有权停止遗留进程/隧道并验证端口关闭。随后串行重试成功形成上表基线，不以更好导航成绩选择运行。

最终四次完整运行均有请求快照、图像引用、事件、轨迹及 HTML；清理后四个受管进程组均无存活成员，远端 30000/30001 端口均不可达。

## 验收结论与复现

提醒时机、开关、历史完整性及完整回归通过。行为目标未达成：首次行动前计划、结构化进度更新、方法不明时知识加载均未在这些真实样本中出现。保持软引导实现，不自动升级为运行时强制调用。
模型对照仅反映固定上下文中的单次选择；四次实飞是初步诊断，不能推断总体成功率或稳定因果收益。

手工入口为 `tests/manual/task_progress_guidance_probe.py` 和 `tests/manual/task_progress_simulation_probe.py`；先用 `--validate-only` 检查环境，再使用新的输出目录执行。后者的固定案例选择只在手工探针中，生产指导不按基准名称分支。
