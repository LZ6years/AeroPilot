# 工具描述清理：前后差异与验证

## 1. 改动边界

按确认计划完成：7 个可见工具说明、定位参数文字、移动速度单位、共用执行说明去重、公共 CaseStudy、图像提示、身份文件，以及五项启用飞行知识和搜索参考。

保留工具类型与名称、参数类型和约束、默认值、枚举、内部 x-flyagent 扩展、Runtime 属性、10 个内部工具定义、曝光和排序、安全实现及终态校验。复杂任务首轮计划仍是软引导，未改成强制流程。

已删除的是冗余句和计划工具的路线式示例，没有文件级删除。input_examples 接口和注册校验保留。保留了工作区原有其他修改，未提交 Git。

## 2. 七个工具的原始描述对照

### knowledge.load

修改前：

~~~~text
Read guidance for judging progress, choosing an action or recovering when a catalog entry addresses a missing method. The catalog is a directory, not loaded guidance. Reconsider relevant guidance when attempts make no useful progress. Returns newly requested body text and declared references; reuse delivered content. Knowledge supplies methods, not detection or distance measurements of the current scene; do not load every entry by default.
~~~~

修改后：

~~~~text
Read an enabled knowledge entry and its declared references. Applicable when a task needs the methods or explanations described by that catalog entry. Returns newly requested body text and references incrementally; previously delivered text is not resent. The catalog is a directory, not loaded guidance. Knowledge provides methods, not current-scene observations or measurements, and does not execute actions.
~~~~

### runtime.report_blocked

修改前：

~~~~text
Stop the instruction as incomplete when evidence, capabilities or safety prevent meaningful progress. A target not yet visible is not itself a blocker if a safe useful next step exists. Do not require a fixed number of attempts or exhaust all tools before stopping. Returns the accepted terminal state or rejection feedback. Runtime handles final stabilization and cleanup.
~~~~

修改后：

~~~~text
Report the instruction as incomplete with a blocking condition and environmental evidence. Applicable when the task cannot continue safely or meaningfully. Returns the accepted terminal state or rejection feedback; Runtime handles final stabilization and cleanup. This ends the instruction without claiming that its requested outcomes were achieved.
~~~~

### runtime.report_completed

修改前：

~~~~text
Report the original instruction completed. Compare observed results with all original requirements, not just the last action, a detected candidate or plan status. Cite environmental evidence. Returns the accepted terminal state or rejection feedback. Runtime handles final stabilization and cleanup.
~~~~

修改后：

~~~~text
Report the original instruction as completed. Applicable when environmental evidence supports all requested outcomes. Returns the accepted terminal state or rejection feedback; Runtime handles final stabilization and cleanup. Plan status alone is not completion evidence.
~~~~

### runtime.update_plan

修改前：

~~~~text
Create or revise a task list. For complex tasks, use this as the first call: ordered requirements, search followed by another goal, route and final-position obligations, or constraints retained across observations benefit from a plan. A simple single immediate goal can act directly. Keep near-term work concrete and unseen later work broad. Preserve stable IDs and completed work; cancel obsolete steps. Update when observed progress or new evidence changes the plan, not after every action. Returns the saved plan and revision; it never executes flight or supplies environmental completion evidence.
~~~~

修改后：

~~~~text
Create or revise a task plan using stable step IDs. For complex tasks, normally use this as the first call; a simple single-step task can act directly. Returns the saved plan and its revision. A plan records intended work and progress; it does not execute flight or provide environmental completion evidence.
~~~~

### skill.visual_grounding.locate

修改前：

~~~~text
Supplement direct visual reasoning only when a decision-relevant target cannot be reliably localized or distinguished, or its measured distance is necessary. Do not delegate general scene understanding or already clear visibility/location to this detector. Use include_distance=false for unresolved localization and true for decision-critical measurement. Confirm candidate identity from the annotated evidence; distance is camera-to-visible-surface depth, not horizontal or target-center distance.
~~~~

修改后：

~~~~text
Locate a natural-language target in the current RGB observation and optionally estimate candidate distances. Applicable to tasks needing candidate locations or target-distance measurements. Returns candidate information, annotated evidence and, when enabled and available, distance estimates. Candidate identity requires interpretation against the task; an empty detection does not prove absence. Distance is camera-to-visible-surface depth, not horizontal or target-center distance.
~~~~

### uav.move_body

修改前：

~~~~text
Translate in the vehicle's current body frame when the task requires changing position. Choose motion from available route or target evidence. Translation is not a heading-only correction or a completion report. Runtime may limit velocity; inspect actual execution and the post-batch observation.
~~~~

修改后：

~~~~text
Translate the vehicle using body-frame velocity components for a specified duration. Applicable to changing position, including horizontal or vertical movement. Returns execution feedback. Safety intervention may change the applied motion; vehicle displacement alone does not establish distance to a target.
~~~~

### uav.turn_relative

修改前：

~~~~text
Change vehicle heading to inspect a view or face a cue. Turning changes bearing, not physical passage or distance to a target. Results report execution; a new observation follows the batch.
~~~~

修改后：

~~~~text
Rotate the vehicle by a signed relative yaw angle. Applicable to heading changes or inspecting a different bearing. Returns execution feedback. Rotation does not translate the vehicle or establish physical passage or target-relative distance.
~~~~

## 3. 参数、示例与执行说明

- vx/vy/vz 在原有共享语义常量中补充 Body-frame velocity in m/s，正负轴向不变。
- query 说明简化为待定位目标的自然语言描述；observation_id 的 CURRENT 引用要求不变。
- include_distance 只说明 true/false 是否估计距离，删除“只有测量请求、接近阈值或定量完成才开启”的策略。
- update_plan 的 input_examples 从单个路线样例变为空；无其他工具示例变化。
- CaseStudy 的 save_checklist 改为 format_records，仍演示嵌套数组和布尔值；不新增可调用工具。
- Provider 的共用 Execution 删除混批禁令、相同结果复用和依赖等待的重复句；ToolUser 中相应通用规则不变。每项工具的执行属性及定位每观察 2 次的配额仍导出。

## 4. 关联来源的逐行差异

以下是相对于本轮开始前冻结文本的差异摘录，不是相对 Git HEAD，也不包含此前用户改动。仅显示增删行，保持出现顺序；不是可执行补丁。

### AGENT.md

~~~~diff
+- Model-visible descriptions follow purpose, applicable need, returned content,
+  and capability limits. Parameter prose explains meaning, units and effects,
+  not tool-selection priority. The complex-task planning preference is explicit;
+  other descriptions must not introduce an unrelated first/last-resort ordering.
+  Shared call rules belong in ToolUser; per-tool execution attributes come from
+  ToolRuntimePolicy. Knowledge explains evidence and flight methods without
+  prescribing hidden actions or duplicating a capability's selection policy.
+Runtime evaluates those declarations generically from the call arguments.
+Evidence precision and geometric meaning must support the requested judgment;
+this does not impose an order between direct visual interpretation and callable
+localization. Image labels explain the sensor representation, not tool choice.
-Runtime evaluates those declarations generically from the call arguments. A
-numeric mention alone does not force immediate perception: use qualitative RGB
-and optional fused-depth evidence while clearly off threshold, and request precise
-candidate distance only when action choice, user-requested measurement, or
-quantitative completion genuinely depends on it.
-
-Direct visual reasoning precedes supplementary localization: request detection
-only for decision-relevant ambiguity, not as a general scene-description proxy.
~~~~

### memory/FLYAGENT.md

~~~~diff
+- If identity, intent, or action safety is materially unresolved, avoid risky
+  movement unsupported by evidence and reassess what can be established safely.
-- If identity, intent, or action safety is materially unresolved, gather a
-  bounded additional observation or hold position instead of guessing a risky
-  movement.
~~~~

### plugins/knowledge/uav_flight/approach_and_standoff/SKILL.md

~~~~diff
+relation, available depth evidence, and change after translation. A quantitative
+distance judgment needs evidence whose precision and geometric meaning match
+the requested tolerance. Camera-to-visible-surface distance is not target-center
+or horizontal distance. UAV displacement from a local origin is not
-relation, fused depth colors, and change after translation. A numeric request
-does not require precise perception while the relation is visibly far from its
-threshold. Near the threshold, when correction magnitude depends on it, or
-before quantitative completion, use available visual grounding with candidate
-distance enabled. Its result is camera-to-visible-surface distance, not target-
-center or horizontal distance. UAV displacement from a local origin is not
+4. Match the precision of the range estimate to the requested relation and
+   the size of the contemplated correction.
+5. After one bounded correction, re-estimate. Reverse only when new evidence
-4. For qualitative or clearly off-threshold requests, use a broad coherent range
-   rather than waiting for exact measurement. Request candidate distance only
-   when a precise decision is now necessary.
-5. After one bounded correction, re-estimate; repeat precise grounding only when
-   another quantitative decision is needed. Reverse only when new evidence
+metric standoff additionally requires fresh target-relative distance evidence
-metric standoff additionally requires a fresh grounded-candidate distance
+If no safe recovery remains, stop and cite it. Use
-Hover only to stabilize. If no safe recovery remains, stop and cite it. Use
~~~~

### plugins/knowledge/uav_flight/orbit_and_surround/SKILL.md

~~~~diff
+Radius judgments need target-relative evidence with precision appropriate to
+the requested tolerance and contemplated radial correction. Qualitative range
+and metric distance support different levels of precision; vehicle displacement
+alone does not measure orbit radius.
-Use qualitative visual range while the radius is clearly unsuitable. Request
-grounded candidate distance only near a metric radius threshold, when radial
-repair magnitude depends on it, or before quantitative completion. Do not spend
-repeated perception turns refining a relation that is already actionably clear.
+re-observe; do not continue a blind shift.
-re-observe; do not continue a blind shift. Hover only to stabilize or compare.
~~~~

### plugins/knowledge/uav_flight/route_traversal_and_passage/SKILL.md

~~~~diff
+not continue blind motion. If no safe evidence-producing
-not continue blind motion. Hover only to compare. If no safe evidence-producing
~~~~

### plugins/knowledge/uav_flight/vertical_landing_and_altitude_control/SKILL.md

~~~~diff
+If height or region cues conflict, compare the available evidence before further
+movement; a smaller adjustment still needs a safe supported direction. If they
-If height or region cues conflict, hover or use a smaller adjustment. If they
+continue a blind descent. If no safe
-continue a blind descent. Hover only to stabilize or compare. If no safe
~~~~

### plugins/knowledge/uav_flight/visual_target_task_reasoning/references/search_under_uncertainty.md

~~~~diff
+  turn. A small local confirmation turn is about 5-15 degrees.
-  turn. Small confirmation means hover or about 5-15 degrees.
+- Peek safely: when an obstacle or corner may hide the target, a changed viewing
+  angle or a short reposition can expose it. Repositioning needs clear free space;
+  choose a view change that preserves useful cues without unsupported movement.
-- Peek safely: when an obstacle or corner may hide the target, prefer yaw or
-  hover-based stabilization first; use a short reposition only when free space is
-  visually clear.
+  reliable, stop broad sweeping and compare the available evidence. A small
+  5-15 degree yaw can expose missing detail while preserving the candidate.
+  Do not describe a candidate and then
-  reliable, switch from broad sweep to hover or a small 5-15 degree yaw before
-  judging so the candidate stays in view. Do not describe a candidate and then
+- Confirmed or centered target: stop searching or use the confirmed
-- Confirmed or centered target: hover, stop searching, or use the confirmed
~~~~

### plugins/knowledge/uav_flight/visual_target_task_reasoning/SKILL.md

~~~~diff
+Candidate identity needs evidence connecting appearance and location to the
+requested target; confidence alone is insufficient. Metric judgments need
+distance evidence with matching geometry and precision.
-Interpret images first. Supplement only decision-relevant ambiguity. Confirm
-candidate identity, not confidence alone; request depth only for a metric decision.
+request. Do not add movement merely to reconfirm an adequate bearing. If vision only
-request. Do not add a confirmation hover or micro-adjustment. If vision only
+re-observe; do not continue blind motion.
-re-observe; do not continue blind motion. Hover only to stabilize or compare.
~~~~

### src/agent/planning/prompts/tool_use_case_studies.txt

~~~~diff
+2. A fictitious format_records capability takes entries:array of objects with label:string
+   and uppercase:boolean. An argument example is
+   {"entries":[{"label":"sample","uppercase":false}]}.
-2. A fictitious save_checklist capability takes entries:array of objects with label:string
-   and done:boolean. An argument example is
-   {"entries":[{"label":"Check the requested outcome","done":false}]}.
~~~~

图像文案的额外变化：RGB 不再要求直接看图优先或限定测距时机；overlay 仅说明假彩色的定性范围、米制色条和精度局限。USER.md、任务 Prompt、ToolUser 保持原样；未启用知识与历史文档保持原样。安全执行结果里的 hold position 仍可描述 Runtime 内部安全悬停，它不是给模型新增的可选工具，因此不删除。

## 5. 验证结果

### 自动测试

- `python -m pytest -q --tb=short`：**1026 passed，10 warnings，29.42s**。
- 11 项新增测试覆盖前后结构一致、隐藏工具原始定义一致、参数单位、执行说明去重、知识/图像提示清理和探针行为。旧精确措辞断言已迁移，没有删除安全或执行校验。
- 一篇知识正文改写后曾超过既有 2400 字符上限 8 个字符，已压缩表述，通过原长度检查，未放宽限制。

### 真实模型配对检查

模型：qwen3-vl-plus。两阶段 corpus_hash 和 provider settings fingerprint 一致；工具参数结构比较通过。每阶段 10 次请求，无重试，共 **20 次**；没有执行返回工具或真实飞行。

默认身份与用户偏好按各版本实际装配，任务/合成状态/历史固定不变；仅测距样例附带仓库录制 RGB 图作为明确标记的合成场景插图。其他案例主要是合成文本状态，不代表真实在线多模态任务。知识请求只评估是否选择加载，未执行 knowledge.load，因此不验证加载后五篇知识的多轮效果。

| 场景 | 修改前首轮工具 | 修改后首轮工具 | 预设期望 |
|---|---|---|---|
| simple_turn_right | uav.turn_relative | uav.turn_relative | uav.turn_relative |
| simple_turn_left | uav.turn_relative | uav.turn_relative | uav.turn_relative |
| simple_translate | uav.move_body | uav.move_body | uav.move_body |
| complex_ordered | knowledge.load | skill.visual_grounding.locate | runtime.update_plan |
| complex_search | runtime.update_plan | runtime.update_plan | runtime.update_plan |
| complex_route | runtime.update_plan | skill.visual_grounding.locate | runtime.update_plan |
| explicit_knowledge | knowledge.load | knowledge.load | knowledge.load |
| explicit_distance | skill.visual_grounding.locate | skill.visual_grounding.locate | skill.visual_grounding.locate |
| supported_completion | runtime.report_completed | runtime.report_completed | runtime.report_completed |
| unsafe_blocked | runtime.report_blocked | runtime.report_blocked | runtime.report_blocked |

简单动作 3/3 → 3/3；明确知识、测距和两个终态场景 4/4 → 4/4；复杂任务首轮计划 **2/3 → 1/3**。合计 **9/10 → 8/10**，全部返回调用通过参数/组批及引用检查。

探针的 unnecessary_calls/omitted_expected_call 只表示相对预设首轮工具的偏离，不足以证明某次其他工具调用在所有任务策略中都“不必要”。这次观察到定位工具首轮选择增加、计划选择减少；不能据单次小样本确认原因，也不能宣称无偏好或总体质量提升。遵守确认计划，没有为追分加强计划优先级或加入新的强制规则。

评测后仅压缩一段未加载知识正文以满足既有长度限制；未改变评测发送过的工具和 system/图像文案。原始评测产物保持原样，新版静态全文反映最终工作区。

原始记录：

- [tests/manual_outputs/tool_description_cleanup_20260922/before/report.json](F:/UAV-Agent/tests/manual_outputs/tool_description_cleanup_20260922/before/report.json)
- [tests/manual_outputs/tool_description_cleanup_20260922/after/report.json](F:/UAV-Agent/tests/manual_outputs/tool_description_cleanup_20260922/after/report.json)
- 各阶段每个案例目录包含 request.json 和 Transcript；报告中的 corpus 保存固定图像输入。

复现入口（输出目录必须不存在，两个阶段各最多 10 次调用）：

~~~~text
python tests/manual/tool_contract_behavior_probe.py --phase before --output-dir <新的修改前目录>
python tests/manual/tool_contract_behavior_probe.py --phase after --baseline <修改前目录>/report.json --output-dir <新的修改后目录>
~~~~

在对应版本分别运行；只对当前版本运行两次不能重现本次历史对照。结构或语料/配置不一致会在 after 发请求前拒绝。

## 6. 审阅入口

- [docs/agent_loop/tool_prompt_audit_20260922_v2.md](F:/UAV-Agent/docs/agent_loop/tool_prompt_audit_20260922_v2.md)：新版完整工具、Prompt、知识及参考。
- [docs/agent_loop/tool_prompt_snapshot_20260922_v2.json](F:/UAV-Agent/docs/agent_loop/tool_prompt_snapshot_20260922_v2.json)：新版结构化快照。
- [docs/agent_loop/tool_prompt_audit_20260922.md](F:/UAV-Agent/docs/agent_loop/tool_prompt_audit_20260922.md)：未覆盖的修改前审计。
