# FlyAgent 方法图 v3（讨论稿）

日期：2026-09-21。使用内置 image_gen 技能生成；不是实验记录或最终矢量图。本次未修改论文正文或项目代码。

## 图稿

- [Unified Framework](unified_framework.png)：任务经过倒三角 TaskParser 形成 Plan；中央 Agent、左侧 Memory、右侧 Skill Library、底部执行与模拟环境。
- [History and Context Projection](history_context_projection.png)：S 表示交互片段，History 是有序片段拼接；Projector 以函数曲线表达旧观测年龄抽样。

## 阅读边界

- 本图的 S_t 是新约定的交互片段，不是 v2 文稿中原先的状态检查点；正文未修改，后续整合需要统一符号。
- Knowledge 为阶段指导，不是查询工具。其余三类汇总表示调用与返回；物理能力的真实执行走底部 Execution。进度控制不直接运动。
- 总体图中的 Result 与环境反馈进入 Core 后会参与下一轮上下文构建；主图省略内部回路线。Memory 同时保存决策和结果；工具返回也经运行时记录，不只记录环境变化。
- 小图各片段以一次模型决策及关联返回为主，user 可选。早期卡片折叠，未展示全部消息，不表示调用缺少对应返回。
- Move / Turn / Locate 为图示短名称，对应 uav.move_body / uav.turn_relative / visual.grounding.locate。意图及预期是公开说明，相关协议字段和 commit 在图中省略。
- Context 卡片是内容摘要，不表示 Projector 额外生成推理摘要；比较与修订由同一 Agent 完成。
- 曲线表达对数年龄采样的指数增长趋势，离散点只为示意，不是配置的精确采样值；实际函数还有取整与唯一性约束。近期保留独立处理，工具图像不走观测抽样。
- 未宣称真实部署或成功恢复，场景与任务片段均为 Illustrative example。

## 生成提示词

### unified_framework

```text
Create an academic scientific infographic, nearly SQUARE canvas 1.1:1, opaque white background. Short crisp English labels. Title "FlyAgent: Unified Framework". Top row three small instruction cards "Turn right", "Find the red car", "Follow the road", then "...". Arrows converge downward into one wide inverted TRIANGLE labelled "TaskParser". Triangle tip -> small "Semantic Plan" card "Goal + phases". Below plan is CENTRAL large purple bordered "FlyAgent Core", with three stacked modules Context -> Shared MLLM (ONE robot icon) -> Decision & Dispatch. Small public-output labels "Strategy / Intent / Expected effect". LEFT of core narrow blue "Memory" panel with overlapping message cards "Observations", "Decisions", "Calls + results", "...". Memory -> Context blue arrow label "Project". Core -> Memory distinct green arrow "Append decisions". RIGHT of core narrow "Skill Library" panel containing four cards vertically: Knowledge / Perception / Physical / Progress Control. Knowledge -> Context BLUE DASHED single arrow labelled "Phase guidance". Core -> library callable subset purple arrow label "Call"; library -> core green arrow label "Result". Knowledge must NOT belong to callable arrow subset: bracket only lower three cards. Bottom row distinct boxes "Execution" then "UAV Environment" with tiny oblique aerial red-car street scene and UAV; scene text "Simulated evaluation" and "Illustrative example". Core Decision & Dispatch -> Execution orange arrow labelled "Action window". Execution -> Environment orange "Actions". Environment -> Core green labelled "Observation + measured motion", route outer bottom/right avoiding library. A green feedback branch to Memory labelled "Append outcomes". Query result returns to Context for another decision, not directly to Execution. Progress Control only manages state, no direct arrow to Environment. No global map, separate reflection model, real-world validation, benchmark names, or invented modules. Academic clean thin lines and gentle blue purple orange green fills; lots of white space, short text not paragraphs. Exactly one core model. Ensure arrowheads only at destination. This is conceptual architecture preview, not measured results.
```

### history_context_projection

```text
Create a refined academic figure on opaque WHITE horizontal canvas 2:1. Title "History and Context Projection". Upper half a long History ribbon labelled "H≤t = S₁ ⊕ … ⊕ Sₜ". Under it early S₁,S₂ as overlapping folded message cards, ellipsis then two LARGE expanded interaction segment cards Sₜ₋₁ and Sₜ. A clear legend "S: interaction segment, not physical state". Sₜ₋₁ contains purple assistant subcard "Intent: approach the red car", "Expected: car remains visible", "tool_calls: Move(...)" followed by orange tool subcard "Result: motion executed". Sₜ contains optional blue user subcard "New observation: car not visible", purple assistant subcard "Intent: verify target location", "tool_calls: Locate(...)", orange tool subcard "Result: returned evidence". Do not claim recovery or success. tool_calls always inside assistant; no tool_call role. Early cards can omit user. Short footnote above lower area "Segments need not start with user; call-result links preserved". Upper History source -> lower left wide "Projector" box. Projector internal dominant inset a real-looking MATHEMATICAL FUNCTION CURVE with horizontal axis "Older-sample slot j" and vertical axis "Observation age a_j". Curve monotonically INCREASES and is CONVEX (exponential rise), discrete points along curve at uniform horizontal slots. NO descending forgetting curve, no image thumbnails anywhere. Label "Log-spaced ages" and side badge "Recent R: keep all"; concise "Rounded + constrained sample ages" note. Small chips in same Projector box "Message mapping", "Image deduplication", "Audit filtering". Projector -> lower RIGHT "Context for next decision" card with pair of visible summary lines "Previous expectation: target visible" "Observed outcome: motion; target lost from view", plus "Query + returned evidence". Under card text "Compared by the SAME Agent" and "Maintain / gather evidence / revise". No separate robot or reflection module. Lower margin two notes "Original history retained" and "Tool images bypass observation sampling". Label whole example "Illustrative example". Blue historical inputs, purple projection/decision, orange calls, green results. Elegant thin borders and clear hierarchy; no photos, benchmark names, reported metrics, fake quantitative ticks. Ensure interaction cards visibly contain multiple message roles. Keep text concise and fully legible.
```

## 小图校正提示词

```text
Edit ONLY two local details; preserve composition, all other labels and colors. In S2 replace 'tool_calls: Look(...)' with 'tool_calls: Turn(...)', replace 'Result: scanning complete' with 'Result: turn executed'; keep intent scan environment, which can be done by turning. In Projector chart remove extra parallel vertical arrow so it has a SINGLE vertical y-axis and a SINGLE horizontal x-axis. Keep rising convex discrete-point curve and axis labels. Do not change other content.
```
