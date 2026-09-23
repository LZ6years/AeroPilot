# Framework 修订 v4

内置 image_gen 编辑，保留 v3 和正文。Memory 改为统一交互日志，Skill 等加入语义图标，TaskParser 改为梯形。

## 顶部指令来源

- Rotate 105 degrees to the right：本地 `.external_refs/UAV-Flow/UAV-Flow-Eval/test_jsons/2025-03-30_11-52-11.json`，图中增加句末标点。
- Navigate to a point 5.0 meters away from the individual：同目录 `2025-03-30_11-52-35.json`。
- 长程指令为压缩改写，来自 `tests/manual/phase_prompt_comparison_instructions.json`，ID `3VE8AYVF8VVTDU3DBYGXREH8D748FB`。原文：lift off above the buildings and head straight on the road. veer off the road near the shipping containers. stop near the shipping containers.

指令用于举例，不表示上述案例的执行效果。Turn(...) 是概念短名，不是精确 API；105 度是任务目标，不表示单次调用必须转105度。Memory 中 e 为事件，S 为交互片段，角色与字段为压缩展示。Knowledge 仍是阶段指导，非主动查询工具。所有候选图仅为视觉预览。

## 编辑提示词

```text
Edit the supplied FlyAgent framework, preserve its overall near-square layout, central Core, all correct directed connections and bottom simulated environment. Requested changes:
1) TaskParser is a TRAPEZOID with wide horizontal TOP and narrower horizontal BOTTOM, NOT triangle, not a pointed funnel. Keep Semantic Plan under it.
2) Top instruction cards more descriptive, enlarge height and use compact readable multiline English text. Exact card 1 "Rotate 105 degrees to the right." with curved rotation arrow icon; exact card2 "Navigate to a point 5.0 meters away from the individual." with person + distance marker icon, NOT car icon; exact card3 "Fly above the buildings and follow the road. Veer off near the shipping containers and stop nearby." with building/road/container route icon. Retain ellipsis. No benchmark names; these are instruction examples not evidence of success.
3) REDESIGN Memory entirely inside its existing boundary: REMOVE all three ugly colored stacks of labeled rectangles. Add elegant archive/history icon beside Memory title, then one unified chronological interaction log, e.g. a slim vertical timeline with alternating tiny user/person, assistant/speech, tool/wrench, result/check symbols and fine text lines. A subtle rear page edge shows accumulated history. Short labels only "Observations", "Intent + expected effect", "Calls + results", and small "S₁ … Sₜ" at bottom. Single cohesive log illustration, not four new boxes or three independent memories. No human brain.
4) Skill Library title accompanied by tool chest / modular library symbol. Add meaningful small icons on each row: Knowledge open book; Perception eye with target reticle; Physical drone with movement arrows; Progress Control checklist flag. Keep Knowledge separated from callable subset bracket, dashed Phase guidance -> Context and solid Call/Result routes unchanged.
5) Add small meaningful icons to Context (assembled documents), Decision & Dispatch (branching arrow), Execution (gear + motion). Keep only one robot model icon. Restrained professional academic style, white background, thin lines, no ornamental clipart overload. Readable typography and better whitespace. Keep Illustrative example and Simulated evaluation labels. Do not add models, tools, maps or modules. Existing external arrows must preserve directions.
```

## 文案校正

```text
Preserve this image exactly except replace these text labels: Knowledge subtitle 'World knowledge' -> 'Phase guidance'; TaskParser subtitle -> 'Instruction to semantic plan'; Memory top row heading 'Observations' (beside user icon) -> 'User input'; Memory final row heading 'Observations' (beside green check icon) -> 'Tool result'; Memory 'call: rotate(105) ...' -> 'call: Turn(...)'; Memory tiny t1,t2,t3,t4 markers -> e1,e2,e3,e4 to denote events, keep S1 ... St at bottom; Decision & Dispatch subtitle -> 'Validate and dispatch calls'. Keep all icons, arrow directions, geometry, task text and overall layout unchanged.
```
