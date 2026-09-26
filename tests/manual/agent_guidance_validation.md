# UAV 指导修改的验证入口

本轮分开验证基础指导修改和离线经验实验，保留历史请求原样。

## 已实现的修改

1. grounding 的实际 provider description 明确当前 RGB 候选、观察支持的短 query、
   空候选、`not_requested`、`inconsistent_depth_surfaces` 和运行失败的不同含义。
   空候选会削弱候选假设，但不证明目标不存在。距离阈值没有修改。
2. 移除 capability 的 `x-flyagent-completion-evidence`、registry 元数据投影和 Runtime
   专用 freshness 检查。当前任务隔离、真实引用和调用结果配对检查保留。曾测距不构成
   结束前重测理由；当前定量判断仍需要有效且几何意义匹配的测量。
3. 五份知识保持稳定名称，description 统一四项文字；正文保留通用判断，路线 reference
   补充道路尽头与路口证据。模型自主判断加载价值，不恢复固定知识选择流程。
4. 常驻提示加入实际反馈估计窗口、重新观察位置、批量依赖和重新估计条件。窗口仍为
   1–3 个动作；Plan 仍为 `id / description / status`，允许反证修正已完成判断。
5. 两个 benchmark 通过现有 `protocol_context` 提供四项任务介绍。常量保留历史 Python
   导入名称以兼容调用方；模型文本不再提 parser/phase。普通任务没有被注入 benchmark 介绍。
6. 独立经验模块见 `plugins/research/flight_experience/README.md`。没有 Core 接入和飞行工具。

## 基线与真实请求

基线索引：`tests/fixtures/agent_guidance_baseline_20260926.json`，含原始 20 例、原请求
所属运行目录、transcript SHA256、模型与动作批次统计。完整本地基线目录为
`tests/manual_outputs/agent_guidance_baseline_20260926`，包含原工作区差异、未跟踪文件
备份和修改前关键源码。原批次实际有 21 个运行目录，其中包含一次重试；20 例对照
以基线索引选择的运行目录为准，不能重复计算重试。

2026-09-23 的主批次模型是 `qwen3.8-flash`。原 20 例有 255 次 grounding、9 次知识加载；
物理批次包含 396 个单动作、48 个双动作和 2 个三动作窗口。调用次数只是诊断指标。
出租车案例 `3NL0RFNU0OL3AF8P06HHQFT8T3F4KA` 结束前在新 observation 上重检，是旧完成
门槛拒绝导致，不能描述成最后实际返航。日志不改写。

固定上下文对照：

```powershell
python tests/manual/agent_guidance_probe.py --source <历史decision.json> --round-index 0 --output <新目录> --prepare-only
python tests/manual/agent_guidance_probe.py --source <历史decision.json> --round-index 0 --output <另一新目录>
```

两组都是新离线请求：baseline 使用捕获的原请求，revised 替换指导和工具描述、
知识目录及 benchmark 上下文。原图、历史动作/结果和过去已加载的知识结果保留。
探针不执行返回的工具，不模拟新飞行，不把这些请求冒充原始记录；不证明闭环提升。
输出 `probe.json` 包含请求、原图引用/散列、公开回复、参数校验与耗时。

## 测试与评估次序

```powershell
python -m pytest -q --tb=short
python tests/manual/flight_experience_probe.py inspect --source <运行目录> --output <新检查目录>
```

再做真实固定上下文请求，然后使用现有 `benchmark_20_case_probe.py` 的监督执行和
清理机制运行小规模闭环，通过后再扩展。同案例完整对照需要固定数据、模型、配置
和控制实现；当前工作区包含本轮之前的控制层修改，因此不能把与 9 月 23 日历史结果
的全部差异都归因于本轮指导。

逐例记录以下指标，不能可靠自动判断的语义项必须人工标注为已核对或未知：

| 指标 | 数据来源与判断范围 |
|---|---|
| 成功率、最终误差 | benchmark 原评测输出；基础设施失败分开统计 |
| 误完成、过冲、无效转向 | 指令、连续原图、实际运动与评测证据，人工核对 |
| 碰撞、安全干预 | 原动作结果与安全日志，不以请求参数代替实际执行 |
| 无效检测 | 分开计空候选、测距失败和工具运行失败 |
| 重复确认 | 同一观察/查询的重复调用；跨视角重复是否必要需人工核对 |
| 窗口数、批量大小 | 真实 physical action batch |
| 计划修正 | 计划调用及状态变化；不能把修正次数多当作质量好 |
| 耗时、调用成本 | 实际调用 elapsed/usage；没有价格时成本未知 |

经验实验单独记录候选数、各审核决定、无效引用、缺图、耗时、usage、人工纠正和
跨场景适用性。它不进入基础改进版的成功率统计。
