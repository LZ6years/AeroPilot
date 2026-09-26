# 独立历史经验实验

读取已有记录，检验经验总结和审核是否有用。它没有注册 capability，没有飞行工具，
不会更新正式知识或影响下一次飞行。模块只导入通用 `src.llm`；手动入口负责加载
现有模型连接配置。这里的“采纳”只表示进入实验产物，仍待人工审核。

```mermaid
flowchart LR
    A[只读历史记录] --> B[独立适配与证据索引]
    B --> C[复盘选帧与原图读取]
    C --> D[候选经验]
    B --> E[独立审核选帧与原图读取]
    D --> E
    E --> F[引用校验与采纳/修订/暂存/丢弃]
    F --> G[隔离实验目录与人工报告]
```

## 输入与边界

- `read_run(run_dir)` 读取 `memory/agent_transcript.jsonl`、`observations.jsonl` 和
  `frames.jsonl`，也支持这些文件直接位于输入目录。多任务记录必须指定 `instruction_id`。
- 旧格式只有 `actions.jsonl` 时，需要调用方提供原始指令并确认目录只含一次任务。
  旧格式没有可靠的任务划分与 checkpoint 对齐，这一限制会写入证据索引。
- 按字段白名单提取原指令、意图、请求参数、执行参数、工具结果、干预原因和实际自运动。
  绝对位置、任意 Context/debug 字段、隐藏目标、参考轨迹及评测结果不进入模型输入。
  扩展日志格式时在适配器内增加明确映射，不能直接转发整个原始对象。
- RGB 以 observation ID 寻址，保留原始字节和 SHA256。支持日志内嵌图片及运行目录内
  的本地图片；路径越界、损坏、缺失或关联不唯一会标记为不可用。
- 原始文件的散列和行引用在 `evidence_index.json` 中。输出不能位于原运行目录内，
  不能覆盖既有输出，也不能是原运行目录的祖先。

## 模型调用和审核

一次有效实验通常调用四次：复盘选帧、带图复盘、审核选帧、带图审核。审核使用独立
上下文，可复用同一模型，也可单独指定模型；同模型的独立调用不等于独立知识来源。
审核可选择反证帧，引用候选所需的原图必须进入其输入。每次最多输入 `max_images`
张原图，候选数由 `max_candidates` 限制。选帧只接收紧凑时序索引，意图摘要明确标记
截断；复盘读取所选时刻及相邻 checkpoint 窗口的完整记录，审核额外读取候选引用的
原始事件。未提供完整内容的索引条目不能作为采纳依据。完整索引仍保存在本地。
旧 action-only 日志缺少 checkpoint 对齐，因此保留完整动作文本并明确记录限制。
极长的记录仍可能超过提供商限制；实验不会把截断的意图摘要当作原始完整证据。

候选字段为 `id / applicability / observed_evidence / recommendation / limitations /
evidence_refs / observation_ids`。必须引用真实时序事件及本轮实际提供的原图。
审核返回 `adopt / revise / defer / discard` 和理由；修订后的引用重新校验。伪造引用、
缺图及未看到的图无法靠模型的“采纳”绕过校验。接口错误或无法解析的审核会使本次
实验失败，采纳结果清空，并保留失败报告。

这些结构校验只能核对来源，不能证明建议正确。人工仍需检查颜色、空间关系、
模型是否复述原来的误判、适用范围是否过大、是否提供了新的决策价值。

## 操作

```powershell
python tests/manual/flight_experience_probe.py inspect --source <原运行目录> --output <新检查目录>
python tests/manual/flight_experience_probe.py run --source <原运行目录> --output <新实验目录> --max-images 8 --max-candidates 6
python tests/manual/flight_experience_probe.py run --source <原运行目录> --output <另一新目录> --observations obs_0033 obs_0036 obs_0037
```

默认读取 `.env` 中的现有连接，分别指定 `--reflection-model`、`--review-model`
可覆盖模型名；连接仍使用同一提供商。`--env` 可选择独立实验配置文件。不要为实验
修改正式飞行配置。`--reflection-input-price` 等四个价格参数单位为美元/百万 token，
只计算输入/输出 token 的估算费用；未提供价格或 usage 时费用记为未知，不记为零。

对明确要检查的片段，可从 `inspect` 输出中选择 `--observations`，跳过模型选帧。
两个独立读者都会收到这些原图及对应时序片段，通常只需复盘和审核两次调用。
产物明确记录人工指定的观察范围；这不是模型自主选帧实验，也不能推断未查看部分。

产物包含：

| 文件 | 用途 |
|---|---|
| `evidence_index.json` | 时序、图像可用性、原文件 SHA256 和来源行 |
| `experiment.json`、`media/` | 调用输入、实际 provider 请求、公开回答、原图、耗时和 usage |
| `catalog.json`、`experience.md` | 仅供实验查看的采纳经验与四项 description |
| `report.html` | 候选、审核、失败或暂存理由与原图 |
| `human_review.json` | 人工填写纠正数量、说明和跨场景验证记录；初始为 pending |

不同场景应使用独立输出目录。不能把同一次飞行中的多个片段当成跨场景验证；
不能把模型自报完成当成评测成功；不能把经验看起来合理写成闭环收益。

未来只有在离线验证有价值后，才考虑通过 capability、handler 与现有插件发现机制
接入只读经验。经验始终是历史指导，不能充当当前任务完成证据。本轮不实现该接口。
