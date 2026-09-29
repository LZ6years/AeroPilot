> Status: Historical

# Qwen3-VL-Plus / Qwen3.8-Flash：工具漏调用对照

本次结果支持“模型选择会明显影响这些样本中的规划和知识调用”。相同录制输入下，Flash 的 36 次响应出现 19 次计划调用、3 次知识调用；历史 VL-Plus 36 次均为 0。Flash 在 TravelUAV 实飞中先规划、加载知识、最终更新计划并完成任务。

证据仍不足以将原问题全部归结为模型能力：Flash 存在非法 JSON 参数、进度更新滞后及终态引用错误；AerialVLN 实飞因进程异常退出而没有导航结果。本实验不测试 Qwen3.8-Max，不自动改变生产默认模型。

## 实现和输入控制

两个现有手工探针增加 `--model`：

- `tests/manual/task_progress_guidance_probe.py`：显式覆盖实验模型时允许与录制模型不同；记录 source/requested/returned model、耗时及 usage。未指定时保留原模型一致性检查。
- `tests/manual/task_progress_simulation_probe.py`：只替换 runner 的实验 ModelConfig；实验 Planner 在原请求审计中补充返回模型标识，不改业务提示词或通信层。
- `tests/test_task_progress_model_override.py`：验证默认行为、不可变覆盖、真实编码后的请求与 manifest 一致、捕获记录不变、返回模型审计、不兼容工具响应保存后停止。

`.env`、生产默认模型、基础指导、周期提醒、知识正文、安全限制和预算均未修改。API 使用现有端点与凭据；所有 Flash 有响应的记录返回模型标识均为 `qwen3.8-flash`。
12 份固定上下文输入变体逐项核对，除 `model` 字段外与上一轮相同，图像引用/哈希也相同。实飞记录的源码哈希与历史引导组相同；ModelConfig 仅 model_name 不同，benchmark 配置仅 output_dir 不同，运行期间无记录源码变化。

温度保持 0，思考模式未显式指定，保留服务端默认。Flash 的返回 usage 包含 reasoning_tokens，不能声称两个模型内部推理预算相同，也不能把默认推理策略的影响单独分离出来。

## 36 次固定上下文结果

原两个 run 各取第 1、6 轮，三组提示、每组三次。每模型 36 次逻辑请求；VL-Plus 沿用历史结果，本次新增 Flash 36 次。未执行这些响应中的工具。

| 提示组 | 响应数/模型 | VL-Plus 计划 / 知识 | Flash 计划 / 知识 | Flash 总 token | Flash 请求耗时合计 |
| --- | ---: | --- | --- | ---: | ---: |
| 旧提示 | 12 | 0 / 0 | 4 / 0 | 131,896 | 90.9 秒 |
| 基础指导 | 12 | 0 / 0 | 5 / 1 | 136,227 | 106.2 秒 |
| 基础指导＋提醒 | 12 | 0 / 0 | 10 / 2 | 137,334 | 109.3 秒 |

Flash 合计 405,457 token，其中输入 387,753、输出 17,704；单次请求平均约 8.51 秒。VL-Plus 合计 392,073 token，旧探针未记录可配对的请求耗时，不比较速度。输入 token 数因模型计数不同而不同，不表示输入文本被修改。
36 次 Flash 均通过工具名称、ID、参数 Schema 检查；这不等于经过 Runtime 完整执行检验。

逐样本：

- **AerialVLN 首轮**：Flash 三组共 9/9 响应先规划，覆盖升高、楼梯、米色屋顶、棕楼/灰屋顶、拖拉机触发右转、大灰楼、回头和脚手架终点顺序。未预编未观察区域移动参数。
- **AerialVLN 第 6 轮**：旧提示和仅基础指导均未规划；加提醒后 3/3 补建计划。补建计划不是完成子目标的独立证据，其“已升至建筑上方”等状态仍需核查实际观察。
- **TravelUAV 首轮**：旧提示 1/3 规划，基础指导 2/3 规划、另 1 次定位与知识并行；基础指导加提醒组 3/3 规划。首轮两个新组输入实际相同，没有注入周期提醒，因此 2/3 与 3/3 的差别不能归因于提醒。
- **TravelUAV 第 6 轮**：旧提示/基础指导均继续定位；带提醒组 2 次加载 `visual_target_task_reasoning`、1 次补建计划。

知识调用内容与视觉搜索任务相关，但离线探针不执行工具，不能证明知识随后被应用；需要以下实飞证据。

逐响应 HTML（历史路径：`tests/manual_outputs/task_progress_flash_20260922/review.html`；产物已删除） ·
输入 manifest（历史路径：`tests/manual_outputs/task_progress_flash_20260922/manifest.json`；产物已删除） ·
响应及 usage（历史路径：`tests/manual_outputs/task_progress_flash_20260922/results.json`；产物已删除） ·
两模型统计（历史路径：`tests/manual_outputs/task_progress_flash_20260922/comparison_summary.json`；产物已删除）

## 两次实飞尝试

两次 Flash 均使用完整新方案、固定原案例，并串行执行。每次重新启动受管场景、由原 runner 初始化；未调整任务预算或重跑追求成绩。历史对照选同样完整新方案下的 VL-Plus。

| 案例/模型 | 状态 | 决策轮次 | 已执行物理动作 | 计划/知识调用 | 总 token | 总耗时 |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| AerialVLN / VL-Plus | 证据不足，未成功 | 30 | 18 | 0 / 0 | 958,436 | 365.5 秒 |
| AerialVLN / Flash | 进程中断，结果未知 | 5 | 3 | 1 / 0 | 55,681（部分） | 未完整记录 |
| TravelUAV / VL-Plus | 证据不足，未成功 | 12 | 5 | 0 / 0 | 202,061 | 215.8 秒 |
| TravelUAV / Flash | runner 判定成功 | 22 | 6 | 2 / 1 | 622,078 | 317.4 秒 |

物理动作计数排除协议拒绝、读取、计划和内部悬停。总耗时含准备、感知、模型、执行与清理；token 为已收到响应的 usage 合计，断连重试的服务端计费消耗未知。

### AerialVLN：首轮规划成立，导航结果缺失

案例 `39GHHAVOMOP8ZV27ILH5P0TV5F6J4A`。
第 1 轮建立七步计划，先后关系及条件触发保留；第 2–4 轮执行升高、转向。第 5 轮提出视觉定位楼梯，之后 Python 进程退出码为 1，无 traceback、无任务终态，也未执行 finally 落盘。
这与先前一次基线补跑的异常形态相似，但根因未定位，不能据此断言是 Flash、感知库或资源不足导致。
未形成计划进度更新或知识使用证据，也不能将此运行计为模型导航失败/成功。
仅依据保存的部分事件报告 3 次物理动作，不伪造 benchmark summary。

中断说明及清理（历史路径：`tests/manual_outputs/task_progress_flash_aerial_20260922/interruption.json`；产物已删除） ·
部分请求/动作审阅页（历史路径：`tests/manual_outputs/task_progress_flash_aerial_20260922/interrupted_review.html`；产物已删除）

### TravelUAV：有规划和知识使用，仍有执行缺陷

案例 `c3f9fc36-7c55-4a48-8213-5adce9b35b12`。

- 第 1 轮先规划，随后按目标方位转向；初始计划区分搜索、候选定位和接近确认。
- 第 3、4 轮定位未返回候选，第 6 轮在搜索方法需要调整时加载 `visual_target_task_reasoning`，ToolResult 成功返回。后续未重复加载。
- 第 7 轮后退并调整高度以扩大路面视野；第 8 轮获得摩托车候选，第 10 轮在保持有用视野的前提下接近，第 11、13、18 轮继续确认。行为与“改变视点获取证据、弱候选出现后停止广泛搜索并确认”的知识内容相符，是加载后使用的支持证据；不能证明只有该次加载才会产生这些动作。
- 第 19 轮才将计划更新为完成，稳定 ID 保留。中途转向和首次候选发现后未及时更新，进度记录滞后；修订的最后一步删除了初版中的红电话亭参照，不能把计划全部完成等同于所有场景描述均已独立验证。
- 第 9、14、15、16 轮输出 `00.3` 或 `00.5` 这样的非法 JSON 数字，均被现有协议校验拒绝；未对返回参数进行修补。
- 第 20、21 轮完成请求包含不被接受的证据引用，被拒绝；第 22 轮修正后完成。请求第 9 轮另有一次连接重试，未另增决策轮次。

最终导航误差 5.35 米、轨迹长 15.15 米，runner success=true；VL-Plus 引导组误差 7.36 米，但因自行报告证据不足而未成功。Flash 总轮次、耗时和 token 更高，本案例不支持“换 Flash 更省时/更省 token”。
该案例初始位置已在 20 米成功距离阈值内，effective_reference_path_length=0、SPL=0；成功依赖终止行为和既有评测规则，不能从这个单例推断整体导航能力。

TravelUAV 完整 HTML（历史路径：`tests/manual_outputs/task_progress_flash_travel_20260922/evaluation/review.html`；产物已删除） ·
实飞对照明细（历史路径：`tests/manual_outputs/task_progress_flash_20260922/simulation_comparison.json`；产物已删除）

## 工程验收与结论

最终 `python -m pytest -q`：**1096 passed，10 warnings，34.96s**；警告为既有 Pillow/msgpack 弃用提示。
两次 Flash 场景均已核查远端进程组无存活成员、端口关闭。AerialVLN 遗留隧道经所属进程命令行核实后停止；TravelUAV 由 runner 正常清理。
源码/配置差异与清理核验（历史路径：`tests/manual_outputs/task_progress_flash_20260922/experiment_checks.json`；产物已删除）

结论：换模型减少了给定上下文中的计划和知识漏调用，TravelUAV 也出现了相应实飞行为。原问题确有模型及其默认推理策略的影响，不能简单概括为提示词完全无效或模型整体“太差”。本次仅一个完整 Flash 实飞结果；默认模型保持原样，后续是否推广需要更多案例与协议稳定性验证。
