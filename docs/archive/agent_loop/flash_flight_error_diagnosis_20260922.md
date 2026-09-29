> Status: Historical

# Flash 实飞错误调查与修复

本地 `.env` 已切换为 `FLYAGENT_MODEL_NAME=qwen3.8-flash`，有效配置读取已验证。端点、凭据、思考模式、任务指导、传感器、安全限制及轮次预算不变。

## 已定位的两类错误

### 非法 JSON 来自模型输出，解析器没有改坏参数

此前 TravelUAV 第 9、14、15、16 轮的原生工具参数包含 `00.3` 或 `00.5`。这些数字不符合 JSON 语法，Planner 的严格解析拒绝了整轮，动作没有执行。此前反馈只有 `arguments are not valid JSON`，没有出错位置。

现在反馈给出工具名、行/列/字符位置、最多 80 字符的附近文本，以及 JSON 数字不能有前导零的说明，要求模型按原意重新提交。合法 `0.3` 和 `-0.3` 都只是语法示例，不由运行时替模型选择符号。不进行自动纠错或执行猜测出来的参数。原始失败响应继续保留。

这定位了错误所在环节，但无法从外部请求日志解释模型为何生成前导零，不能把“增加反馈信息”称作已修复模型生成能力。

### 终态引用混用了不同类型的 ID

此前第 20、21 轮的 `evidence_refs` 混入 `frame_0013`、图片文件路径等。原校验要求列表中每个引用都属于本任务的环境证据集合；合法观察 ID 与非法图片引用混用，仍然整次拒绝。

实际接受的类型是本任务 observation_id，以及已有成功环境工具结果的 tool_call_id/result_ref。图片帧 ID、图片路径本身、计划和知识引用不属于该集合。

现在工具参数说明明确这些类型；拒绝反馈列出无效项、重复项和最多五个已有合法引用示例。既有证据集合、新鲜度检查及完成判断保持不变，拒绝不会更新终态。

## 视觉定位期间进程退出：已定位本地 SAM，底层根因未解决

恢复此前 Flash AerialVLN 第 5 轮对应的 RGB、深度、观察 ID 和原查询，使用同一能力配置。复用原视觉定位探针，未调用语言模型或启动模拟器。

| 复现 | 调用数 | 进程退出码 | 耗时 |
| --- | ---: | ---: | ---: |
| 独立进程 cold1 | 1 | 0 | 28.8 秒 |
| 独立进程 cold2 | 1 | 0 | 29.8 秒 |
| 独立进程 cold3 | 1 | 0 | 29.8 秒 |
| 同进程 warm5 | 5 | 0 | 76.9 秒 |

八次调用全部成功。阶段记录覆盖图像读取、Torch 导入、GroundingDINO/SAM 加载和推理、深度处理及产物写入；实际设备为 CPU。采样的最高进程 RSS 约 3.19 GiB、最低系统可用内存约 2.60 GiB，仅描述本次复现，不能据此推断先前退出时的资源状态。

原异常没有 traceback 或可靠的系统故障记录，只有调用发起后进程退出码 1。当前没有证据支持修改模型库、切换设备或断言 OOM/CUDA 故障；没有针对假设修改感知实现。

随后受监督的 TravelUAV 实飞在第 5 轮定位工具内崩溃。父进程捕获退出码 **3221225477 / 0xC0000005**，stderr 保存 `Windows fatal exception: access violation`。GroundingDINO 推理已结束，SAM 加载已完成，`sam.inference begin` 后无结束记录。实际设备 CPU；依赖为 Torch 2.5.1、Transformers 4.57.6、NumPy 1.26.4、Pillow 12.3.0、Python 3.12。

栈顶是本机 `transformers/models/sam/modeling_sam.py:875`，即 SAM 视觉注意力对相对位置张量执行 `reshape` 的位置，不能据此断言后续 SDPA 算子就是根因。[微软说明](https://learn.microsoft.com/en-us/shows/inside/c0000005)将此退出码解释为非法内存访问；它本身不等于内存不足、权限不足或 CUDA 故障。崩溃前采样仍有约 3 GiB 系统可用内存，但 RSS/可用内存采样也不足以排除提交内存或原生库问题。

用新捕获的 TravelUAV `obs_0003`、原 RGB/深度及原查询继续离线拆分，均无模拟器或语言模型调用：

| 新现场复现 | 结果 | 含义 |
| --- | --- | --- |
| 原完整路径 cold1/cold2/cold3 | 崩溃 / 通过 / 通过 | 相同输入可离线复现，但不是每次发生 |
| SAM 注意力改为 eager，三次 | 三次均崩溃 | 无效的绕过方案，未用于生产 |
| 仅 SAM 图像编码器，一次 | 通过，18.5 秒 | 不足以排除间歇性故障或确认 GroundingDINO 是诱因 |

eager 栈顶为同文件第 824 行的相对位置张量 `reshape_as`，与原路径处于同类步骤。该诊断使用库的 `attn_implementation` 参数（[官方接口说明](https://huggingface.co/docs/transformers/v4.57.1/en/attention_interface)）；诊断覆盖还提前导入 SAM 类，因此通过率不应作为严格性能或因果比较。结果明确否定“切换 eager 即可解决”，没有据此修改生产后端，也没有自动降级 CPU（原本就是 CPU）。

目前证据可以区分模型输出错误与本地原生故障，并定位最后执行阶段；尚没有原生转储/C++ 调用栈来确定损坏内存的源头。旧 AerialVLN 退出码 1 不能直接归为这次 0xC0000005 的同一根因。由于新现场仍可崩溃，未继续启动额外实飞或声称感知故障已修复。

实飞崩溃栈（历史路径：`tests/manual_outputs/flash_error_diagnosis_20260922/travel/stderr.log`；产物已删除） ·
新现场输入（历史路径：`tests/manual_outputs/flash_error_diagnosis_20260922/travel_reproduction/input.json`；产物已删除） ·
原路径离线崩溃（历史路径：`tests/manual_outputs/flash_error_diagnosis_20260922/travel_reproduction/cold1/stderr.log`；产物已删除） ·
eager 崩溃（历史路径：`tests/manual_outputs/flash_error_diagnosis_20260922/travel_reproduction/eager1/stderr.log`；产物已删除）

新增诊断父进程保留 stdout/stderr、资源样本、退出码和超时标志；子进程启用无缓冲输出及 faulthandler。实飞使用隔离 run 目录，退出后核对场景注册信息中的 PID/场景身份，再调用原受管清理机制，并核验进程组和端口；本地 SSH 隧道还校验 PID 创建时间与命令行，避免误清理其他任务。

诊断入口位于 `tests/manual/process_diagnostics.py`、`supervised_flight_probe.py`；生产感知代码不加载诊断钩子。
复现输入（历史路径：`tests/manual_outputs/flash_error_diagnosis_20260922/grounding/input.json`；产物已删除） ·
cold1 阶段记录（历史路径：`tests/manual_outputs/flash_error_diagnosis_20260922/grounding/cold1/trace/stages.jsonl`；产物已删除） ·
warm5 进程记录（历史路径：`tests/manual_outputs/flash_error_diagnosis_20260922/grounding/warm5/process.json`；产物已删除）

## 36 次反馈恢复对照

使用四次非法 JSON 和两次终态拒绝之后的真实请求上下文。每个样本原反馈/改进反馈各重复三次，共 36 次 Flash 逻辑请求；图像、工具定义及其余历史固定，只替换最近一次相关反馈。终态 ToolResult 和重复的 protocol feedback 同步替换。没有执行返回动作，没有使用未来环境证据。

| 检查 | 原反馈 | 改进反馈 |
| --- | ---: | ---: |
| 四个 JSON 样本返回合法调用 | 6/12 | 7/12 |
| 两个终态样本返回合法引用 | 3/6 | 6/6 |
| 仍选择原工具 | 18/18 | 18/18 |
| 返回总 token | 704,438 | 706,596 |

JSON 改善有限，且第 14 轮样本在改进反馈下仍 3/3 失败；没有继续调提示词重跑以追求更好数字。终态引用的结果支持明确列出无效项和合法类型，但样本过小，不能宣称问题已永久消失。

合法返回往往调整了速度、时长或垂直方向，不能将其一概计为“只改格式”。例如第 9 轮一条改进反馈响应保留 `vx=2, vy=0.8, duration=2.5`、将垂直速度选为 `-0.3`；另一条选为 `+0.3`。第 14–16 轮原公开意图提到下降，负 vz 与该意图更一致，不能单凭与“机械去掉前导零”所得正值不同就判定擅改意图。探针中的 decimal hypothesis 仅用于对照分析，绝不作为可执行修复。

对照 HTML（历史路径：`tests/manual_outputs/flash_error_diagnosis_20260922/feedback/review.html`；产物已删除） ·
固定输入及两组反馈（历史路径：`tests/manual_outputs/flash_error_diagnosis_20260922/feedback/manifest.json`；产物已删除） ·
逐次响应（历史路径：`tests/manual_outputs/flash_error_diagnosis_20260922/feedback/results.json`；产物已删除） ·
统计（历史路径：`tests/manual_outputs/flash_error_diagnosis_20260922/feedback/summary.json`；产物已删除）

## 两次受监督实飞

均使用完整新指导、Flash、原固定案例，串行执行并重新创建场景。每次请求、轨迹、RGB/深度、资源采样和清理状态均写入独立目录。历史对照的反馈与终态参数说明不同，本轮是修复后的诊断验证，不作为仅更换模型的严格配对实验。

| 指标 | AerialVLN 多地标 | TravelUAV 紫色摩托车 |
| --- | --- | --- |
| 任务 | 39GHHAVOMOP8ZV27ILH5P0TV5F6J4A | c3f9fc36-7c55-4a48-8213-5adce9b35b12 |
| 状态 | 正常结束，基准不成功 | 第 5 轮 SAM 原生崩溃，中断，无最终成绩 |
| 决策数 / 已执行物理动作 | 36 / 19 | 5 / 2 次转向 |
| 规划 | 首轮；第 6、16、24、33 轮更新 | 首轮；中断前未更新 |
| knowledge.load | 0 | 0 |
| JSON 拒绝 | 0 | 中断前 0 |
| 终态 | 第 34 轮因新鲜证据不足拒绝；第 35 轮定位后第 36 轮通过 | 未提交 |
| 父进程计时 | 541.7 秒 | 124.6 秒 |
| prompt / completion / total token | 1,676,040 / 16,010 / 1,692,050 | 51,479 / 2,145 / 53,624 |

AerialVLN 首轮规划覆盖指令中的顺序子目标，之后有进度更新，但不能认为这些更新全部有充分观察支持。例如部分棕色建筑/车辆查询没有有效检测或得到 `ground` 标签，后续候选置信度有限，模型仍推进计划。最终距目标 **54.996 米**，轨迹长 248.446 米、nDTW 0.377719；oracle_success 为真，表示曾经过成功范围，最终又离开。评测 steps=20 包含一个虚拟 stop，真实飞行动作为 19 次。终态引用合法不意味着视觉判断或导航语义正确。

TravelUAV 首轮计划包含朝向目标方位、确认摩托车及周边、接近目标，随后转向并定位；尚未完成接近阶段即中断，无法评价最终目标确认与完成引用。两例均未加载知识，因此没有“知识已被用于后续决策”的证据。Aerial 的路线识别困难至少部分属于现场证据不足，不能仅凭零次调用就认定每一轮都违反 knowledge.load 适用要求；方法指导在持续歧义下是否能帮助，仍待验证。

父进程在两个案例均完成身份核对和清理：远端所属进程组成员为空、服务端口不可达；Aerial 隧道已自行退出，Travel 遗留隧道按 PID/创建时间/命令行核对后终止。Travel 的实际原生崩溃验证了监督器能留存失败，而不是只有模拟退出测试。

Aerial 基准 HTML（历史路径：`tests/manual_outputs/flash_error_diagnosis_20260922/aerial/evaluation_output/evaluation/review.html`；产物已删除） ·
Aerial 监督与图像（历史路径：`tests/manual_outputs/flash_error_diagnosis_20260922/aerial/review.html`；产物已删除） ·
Travel 部分轨迹与响应 HTML（历史路径：`tests/manual_outputs/flash_error_diagnosis_20260922/travel/review.html`；产物已删除） ·
Travel 清理记录（历史路径：`tests/manual_outputs/flash_error_diagnosis_20260922/travel/supervision.json`；产物已删除）

## 验证与重放入口

自动测试覆盖 JSON 前导零拒绝、合法数字不变、反馈位置、拒绝轮不执行动作、混合/跨任务/知识/计划/重复引用拒绝且终态不变，以及异常退出、超时区分、所属场景清理和中断 HTML。反馈专项测试 19 项通过。

最终完整回归：`python -m pytest -q --tb=short`，**1115 passed，10 warnings，34.61 秒**。警告为既有 Pillow/msgpack 弃用提示。变更文件差异检查通过，报告 UTF-8 与本地链接检查通过。

所有新结果位于 `tests/manual_outputs/flash_error_diagnosis_20260922/`，旧产物均保留。手工探针示例（使用新的输出目录）：

```powershell
python tests/manual/visual_grounding_capability_probe.py --rgb <原RGB路径> --depth <原深度路径> --query <原查询> --include-distance --output-dir <新目录> --trace-dir <新追踪目录>
python tests/manual/supervised_flight_probe.py --benchmark travel --output-dir <新目录>
```

当前剩余项是 SAM 原生内存访问异常的底层根因与稳定修复；调查已保存间歇性复现及无效绕过证据。模型非法 JSON 的恢复仍不稳定，规划/知识指导也不能保证任务成功。没有以错误次数下降或单例完成宣称整体能力提升。
