# 真实 AerialVLN 导航与分组历史比较

入口负责冻结、准入、串行监督与只读报告；导航、评分和场景清理由生产组件执行。
原十例调用方式继续可用。追加困难清单和插件配置后运行 20 例，所有数量读取 manifest。

`--hard-only --hard-case-ids-file` 只运行锁定困难十例，manifest 仅包含 hard10，
技术审计检查前两例。`--previous-root` 可以指向已完成的二十例目录：按身份筛选困难组，
要求十例齐全、各例有终态与结果且顺序相同；重复、缺失或重排均拒绝。默认原十例调用
仍要求上轮具有完全相同的案例列表。困难组主对照为上轮同十例，第二对照为旧全量。

```powershell
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_plan_context_probe.py prepare --root 'H:\FlyAgent-Eval\avln_plan_check_hard10_new' --previous-root 'H:\FlyAgent-Eval\avln_pixel_place_plan_20_20261011_190946' --plugins-config configs/plugins.pixel_map.json --hard-case-ids-file tests/manual/aerial_vln_hard_instruction10.json --hard-only
```

计划工具描述按需使用：较长、困难或多阶段任务可记录计划，不要求首次动作前创建。
已有计划在进度或方法改变时更新，不要求每次动作前额外核对。
工具描述负责适用时机，主提示和周期提醒只提示维护已有计划；
Runtime 不判断任务复杂度，也不新增确认字段。
报告记录首次提交前实际物理窗口数；此记录不能证明模型内部核对过或路线语义正确。
冻结清单额外记录相对上轮的源码差异，不改写上轮源码和结果。

原二十例模式保持原十例顺序，随后十例 high。困难清单从旧全量已完成的 high 案例抽样，场景
2、5、10、12、14 各两例；种子 20261011，按 SHA256("种子:场景:high:episode_id")
排序，不使用成绩。准备时重新校验清单与抽样规则，锁定文件哈希。

原十例主对照为 `avln_single_action_teaching_10_20261011_164226`（SR 4/10、Oracle 7/10），
困难十例主对照为旧全量相同案例（3/10、3/10）。全部 20 例只与旧全量相同 20 例比较
（6/20、10/20），不混合版本。历史独立尝试全部保留，重复导出按实际运行身份去重。

从仓库根目录运行，以下命令的新目录不得替换此前实验：

```powershell
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_plan_context_probe.py prepare --root 'H:\FlyAgent-Eval\avln_pixel_place_plan_20_new' --previous-root 'H:\FlyAgent-Eval\avln_single_action_teaching_10_20261011_164226' --plugins-config configs/plugins.pixel_map.json --hard-case-ids-file tests/manual/aerial_vln_hard_instruction10.json
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_plan_context_probe.py admit --root 'H:\FlyAgent-Eval\avln_pixel_place_plan_20_new'
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_plan_context_probe.py run --root 'H:\FlyAgent-Eval\avln_pixel_place_plan_20_new'
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_plan_context_probe.py report --root 'H:\FlyAgent-Eval\avln_pixel_place_plan_20_new'
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_plan_context_acceptance.py --root 'H:\FlyAgent-Eval\avln_pixel_place_plan_20_new'
```

使用 prepare 输出的实际目录。当前完整源码、提示、知识与插件一起冻结；指定插件配置
仅复制到新快照的 `configs/plugins.json`，不改工作区默认配置或历史源码。密钥不入快照。
模型沿用 gpt-6.1-sol 与原接口参数；单动作窗口、300 窗口上限、20 张总图预算及压缩设置保留。
STMR 关闭，远端 GPU 0，端口本地 35001／远端 30001。

插件启用飞行知识、pixel_depth、place_memory，关闭 visual_grounding。
地点图 1600×1100，局部 80×80 米，节点水平 10 米／垂直 5 米合并，每点最多两张照片。
图只表达实测路径、当前位置、朝向和模型填写的地点判断，照片是历史证据。
像素测距只接受 CURRENT 原始 RGB 像素，每观测至多两次；表面射线距离不是目标水平距离。
知识按需复用，地点保存不强制；计划默认 optional，不设 Runtime 强制门槛。
不用调用数量或 SR 决定技术准入。

准入回放所选上轮目录的全部实际请求（教学十例为 208 份，地点图二十例为 689 份），
旧工具名称只在离线审计 registry 中保留。原始请求、
图片及 Transcript 不改写，新投影明确标记为离线检查。保存的真实 RGB／深度在独立沙箱
验证像素、观测 ID、标记图、地点保存／查阅和照片引用；注入的无效深度明确标为派生夹具。
optional 模式使用同一真实接口完成三组创建、推进、描述纠正、排序、取消，共 15 次
合法计划调用，首次通过。计划失败停止准入，不切换严格模式。
off 模式将计划专项标记为不适用，改为三次原生调用参数准入（转向、移动、知识），
同样使用保存的真实 RGB，首次通过且不执行返回工具。两种模式均保留像素／地点沙箱准入。

保存上下文行为检查保留为历史强化提示诊断，覆盖简单目的地、复杂路线创建、同阶段继续、
跨阶段更新、判断纠正、完成核对与阻塞保留。只替换所属提示和计划工具描述；
简单任务显式标记为派生指令。图片、历史和原始记录保持不变，不执行返回的工具：

```powershell
& 'F:\conda_env\flyagent\python.exe' tests/manual/plan_stage_behavior_probe.py --root 'H:\FlyAgent-Eval\avln_plan_check_hard10_new' --history-root 'H:\FlyAgent-Eval\avln_pixel_place_plan_20_20261011_190946'
```

行为是否匹配只作诊断假设；恢复按需后，创建时机和动作前核对不作为准入要求。
此历史模板不用于 off 模式，不能替代对里程碑语义和真实导航的审阅。

每例独立进程串行执行，前两例和第一例困难任务检查真实请求、当前 RGB、唯一任务状态、
唯一最新地图、图片预算、轨迹和场景清理。导航失败与模型报错保留为正式结果，不重跑提分。
基础设施中断只有确认清理后才可 `run --resume`，新建尝试目录，最多三次，保留全部尝试。

报告输出 `comparison.json/md/html` 和 `context_modules.json/html`，连接请求、地图版本、
测距标记图、轨迹、计划时间线、照片与清理记录。计划间隔分别统计实际物理窗口及成功动作数；
展示结束时未完成步骤，路线语义由人工审阅。输入／输出 token 只统计实际 API usage，缺失标未知，
不把整轮成本归因于某个工具。临界到达仅标记正常结束且 20 < 距离 ≤ 25 米，正式 20 米
SR 不变，约 10 米建议只影响后续请求。

报告反映当前组合版本整体变化，不能单独证明地点图、测距或计划的因果收益。
optional 模式困难导航计划使用较少时，应报告“导航中计划验证覆盖不足”；off 模式计划不作为
验收指标。十例困难样本不能代表全部 high。
派生报告读取器单独记录哈希，历史源码、评分与原始记录保持完整。

离线审计还核对原始请求中实际送达的图片字节。旧记录器遗漏的 CONTEXT IMAGE 元数据
从同一份捕获请求的相邻说明恢复；保存地点后立即返回的地图仍保留原 TOOL EVIDENCE
标签，并以该次已捕获的 latest_group 选择和图片哈希核对地图身份。报告单列这种标签差异，
不改写请求、不据此修改导航快照，也不把地点标签或历史照片解释为当前相机视野。

## 无计划困难十例对照

`--plan-mode optional|off` 默认为 optional。off 只修改新快照的计划曝光和三类决策提示，
逐项保存修改前／后哈希；已有实现、任务状态结构及历史记录保留。
准入、监督器和子进程均从该冻结入口启动。报告可重复使用 `--reference-root` 指定额外
完整历史版本，校验案例身份、顺序及终态；guard 锁定其 manifest 和 progress 哈希。

```powershell
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_plan_context_probe.py prepare --root 'H:\FlyAgent-Eval\avln_no_plan_hard10_new' --hard-only --plan-mode off --hard-case-ids-file tests/manual/aerial_vln_hard_instruction10.json --plugins-config configs/plugins.pixel_map.json --previous-root 'H:\FlyAgent-Eval\avln_pixel_place_plan_20_20261011_190946' --reference-root 'H:\FlyAgent-Eval\avln_plan_check_hard10_20261011_225505'
```

然后使用 prepare 输出目录执行 admit、run、report 和 acceptance。每份实际请求检查计划
工具未曝光、task_plan 为 null、计划专属提示已移除。意外调用作为拒绝或模型错误记录，
不恢复工具，不重跑导航失败。显式版本表分开展示按需、加强提示和无计划版本的导航与成本。
该无计划对照使用其冻结的 v2 协议、知识、地图、测距及评分；不以后续协议修改重写旧请求。

## v3 协议与中间停止两例回归

v3 删除全部 landing 特例：不再把任意 landing 解释为结束、统一免除接地，或禁止推导下降。
完成条件改为完整指令路线及其最终位置关系；有明确后续要求时，中间到达或停止不能结束任务。
通用知识、执行工具和按需计划保持现状，复测同时审阅下降、接地判断及阻塞行为。

`--subset-case-ids-file` 在完整清单验证后筛选有序子集，拒绝空、未知、重复或乱序 ID。
manifest 保留源清单和子集身份、哈希，报告及技术检查使用实际数量。
两例清单为困难组原第6、9例；主对照为旧按需版，另列加强提示和无计划版。

```powershell
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_plan_context_probe.py prepare --root 'H:\FlyAgent-Eval\avln_protocol_v3_land2_new' --hard-only --plan-mode optional --hard-case-ids-file tests/manual/aerial_vln_hard_instruction10.json --subset-case-ids-file tests/manual/aerial_vln_land_regression2.json --plugins-config configs/plugins.pixel_map.json --previous-root 'H:\FlyAgent-Eval\avln_pixel_place_plan_20_20261011_190946' --reference-root 'H:\FlyAgent-Eval\avln_plan_check_hard10_20261011_225505' --reference-root 'H:\FlyAgent-Eval\avln_no_plan_hard10_20261012_100150'
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_protocol_behavior_probe.py --root 'H:\FlyAgent-Eval\avln_protocol_v3_land2_new' --history-root 'H:\FlyAgent-Eval\avln_pixel_place_plan_20_20261011_190946'
```

该接口探针仅替换保存请求中的协议，保留原指令、状态、工具及图片，并记录原始证据哈希。
返回工具只校验、不执行；可以报告阻塞，不能在所选中间停止处宣称整条路线完成。
通过上下文行为检查及原有 admit 准入后，再执行 run、report 和 acceptance。
地图沿用实测路径与地点记忆，不含道路、建筑或目标真值；STMR关闭。
技术完成不要求 SR 成功或创建计划；原始指令、动作、观测和终止理由仍需人工审阅。
# 地点工具20例复测

使用原十例加锁定困难十例，显式 `--plan-mode optional`。主对照是
`H:\FlyAgent-Eval\avln_pixel_place_plan_20_20261011_190946`，第二对照由
`--baseline H:\FlyAgent-Eval\aerialvln_s_20261007` 提供。完整20例要求身份、
顺序和终态匹配；manifest 的 overall_primary_baseline 明确总体来源，
报告不再按样本数选择基线。旧十例、hard-only、subset 模式继续支持。

```powershell
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_plan_context_probe.py prepare --root H:\FlyAgent-Eval\avln_place_memory_tools_20_TIMESTAMP --previous-root H:\FlyAgent-Eval\avln_pixel_place_plan_20_20261011_190946 --plugins-config configs/plugins.pixel_map.json --hard-case-ids-file tests/manual/aerial_vln_hard_instruction10.json --plan-mode optional
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_plan_context_probe.py admit --root H:\FlyAgent-Eval\avln_place_memory_tools_20_TIMESTAMP
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_plan_context_probe.py run --root H:\FlyAgent-Eval\avln_place_memory_tools_20_TIMESTAMP
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_plan_context_acceptance.py --root H:\FlyAgent-Eval\avln_place_memory_tools_20_TIMESTAMP
```

准入包含历史实际请求审计、保存证据沙箱、15次计划生命周期调用，以及
三组独立地点链（列表→保存→同参列表→查看→修订→同参查看）。定向准入
不计入自主使用。任何准入失败均保留记录并阻止导航。

新增 place_memory_audit.json/html，按原始工具调用和当时回执重建目录，
核对保存后重复读取，禁止从最终 graph.json 反推历史。照片原始文件一致性
沿用 capability_integrity。列表不附图，地图仍遵循最新版本媒体预算。
报告列自主调用覆盖；是否帮助重访、纠错或减少搜索须结合实际轨迹人工审阅，
调用少时说明覆盖不足。当前 v3 协议、提示及全部工作区差异均属于组合变化，
不得将相对旧协议的变化单独归因于地点工具。

接口超时等终态可能没有 SR/Oracle 评分。报告的成功率分母使用各指标的
有效评分数，不把缺失填为失败；`terminal_unscored_cases` 单列未评分终态，
`scored_pair_comparison` 只比较双方均有 SR 的相同案例。完整主对照仍保留，
有效子集不能宣称为完整20例的成功率。报告读取器若在运行后修正，保留
读取器源码及哈希；不修改冻结导航源码、原始请求或原始评分。

## 困难十例并行复测

`aerial_vln_hard10_parallel_probe.py` 复用上述冻结、准入与报告，并调用现有
parallel_campaign 控制器。唯一可写进度为原生 tasks/attempts 队列；比较报告
和验收只在内存中投影为 cases，不维护第二份运行状态。

```powershell
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_hard10_parallel_probe.py prepare --root H:\FlyAgent-Eval\avln_hard10_parallel_TIMESTAMP
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_hard10_parallel_probe.py admit --root H:\FlyAgent-Eval\avln_hard10_parallel_TIMESTAMP
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_hard10_parallel_probe.py run --root H:\FlyAgent-Eval\avln_hard10_parallel_TIMESTAMP
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_hard10_parallel_probe.py report --root H:\FlyAgent-Eval\avln_hard10_parallel_TIMESTAMP
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_plan_context_acceptance.py --root H:\FlyAgent-Eval\avln_hard10_parallel_TIMESTAMP
```

固定原困难十例顺序，前四例分两波做正式样本内技术检查，通过后继续六例。
采用 gpt-6.1-sol、optional 计划、v3 协议、48K/64K/32K 压缩和20张图预算。
每任务单物理窗口、只读 max_workers=1、最多300窗口；两通道分别使用
GPU0/2、本地35001/35002、远端30001/30002。启动前记录环境和资源占用，
不停止其他实验。导航失败和模型接口错误保留终态，只有确认清理的基础设施
中断允许恢复，最多三次尝试。队列的 terminal_provider_errors 为显式实验
选项，旧任务默认保留原有服务恢复策略。

主对照为 `avln_place_memory_tools_20_20261012_142206` 的同十例，第二对照为
`avln_pixel_place_plan_20_20261011_190946` 的同十例。未知评分不填零。
报告保留完整意图和来源，人工审阅运动路径、转向衔接与历史信息缺口；
不以 inspect 调用次数证明效果，不将组合变化归因于地图，也不承诺物理防穿楼。

并行控制器沿用生成逐例报告后清理原始深度数组的策略。审计读取器仅在
receipt.json 明确记录清理时，将无法重算的像素距离标为未知；RGB、标记图、
调用参数和工具回执仍保留。缺少清理凭据的原始深度缺失继续作为错误。
验收的 limitations 和比较报告会列出这类不可复核项，不把它们算成核验通过。
控制器修复可以通过既有 execution_overlay 单独冻结；原导航 source 不覆盖，
已有终态不重跑。中断尝试的调用成本须另计，不能仅凭最终尝试汇总宣称省时。

### 同冻结版本的串行困难十例复测

`aerial_vln_hard10_serial_retest.py prepare --root <新目录>`逐文件复制
`avln_hard10_parallel_20261012_210310/source`，不带入后续照片投影修复。
`run --root <目录>`复用监督控制器和原生恢复队列，固定每例上一轮的GPU/端口，
以`dispatch_concurrency=1`保证同一时刻只运行一个任务。宿主保留A/B两个通道，
两个可用工作线程不等于两个活跃导航任务；每次派发记录独立审计。

原接口准入按相同源码和模型复用，来源与哈希写入`admission_reuse.json`，
不冒称新准入；环境、GPU和端口重新检查。`report`直接以原并行十例为主对照，
旧20例同十例为第二对照。全部命令使用`F:\conda_env\flyagent\python.exe`。

JSON和逐例表增加`arrival_distance_review`：终点水平距离、距20米有符号差值、
超出20米的非负距离，以及`20 < 距离 <= 25`的已评分临界失败（含主动停止）。
正差值是超出，负差值是已进入；不将未知评分补零，也不改变正式SR。
