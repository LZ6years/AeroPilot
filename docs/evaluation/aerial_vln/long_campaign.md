# AerialVLN-S 长评测与恢复

评测前先运行离线恢复测试与真实 Provider 小请求验证，再冻结当前代码、提示词、
知识及配置。冻结目录不保存 `.env` 或 API Key。Windows 统一使用
`F:\conda_env\flyagent\python.exe`；串行任务子进程或双路共享宿主记录 CUDA 诊断，
每条实际任务保留对应的诊断副本。

本地完整数据集为 **864 条 instruction episode**：`val_seen` 333 条、
`val_unseen` 531 条，共 13 个场景。共享 trajectory 的不同 instruction 不去重。
难度仍在完整 split 上计算，再按精确 episode ID 选择批次。

## 请求恢复

`src/llm` 控制真实尝试，SDK `max_retries=0`。连接、流式 TLS 和临时 HTTP
错误使用连接预算，最多追加 8 次，退避 1/2/4/8/16/30/30/30 秒；超时独立追加
1 次。整个请求恢复窗口最多 600 秒，每次请求超时受剩余窗口限制。
证书校验保持启用；证书、鉴权、配置错误不重试。压缩客户端仍为 300 秒、
超时不重试，并沿用自身连接配置。

未完整结束的流式片段丢弃，重新发送相同输入，只有完整响应进入 Transcript 并执行工具。
终局完整后的尾部断开接受该响应，不重复请求。日志保存真实次数、类别、退避、耗时及
耗尽原因。模型协议拒绝计数和物理窗口计数不受传输尝试影响。

## 队列与计分

临时 Provider 请求耗尽后，该任务进入 `retry_pending`，先验证场景清理，再继续其他
未执行任务。每条最多 3 次完整尝试，分别从原始起点开始，保存到独立 attempt 目录。
连续 3 次 Provider 失败后，按 30/120/300 秒执行服务检查；未恢复则 `paused`，
保留其余任务为 pending。未知基础设施故障或无法证明清理完成也暂停主机派发。

每个 `(split, episode_id)` 仅采用第一个有效评分。导航失败、blocked 和窗口上限
属于有效结果，继续队列。基础设施中断的轨迹保留，但不进入 SR 分母。
报告生成失败独立修复，不重新飞行。子进程整体期限 7200 秒，无进展期限 900 秒。

## 两阶段执行

产物根目录：`H:\FlyAgent-Eval\aerialvln_s_20261007`。

1. 混合 15 条预跑：9 seen、6 unseen，覆盖全部 13 场景。
2. 按用户后续要求，15 条完成后先暂停并交付报告，再启动 seen 剩余 324 条，合计 333 条。
3. unseen 剩余 525 条单独启动，合计 531 条；复用同一冻结版本的 6 条预跑结果。

原冻结入口串行；本次 seen 可使用下述外部双路入口。每条任务前后立即原子保存进度，
每 25 次尝试检查存储和覆盖。
预跑后用实际产物大小预算剩余产物的双倍空间，加一个临时深度峰值，并至少保留
10 GiB 可用空间；不足时暂停。导航成功率不是继续 seen 的门槛。

两套 split 保存全部原始分辨率 RGB 和工具证据图片，HTML 使用相对路径直接引用，
不复制图片、不嵌入 Base64、不保存完整 Provider 请求 debug。seen 展示完整公开说明、
动作、工具结果和实际/参考轨迹；unseen 展示全部图片时间线与动作摘要。
默认审计模式及历史产物保持兼容。仅在任务评分、报告、场景清理均成功后删除该次
尝试目录内的临时 depth `.npy`。中断任务保留深度与部分轨迹，供恢复审查。

## 命令

从仓库根目录完成测试和 `probe` 后，保存通过的 `verification.json`（测试命令、
结果与时间）；`prepare` 会执行只读远端预检并冻结源代码。

```powershell
& 'F:\conda_env\flyagent\python.exe' -m evaluation.benchmarks.aerial_vln.campaign probe
& 'F:\conda_env\flyagent\python.exe' -m evaluation.benchmarks.aerial_vln.campaign prepare
```

正式启动和续跑均从冻结源码目录执行：

```powershell
Set-Location 'H:\FlyAgent-Eval\aerialvln_s_20261007\source'
& 'F:\conda_env\flyagent\python.exe' -m evaluation.benchmarks.aerial_vln.campaign run --stage seen
# seen 完成后另行执行：
& 'F:\conda_env\flyagent\python.exe' -m evaluation.benchmarks.aerial_vln.campaign run --stage unseen
# 只补报告，不启动模拟器：
& 'F:\conda_env\flyagent\python.exe' -m evaluation.benchmarks.aerial_vln.campaign repair-reports
```

创建根目录下的 `PAUSE` 文件会请求暂停，父进程停止当前子进程并验证清理。
确认恢复后删除 `PAUSE` 再执行同一 `run` 命令。OS 文件锁防止两个派发器同时运行；
中断续跑会先核验并终止遗留的、身份匹配的本次子进程，再清理其自有场景。
查看 `progress.json`、`dispatcher.log`、各 split `summary.json/index.html` 与各
attempt 的 `receipt.json/cleanup.json`，可区分评分、待重试、服务暂停及报告待补状态。

本次已启动的冻结派发器使用 `tests/manual/aerial_vln_pilot_handoff.py` 交接监护：
最后一条预跑子进程正常退出后，在父进程验证远端清理及派发下一条之前写入 `PAUSE`。
它不更改冻结飞行代码或已有结果，生成根目录 `pilot_report.html/json` 和
`pilot_report_ready.json`。交付报告前保留 `PAUSE`；报告交付后续跑同一 seen 命令。
该监护也会为基础设施暂停生成已完成部分的报告。

本次预跑中另发现 SSH `mkdir` 超时发生在场景启动之前。可对交接监护启用
`--recover-startup`，使用 `src/sim_server/remote/recovery.py` 的只读清理核验：
本地生命周期必须记录回滚、没有启动 spec，原尝试的进程与隧道必须退出；远端
必须确认无注册文件、无启动 spec、无匹配场景的进程，且 RPC 端口拒绝连接。
核验依赖 SSH，不能把 SSH 不可达当作清理成功；核验连接失败按 30/120/300 秒
延后检查，仍无法核验时保持暂停。已启动、未知原因和手动 `PAUSE` 都不能由该路径恢复。

核验通过后保存独立 `cleanup_recovery.json` 和队列恢复事件，保留原始失败回执，
再启动同一冻结派发器。任务保留三次完整尝试上限，已有评分不重复飞行或计分。
监护本身是冻结飞行流程外的诊断工具，不修改冻结源码、模型配置或评分结果。
若原端口被另一组实验占用，监护记录该 SSH 隧道与父派发器的 PID 身份并等待整组
实验退出，避免趁任务之间的端口空闲抢占模拟器；不终止其他实验。等待状态保存在
`shared_simulator_wait.json`，释放后重新核验清理并自动续跑剩余预跑。

交付预跑报告后，保存 `pilot_report_delivered.json`，seen 全量使用
`--monitor-split val_seen --recover-startup` 延续同一套清理核验和启动恢复监护。
此模式不再设置预跑交接的 `PAUSE`，只恢复明确启动的同一 split；seen 完成后退出，
不会派发 unseen 剩余任务。`split_observer_status.json` 保存监护进度。

```powershell
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_pilot_handoff.py --output 'H:\FlyAgent-Eval\aerialvln_s_20261007' --recover-startup
```

真实验证中的故障注入日志明确标注 `injected_tls_failures`；它验证本地重试后
真实请求成功，不宣称真实网络曾发生故障。

## seen 双路共享模型

新增外部控制器只执行剩余 seen，默认两路，也支持单路恢复。远端 A 路为
GPU 0、本地隧道 35001、AirSim 30001；B 路为 GPU 2、本地隧道 35002、
AirSim 30002。GPU 1 禁用。启动时另存端口和 GPU 执行配置，并校验原冻结源码、
模型配置和 split 身份；原 source 目录不修改。控制器及恢复助手另存不可变快照，
后续宿主重启继续使用同一快照。

远端 Vulkan 可能把软件适配器插入显卡列表，不能直接把 `nvidia-smi` 编号作为
`GraphicsAdapter`。控制器通过现有 NVIDIA ICD 枚举 Vulkan 设备，按 UUID 匹配
GPU 0/2，冻结实际适配器索引与 PCI 地址；CUDA 可见设备也按 UUID 限定。
启动前再次核验映射，试跑期间每 15 秒、正式运行每 60 秒核验自有进程组的实际
GPU。发现 GPU 1 或其他未经授权显卡时立即停止新派发并清理。SDL 显示上下文
可能同时出现在 GPU 0，允许 B 路出现 GPU 0/2，但必须有 GPU 2 的实际证据。

父控制器持有唯一队列锁，是进度的唯一写入者。宿主进程的两个任务线程分别
构造 Runtime、Provider、Transcript、模拟器和独立尝试目录，仅共享现有视觉
模型缓存与 CUDA 推理锁。本机 CUDA 0 上只加载一份 GroundingDINO 和 SAM；
模型请求及飞行并行，GPU 前向推理串行。宿主记录实际 flyagent/CUDA 张量诊断，
每条任务保存该诊断、线程归属和指令哈希。

共享宿主只读取已经完整缓存的模型权重。启动前核验缓存快照及文件哈希，设置
`HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，避免元数据 HEAD 请求的 TLS
故障阻塞模型加载。此设置仅影响本机模型缓存读取，Provider 请求仍按原配置在线执行。

HTML 由父控制器按原完整格式生成，引用各任务原始图片；线程不绘图、不复制媒体，
不保存完整请求 debug。首个有效评分、三次尝试上限、TLS 重试、Provider 服务检查、
临时深度清理及 seen 完成后停止的规则继续适用。报告失败保留评分，补报告不飞行。

先选择四条未执行 seen，覆盖低、中、高难度，分两轮双路试跑。技术门槛检查
单份模型身份、指令与图片归属、报告及清理，并通过远端 NVML 的进程组证据核实
GPU 0/2 分配。导航成功率不是门槛，有效试跑直接进入 seen 总评测。

每条线程独立监测 7200 秒期限和 900 秒无进展期限。线程卡死时回收自有宿主，
保留已落盘的完整评分，未完成任务逐条清理后进入恢复队列；清理无法验证则暂停
同主机新任务。宿主检测父控制器退出后自行退出，续跑仍须核验遗留场景。
持续并行限流或显存不足时，核验清理后降为一路；手动暂停不自动恢复。

并行启动失败可核验已经返回的任务线程：只检查该任务的运输进程和已预留端口，
共享宿主及另一条线程属于其他所有者。即使失败发生在隧道或 settings 创建之前，
仍须证明本地回滚、无启动 spec、远端无对应进程与文件、两端端口关闭；不能仅凭
线程返回或空清理列表继续派发。

SSH 启动超时也可能发生在 spec 和远端登记写入之后、SceneInstance 落盘之前。
这时使用另一条只读核验路径：要求完整启动回滚，核对本地与远端 spec、PID、
进程组、启动时间和专属 settings，重新检查远端自有进程组确已停止、无匹配进程、
无不可读取进程以及两端端口关闭。任一证据缺失仍暂停；不以旧回执或空列表替代核验。

首次切换先等待旧串行子进程正常结束，在确认无活跃飞行的调度边界设置暂停，
让旧父进程完成清理、保存评分后退出。短暂挂起只用于核实本次父调度器的边界，
不挂起飞行子进程，也不操作其他实验。旧观察器退出后，交接完成方可启动新控制器。

~~~powershell
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_parallel_campaign.py handoff --output 'H:\FlyAgent-Eval\aerialvln_s_20261007'
& 'F:\conda_env\flyagent\python.exe' tests/manual/aerial_vln_parallel_campaign.py run --output 'H:\FlyAgent-Eval\aerialvln_s_20261007' --workers 2
~~~

查看根目录 parallel 下的 execution_overlay.json、status.json、trial_check.json、
gpu_samples.json 和 progress_check.json；原 val_seen/index.html 仍是可视化入口。
存储预算加上两条并发尝试的临时峰值，每 25 条更新覆盖率、实际吞吐和容量。
未启动的 unseen 保持原状态。

Windows 上共享状态读取和原子替换可能短暂遇到文件占用。外部控制器及宿主的
JSON 读写、队列进度保存使用独立的短退避：10/20/50/100/200/500/1000 毫秒，
最多 8 次操作。成功恢复记录次数和耗时；持续拒绝访问仍报告并保留恢复状态，
不能把权限错误解释为空队列或空清理证据。这些重试不消耗 Provider 或任务尝试。
退出时同步刷新 status.json 与 exit.json，清空运行路列表，避免显示过期状态。

## 无人值守 seen 守护

外部 `watch` 入口只读取进度，不分配任务、不改评分、不直接操作场景。它持有
独立 OS 文件锁；已有控制器仍在运行时仅接管监督，不启动第二个控制器。控制器
异常退出时按 30、120、300、900、1800 秒退避，最后一个间隔循环使用；退避状态
落盘，守护进程自身重启也不会失去冷却时间。有效评分增加后重置退避。

所有续跑均先通过原控制器的冻结哈希、GPU UUID、存储、所有权及遗留场景清理
核验。清理仍无法核验时仅延后重查，不派发新任务。Provider 持续不可用时先做
30 秒服务探测，服务未恢复不启动飞行，也不消耗任务尝试。控制器超过 20 分钟
未更新进度或状态时，只回收 PID、启动时间和命令一致的本次自有控制器，随后
按上述清理流程恢复。模型请求和压缩的正常重试窗口不会触发此门槛。

启动 spec 尚未写入时，历史 SSH stop 超时可能留下清理错误，却没有实际启动场景。
已结束的执行路可重新核验：要求 backend 关闭和端口释放记录，错误仅来自临时运输
故障的 stop 操作，本地自有运输进程已退出，远端无启动文件、对应进程或不可读取
进程，且两端 RPC 端口均关闭。新核验保留原始生命周期错误和初次清理回执，
不靠反复读取旧错误永久阻塞恢复。未知、鉴权、其他清理阶段错误仍不能通过。
Windows 本地端口使用系统 TCP 表证明无端点；表不可读取或仍有端点则拒绝证明。
连接超时始终不能作为端口关闭证据。

队列统一读取基础设施错误中的单个 `error` 和清理阶段的 `errors[]`。场景清理
已经核验、且全部错误均为临时超时或连接中断时，保存本次无效尝试并延后重跑，
另一条执行路继续派发。最终观察超时同样适用；错误轨迹不能进入 SR/OSR。
三次完整尝试上限及首个有效评分规则保持一致。

需暂停的错误另存可选 `pause_cause`，记录任务和尝试编号；守护核对该编号的
原始回执，并使用队列的同一分类判断能否恢复。混合未知错误、鉴权错误或缺少
清理证据不会被归为可恢复。旧进度无需迁移字段；旧泛化暂停须先核查证据，再
显式启动一次控制器，由它保存新的运行状态。重新运行时清除过期暂停原因。

Windows 当前用户任务在登录时和每 5 分钟启动隐藏守护进程，OS 锁消除重复启动。
守护与聊天窗口独立；任务不请求管理员权限或账户密码。电脑关机、断网或休眠时
无法继续计算，登录或唤醒后可恢复。seen 完成后停止，计划任务随后自行禁用。

手动 PAUSE、鉴权或配置错误、GPU 越界、技术验证失败及任务三次尝试耗尽均保留
等待处理状态。磁盘不足时重查容量，无法读取进度或冻结证据时禁止启动。手动暂停
不会被守护覆盖：继续时删除 PAUSE，再创建 `parallel/watchdog/RESUME`，或显式
启动 `run` 控制器。日志只在状态变化、重启和退避时追加，状态文件覆盖保存。

Python 宿主明确使用 UTF-8。旧宿主或原生依赖的 stderr 可能含其他编码的进度条，
诊断读取将非法 UTF-8 字节保留为转义文本，不因日志解码失败遮蔽实际退出原因；
原日志字节保持不变。旧控制器仅因读取 stderr 解码而退出时也可进入恢复。

~~~powershell
# 先通过离线测试并冻结最新外部执行扩展，再安装本次评测的当前用户任务。
& '.\tests\manual\install_aerial_vln_watchdog.ps1' -OutputRoot 'H:\FlyAgent-Eval\aerialvln_s_20261007' -Workers 2
~~~

查看 `parallel/watchdog/status.json`、`events.jsonl`、`scheduled_task.json`，或原
`val_seen/index.html`。监督日志及控制器日志也存于 H 盘，完整 HTML 和原始图片
继续使用原评测目录；守护不创建图片副本或完整请求 debug。

~~~powershell
& 'F:\conda_env\flyagent\python.exe' -m pytest -q
& 'F:\conda_env\flyagent\python.exe' -m pytest -q tests/diagnostics/test_aerial_vln_pilot_handoff.py
~~~
