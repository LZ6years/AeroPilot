# 只读并发、设备闸门与诊断

## GPU 环境正式对照

所有本地命令显式使用 `F:\conda_env\flyagent\python.exe`。在该环境完成相关测试和
完整 pytest 后，停止其他项目实验，再启动独立输出目录：

```powershell
& 'F:\conda_env\flyagent\python.exe' tests/manual/read_only_concurrency_probe.py run --require-cuda --input-manifest F:/UAV-Agent/tests/manual_outputs/ro_20260927_02/manifest.json --root F:/UAV-Agent/tests/manual_outputs/ro_gpu_DATE --review-root F:/UAV-Agent/tests/manual_outputs/reviews/rog_DATE
```

`--input-manifest` 复制上一轮四组真实图像及调用，并校验原输入哈希，不重新抽样。
`--require-cuda` 增加独立子进程接入验证：GroundingDINO 和 SAM 的参数设备必须为
CUDA，且实际完成前向调用、深度处理和证据写入。环境版本、GPU、权重缓存来源和
文件哈希进入审计。只使用已缓存权重，不下载、不更换精度、不自动 CPU 回退。

接入通过后进行原 60 批视觉对照，使用本轮 GPU 串行公开结果作严格基线。随后
执行 1／2／4 线程各一次构造混合批次（两个真实视觉调用、同知识条目两次读取），
验证增量内容有序交付。接入及混合批次单列，不混入 60 批的性能统计。第一批包含
模型首次加载；解释器／CUDA 初始化属于单列的子进程 preflight，不算热批次耗时。
诊断钩子在 CUDA forward 前后同步，测量已完成的推理而非 CPU 提交时间；同步仅用于
实验，所有条件一致，仍有诊断开销，不改生产后端。

正式计时期间不运行 pytest、其他项目实验或修改冻结源码。检测到其他项目 Python
实验时暂停，不结束它们。缺失权重、错误环境、后端失败、数值差异和异常退出都保留
证据并暂停，不以降低精度、增加容差或挑选重跑结果换取通过。

最终状态以本目录 `summary.json` 和 HTML 为准：prepared、running、passed、paused。
阶段工作时间、设备等待、批次墙钟分别报告，禁止重复相加。四次热运行的 P95 仅为
小样本描述；旧 CPU 结果也使用不同依赖版本，不能作为隔离 GPU 加速比的基线。

2026-09-27 的正式 GPU 对照已完成：原始目录 `tests/manual_outputs/ro_gpu_20260927_1624/`，
短路径审阅入口 `tests/manual_outputs/reviews/rog_0927_1624/index.html`。60 批对照与 4 批
接入/混合检查通过，实际 CUDA 工作及输出一致性已验证；尚不足以建议生产启用四线程。
完整性能表、首次浏览器失败及后续验收记录位于同目录的 `report.html` 与
`review_revalidation.json`。原始推理快照未因审阅修复而变更，也没有重跑推理。

审阅探针优先使用已有 websocket-client；Windows 环境未安装时使用同目录
`cdp_websocket.ps1` 调用系统 .NET ClientWebSocket，不安装依赖或切换 Python 环境。

生产默认仍为同步执行。`ReadOnlyExecutionConfig` 由调用方显式传给
`TaskRuntime` 或两个 benchmark runner；不会读取 `.env` 或自动启用示例文件。
示例 `configs/read_only_execution_config.example.json` 配置四线程、最多两个能力调用。
同设备后端加载和推理始终共享一个进程内闸门，不按线程复制模型。

```python
from src.runtime.configs import load_read_only_execution_config
execution = load_read_only_execution_config("configs/read_only_execution_config.example.json")
# TaskRuntime(..., read_only_execution_config=execution)
# TravelUavBenchmarkRunner(config, read_only_execution_config=execution)
```

只接受已经通过整轮校验的只读批次。执行器为每个调用预留资源，跳过正在等待
能力名额的调用以执行后面的知识读取。同一知识条目按声明顺序交付增量内容；
任务状态、行动记录、证据登记、Transcript 均由主线程按原调用顺序提交。
执行器逐项通知主线程，不再等整批返回才保存。乱序完成结果先写入审计专用的
`memory/tool_completions.jsonl`；前序齐全后才进入正式历史。该文件不改变模型输入
或合法证据集合。保存失败停止新调用提交，不重试工具；详见
[及时保存离线探针](tool_completion_probe.md)。
普通调用异常只生成该调用失败结果；退出时取消未开始工作并等待已开始工作。
线程不支持安全强杀，原生进程崩溃仍由实验父进程保存诊断。

## 离线真实记录对照

```powershell
& 'F:\conda_env\flyagent\python.exe' tests/manual/read_only_concurrency_probe.py run --root F:/UAV-Agent/tests/manual_outputs/ro_DATE --review-root F:/UAV-Agent/tests/manual_outputs/reviews/ro_DATE
```

从历史 `benchmark_20_flash_20260923_v2` 确定性选取 AerialVLN、TravelUAV 各两个
实际成功返回的双视觉批次，冻结原调用、RGB／深度和源码。AerialVLN 包含检测与
距离估计；TravelUAV 原双调用均请求距离，不改造原参数。每个样本每种 1／2／4
线程配置使用独立受监督子进程，五次重复，第一次加载单列。共 60 批、120 次视觉
工具调用，不请求语言模型，不访问环境。混合知识与能力的调度由事件屏障自动测试
验证，属于构造输入，不伪称来自历史模型。

每批比较公开候选、置信度、距离及状态；不把路径差异当作推理差异。任意后端错误、
设备重叠、未闭合计时或子进程异常均暂停，不进行隐式重试或 CPU 回退。
原始目录包含 manifest、冻结源码、输入哈希、逐批结果、stdout/stderr、父进程资源、
GPU 采样与子进程分段计时。短路径审阅目录包含时间线及完整审计页，标记无导航评分。
Chrome／Edge 实际检查两类样本的时间线图片；全部本地链接及路径长度同时校验。

## 真实场景对照

离线正确性及资源稳定性通过后，执行历史 ModernCityMap 第 13 例：

```powershell
& 'F:\conda_env\flyagent\python.exe' tests/manual/airsim_task_profile_probe.py run --experiment read-only-concurrency --root F:/UAV-Agent/tests/manual_outputs/ro_flight_DATE --review-root F:/UAV-Agent/tests/manual_outputs/reviews/rof_DATE
```

后台衔接可附加 `--after-offline <离线目录>`：核对 12 个条件完整通过后才启动。
若已有其他实验占用资源，可加 `--after-process <已有父进程PID>`，等待其本地进程树
退出；只观察，不结束别人的进程。门槛状态写入 `<原始目录>.gate.json`。离线失败或
父进程退出但没有通过记录时暂停，不自动转入实飞。

串行／四线程各一次，两次稳定状态复用均关闭。沿用模型、传感器、预算、所有权清理
及原有完整审阅出口。每次独立场景；基础设施、清理或审阅失败暂停。默认不传
`--experiment` 时仍为原来的稳定状态复用实验。自由模型决策可能产生不同轨迹和调用
数量，不能用总任务耗时差单独证明线程优化收益。

## 计数与性能解释

- 提交调用：来自原始模型记录，包括后续拒绝；离线重放只选已接受调用。
- 实际执行：来自 `read_call_started`，按 decision/call ID 关联。
- 实际模型工作：后端首次加载事件、`forward` 计时和设备活动峰值。

批次墙钟与各线程工作时间分别汇总，不直接相加嵌套区间。设备等待属于后端调用
时间的组成部分。父进程 `nvidia-smi` 显存／利用率是设备总量；Torch allocated／
reserved／peak 由实际推理子进程采集，是该进程共享 allocator 值。缺失字段保持
null，采样失败保留错误，不能通过显存曲线推断工具次数。没有明确收益仍保留串行默认。
