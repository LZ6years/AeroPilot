# AirSim 完整任务耗时对照（第三批）

本探针仅用于诊断。使用历史批次 `benchmark_20_flash_20260923_v2` 的第 13 例 ModernCityMap，同一初始状态分别关闭、开启稳定状态复用，各执行一次完整 TravelUAV 任务。模型仍按当前 `.env` 读取；不修改 `.env`、默认开关、提示词、传感器、300 个物理窗口以及现有非物理／协议拒绝预算。

## 运行

在项目根目录执行，提供尚不存在的输出目录；SSH 身份沿用已有进程环境或本地配置：

```powershell
python tests/manual/airsim_task_profile_probe.py run --root tests/manual_outputs/air3_<timestamp> --review-root tests/manual_outputs/reviews/air3_<timestamp>
```

新增 `--experiment read-only-concurrency` 可复用本入口做串行／四线程完整任务对照，
两次均关闭稳定状态复用，并冻结 `effective_read_execution.json`。
不提供此参数时保留上面的原实验。前置离线对照及解释边界见
[只读并发验证](read_only_concurrency.md)。两类实验现在都启用可选父进程 GPU 采样；
Torch allocator 数据只在实际 CUDA 推理进程内采集，CPU 运行不伪造 GPU 显存。

`--prepare-only` 只冻结源码／配置、验证历史数据哈希并检查远端启动条件，不请求模型、不启动场景。这份准备目录不能直接作为执行目录；正式执行使用新目录重新冻结。探针不自动重跑失败任务，也不设任务总墙钟截断。

正式执行先恢复历史第 13 例的一次真实图像请求验证模型接入和原生工具协议，返回调用不执行。随后两个子进程串行运行，源码来自同一实验快照；当前模型配置在每次启动前重新核对。首次任务完成后，必须验证所属场景进程、隧道、端口释放，以及 Chrome、Edge 两种审阅页的时间线图片，才启动第二次。基础设施、清理或审阅故障暂停；正常的导航失败按实际结果保留。

## 产物与计时含义

- `manifest.json`：案例、数据哈希、源码／配置快照哈希和脱敏模型配置。快照不复制 API Key。
- `01/`、`02/`：独立运行、资源日志、stdout/stderr、退出／清理记录、请求审计与轨迹。
- `timings.jsonl`：单调时钟开始／结束记录，含父阶段、线程和决策序号；不记录请求参数、图片内容或密钥。
- `timing_summary.json`：每线程的阶段数量、包含子阶段的耗时、扣除直接子阶段后的耗时、P50／P95、首次调用耗时、异常及未完成阶段。
- 原有 `runs/*/artifacts/*/performance.jsonl`：AirSim 细分 RPC、采图、解码、写盘记录，独立汇总到总报告的 `airsim_stages`，不能再次加到主线程总耗时中。
- 短路径审阅目录：`index.html`、`cases/01` 与 `cases/02` 中的 `review.html`、`audit_review.html`，复制图片及资源，相对链接且路径不超过 240 字符。异常退出从已落盘证据导出“无最终评分”页面。

`model.request` 包含实际 HTTP 尝试与重试等待；`model.http_attempt` 包含响应读取。`runtime.physical_window` 包含动作、稳定和动作后观察。`sensor.observe` 内的图像 RPC／解码／写盘由已有 AirSim 计时进一步解释。视觉定位分别记录加载、流水线、深度处理和产物写入；流水线扣除加载后仍包含预处理、推理和后处理，不能称为纯 GPU 推理时间。

父阶段是包含子阶段的耗时，不应相加。后台日志线程与主线程重叠，单列而不加到任务墙钟；`audit.flush_wait` 才是主线程等待日志刷出的时间。决策标签从 Planner 入口更新，该入口前的上下文组装仍带上一轮标签。计时钩子也有开销；未配对的开始记录表示中断，旧记录缺失计时不按零处理。

两个完整任务由模型分别决策，动作、轨迹和调用次数可能不同；这是定位瓶颈的试验，不能当作相同动作回放，也不能据此宣称成功率改善或自动启用复用。是否增加只读并发、调整 GPU 调度或减少 RPC，须在结果与正确性复查后决定。

## 工程验证

```powershell
python -m pytest tests/test_task_profile.py tests/test_airsim_phase2_probe.py tests/test_airsim_recovery_state.py -q
python -m pytest -q
```
