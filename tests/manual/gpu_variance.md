# 固定线程 GPU 波动复测

用于排查同一视觉输入在不同进程中的推理耗时变化。复用真实输入、CUDA 检查、设备闸门、
监督进程及短路径审阅出口；不启动模拟器或请求语言模型。生产默认、精度和 GPU 策略均不变。

```powershell
& 'F:\conda_env\flyagent\python.exe' tests/manual/gpu_variance_probe.py --input-manifest F:/UAV-Agent/tests/manual_outputs/ro_gpu_20260927_1624/manifest.json --sample 1 --trials 3 --repeats 10 --root F:/UAV-Agent/tests/manual_outputs/gv_DATE --review-root F:/UAV-Agent/tests/manual_outputs/reviews/gv_DATE
```

默认三个独立子进程，每进程十批，固定一个工具线程；每进程首批为冷启动，其余九批
分别统计。四组历史输入全部校验并保留，仅指定样本执行，其余页面无执行记录。
所有公开结果严格与第一个进程串行结果对比，不放宽浮点容差。错误立即暂停并保留部分页面。

先运行相关测试及完整 pytest，然后冻结源码、输入和权重；计时期间不同时运行项目实验。
父进程通过现有 GPU 监控器的可选采样器记录频率、功耗、温度、总显存、利用率和可见
计算进程。详细查询失败不丢弃基本样本；未知进程显存和功率限制保持 null。
三个 NVIDIA 查询顺序执行，约一秒采样；与批次的关联属于粗粒度相关证据，不是内核级
测量。WDDM 下进程列表和显存可见性有限，不能据此宣称独占 GPU 或确认显存换页。

审阅入口为 index.html，分段及 GPU 关联记录为 performance.html 和 summary.json。
每个条件保存完整计时、资源采样、工具结果、图片及子进程退出记录。Chrome/Edge 检查
实际执行样本的时间线与审计页。重复实验应使用新的目录，禁止覆盖旧证据。

可选 `--cuda-trace` 在每进程最后一批启用 PyTorch CPU/CUDA profiler，并将该批排除
出普通热统计。十批配置下，第一批冷启动、随后八批普通热运行、最后一批诊断跟踪。
输出原始 Chrome trace 和内核/复制/主机同步摘要，并为现有阶段附加 record_function
标记。跟踪批次含采集与导出开销，不用于宣称提速；GPU 活动按区间并集计算，主机
等待与设备活动不能相加。缺少真实 CUDA kernel 事件时暂停，不接受 CPU-only 替代。
当前仅支持本探针固定单线程；不将此跟踪方案用于多线程归因。

可选 `--cuda-events` 不依赖 CUPTI，仅使用现有 Torch CUDA Event；与 `--cuda-trace`
互斥。末三批诊断、其余批次按原方式计时；至少五批，默认十批时普通热统计为六批。
记录 forward 的当前流 elapsed time、CPU 墙钟、已有显式同步，以及 pipeline 层的
Tensor.to/cpu 传输。内部模型复制未单独拆分；流时间包含调度及主机提交间隙，不能
称为内核净计算时间。同步属于 forward 内部子区间，不能相加。诊断增加阶段同步，
必须与普通运行分开报告。所有补丁仅存在于单线程实验子进程的诊断上下文中，异常
退出上下文也恢复方法；独立 JSON 保存每条记录和分组汇总。

`--cuda-events --sam-modules` 进一步测量 SAM 的 image_encoder、prompt_encoder 和
mask_decoder。仅在诊断批次临时包装已安装 Transformers 的模块方法，不修改依赖文件。
模块区间嵌套在 SAM forward 中，不能把它们和 forward 再次求和；模块边界同步也有
扰动，需结合普通批次与严格输出对照判断。

`--sam-coordinate-ab --repeats 13` 为独立实验：不使用 profiler/event 新增同步，在
同进程按冷启动原实现、随后 ABBA×3 交替比较 SAM 相对位置坐标的 CPU/GPU 构造。
仅在实验上下文中改写已安装 get_rel_pos 的两个 arange 的 device 参数；AST 结构
不符合预期即拒绝，依赖文件与生产适配器不改动。先在 CUDA 上对 14/64 等尺寸及
非等长坐标逐元素严格比较，再进行完整工具结果比较。每条件分变体报告，混合的
warm P50/P95 不作为 A/B 结论。该实验不缓存图像、不改精度、不预设优化有效。

## 使用已安装的 Nsight Systems

`nsight_stage_probe.py run` 接受原输入/输出参数，以及 `--nsys <现有 nsys.exe 路径>`。
仍由 flyagent Python 启动项目进程。仅采集每个子进程最后一个热批次的 CUDA/NVTX，
关闭 CPU sampling/context switch，不提权、不安装工具。CUPTI SQLite 中必须有真实
kernel 记录，否则实验暂停。原始 nsys-rep、SQLite 和 API/内核/复制汇总独立保存。
跟踪批次排除出普通热统计；累计 API 时间与 GPU 时间存在包含/重叠关系，不能相加。
监督器的根进程变成 Nsight launcher，因此根进程 RSS 不能冒充目标 Python 的 RSS。

`--capture-on-slow` 为有上限的 Nsight 诊断选择：跳过冷批，在非冷批超过 3 秒时采集
下一批；否则采集末批。每进程只采一次，记录 capture_selection.json，不能当作
随机抽样或保证下一批仍慢。项目进程直接使用冻结脚本入口，避免 Windows Nsight
转发 Python -c 内联代码时丢失引号。

普通 `gpu_variance_probe.py --windows-memory` 可选启动隐藏的 Windows GPU 进程
内存计数器采样进程；只读，不改变其他进程或 GPU 策略。保存 Dedicated Usage、
Shared Usage 和 Total Committed 原值及状态码，按目标 PID 和适配器实例独立汇总。
不跨实例合并，也不把未知值当零。Shared Usage 本身不能证明换页；不得把三种
口径相加。目标进程没有样本时暂停诊断。采样结束仅关闭本探针拥有的 PowerShell
进程。原始计数器和错误日志随短路径审阅包一起导出。
