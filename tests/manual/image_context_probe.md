# 图片上下文离线测量

```powershell
& 'F:\conda_env\flyagent\python.exe' tests/manual/image_context_probe.py run --input-manifest tests/manual_outputs/cb_20260927_2028/manifest.json --root tests/manual_outputs/ic_<timestamp> --review-root tests/manual_outputs/reviews/ic_<timestamp>
```

先运行相关测试与完整 pytest，再启动实验。使用新目录，不覆盖历史证据。父子进程均使用
flyagent 解释器，子进程从冻结源码启动。无模拟器、模型请求或视觉推理；不读取 `.env`。

从六个冻结决策的观察 ID 追溯 observation → frame 引用，保留图片的文件／内存载荷形式。
逐案例扫描全部决策的当前 RGB 内容哈希，统计相邻重复，称为“可复用机会”，不能视为
旧运行真实编码次数。原始帧、元数据、输入与源码哈希独立保存。

每张图三个独立进程，每进程首调用单列，30 次普通热测量；另外 30 次诊断测量读取、
解码／RGB 转换、PNG 编码、Base64 和输入元数据组装。诊断结果必须与生产函数的
Data URL 逐字节一致。Python 分配峰值在另一次调用记录，RSS 是调用后的快照，非峰值。
父进程资源日志保留进程采样；首次调用不表示操作系统冷文件缓存。

只有存在相邻同图且至少三张图转换中位耗时 ≥2 ms，才进入缓存实验。此阶段不改变生产
代码或默认配置。六个历史完整上下文输入仍缺失，此处仅做图片组件重放。

输出包含普通与诊断原始耗时、分位数、转换次数、输入大小、资源与退出记录，以及短路径
图片审阅页面。失败写入 manifest 的 paused 状态，不以组件成功代替导航或完整构建验收。

## 缓存对照（第一阶段门槛通过后）

```powershell
& 'F:\conda_env\flyagent\python.exe' tests/manual/image_context_probe.py compare --input-manifest tests/manual_outputs/ic_<timestamp>/manifest.json --root tests/manual_outputs/icc_<timestamp> --review-root tests/manual_outputs/reviews/icc_<timestamp>
```

新增图片模块配置 `rgb_encoding_cache_max_bytes` 默认为 0（关闭）。配置必须是非负整数，
拒绝布尔值。对照使用 33554432（32 MiB），不写入生产配置。每实例最多四个 LRU 条目，
预算统计 ASCII Data URL 载荷字节，不等于总进程 RSS；字典、哈希键和临时编码内存另有成本。

每次读取原始字节并计算 SHA256，同图跳过解码和编码；来源优先级及尺寸校验保留。
只缓存不可变编码结果和尺寸，不缓存任务标签、完整上下文或像素对象。运行、场景、
任务、reset_epoch 变化清空；重新配置替换缓存。深度叠加模式不使用缓存。

每样本三个独立进程，各做重复同图、强制未命中、六图循环（超过四项容量）三组。
每组交替关闭／开启，每种首调用单列，随后 30 次；清空缓存操作在强制未命中的计时外。
逐次比较完整图片输入及像素；额外使用真实历史图片组成明确标注的构造状态，经过 Builder
和 Planner 假发送入口比较完整请求。无网络发送；不是原历史任务状态的完整恢复。

普通结果要求重复路径 P50 同时降低 ≥20% 且 ≥1 ms，P95 不退化超过 10%；未命中额外
P50 不超过 max(10%, 0.5 ms)。逐进程和汇总均判断，不为通过阈值重跑。未通过只代表
当前有限样本不足以建议下一步启用，不改精度、不扩大缓存、不修改生产默认。
