# Model Context V2 组装探针

该探针只测量当前 V2 契约，不读取 V1 run、不导入 V1 源码，也不尝试迁移旧 artifact。
它构造固定长度的 checkpoint 历史，通过真实 `HistoryProjector`、
`RequestAssembler` 和 `MediaHydrator` 记录：

- checkpoint 构建次数；
- Provider 请求文本与序列化大小；
- 被选择媒体的读取、编码和 SHA-LRU 命中次数；
- 峰值 Python 分配内存；
- 请求组装端到端时间。

探针不启动模拟器、不调用真实模型，也不把上下文缩小直接解释为任务成功率提升。

## 运行

从仓库根目录执行：

```powershell
$probeOutput = Join-Path $env:TEMP 'flyagent-context-v2-probe'
& 'F:\conda_env\flyagent\python.exe' tests/manual/context_assembly_probe.py `
  --output $probeOutput `
  --history-length 100
```

输出写入 `<output>/context_assembly_metrics.json`。`--output` 目录中同时会保存探针使用的
固定 RGB 文件和 V2 Transcript；持久化 checkpoint/Transcript 不含 base64。

如果已有独立冻结的 V1 数字结果，可通过 `--v1-metrics <json>` 传入。脚本只读取其中的
数值作为报告基线，不读取旧 run，也不提供 V1 reader。

## 解释边界

同一图片 SHA 在一个 instruction 的缓存生命周期内应只编码一次；不同 observation 即使
引用相同字节，仍保留各自的时序身份。只有历史策略选中的 `MediaRef` 才会被读取。
真实模型效果需使用相同任务集另做 A/B，本探针只证明组装行为、资源消耗和持久化边界。
