# 上下文读取作用域验证

所有命令使用 `F:\conda_env\flyagent\python.exe`。此探针不启动模拟器，不调用语言模型，
不读取 `.env`。生产改动仅在一次 `ObservationContextBuilder.build()` 内复用同文件文本和
行数组；记录仍独立解析。退出最外层作用域后释放缓存，下一次构建重新读取。

## 历史输入核查

```powershell
& 'F:\conda_env\flyagent\python.exe' tests/manual/context_build_probe.py --batch tests/manual_outputs/benchmark_20_flash_20260923_v2 --root tests/manual_outputs/cb_<timestamp> --review-root tests/manual_outputs/reviews/cb_<timestamp>
```

按原顺序选择两类基准最早有完整首、中、末请求及图片的案例。中间轮取索引
`(轮数 - 1) // 2`，每轮取首次请求；不按任务成绩选择。请求图像通过原恢复机制检查路径、
数量和 SHA256。原请求、源码快照及输入哈希独立保存，旧记录不改动。

现有历史记录只有逐轮输出 `model_context` 和请求，FlightMemory 和任务状态是最终快照。
CLI 明确暂停六个性能样本，仍导出原始请求图片和审计页供复查。没有完整输入时不反推、
不拿最终状态代替、不将请求恢复等价于上下文重建，`timed_builds=0`。退出成功只表示
审阅包生成成功，实验状态必须读取 manifest 中的 `state=paused`。

## 构建对照接口及工程测试

`compare_builds(builder, state, memory, capture_request, output, repeats=10)` 用于完整、已验证
的输入。调用方提供只拦截请求的回调，禁止真实客户端。每轮交替 baseline/candidate，
前者采用修改前逐引用整文件读取算法，后者启用作用域；首轮单列，比较完整上下文和
实际请求（包含图片字节），任何差异立即写入记录并停止。

普通耗时不启用诊断钩子。另做诊断构建，记录各模块、文件读取、JSON 解析和图片编码；
采用现有 TaskProfile 的父子阶段语义，不把包含时间重复相加。Python 分配峰值由另一次
tracemalloc 构建记录，不当作系统总内存，不混入普通耗时。只读文件文本缓存的瞬时
内存成本应结合该峰值判断。

`tests/test_context_build_probe.py` 使用明确构造的 12 观察输入验证实际 Builder 和 Planner
请求边界，共十轮交替；不把这个测试当作六个历史样本的性能结果。测试还覆盖差异拒绝。
历史回放缺少逐轮输入的限制保留，需有可信的构建输入记录后才能完成历史性能验收。
