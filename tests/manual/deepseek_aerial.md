# DeepSeek-V4.1-Flash 真实 Benchmark

使用阿里云百炼 OpenAI-compatible 端点时，本地 `.env` 只保存端点和
Key，进程显式选择以下模型协议：

```text
FLYAGENT_MODEL_NAME=deepseek-v4.1-flash
FLYAGENT_MODEL_THINKING_MODE=disabled
FLYAGENT_MODEL_THINKING_WIRE_FORMAT=enable_thinking
FLYAGENT_MODEL_STRUCTURED_OUTPUT_FORMAT=json_object
```

`enable_thinking` 是该端点的思考开关字段。当前端点不接受 compactor 的
`response_format.type=json_schema`，因此使用 `json_object`；返回内容仍必须通过
FlyAgent 的严格 `CompressionValidator`，不会因提供商约束变宽而直接接受。

## 准备与接入

```powershell
& 'F:\conda_env\flyagent\python.exe' tests/manual/model_admission_probe.py `
  --source <历史decision_0001.json> --output <新批次目录>/admission
& 'F:\conda_env\flyagent\python.exe' tests/manual/history_compaction_probe.py `
  --source-run <长历史 run> --output <新批次目录>/compaction
```

模型接入探针恢复真实图片和工具声明；只发送两轮请求，不执行工具。第二轮对第一轮实际返回
的调用明确反馈未执行，不能把它解释为导航成绩。压缩探针必须另外验证
`json_object` 输出通过语义、证据和边界校验。任一接入错误停止，不自动降级。
失败产物保留；修复凭据后使用新的探针目录，人工确认通过后将通过的 admission.json
复制到批次 admission 目录，保留失败记录。批量入口强制检查接入通过且配置匹配。

## 执行与审阅

```powershell
$env:FLYAGENT_SSH_KEY = '<path-to-private-key>'
& 'F:\conda_env\flyagent\python.exe' tests/manual/benchmark_20_case_probe.py run `
  --root <新批次目录> --review-root <短路径审阅目录>
```

后台运行使用隐藏窗口、独立父进程和 stdout/stderr 文件。每例串行监督并核验清理，
前两例自动通过 Chrome、Edge 图像检查后继续。基础设施故障暂停；恢复使用 --resume，
不覆盖已有尝试。源码变化后不能直接恢复冻结批次，需明确记录并重新冻结。

审阅目录采用 cases/01～10，同时生成 review.html 与 audit_review.html；图片资源
完整路径不超过 240 字符。未运行案例没有过程图片，异常退出页面标记无最终评分。
summary.json 和 comparison.json 记录当前结果及历史逐例对照。人工行为复查仍须阅读
时间线，核查计划顺序、观察支持及知识应用，不能以调用次数替代判断。

历史 Qwen 使用当时的提供商默认模式，DeepSeek 显式非思考；两者不能视为相同
内部推理预算。保留 source_changes_from_baseline，新增的审计范围不代表历史源码必然改变。
