# DeepSeek AerialVLN 同案例测试

使用历史批次的原 10 例，关闭思考。模型名为 `deepseek-flash`，端点为
`https://api.deepseek.com`；Key 仅保留在本地 `.env`。
`FLYAGENT_MODEL_THINKING_MODE=disabled` 必须显式配置。此配置支持 enabled、disabled
或未设置，但 enabled 仅是参数透传，不代表已实现 DeepSeek 思考工具历史回传。

## 准备与接入

```powershell
python tests/manual/benchmark_20_case_probe.py prepare --root <新批次目录> --source-manifest tests/manual_outputs/benchmark_20_flash_20260923_v2/manifest.json
python tests/manual/model_admission_probe.py --source <历史decision_0001.json> --output <新批次目录>/admission
```

探针恢复真实图片和工具声明；只发送两轮请求，不执行工具。第二轮对第一轮实际返回
的调用明确反馈未执行，不能把它解释为导航成绩。任一接入错误停止，不自动降级。
失败产物保留；修复凭据后使用新的探针目录，人工确认通过后将通过的 admission.json
复制到批次 admission 目录，保留失败记录。批量入口强制检查接入通过且配置匹配。

## 执行与审阅

```powershell
$env:FLYAGENT_SSH_KEY = '<path-to-private-key>'
python tests/manual/benchmark_20_case_probe.py run --root <新批次目录> --review-root <短路径审阅目录>
```

后台运行使用隐藏窗口、独立父进程和 stdout/stderr 文件。每例串行监督并核验清理，
前两例自动通过 Chrome、Edge 图像检查后继续。基础设施故障暂停；恢复使用 --resume，
不覆盖已有尝试。源码变化后不能直接恢复冻结批次，需明确记录并重新冻结。

审阅目录采用 cases/01～10，同时生成 review.html 与 audit_review.html；图片资源
完整路径不超过 240 字符。未运行案例没有过程图片，异常退出页面标记无最终评分。
summary.json 和 comparison.json 记录当前结果及历史逐例对照。人工行为复查仍须阅读
时间线，核查计划顺序、观察支持及知识应用，不能以调用次数替代判断。

历史 Qwen 使用当时的提供商默认模式，本轮 DeepSeek 显式非思考；两者不能视为相同
内部推理预算。保留 source_changes_from_baseline，新增的审计范围不代表历史源码必然改变。
