# 远端通过本机 Sakuracat 调用 API

远端生成文件统一放在 `$DATADIR` 对应分类目录。本机服务器已验证路径为
`/home/Newdisk/wangzequan`，非交互式 SSH 需显式 export；不要移动运行中的冻结目录。

Windows PowerShell 保持以下 SSH 命令运行，把服务器回环端口转到本机代理：

```powershell
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -i C:/Users/HP/.ssh/id_rsa_npu_wzq -p 10245 -R 127.0.0.1:17890:127.0.0.1:7890 wangzequan@10.68.240.49
```

本次已经建立该隧道；同一个远端端口不要重复启动隧道。
Sakuracat 保持运行，且其规则必须让目标 API 域名经过预期节点；系统代理开关本身不保证规则选择代理。
本机休眠、代理退出或 SSH 隧道断开都会影响此线路，不适合作为无人值守全量评测的唯一网络依赖。

在远端**新启动**的任务 shell 中设置（不修改已运行任务的连接）：

```bash
export DATADIR=/home/Newdisk/wangzequan
export HTTPS_PROXY=http://127.0.0.1:17890
export HTTP_PROXY=http://127.0.0.1:17890
export NO_PROXY=127.0.0.1,localhost,::1
export https_proxy="$HTTPS_PROXY" http_proxy="$HTTP_PROXY" no_proxy="$NO_PROXY"
mkdir -p "$DATADIR/flyagent/travel_uav" "$DATADIR/flyagent/diagnostics"
```

现有 OpenAI SDK 的 HTTP 客户端默认读取环境代理，无需修改模型提示词或导航逻辑。
在 tmux 中启动时，应把上述设置写进启动脚本，避免 tmux server 保留旧环境。
AirSim 使用本地 RPC；NO_PROXY 保留本机接口直连。

2026-10-16 对照诊断在
`$DATADIR/flyagent/diagnostics/20261016-api-route/`：
`unauthenticated_timing.json`、`models_timing.json`、`real_request_comparison.json`。
首页或 models 返回码只能验证该端点，不能代表 Chat Completions 的推理速度。
真实请求对照使用完全相同 payload，记录请求哈希、首事件、首输出及总耗时；
仅重放请求，不执行模型返回的工具动作。少量样本不能证明长期稳定性。

本次相同真实请求（2 张图片）的交错对照，禁用请求重试：

| 路线 | 第一次 | 第二次 |
|---|---|---|
| 远端直连 | 91.18 s 超时，未收到流事件 | 16.93 s 成功 |
| 本机 Sakuracat | 11.91 s 成功 | 17.26 s 成功 |

执行顺序为直连、代理、代理、直连。代理成功两次平均 14.58 s；
直连只有一个成功样本，不能排除超时后只比较成功均值，也不能凭四次请求确定因果。
结果支持进一步采用代理做稳定性验证，但服务端排队、推理和缓存也可能影响差异。
本次隧道由 Windows 后台 SSH PID 87408 创建；该 PID 仅为本次记录，不能作为未来固定标识。
