# AirSim 动作可靠性与计时：第一批

这一批修改属于适配器层，针对 paused_multirotor。AerialVLN 的 pose_kinematic、
UnrealCV、SSH 部署、模型配置、任务预算及工具批量语义保持现有行为。

## 动作执行与判定

- 转向显式传入 8 秒服务端超时及 5° 容差。服务端时间与客户端墙钟超时不是同一时钟；
  慢速仿真仍可能先遇到客户端超时。此值不承诺适用于所有场景性能。
- 动作 RPC 异常或返回 false 时，先尝试暂停，再取消最后任务，最后执行冻结。
  取消和冻结错误分别保留，不覆盖原始 RPC 错误。
- 取消或冻结失败后，后续物理动作拒绝执行，需重建场景/适配器；不能靠重发动作恢复。
  零速度刹车仍允许尝试冻结，但不会清除故障状态。
- 实测位移超过 max(5 米, 预期位移的 3 倍) 判为 motion_discontinuity，并要求重建场景。
  预期至少 0.1 米而实际低于 max(0.01 米, 预期的 10%)，判为无效果、低进展或碰撞阻塞。
  转向目标残差超过 5.5° 判为未到达目标；多出的 0.5° 用于控制完成后读回的余量。
- 这些是可通过 ActionVerificationPolicy 调整的粗粒度异常界限，不是动力学定律，
  不验证任务路线，也不自动修改模型参数。需要真实场景验证其误报与漏报。

## 新计时字段

action_diagnostics.timings_sec 包括 before_snapshot、resume、submit、future_wait、
action_wait、cancel、freeze、after_snapshot、total；转向还记录 target_pose。均为墙钟秒。
action_wait 包含 resume/submit/future_wait；total 包含各顶层阶段，不能把所有字段相加。
原 elapsed_sec 仍表示动作窗口等待，便于读取历史记录。simulation_elapsed_sec 单独记录
动作前后服务器时间戳差；不可将它作为网络 RTT。

total 不包括 Runtime 的额外稳定检查、采图、模型与视觉推理。第一批没有删除这些操作，
没有增加物理动作并发，也不宣称整体运行时间已下降。

## 检查入口

```powershell
python -m pytest tests/test_airsim_actions.py tests/test_airsim_uav_adapter.py -q
python tests/manual/airsim_timing_report.py --batch-root <已有批次目录> --output <新JSON路径>
```

计时报告只读历史审计；旧记录没有分段计时时明确标记缺失。
真实验证复用 airsim_action_regression_probe.py，固定场景、初始位姿和动作，保存新目录。
以正常动作、转向超时、碰撞及初始化后首次动作比较失败率和 P50/P95，保留清理记录。
脚本无需模型；本次自动测试和离线核查不能替代真实引擎验证。

## 第二批：故障传递、稳定凭据与真实对照

`PrimitiveResult.execution_state` 和 `ActionResult.execution_state` 传递控制证据。
取消失败、冻结失败和粗粒度位姿跳变设置 `requires_scene_reset`。Runtime 保留已执行
结果，将剩余同轮调用标记为 skipped，抛出 `SceneRecoveryRequired`；不再请求模型、
悬停或采图。Benchmark 将其归为基础设施中断，保留轨迹、取消最终评分，由父进程
检查所拥有的进程组、远端端口及本地隧道端口。普通碰撞和低进展仍是普通动作失败。

动作成功且冻结后的位姿读取有效时，AirSim 后端签发一次性随机凭据。凭据只匹配
本场景实例和当前 agent；下一次控制写入、读取断连、重置或关闭会使它失效。
稳定器自行检查速度有限且小于既有阈值；要求额外 hover、缺失凭据或过期凭据均
使用原流程。动作后的 RGB 和深度采集保留。开关位于
`configs/uav_stabilization_config.json` 的 `reuse_verified_post_action_state`，默认 false。
这不是远程状态缓存，也不承诺其他进程直接控制同一模拟器时仍可复用。

后端把动作诊断从旧 PrimitiveResult 的 data 容器移入 ActionResult.execution_state；
Runtime 无需识别 AirSim 私有诊断字段。模型仍只接收 data 中的位姿与失败语义。
完整诊断保存在动作审计及 `artifacts/<scene>/performance.jsonl`。

计时采用 monotonic 时钟。performance.jsonl 的每条记录对应一次动作或观察操作，
包含 SDK 调用次数、父子 span、耗时和异常类型；异步 future 等待不是第二次 RPC。
capture.decode_and_write 为包含写盘的父阶段，capture.artifact_write 是其子阶段，
报告扣除直接子阶段以得到独占耗时。Runtime 另记稳定检查和 memory artifact 写入时间。
缺失的历史计时保持缺失，不视为零。连接及引擎启动成本不与稳定态动作混合。

### 无模型实验入口

```powershell
python tests/manual/airsim_phase2_probe.py --mode case16 --repetitions 3 --root tests/manual_outputs/air2_new/case16 --review-root tests/manual_outputs/reviews/air2_new/case16
python tests/manual/airsim_phase2_probe.py --mode normal --repetitions 10 --root tests/manual_outputs/air2_new/normal --review-root tests/manual_outputs/reviews/air2_new/normal
python tests/manual/airsim_phase2_probe.py --mode timeout --repetitions 1 --root tests/manual_outputs/air2_new/timeout --review-root tests/manual_outputs/reviews/air2_new/timeout
```

SSH 使用现有 MapPackage 的 host/user/port 和环境中的 FLYAGENT_SSH_KEY，不修改 .env。
入口从既有 20 例 manifest 读取案例 13/16 身份，从原 TravelUAV 索引恢复位姿和目标；
复用 airsim_action_regression_probe.py，每组新建子进程和场景。
normal 的 repetitions 表示配对数量，10 表示关闭/开启各 10 组，严格交替；每组固定
90° 转向及 1 m/s、1 秒前进。case16 最多 3 次。timeout 只把转向截止时间改为
0.001 秒，以测试有限超时、取消、冻结和下一步边界，不改变生产策略。

探针签名新增 sequence-file、case-id、reuse-verified-post-action-state、
diagnostic-snapshots 和 yaw-timeout。诊断快照额外 RPC 只用于案例 16/超时实验，
不参与正常路径性能对照。每组异常停止后先清理；清理未证实时不会启动下一组。
原始 report.json、stdout/stderr、资源日志及监督记录始终保留。

长时间测试将 Python、JSON、提示词和紧凑索引复制到独立 source 快照，记录逐文件哈希；
不复制凭据或修改共享工作区。这样可以避免其他开发同时编辑代码而污染配对对照。
源码快照属于实验输入，并不是生产配置。每个新批次必须使用新的输出目录。

短路径出口提供 review.html、audit_review.html、原始审计和图片；限制导出路径为
240 字符。可用 review_browser_probe.py 的 `--cases 01 02` 验证代表页；浏览器探针
根据临时 profile 识别其进程，支持 Edge 启动器重启，不关闭用户的浏览器窗口。

本批结果必须分别报告案例 16 是否重现、正常对照 RPC/P50/P95/失败与残差、有限
超时边界。速度改善不代表导航成功率改善；自动测试通过也不能替代真实引擎证据。
