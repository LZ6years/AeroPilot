# 通用导航决策验证

`navigation_decision_probe.py` 验证当前模型、工具契约和提示在固定输入下的决策。
历史样本保留原始指令、Assistant 文字、工具结果和对应 RGB，只读取选定观察之前的记录；
不向模型提供评测坐标、参考轨迹或后续结果。历史中已加载的知识也保留原文，
因此这组实验检查新契约接续历史上下文的表现，并非一次全新任务的严格提示 A/B。

```powershell
& 'F:\conda_env\flyagent\python.exe' tests/manual/navigation_decision_probe.py prepare --output tests/manual_outputs/navigation_decision_NEW
& 'F:\conda_env\flyagent\python.exe' tests/manual/navigation_decision_probe.py run --output tests/manual_outputs/navigation_decision_NEW
```

准备阶段固定四个历史片段、十个合成相反条件及文件散列。运行最多给每个样本三次模型调用；
仅在本地执行知识读取和计划存储确认，物理、感知及完成调用均不实际执行。
合成样本使用文字明确给出的观察条件，验证条件推理，不能作为 RGB 识别正确率。
原图、实际 provider 请求、公开响应、usage、校验失败和各轮结果写入独立目录。
再次运行会跳过已有结果，不覆盖失败；改动源码、配置或证据后应准备新目录。

人工验收应检查建议是否符合输入条件，而非是否出现固定措辞或工具调用顺序。
`valid` 仅代表工具契约通过，不代表导航判断正确。未执行物理调用的探测不能证明闭环收益。

## 真实闭环与控制诊断

使用 `flight_experience_batch_probe.py` 的 `prepare` / `fly` 固定原有 5＋5 身份，
不调用 `experience`。定向子集单独保存 manifest 和完整来源清单，不能并入正式十例成绩。
源码或配置变化后中止原批次，重新冻结；保留全部旧尝试与其清理结果。

真实感知测试前记录 `sys.executable`、PyTorch / CUDA 版本及 CUDA 可用性。
下载站点异常时，可先单独验证本地缓存包含配置指定的完整权重，记录缓存版本和散列，
再在独立实验进程中显式设置 `HF_HUB_OFFLINE=1`；这不是更换模型或关闭距离校验。
不修改正式配置以掩盖网络故障，不将此前工具运行失败解释为空候选。

`airsim_action_regression_probe.py` 的 `--skip-target` 与 `--diagnostic-pose` 仅用于人工诊断。
其隐藏资产信息不进入导航模型。比较实际偏航、控制返回、碰撞时间变化及取消/冻结结果；
保留的碰撞物体名不代表当前正在接触。诊断变体不参与正式 benchmark 评分。
