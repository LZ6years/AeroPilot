# TravelUAV 停止、历史测距与报告修复记录

本次修改面向新任务。原图片预算实验、冻结源码、原始轨迹和评分均保留，
不将离线回放或新提示词的结果混入旧实验。

## 已实施

- TravelUAV 协议标识升级为 `front_view.no_assistant.v3`：要求距离不超过
  20 m，并说明三维直线距离包含高度差。像素表面深度不等于无人机到目标
  参考位置的距离；判断需考虑相机位置、采样表面偏移和不确定性。
- 周边描述用于目标身份核验，可结合不同时间的观察；最终图像不要求同时
  展示全部背景。明确的最终空间关系与路线要求仍须满足。
- 像素测距仍使用 `observation_id/pixel_x/pixel_y`。允许读取当前任务完整
  持久化历史中的配对 RGB/深度，输出来源时间与 CURRENT/HISTORY 标记。
  `ToolRuntimePolicy.observation_id_parameter` 将调用限额绑定到来源观察，
  不因当前画面变化而重置。没有声明该参数的工具维持原计数方式。
- RPC 成败与运动连续性分别验证。异常位移不能因控制器失败而跳过检查，
  也不能因出现新碰撞而自动恢复；要求重建场景并阻止后续动作。原 RPC、
  碰撞和测量诊断保留。移除了将大位移直接视为可恢复碰撞的分支。
- 可移植报告将地图、地点卡片、原始照片与动作存档内嵌到案例页。地图热点
  跳转和动作展开使用本页锚点与原生 details，不依赖外部 iframe 或服务。

## 检查结果

### 代码与模型回放

全量默认测试：**1684 passed，1 failed**。剩余失败是
`tests/test_boundaries.py::test_default_tests_do_not_import_manual_scripts`，
来自现有 `test_blocked_success_audit.py`、`test_gpu_display_context.py`、
`test_travel_dataset_handoff.py` 对手动脚本的导入；这些文件未在本次修改。
报告的公共导出逻辑已放入 evaluation 层，避免新增这种依赖。

新增覆盖历史 61 帧、跨任务拒绝、配对帧不匹配、深度缺失、按来源计数、
跨观察限额、控制器失败加大位移、碰撞加大位移以及普通小碰撞。
独立诊断测试覆盖 SDK 部分接口不可用时仍能保存其他位姿信息。

模型请求明确标记为“从持久化 transcript 重建的离线请求”，未执行工具：

- 第一轮三例回放中，蓝车案例改为继续核验位置；历史测距案例选择转向恢复
  视野。黑车案例仍误把 18.6 m 表面距离估计为达标。
- 在补充参考位置几何与不确定性要求后，同一黑车输入选择进一步接近，
  不再直接完成。此单次结果不能证明总体 SR 改善。

原截图案例的历史 RGB 仍可读取，但配对深度 NPY 已被旧实验留存策略删除。
对此返回 `metric_depth_unavailable`；不会使用其他帧补值或伪造历史距离。
新任务保存有配对深度时可正常测量，确定性测试已验证。

### 报告与浏览器

新版报告：75 个案例，6375 张内嵌过程图、地图和存档照片。
原 75 份过程报告的 SHA256 均与此前导出清单一致。

Browser 插件不可用，使用已安装 Playwright 与 Edge。检查桌面 1440×1000、
移动端 430×932，并将黑车、摩托车、帐篷三个案例 HTML 单独复制到其他目录：

| 检查 | 结果 |
| --- | --- |
| 页面标题、索引 75 个链接、非空内容 | 通过 |
| 地图展开、热点跳转至地点 | 通过 |
| 原始照片显示、动作明细展开 | 通过 |
| 缺失图片、外部图片依赖、残留 iframe | 均为 0 |
| 页面异常与控制台错误 | 均为 0 |

## 异常位移：已证实与未证实

对全部 300 条 TravelUAV 轨迹的筛查发现 11 个转向窗口的三维位移超过
50 m，10/20/30 张预算分别 5/2/4 例。50 m 仅是离线筛查阈值，运行时仍用
已有运动验证边界。用保留的前后状态和 RPC 回执回放新执行器后，11 例均
标记 `motion_discontinuity`，要求重建场景，且原始 RPC 信息完全保留。

运行了 11 次不调用 LLM 的真实场景探针：三个场景的完整独立位姿采样、
两个异常场景各三次最小前置采样，以及两个异常场景按生产初始化顺序并
暂停 25 秒后的转向。远端生成目录显式置于
`/home/Newdisk/wangzequan/diagnostics/travel_control_state_20261018`。

其中 NYC 桥上黑车场景捕获一次 6.58 m 的垂直下坠：状态估计、物理状态、
渲染位姿和相机位姿一致，控制器返回 false 并出现道路碰撞。新校验已将其
拦为需重建场景的异常。这一例属于真实物理位置变化，不能解释为图像或
坐标日志单独跳动。

救护车与帐篷的原始 105/1698 m 水平跳变尚未复现，因此没有擅自修改
初始化/冻结顺序、位姿来源或切换物理引擎。旧轨迹缺少独立渲染/相机
位姿，仍不足以最终区分物理跳变与状态同步错误。

远端二进制字符串存在 `PhysicsEngineName`、`FastPhysicsEngine`、
`ExternalPhysicsEngine`，不存在配置中拼写为 `PhysiceEngineName` 的键。
这支持该键拼写存在问题，但不足以单独解释间歇性大位移；未直接将其改成
ExternalPhysicsEngine。后者与本项目原生动力动作的控制契约不同。

所有探针完成清理；独立 SSH 检查确认 39641 端口不再监听，诊断目录对应
的进程不存在。源码、凭据、已有运行和其他服务未清理或迁移。

## 产物与复查

- [可移植失败报告索引](G:/FlyAgent-Eval/image_budget_200_20261013_v2/analysis/travel30_failures_portable_v2_20261018/index.html)
- [11 条历史动作回放](G:/FlyAgent-Eval/image_budget_200_20261013_v2/analysis/travel_control_state_fix_20261018/historical_motion_replay.json)
- [真实探针汇总](G:/FlyAgent-Eval/image_budget_200_20261013_v2/analysis/travel_control_state_fix_20261018/live_probe_summary.json)
- [浏览器检查](G:/FlyAgent-Eval/image_budget_200_20261013_v2/analysis/travel_control_state_fix_20261018/report_qa.json)
- [黑车边界的新协议回放](G:/FlyAgent-Eval/image_budget_200_20261013_v2/analysis/travel_control_state_fix_20261018/model_replay_boundary_v3/results.json)

全部本地 Python 命令使用 `F:\conda_env\flyagent\python.exe`。
探针记录了该解释器、PyTorch 2.6.0+cu126、CUDA 12.6 与 CUDA 可用状态；
探针使用仿真渲染而非感知模型推理，不将其描述为感知 GPU 性能测量。
