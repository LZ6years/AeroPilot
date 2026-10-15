# 像素测距与任务内地点记忆

两项能力由 capabilities.pixel_depth 和 capabilities.place_memory 提供。
实验使用 configs/plugins.pixel_map.json；普通配置保留旧测距能力。
关闭旧插件不会关闭米制深度采集。坐标基于原始 CURRENT RGB，返回
DepthPerspective 的表面射线距离，不是目标中心或水平距离。

地点图读取任务起点到当前观察之间的完整日志。节点表示无人机观察
位置，合并范围为水平 10 米、垂直 5 米。save_place 单独执行，保存
插件文件，不修改 MissionState；inspect_place 返回 historical_memory，
其调用不能成为新终局证据。每点最多两张原图引用。备注与实际动作分开。

主动存档的适用时机写在 save_place/capability.json 中，该说明随工具声明
直接发送给模型。SKILL 正文用于同步文档，不依赖它被自动加载。关键地标、
路口、离开重要观察位置、有新信息的重访和需要记住的搜索结果适合保存；
普通移动与没有新信息的观察不强制保存。程序不生成语义标注。

模型每轮自动收到最新 1600×1100 地图与简短用途说明，不再自动收到地点
详情列表或存档状态 JSON。当前 RGB 和原始指令仍按原路径提供。地图显示
实际当前位置的蓝色方向箭头、绿色起点、实测路线、地点编号和既有标注的
短摘录。10 米基础网格按缩放减少密度并标明间距；局部图固定为 80×80 米。
初始朝向固定向上，不根据当前 yaw 旋转地图，不推测道路、建筑或可通行空间。

layout.py 只负责确定性布局；render.py 负责不可覆盖的版本、媒体引用及
可点击 HTML。重叠节点保留全部编号，在 HTML 中可选择对应地点。图片标签
通过通用 latest_group 合同识别累计上下文，不使用当前相机的像素边界或
当前可见性声明。像素测距只在 CURRENT 原始 RGB 中选点，不能在地图上选点。

list_places 只列模型主动存档的地点，返回完整短备注和 P-ID；不列自动记录但
未存档的节点。offset 和 limit 必填，从 0 开始，limit 为 1～20，按编号数字
升序分页，next_offset 为 null 时结束。每次直接读最新存档，不生成图片。
模型忘记编号时可先列表，再在后续轮次选择编号查看。

inspect_place 按需返回地点详情、实际动作和最多两张历史照片。save_place
立即返回保存内容及新地图；无需飞行动作即可在下一次决策看到更新。详细
状态保留在明确调用的工具结果与人类存档中。历史照片及模型备注仍属于
历史记忆；旧存档缺少时间边界时不推造存档时间。update_plan 继续由模型
自行选择使用，地图更新和节点创建均不会替模型改写计划。

首次 save_place 使地点进入目录；再次保存整体替换备注、地标和证据引用，
空照片数组保留原代表照片。照片必须属于该地点；跨地点环境证据可以用于
比较。错选照片返回无效编号、最近最多十个合法编号、合法总数，以及通过
inspect_place 查看完整选项的方法，失败不写入。回执和图重建均使用实际保存
的照片，每个选定 observation 对应同一张原始 RGB。

inspect_place 返回原始 evidence_refs、全部可选照片 observation ID，以及
照片采集时的初始地图相对位置（右、前、上，米）与无人机相对初始航向
（弧度），不将无人机航向当作相机光轴。旧数据仅从同一历史 observation
恢复，缺失则为 null。已知编号的未存档节点仍可查看，并标记未存档。

用途和参数说明通过 capability.json 的 Provider 工具声明进入模型；SKILL.md
同步作开发文档，不自动全文注入。save_place 仍为独占 runtime_control，
列表和查看为 parallel_safe 只读工具。默认 max_workers=1，同轮独立只读
调用顺序执行，依赖前一结果的调用分轮。相同参数允许同轮或跨轮重新读取；
调用 ID 仍须唯一。现有每观察额度按实际完成的调用次数计数（含失败），
校验拒绝及未启动调用不计数，地点列表和查看不设置额度。

MediaRef.latest_group 保留最新累积图；历史引用和图片版本不可覆盖。
地图不替代 CURRENT RGB，不含隐藏目标。记忆按 run_dir/instruction_id
隔离，重试不继承失败尝试的地图。

当前20例复测使用原十例加锁定困难十例，保留 optional 计划及 v3 协议。
运行和验收入口见 [20例复测说明](../../tests/manual/aerial_vln_plan_context_probe.md)。
定向接口准入与正式导航自主调用分别统计；模型接口错误保留终态和缺失评分。

## 地图展示与查询

当前地图绘制版本为 `simple-place-map-v1.2`：去掉图片顶部大标题和长说明，
收回留白；保留地点编号、短标签、轨迹、当前位置、航向、比例尺和局部图。
编号表示无人机观察位置，不是建筑物坐标。旧图片及引用保持原样。
静态模块说明和工具 description 教模型何时查询：当前上下文和短标签不足以
辨认旧地点时，用 inspect_place 取回完整备注、旧照片和历史姿态；忘记已存档
编号时才先 list_places。已有信息足够时无需查询，不以调用次数衡量效果。

困难十例并行复测说明见[复测文档](../../tests/manual/aerial_vln_plan_context_probe.md)的“困难十例并行复测”部分。
逐例诊断页按回合完整展示公开意图，区分模型原文、参数生成说明与未捕获响应。

## 历史四例预览

以下保留2026-10-09/10的预览流程及当时配置，不是当前20例复测的运行设置。
原 100 条目录和原四条主动存档预览作为只读对照，不由本轮恢复或继续。
review.html 保留完整过程并嵌入地点存档。comparison.html / report.html
采用有效评分分母，另列 completed 子集、基础设施错误及工具调用次数。
PNG 哈希包含绘制版本，原图不覆盖；每个版本保存图数据快照供审计。

主动存档使用独立四条预览队列，不会派发第五条。固定从原 100 条清单
选择首条低难度、前两条高难度和首条中难度，按低、高、中、高执行。

```powershell
& 'F:\conda_env\flyagent\python.exe' -X utf8 tests/manual/aerial_vln_pixel_map_campaign.py redraw --source H:\FlyAgent-Eval\avln_pixel_map_save_preview_4_20261009 --output H:\FlyAgent-Eval\avln_simple_place_map_preview_4_20261010
& 'F:\conda_env\flyagent\python.exe' -X utf8 tests/manual/aerial_vln_pixel_map_campaign.py prepare-preview --output H:\FlyAgent-Eval\avln_simple_place_map_preview_4_20261010
& 'F:\conda_env\flyagent\python.exe' -X utf8 tests/manual/aerial_vln_pixel_map_campaign.py calibrate --output H:\FlyAgent-Eval\avln_simple_place_map_preview_4_20261010
& 'F:\conda_env\flyagent\python.exe' -X utf8 tests/manual/aerial_vln_pixel_map_campaign.py run --output H:\FlyAgent-Eval\avln_simple_place_map_preview_4_20261010 --workers 2
```

离线重绘只读原图快照，在 offline_preview/index.html 汇总每条任务起始、
中段、最终地图；原始照片使用引用，不复制。其图片从未用于旧模型决策。
新版预览生成 report.html、save_place_check.html/json、完整逐条 review.html
及地图存档。冻结只复制明确的功能文件和插件，不夹带无关工作区修改。
至少两条任务出现模型自主有效存档且原图与记录对应才通过引用核验；
语义是否对应真实场景仍需查看原图。未通过时不自动补写标注，SR 不作为
技术门槛。按用户最新要求，预览直接运行控制器，不开启守护或 Windows
自动重启任务。仅运行四条；GPU 0/2、本地 35001/35002、远端 30001/30002。
GPU 1 禁用。四条结束后控制器退出，不启动剩余 100 条，旧版结果不合并。

真实同批采集确认 AirSim 原始颜色为 BGR：交换红蓝后与 PNG 完全一致。
本轮同时修正 raw BGR/BGRA 解码；旧 333 条图像与评分保持原样，对照
报告应把颜色修正列为组合变化，不能把增益全部归因于像素测距或地图。

### 历史照片的实际送达

`inspect_place` 返回照片引用不等于图片已经进入下一次模型请求。图片投影保留
原观察与最新工具取回两种候选来源：起点、当前、最近观察及最新地图先占位，
其余取回照片按最新工具结果参与既有证据预算；同一图片只发送一次，并在所选
来源对应的事件位置发送。总预算20张和工具证据上限4张不变。

2026-10-12困难十例结束后发现旧照片的首次观察来源掩盖了重新查询的优先级，
本项补修在正式导航后完成。原十例仍使用冻结旧投影；实际请求与补修离线重建
必须分开审计，不能把离线送达成功当成正式导航效果。
