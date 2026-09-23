# FlyAgent Method v2（讨论稿）

日期：2026-09-21。中文正文底稿，保留英文节标题与 LaTeX 公式。v1 保留不变。本次仅整理文稿；按最新要求暂停绘图，已有预览不纳入本稿，图示位置仅保留文字占位。

## 3. Method

FlyAgent 是一个空中语言任务的统一 Agent 框架。它不为预先列举的任务类别分别规定完整的执行流程，而是通过统一的任务表示、可复用技能与执行反馈，让同一个决策主体持续组织任务。本文借助现有 Benchmark 的不同任务检验该工作机制；评测类别描述验证范围，不定义框架的任务边界。这里的统一性也不意味着已经验证所有可能的空中任务。

给定语言指令 $I$，系统将其解析为语义计划 $G$。在第 $t$ 个决策时刻，将截至当前的事件历史 $H_{\leq t}$ 投影为模型上下文 $c_t$：

$$
G=\operatorname{Parse}(I),\qquad
c_t=\operatorname{Project}(H_{\leq t}).
\tag{1}
$$

共享多模态模型根据该上下文和可调用接口集合 $\mathcal U$ 生成决策 $y_t$。运行时执行并收集反馈 $f_t$，随后将本轮交互写入历史，供下一轮判断使用：

$$
y_t=\pi_\theta(c_t;\mathcal U),\qquad
H_{\leq t+1}
=\operatorname{Append}(H_{\leq t},y_t,f_t).
\tag{2}
$$

$\theta$ 表示任务执行中保持冻结的模型参数。$H_{\leq t}$ 已包含当前状态检查点，其中携带任务、阶段、策略、观测、运动反馈和已加载指导等信息；因此不在式（1）之外重复叠加同一份“当前状态”。系统指令和工具定义作为模型请求的其他组成部分另行提供。

式（2）中的 $f_t$ 是执行证据的概念集合，而不是代码中的单个 ToolResult 对象。只读查询可以返回新证据而不产生运动；物理执行则可能产生工具结果、运动测量和新的视觉观测。决策轮次 $t$ 因而不等于飞行步数、观测帧号或物理时间。

方法由两部分组成：3.1 说明不同任务与能力如何进入统一表示；3.2 说明历史如何形成当前上下文，以及模型如何根据实际执行结果修订后续行为。

> **Figure 2 占位：总体 Framework。** 暂不绘制或引用已生成预览。后续由用户确认布局；需呈现共享 Agent、任务与技能支持、运行循环、环境执行和反馈。不能画成按任务类别分别进入专用流程。

### 3.1 Unified Task and Skill Representation

**（1）统一任务表示**

TaskParser 将自然语言转为任务意图、最终目标与语义阶段。它统一的是任务的描述方式，不提前生成不可变的完整动作序列：

$$
G=\left(z,g,\{\phi_k\}_{k=1}^{K}\right),\qquad
\phi_k=(g_k,D_k,C_k).
\tag{3}
$$

其中，$z$ 表示任务意图，$g$ 包含目标对象、属性和要求的最终状态；$g_k$、$D_k$、$C_k$ 分别表示阶段目标、完成条件与继续条件。式（3）是语义计划的核心抽象，省略了阶段名称、索引、解析校验和返回元数据等字段。

例如，“Find the red car, approach it, then hover nearby.” 要求保留红色车辆这一目标及附近悬停这一终态。为解释结构，可以示意为寻找、接近、悬停等阶段，但实际阶段数量和名称应以一次真实解析结果为准，不能为构图强行拆分。

语义计划回答“需要完成什么”。当前激活阶段由运行时维护；阶段内策略和下一动作窗口则在执行中决定。三者不能混用：任务阶段不是局部策略步骤，局部策略也不是已经执行的运动。

**（2）共享 Skill Library 与接口分类**

FlyAgent 将可复用的指导和执行能力组织为 Skill Library。这里的“共享”不是所有任务必须加载相同的知识或调用相同的工具，而是同一套资源与接口可以被不同任务复用。本文按用途区分 Knowledge、Perception、Physical 和 Progress Control；这是一种论文表达，不等同于代码的插件目录、继承结构或 effect 枚举。

| 分类 | 当前实现示例 | 提供什么 | 是否产生物理运动 | 如何进入循环 |
|---|---|---|---|---|
| Knowledge | 阶段相关飞行指导 | 工作原则、判断依据与操作注意事项 | 否 | 根据阶段选择指导，缓存选择结果，并在上下文构建时加载对应内容 |
| Perception | visual.grounding.locate | 目标定位等证据，可包含结构化结果与图像证据 | 否 | Agent 提交只读查询，返回后再次决策 |
| Physical | uav.move_body、uav.turn_relative、uav.hover | 平移、转向或悬停，以及执行反馈 | 是，或维持悬停 | 通过有序动作窗口执行，随后收集观测与运动反馈 |
| Progress Control | runtime.commit_phase_turn、runtime.report_blocked | 决策提交、完成判断或阻塞报告 | 工具本身不产生运动 | 更新运行状态；commit 也可先于物理动作窗口 |

Knowledge 与其他三类的关键区别是：**当前知识不是 Agent 主动调用的查询工具。** 运行时依据当前阶段选择适用指导，并复用缓存的选择结果。它们是工作指导，不是环境事实；指导中写有“目标应当可见”，不能作为目标当前确实可见的证据。缓存的是阶段选择结果，不应推断指导内容在所有请求中只出现一次。

Perception、Physical 与 Progress Control 对应只读取证、物理操作与运行控制三种用途。具体代码通过工具策略声明执行影响、结果返回时机、批处理方式和是否向模型暴露，而不是仅靠工具名称判断。运行时内部还有 lifecycle 等接口，例如连接、初始化与部分飞行生命周期操作；它们不是本表三类模型决策用途之外的新方法贡献，也不能全部画成 Agent 可自由调用的工具。

知识与可调用能力共同支持任务，但进入上下文或执行器的路径不同。当前可选感知能力以视觉定位为例；“可扩展”只表示能够接入更多能力，不表示已经验证多个独立插件之间的路由效果。按需知识查询仍是后续实验设想，本稿不将其写成已有机制，也不声称已减少 token。

### 3.2 Execution-Grounded Agent Loop

**（1）从事件历史构建决策上下文**

模型需要知道此前期待发生什么，也需要知道执行后实际得到了什么。FlyAgent 因而保存按时间排列的交互事件，并在每次请求前构建模型可见视图。这里必须区分三个对象：

- **事件历史 $H_{\leq t}$**：截至当前保存的原始交互事件；顺序有意义，不是无序集合。
- **状态检查点 $S_t$**：某一时刻的任务阶段、策略、观测、运动与指导等信息，是历史中的一种事件载荷。
- **模型上下文 $c_t$**：事件历史经过投影得到的消息序列，不是原始记录的逐字段复制，也不是无人机物理状态本身。

消息流可以用如下示意理解：

~~~text
User instruction
→ State checkpoint
→ Assistant [public content + tool_calls]
→ Tool result(s)
→ Optional tool-evidence image
→ Next checkpoint / next decision
~~~

这里的 tool_call 是 assistant 消息的字段，不是第四种独立消息角色。原始指令与状态检查点均可映射为 user 消息，但 user 消息不代表全部由人现场输入。只读工具返回后可以在同一观测上继续决策，不要求每次查询都生成新物理观测。上述链条用于解释典型交互，不是固定的事件排列模板。

完整记录和投影的对应关系如下：

| 原始事件或内容 | 模型请求中的表示 | 投影时的处理 |
|---|---|---|
| user_instruction | user 消息 | 保留原指令 |
| state_checkpoint | 带检查点标记的 user 消息 | 组织状态文本；选择、去重并标识观测图像 |
| assistant_turn | assistant 消息及其 tool_calls | 保留工具调用 ID；将规范工具名转为模型接口名；仅回放模型实际提供的公开内容 |
| tool_result | 与 tool_call_id 匹配的 tool 消息 | 返回结构化结果，移除 audit_details 等不用于模型判断的诊断字段 |
| 工具返回图像 | 紧随相应结果的 user 多模态证据 | 与结果 ID 关联，独立于观测图像抽样；只附加可读取图像 |
| protocol_feedback | user 消息 | 将协议错误作为下一次修正的输入 |
| provider_turn_rejected、task_terminal | 不投影为普通请求消息 | 原始记录保留，避免回放无效调用或混入终态审计 |

Project 还会去掉已由工具 schema 表达的重复接口声明，但保留动态运行状态。对同一个 observation_id 的多次检查点，文字检查点仍按顺序回放，观测图像只在该观测最后一个检查点附加并去重；最新观测标记为 CURRENT，其他选中观测标记为 HISTORY。当前图像不可用时，不能用旧图替代并标为 CURRENT。

因此，“保留完整历史”指保存原始事件，而不是所有原始信息都进入每次请求。投影作用于独立快照，不修改持久记录；代码生成的选择诊断也不是额外输入给 Agent 的任务证据。

**观测图像选择。** 设截至当前有 $N$ 个按首次出现顺序排列的唯一观测，预算为 $B$，近期连续保留数为 $R$，旧观测槽位为 $M=B-R$。图像选择基于观测身份与序号，而不是图像内容相似度或语义重要性。若 $N\leq B$，保留全部候选观测；否则保留年龄为 $0,\ldots,R-1$ 的近期观测，并为旧观测计算候选年龄：

$$
\widetilde a_j=
\left\lfloor
\exp\left(
\log(R+1)+\frac{j}{M-1}
[\log N-\log(R+1)]
\right)-1+\frac12
\right\rfloor ,
\qquad j=0,\ldots,M-1.
\tag{4}
$$

年龄 0 表示最新观测。为避免取整后重复，并为剩余观测留下足够位置，代码进一步约束：

$$
a_j=\min\left\{\max(\widetilde a_j,p_j+1),\,N-M+j\right\},
\qquad
p_j=
\begin{cases}
R-1,&j=0,\\
a_{j-1},&j>0.
\end{cases}
\tag{5}
$$

最终选中集合按时间顺序排列：

$$
\mathcal I_N=
\begin{cases}
\{0,\ldots,N-1\},&N\leq B,\\
\operatorname{sort}\left(
\{N-1-a:\ a\in
\{0,\ldots,R-1\}\cup\{a_j\}_{j=0}^{M-1}\}
\right),&N>B.
\end{cases}
\tag{6}
$$

当前配置为 $B=20,R=3$，包括当前观测在内的最近三次观测连续保留，较早观测按对数年龄间隔选取，并保留首个观测。配置要求 $R\geq1$ 且 $B\geq R+2$，因此式（4）的分母有效。未超预算时不进行稀疏选择。

选择针对观测 ID；实际发送的图像还受文件可用性和去重规则影响。工具返回的定位证据图像另行处理，不通过这组观测抽样规则。该规则不是学习式遗忘模型，也没有在本文中据此宣称端到端速度或成本提升。

> **Figure 3 占位：Context Construction。** 只保留这一张方法细节图，后续再确定画法。需说明历史快照 $H_{\leq t}$、其中的状态检查点、Project 和最终消息上下文的区别，并展示观测选择函数与独立工具图像路径。原 v1 的策略修订图和历史图不再作为本章两张独立小图引用。

**（2）决策、执行与策略修订**

基于上述上下文，Agent 的选择按目的可概括为取证、物理行动与进度管理：

$$
y_t\in
\left\{
\operatorname{Query}(Q_t),\
\operatorname{Act}(E_t,\sigma_t,w_t,B_t),\
\operatorname{Manage}(E_t,b_t)
\right\}.
\tag{7}
$$

$Q_t$ 表示只读查询；$E_t$ 表示对阶段完成条件的证据判断；$\sigma_t$ 表示局部策略；$w_t$ 表示下一动作窗口的意图和预期；$B_t$ 表示有序物理调用；$b_t$ 表示进度管理选择。这里 $B_t$ 是动作序列，与式（4）—（6）的标量图像预算 $B$ 不同。

式（7）是论文中的目的分类，不是三个同名 API，也不意味着底层工具类型互斥地只出现一个：物理行动分支需要先提交本轮判断，再执行物理工具。只读查询与依赖其结果的动作不能提前作为同一轮无条件动作提交。

局部策略和动作窗口可写为：

$$
\sigma_t=([s_t^1,\ldots,s_t^{m_t}],q_t,\rho_t),
\qquad
w_t=(\iota_t,\widehat e_t).
\tag{8}
$$

策略包含自然语言步骤、当前步骤索引 $q_t$ 和理由 $\rho_t$；窗口说明包含意图 $\iota_t$ 与预期效果 $\widehat e_t$。当前实现支持 1–4 个策略步骤和每次 1–3 个有序物理调用。预期效果是供后续检查的自然语言描述，不是已实现的数值运动预测器。

只读取证后，返回证据进入事件历史，Agent 重新判断是否需要继续查询或执行动作。物理执行前，运行时检查工具参数、提交协议与安全约束；执行后收集工具状态、新观测与实测运动。进度管理可以推进阶段、完成任务或报告阻塞，但不能把“工具执行成功”等同于“任务目标已实现”。

阶段条件的判断可为 supported、unknown 或 contradicted。相应证据和协议用于约束阶段推进，而不是独立反思模型。实际工具返回会记录状态与数据，并可包含请求参数、执行参数、安全干预和结果引用；物理观测与状态检查点仍可单独保存。

通过式（1）—（2）的循环，之前的意图与预期、实际返回以及当前观测共同影响下一轮策略。例如，Agent 原本预期前进后红色车辆仍在视野内；若实际产生位移但车辆不可见，它可以先补充定位，再根据证据决定是否调整接近方向。证据不足时仍可能继续取证或报告阻塞，不能预设恢复成功。

本文将这种**由执行证据引起后续策略、能力使用或进度判断变化**的过程称为 Execution-Grounded Agent Loop。日志提供可追溯记录，但单纯保存日志并不是方法有效性的证明；方法也不依赖展示隐藏思维链。其作用仍需同模型、同工具条件下的受控对照和真实行为分析验证。

## 写作核对与实现依据（不直接纳入正文）

### 核心符号

| 符号 | 含义 |
|---|---|
| $I,G,\phi_k$ | 指令、语义计划、任务阶段 |
| $H_{\leq t},S_t,c_t$ | 原始事件历史、状态检查点、模型可见上下文 |
| $\pi_\theta,\mathcal U,y_t,f_t$ | 冻结决策模型、工具接口集合、决策、执行证据 |
| $N,B,R,M$ | 唯一观测数、观测预算、近期保留数、旧观测槽位数 |
| $\widetilde a_j,a_j,\mathcal I_N$ | 候选年龄、约束后年龄、时间有序选择索引 |
| $\sigma_t,w_t,\widehat e_t$ | 局部策略、动作窗口说明、预期效果 |
| $E_t,Q_t,B_t,b_t$ | 完成证据判断、查询、动作序列、进度管理选择 |

### 代码依据

| 说明 | 代码位置 |
|---|---|
| 任务解析 | [task_parser.py](../../src/agent/semantic/task_parser.py) |
| 阶段指导选择与选择缓存 | [skills.py](../../src/memory/context/modules/skills.py) |
| 工具注册与模型暴露 | [registry.py](../../src/tools/registry.py) |
| 工具执行影响、时机与批处理策略 | [schemas.py](../../src/tools/schemas.py) |
| 原始事件、公开内容和 ToolResult | [transcript.py](../../src/agent/transcript.py) |
| 事件到消息投影、过滤、图像去重 | [history_projection.py](../../src/agent/history_projection.py) |
| 对数年龄抽样函数 | [image.py](../../src/memory/context/modules/image.py) |
| 当前预算与插件配置 | [plugins.json](../../configs/plugins.json) |
| 提交、物理窗口与任务推进 | [task_runtime.py](../../src/runtime/runners/task_runtime.py) |

本稿依据静态代码核对，不代表完整任务运行已验证。Knowledge 选择缓存不等于模型上下文中不再重复出现指导，也不等于已实现 token 压缩。自身运动测量不等于对目标距离的真值。图像投影不是语义检索、状态估计器或全局地图。

### 实验应支持什么

| 主张 | 验证方式 |
|---|---|
| 同一种工作方式可用于不同任务 | 利用现有 Benchmark 分别评估；保持核心机制一致，不外推为所有任务已验证 |
| 执行结果确实影响后续行为 | 与同模型、同工具的直接决策、重规划及 ReAct-style 基线对比 |
| 意图、预期与结果的关联有作用 | 消融预期说明、执行证据或策略修订；分析真实轨迹中的行为变化 |
| 图像选择保留了有效历史信息 | 若开展实验，在相同图像预算下比较近期、均匀和当前选择规则，重点比较任务效果 |

不虚构结果，不把已有 Agent 的一般循环或日志机制宣称为首创。按需知识查询仅作为未来可选实验方向，不在当前方法、图示或工具表中充当已实现功能。

## 图示占位与后续工作

根据最新要求，本次不继续绘图，也不把本轮已生成的候选预览作为定稿嵌入。

- **Figure 1（Introduction）**：原问题图继续独立保留；不算第三章方法模块，本次不改动。
- **Figure 2（Method 总体图）**：待用户提供新的绘图想法后再设计。内容依据 3.1 和 3.2，不纳入全局地图、额外反思模型或按需知识查询。
- **Figure 3（Method 唯一细节图）**：集中说明 Context 的保存、投影、消息映射与图像选择；不再另外新增策略修订小图。

暂定英文图注供后续配图时修改：

**Figure 2.** FlyAgent uses a shared task representation and skill library within an execution-grounded agent loop. Phase-relevant knowledge supplies guidance, while callable interfaces support evidence acquisition, physical actions, and progress control. Projected history conditions the shared multimodal agent, and execution feedback informs subsequent decisions. Task examples do not define task-specific execution branches.

**Figure 3.** Projection of the event history into the decision context. State checkpoints, assistant turns, and tool results are mapped to chronological provider messages, preserving call-result associations while filtering audit-only information. Observation images are selected by a recency-plus-logarithmic-age rule and deduplicated; tool-evidence images follow a separate path. Original records remain unchanged. Any synthetic scenes in the eventual figure must be labeled as illustrative.
