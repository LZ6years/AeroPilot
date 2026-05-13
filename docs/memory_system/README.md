# FlyAgent 记忆系统文档集

本目录用于设计 FlyAgent 的无人机记忆系统。它是当前 M4/M5 的上下文扩写版：M4 已经有短期滚动执行，M5 已经有原始观测和 context builder，但这两层还不足以表达真实无人机在同一 scene 中连续执行多条指令时的历史状态、地图线索和压缩记忆。

核心结论：

- 新指令不是新世界。普通连续指令只更新 `active_instruction`。
- `FlightMemory` 持续保存当前 scene/UAV 的历史指令、图像摘要、姿态摘要和动作效果摘要。
- 用户显式 reset pose、重启 scene 或切换 map 时，才刷新飞行连续记忆。
- `MapMemory` 不因 reset pose 清空，除非切换 map 或用户显式清除。
- 模型上下文不塞完整历史，而是使用“最近原始窗口 + 滚动摘要 + map data 线索 + 规则/偏好”。
- 历史压缩和上下文编排使用可插拔模块，科研用户可以只替换某一类数据处理方式，例如只替换图片历史记忆。

## 文档目录

1. [需求文档](./01_requirements.md)
2. [记忆分层设计](./02_memory_layers.md)
3. [上下文编排](./03_context_composition.md)
4. [历史压缩策略](./04_history_compression.md)
5. [地图记忆设计](./05_map_memory.md)
6. [MVP 计划](./06_mvp_plan.md)
7. [图示汇总](./07_diagrams.md)
8. [可插拔上下文模块](./08_pluggable_context_modules.md)

## 与 M4/M5 的关系

```mermaid
flowchart LR
    M4["M4<br/>短期滚动执行器"] --> M5["M5<br/>原始观测与状态管理"]
    M5 --> MEM["Memory System<br/>连续飞行记忆与上下文压缩"]
    MEM --> M6["M6<br/>模型动作规划"]

    M4 -.已有.-> A["short-horizon actions"]
    M5 -.已有.-> B["RawObservation / latest_context"]
    MEM -.补齐.-> C["FlightMemory / MapMemory / ContextModuleRegistry / ContextComposer"]
```

M4/M5 解决的是“每轮怎么执行和观测”。记忆系统要解决的是：

- 多条指令之间如何延续当前 UAV 物理状态。
- 历史图像、姿态、动作如何压缩成模型可用信息。
- 飞过区域和地标线索如何保留但不误当事实。
- `FLYAGENT.md` 与 `USER.md` 这类规则/偏好如何进入上下文。
- 不同研究方法如何替换单类上下文处理，而不破坏飞控和安全边界。

## 第一版目标

第一版先把需求和设计写清楚，不实现完整 runtime 代码。后续实现时，应优先落地：

```text
FlightMemory JSON
ContextComposer 固定外壳
ContextModuleRegistry 注册机制
默认五类上下文模块
可替换的历史压缩策略
MapMemory schema
连续指令与 reset 语义
```
