# UAV-Flow evaluation

UAV-Flow adapter 把外部 case 和指标映射到 FlyAgent 的 TaskRuntime/EnvClient 边界。
它不替换 Planner、Context V2、ToolSpec 或 Safety，也不向模型注入 benchmark 真值。

正式 Runtime 通过 `RuntimeFactory` 创建。每个 instruction 使用原生工具回合和物理
窗口；状态以 Transcript、checkpoint、observation/action records 和 terminal evidence
审计。外部 gym/UnrealCV 细节留在 evaluation adapter 或 simulator backend。

真实引擎手工验收与历史迁移说明不属于默认产品回归，参见
[测试分层](../../testing.md)。
