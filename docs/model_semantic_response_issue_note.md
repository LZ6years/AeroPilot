# Model 语义回复问题记录

本文记录 2026-05-19 真实 UE + Qwen 测试中暴露的问题。当前先记录现象和证据，不立即把问题硬编码成规则补丁；后续再设计更稳定的下层语义契约和执行闭环。

## 背景

当前链路已经可以跑通：

```text
自然语言指令
-> TaskParser(model)
-> PlannerContext
-> ModelActionPlanner
-> ToolContractValidator
-> SafetySupervisor
-> ToolDispatcher / AirSim UE
-> runtime automatic observation
-> post-action phase judge
```

最近新增的 UAV stabilization 也已经接入：每个真实飞行动作后，runtime 会等待速度低于阈值，再进行下一步动作或最终 observation。稳定等待不是模型动作，也不会写入 memory history。

本次问题不是工具执行失败，也不是安全层拒绝，而是模型对复合自然语言指令的语义阶段理解出现偏差。

## 复现 Run

真实 model + UE 测试目录：

```text
runs/model_ue_stabilization_v02_validation_20260519_101617_9dbfa0
```

本轮测试指令包括：

```text
向前飞行5米，然后悬停
向右转40度，然后悬停
向前飞行5m后向右转40度，最后再向前走2m
```

离线 evaluator 结果显示基础链路是通过的：

```text
completed: true
action success rate: 1.0
forbidden tools: 0
rgb artifacts: 12
depth artifacts: 12
```

因此不能简单归因于 UE 未启动、ToolDispatcher 失败、动作 contract 非法或 artifact 未采集。

## 具体问题

第三条指令原文含义应该是：

```text
先向前飞行 5m
再向右转 40 度
最后再向前走 2m
```

但是模型语义解析/阶段生成后的最后阶段变成了：

```json
{
  "name": "adjust_altitude_down",
  "meaning": "descend vertically by 2 meters",
  "completion_criteria": [
    {
      "kind": "translation",
      "axis": "down",
      "target": 2.0,
      "unit": "m",
      "tolerance": 1.0,
      "correction_allowed": true
    }
  ]
}
```

对应的 `semantic_goal` 也被写成了：

```json
{
  "target": {
    "description": "move forward 5 meters, then rotate right 40 degrees, then descend 2 meters",
    "attributes": {}
  },
  "final_state": "descended_position"
}
```

也就是说，模型把“最后再向前走 2m”错误理解为了“下降 2m”。

## 运行层看到的结果

第三条任务实际执行的动作序列中，前两段基本符合预期：

```text
move_forward: uav.move_body {"direction": "forward", "duration": 2.5}
move_forward: uav.move_body {"direction": "forward", "duration": 1.25}
rotate_right_40deg: uav.turn_relative {"yaw_deg": 30.0}
rotate_right_40deg: uav.turn_relative {"yaw_deg": 10.0}
```

但是最后阶段执行成了：

```text
adjust_altitude_down: uav.move_body {"direction": "down", "duration": 1.0}
```

post-action phase judge 随后根据错误的 `completion_criteria.axis = down` 判断该阶段完成。这说明 phase judge 的行为是按输入标准执行的，问题源头更靠前：`TaskParser(model)` 或语义阶段生成。

## 当前判断

这不是 `ToolContractValidator` 的职责范围。`ToolContractValidator` 只能判断：

```text
工具是否允许
参数格式是否合法
参数是否超限
observe 是否非法出现
phase 字段是否匹配当前阶段
```

它无法判断自然语言里的“向前走 2m”是否被模型误解析成“下降 2m”。

这也不是 `SafetySupervisor` 的职责范围。Safety 只能做：

```text
限幅
安全替换
累计 yaw scan 预算
hover 等安全兜底
```

它不应该根据用户原始语义去重写模型阶段。

## 与 Stabilization 的关系

本轮还观察到：`velocity_threshold_mps = 0.2` 对长距离 `uav.move_body` 比较严格，长 forward 动作可能 timeout，6 秒后仍有约 `0.76-0.85 m/s` 的残余速度。

但是这和“最后 forward 被解析成 down”是两个不同问题：

```text
stabilization 问题：动作后速度是否足够稳定
semantic response 问题：模型是否理解并生成了正确阶段
```

前者影响 observation 和位移测量质量；后者直接改变任务目标。当前记录的主要问题是后者。

## 暂不立即做的事

当前先不做以下硬规则：

```text
不硬编码“再向前走”必须转成 forward
不写针对这一个中文短语的特殊修复
不让 SafetySupervisor 判断自然语言语义
不让 ToolContractValidator 做语义纠错
不新增固定规则 planner 覆盖模型输出
```

原因是这些都会让下层越来越像一堆特殊 case，短期能修一个例子，长期会破坏系统泛化能力。

## 后续可能方向

后续需要思考的是做一个更稳定的“下层语义契约”，而不是只修 prompt。

可能方向：

1. 让 `TaskParser` 输出更结构化的 motion intent。

   例如每个 phase 除了自然语言 `meaning`，还必须有稳定的执行语义：

   ```json
   {
     "motion_intent": {
       "type": "translation",
       "axis": "forward",
       "distance_m": 2.0,
       "reference_frame": "current_body"
     }
   }
   ```

2. 对 `completion_criteria` 做语义一致性校验。

   例如 `meaning` 里出现 forward，但 `completion_criteria.axis = down` 时，标记为 parser inconsistency，进入 repair，而不是直接执行。

3. 给 parser 增加独立 repair。

   当前 repair 主要修 action contract。后续可以考虑 `SemanticPlanRepair`，只修 parser 输出，不碰 UAV tool。

4. 将中文运动词汇规范化为稳定中间语义。

   例如：

   ```text
   向前 / 往前 / 再向前走 / 继续往前飞 -> axis=forward
   上升 / 向上飞 -> axis=up
   下降 / 向下飞 -> axis=down
   左移 / 向左平移 -> axis=left
   右移 / 向右平移 -> axis=right
   ```

   这个可以作为 parser contract 的解释和校验，不一定要做成手写 planner。

5. 在 evaluator 中增加 parser-level failure 分类。

   当前 evaluator 更关注执行结果。后续应区分：

   ```text
   parser_semantic_mismatch
   planner_action_mismatch
   runtime_execution_error
   safety_intervention
   phase_judge_error
   ```

这样才能看清楚失败到底来自模型语义、动作规划、飞控执行还是阶段判断。

## 当前推荐下一步

先保留这份记录。下一步不要急着修成特殊规则，而是先设计 `SemanticPhase` / `completion_criteria` 的稳定契约：

```text
用户自然语言
-> 可解释的语义阶段
-> 稳定 motion intent
-> completion criteria
-> action planner 根据剩余进度补动作
```

等这个契约稳定后，再让 model 只在契约范围内表达语义，减少“向前 2m 被理解成下降 2m”这类错误。
