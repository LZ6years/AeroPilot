# MapMemory.md

MapMemory 保存当前项目中和地图、场景、区域、地标相关的可复用线索。它不是一次 run 的临时历史，也不是完整 SLAM 地图；它是无人机在同一 map / scene 中可以参考的空间记忆。

## MapMemory 的可信度

MapMemory 可以被认为是比较可信的线索，尤其是来源清楚、坐标系明确、最近被确认过的条目。

但是 MapMemory 仍然不是当前视觉事实。无人机在执行靠近、穿越、降落等高风险动作前，仍应结合当前最新 RGB / depth / pose 重新确认。

如果最新观察和 MapMemory 冲突，应优先相信最新观察，并把对应 MapMemory 条目标记为需要重新确认，或在后续更新中降低 confidence。

## 适合保存的内容

- 某个 map_id / scene_id 下曾经见过的地标。
- 某个区域可能存在的目标、入口、车辆、路灯、障碍物等线索。
- 和线索相关的 pose_hint、region_hint、coordinate_frame。
- visual_evidence_ref、depth_evidence_ref、last_seen_at。
- confidence 和 requires_reconfirmation。
- 风险说明，例如 distance_unknown、target_may_have_moved、coordinate_frame_uncertain。

## 不适合保存的内容

- 当前 run 的最新 pose。
- 当前 run 的最近 action。
- 当前图像帧的完整内容。
- 要求 planner 立刻执行的动作命令。
- 没有 map_id 或来源的空间事实。

## 使用原则

- map_id 不匹配时，不召回旧 map 的记忆。
- scene_id 不匹配时，只召回允许跨 scene 使用的线索。
- coordinate_frame 不清楚时，只能当作低风险搜索线索。
- requires_reconfirmation=true 的条目可以指导观察方向，但不能单独支持高风险动作。
- 多次最新观察都和某条 MapMemory 一致时，可以提高 confidence。
- 最新观察明确否定某条 MapMemory 时，应更新或降权该条目。
