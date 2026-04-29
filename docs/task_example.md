# 示例任务闭环：右转面向黑色路灯后停下

## 1. 用户指令

```text
前面右转会有一个黑色的路灯，右转面向路灯后停下
```

## 2. 任务解释

该指令包含四个阶段：

1. 向右转。
2. 在右转后的视野中搜索黑色路灯。
3. 调整 UAV 朝向，使其面向路灯。
4. 悬停停止。

## 3. 结构化任务

```json
{
  "intent": "navigate_to_visual_target",
  "target": {
    "description": "black street lamp",
    "color": "black",
    "class": "street_lamp"
  },
  "steps": [
    {
      "type": "turn",
      "direction": "right",
      "angle_deg": 90
    },
    {
      "type": "search_target",
      "description": "black street lamp"
    },
    {
      "type": "face_target",
      "description": "black street lamp"
    },
    {
      "type": "hover"
    }
  ],
  "safety": {
    "max_speed_mps": 1.5,
    "min_depth_m": 2.0,
    "timeout_sec": 60
  }
}
```

## 4. 执行流程

### 4.1 初始化

工具调用：

```text
uav.connect
uav.takeoff
uav.get_pose
uav.get_frames(types=["Scene", "DepthPlanar", "DepthVis"])
```

期望结果：

- UAV 可控。
- UAV 已在安全高度。
- 获取当前画面、深度和姿态。

### 4.2 右转

工具调用：

```text
uav.turn_relative(angle_deg=90)
uav.hover
uav.get_pose
```

注意：

- 如果没有现成 `turn_relative`，可通过当前 yaw + `rotateToYawAsync` 包装。
- 右转完成后必须短暂停稳，再开始识别。

### 4.3 搜索黑色路灯

工具调用：

```text
uav.get_frames(types=["Scene", "DepthPlanar", "DepthVis"])
vision.detect(description="black street lamp")
```

如果没有找到：

```text
uav.turn_relative(angle_deg=15)
uav.hover
vision.detect(description="black street lamp")
```

建议最多扫描：

- 左右各 45 度。
- 或累计 6 次，每次 15 度。

失败策略：

- 找不到目标时悬停。
- 返回 `TARGET_NOT_FOUND`。
- 不继续向前飞。

### 4.4 面向目标

输入：

- 目标 bbox。
- 当前图像宽高。
- 当前 UAV yaw。
- 当前相机 pitch。

计算：

- bbox 中心点。
- bbox 中心相对图像中心的水平偏移。
- 将水平偏移转换为 yaw 修正角。
- 将垂直偏移转换为相机 pitch 修正。

工具调用：

```text
uav.turn_relative(angle_deg=small_yaw_delta)
uav.rotate_camera(pitch_rate=small_pitch_rate, duration=0.1)
uav.hover
```

完成条件：

- bbox 中心距离画面中心小于宽高的 5%。
- 深度距离大于安全阈值。
- 连续 2 帧目标检测一致。

### 4.5 停下

工具调用：

```text
uav.hover
uav.get_pose
uav.get_frames(types=["Scene"])
```

输出：

```json
{
  "status": "completed",
  "message": "已右转并面向黑色路灯悬停",
  "final_phase": "hovering"
}
```

## 5. 伪代码

```python
def run_lamp_right_turn_task(agent):
    agent.call("uav.connect")
    agent.call("uav.takeoff")

    agent.call("uav.turn_relative", angle_deg=90)
    agent.call("uav.hover")

    target = None
    for _ in range(6):
        frames = agent.call("uav.get_frames", types=["Scene", "DepthPlanar", "DepthVis"])
        target = agent.call("vision.detect", description="black street lamp", frame=frames["Scene"])
        if target.ok and target.payload.get("bbox"):
            break
        agent.call("uav.turn_relative", angle_deg=15)
        agent.call("uav.hover")

    if not target or not target.payload.get("bbox"):
        agent.call("uav.hover")
        return {"status": "failed", "reason": "TARGET_NOT_FOUND"}

    for _ in range(10):
        frames = agent.call("uav.get_frames", types=["Scene", "DepthPlanar"])
        bbox = agent.call("vision.detect", description="black street lamp", frame=frames["Scene"]).payload["bbox"]
        offset = agent.call("vision.estimate_target_offset", bbox=bbox)
        distance = agent.call("depth.estimate_bbox_distance", bbox=bbox, depth=frames["DepthPlanar"])

        if distance.payload["meters"] < 2.0:
            agent.call("uav.hover")
            return {"status": "completed", "reason": "SAFE_DISTANCE_REACHED"}

        if abs(offset.payload["yaw_deg"]) < 3:
            agent.call("uav.hover")
            return {"status": "completed", "reason": "TARGET_CENTERED"}

        agent.call("uav.turn_relative", angle_deg=offset.payload["yaw_deg"])
        agent.call("uav.hover")

    agent.call("uav.hover")
    return {"status": "completed", "reason": "MAX_ALIGNMENT_STEPS_REACHED"}
```

## 6. 关键安全点

- 不允许目标未确认时向前快速飞。
- 不允许深度距离过近时继续靠近。
- 目标丢失后必须悬停并重新扫描。
- 工具调用超时必须悬停。
- 用户说“停下”时，停下应解释为 `hover`，不是 `land`。

