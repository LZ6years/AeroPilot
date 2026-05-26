# Visual Instruction Eval

This checks the default image context module path: metadata plus latest real RGB `model_inputs[type=image]`.

## Summary

```json
{
  "case_count": 6,
  "passed_structural_cases": 6,
  "average_visual_grounding_score": 1.0,
  "semantic_checked_cases": 0,
  "metric_pass_counts": {
    "has_image_model_input": 6,
    "image_input_is_data_url": 6,
    "json_text_omits_base64": 6,
    "planner_saw_image_part": 6,
    "no_fake_visual_claim_when_image_missing": 6
  }
}
```

## entrance_candidate_glass_building

![entrance_candidate_glass_building](C:/Users/HP/Desktop/UAV-Agent/tests/fixtures/visual_description_images/frame_000003.png)

- instruction: 请根据当前画面判断玻璃幕墙建筑底层是否存在入口候选区域，只描述可见视觉事实。
- image_input: True
- image_part: True
- json_text_omits_base64: True
- score: 1.0

## yellow_bus_tree_landmarks

![yellow_bus_tree_landmarks](C:/Users/HP/Desktop/UAV-Agent/tests/fixtures/visual_description_images/frame_000007.png)

- instruction: 请识别画面中可作为视觉参照的黄色车辆和大树，并说明它们在画面中的相对位置。
- image_input: True
- image_part: True
- json_text_omits_base64: True
- score: 1.0

## open_foreground_paving

![open_foreground_paving](C:/Users/HP/Desktop/UAV-Agent/tests/fixtures/visual_description_images/frame_000001.png)

- instruction: 请判断画面前景铺装区域是否开阔，并指出附近会影响视觉理解的车辆或树木位置。
- image_input: True
- image_part: True
- json_text_omits_base64: True
- score: 1.0

## no_clear_entrance_sign

![no_clear_entrance_sign](C:/Users/HP/Desktop/UAV-Agent/tests/fixtures/visual_description_images/frame_000005.png)

- instruction: 请确认当前画面是否能看到明确入口标识；如果没有，只说明哪些建筑底层或车辆可作为视觉参照。
- image_input: True
- image_part: True
- json_text_omits_base64: True
- score: 1.0

## vehicles_are_not_entrance_targets

![vehicles_are_not_entrance_targets](C:/Users/HP/Desktop/UAV-Agent/tests/fixtures/visual_description_images/frame_000009.png)

- instruction: 当前阶段目标是建筑入口，请区分车辆和入口候选，不要把黄色车或白紫车辆当作入口目标。
- image_input: True
- image_part: True
- json_text_omits_base64: True
- score: 1.0

## right_side_vehicle_and_building_context

![right_side_vehicle_and_building_context](C:/Users/HP/Desktop/UAV-Agent/tests/fixtures/visual_description_images/frame_000003.png)

- instruction: 请描述右侧白色车辆、红色车辆和后方建筑底层之间的画面位置关系，用于判断入口候选是否被车辆干扰。
- image_input: True
- image_part: True
- json_text_omits_base64: True
- score: 1.0
