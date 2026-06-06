---
name: obstacle_and_safety_awareness
description: Use when the UAV may be near people, vehicles, buildings, trees, narrow passages, occlusions, or other flight safety constraints.
---

# Obstacle And Safety Awareness

Use this guidance whenever the instruction or scene implies collision risk,
human proximity, occlusion, clutter, narrow passages, or uncertain free space.
Safety knowledge should shape action size and confidence, but it should not
override observed facts or invent obstacles.

Prefer conservative actions when obstacles or people are close, partially visible,
or likely outside the current field of view. Before moving through a narrow gap,
around a corner, or toward a person, favor observations that improve clearance
confidence. Keep altitude, direction, and distance changes modest when the visual
field is ambiguous.

For human-centered tasks, avoid aggressive approach behavior. For vehicle or
street scenes, consider that objects may occlude people or route cues. For trees,
buildings, and poles, remember that image edges can hide collision risk.

Do not assume a path is clear because no obstacle metadata exists. Do not use
safety as a reason to ignore the primary task; instead choose safer ways to gather
evidence or make progress.
