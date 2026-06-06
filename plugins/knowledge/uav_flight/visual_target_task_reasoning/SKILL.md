---
name: visual_target_task_reasoning
description: Use when a UAV task depends on seeing, facing, observing, centering, or keeping attention on a visual target.
---

# Visual Target Task Reasoning

Use this guidance when the instruction is target-relative and the target should
be interpreted from current visual evidence. Common examples include facing a
person, looking toward a vehicle, observing a landmark, keeping a target in view,
or deciding whether a turn made the target better aligned.

Treat visual-target tasks as evidence-driven tasks. The image, visual
description, recent observation, and recent action history are more important
than raw file paths or a single yaw value. A turn is useful only if it improves
the target-facing relation or creates a better observation.

Prefer small corrective actions when alignment is uncertain. If the target is
visible, reason about whether it is left, right, centered, partially occluded, or
too close to the image edge. If the target is not visible, switch to search logic
instead of claiming the task is complete.

Do not invent target visibility from metadata. Do not mark a target-facing task
complete only because a yaw command was issued. Do not use benchmark-only
coordinates, hidden target positions, or reference paths as planner evidence.

Visual-target tasks usually need three checks: whether the target is actually
visible or described in the latest observation; where it appears in the image if
visible; and whether the last action improved or worsened the target relation.
This guidance is not a detector and must not be treated as observed fact.
