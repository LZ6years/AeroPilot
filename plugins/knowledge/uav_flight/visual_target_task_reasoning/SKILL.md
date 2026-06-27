---
name: visual_target_task_reasoning
description: Use when a UAV task depends on seeing, facing, observing, centering, or keeping attention on a visual target.
---

# Visual Target Task Reasoning

Use this guidance when the instruction depends on seeing, facing, centering, or
keeping attention on a visual target. Treat these tasks as evidence-driven: the
latest image or visual description is more important than raw paths, old map
hints, or a single cumulative yaw value.

Before choosing a visual-target action, check:
1. whether the target is visible in the latest observation;
2. whether it appears left, right, centered, occluded, or near the image edge;
3. whether the last action improved or worsened the target relation;
4. whether any available recent motion history suggests repeated or alternating
   corrections without clearer visual progress.

If the target is not visible, switch to search logic instead of claiming the
target-facing task is complete.

Image-side alignment is for visual centering or facing, not for every possible
turn. When the current intent is to face, center, or keep attention on a visible
target:
- left target -> left/negative yaw, or hover if the evidence is already stable.
- right target -> right/positive yaw, or hover if the evidence is already stable.
- centered or roughly centered target -> hover, confirm, or let the current
  phase decision advance instead of making another tiny yaw correction.

Self-check before yaw: if the user-facing message or action reason says the
target is on the left, the yaw choice for centering should also be
left/negative. If it says the target is on the right, the yaw choice for
centering should be right/positive. If the chosen yaw goes the opposite way,
first check whether the intent is really broad search, checking another side,
avoidance, or a camera-frame exception. If it is not, rewrite the plan as
hover/confirm or a same-side small yaw.

Do not use current cumulative yaw as if it were the target's image-side offset.
The latest image side is the primary cue for visual centering; cumulative yaw
can affect correction size or the choice to stop, but it should not invert the
image-side action for a visible target.

Do not chase perfect centering. If repeated small yaw corrections have already
kept the target visible without making the relation meaningfully clearer, prefer
hover/confirm and let the current phase decision decide whether the
visual-target phase is good enough to advance.

When recent motion history is available, use it as history only. It can warn
that the UAV is oscillating or repeatedly applying the same small correction,
but it is not a detector and should not override the latest visual evidence.
If the target remains visible during repeated left/right micro-corrections,
avoid another mechanical micro-yaw unless the latest image shows a clear
remaining alignment problem.

Do not invent target visibility from metadata. Do not mark a target-facing task
complete only because a yaw command was issued. This guidance is not a detector
and must not be treated as observed fact.
