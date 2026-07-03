---
name: visual_target_task_reasoning
description: Use when a UAV task depends on finding, confirming, seeing, facing, observing, centering, or keeping attention on a visual target.
summary: Select for visual target search, confirmation, facing, centering, observation, or attention tasks.
---

# Visual Target Task Reasoning

Use this guidance when the instruction depends on finding, confirming, seeing,
facing, centering, observing, or keeping attention on a visual target. The latest
image or visual description is more important than old hints or cumulative yaw.

Classify the latest target state before choosing an action:
1. target absent: no target-like cue is visible;
2. weak candidate: something may match the target but is ambiguous;
3. visible target: the target is reasonably identifiable, even if it is a little
   left or right of center;
4. stable target: the target is visible and roughly aligned; this includes a
   possible choice to stop, and a small left/right offset alone is not enough to
   keep micro-adjusting.

For target absent or weak-candidate states, search or confirm instead of
claiming the visual objective is complete. When detailed search scale matters,
request the search-under-uncertainty reference.

For visible or stable target states, reason about whether the target appears
left, right, centered, occluded, or near the image edge, and whether the last
action improved or worsened the visual relation.

If a visible target stays in view across several small or alternating yaw
corrections, treat it as a stable target unless the latest image shows it is
clearly off-view, strongly occluded, or not identifiable.

Do not chase perfect centering. If repeated small yaw corrections keep the
target visible without clearer progress, prefer hover/confirm and treat the
visible, stable alignment as enough evidence for the current visual objective.

When recent motion history is available, use it as history only. It can warn
about oscillation or repeated small corrections, but it is not a detector and
should not override the latest visual evidence.

Image-side alignment is for visual centering or facing, not for every possible
turn. When the current intent is to face, center, observe, or keep attention on a
visible target:
- left target -> left/negative yaw, or hover if the evidence is already stable.
- right target -> right/positive yaw, or hover if the evidence is already stable.
- centered or roughly centered target -> hover, confirm, or treat the stable
  visual alignment as sufficient evidence instead of making another tiny yaw
  correction.

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

Do not invent target visibility from metadata. Do not mark a target-facing task
complete only because a yaw command was issued. This guidance is not a detector
and must not be treated as observed fact.
