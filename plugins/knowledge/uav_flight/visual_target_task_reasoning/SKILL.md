---
name: visual_target_task_reasoning
description: Use when a UAV task depends on finding, confirming, seeing, facing, observing, centering, or keeping attention on a visual target.
summary: Select for visual target search, confirmation, facing, centering, observation, or attention tasks.
---

# Visual Target Task Reasoning

## Application Boundary

Use for finding, confirming, facing, inspecting, or attending to a visual
target. Facing needs identity and usable bearing; inspection may require
requested detail. Visual alignment alone cannot prove route, distance, orbit,
or landing progress.

## Evidence Model

Classify current evidence as target absent, weak candidate, directional target
cue, or inspection-ready. A directional cue identifies the target and gives
usable image bearing despite hidden nonessential detail. Clipping or imminent loss
from view can invalidate bearing; incidental occlusion does not. Motion history
can reveal oscillation but cannot create target-relative bearing. Usable current
image bearing is operational evidence for visual alignment.

## Decision Procedure

1. Classify the latest evidence before acting.
2. With no candidate, sweep consistently into new views. Absence alone does not
   justify reversal. Reverse only after sweeping past a candidate, reaching a
   scene or flight boundary, or exhausting new area. Use
   [search guidance](references/search_under_uncertainty.md) for scale.
3. With a weak candidate, stop broad search and hover or make one small
   same-side confirmation.
4. For facing, correct only when clipping, imminent loss, or ambiguity makes
   bearing unusable. Do not correct merely for lateral offset or alternate
   around an already usable bearing.

## Completion And Stop

For facing, completion requires only an identifiable cue with usable bearing
in the forward view. Moderate left or right offset is explicitly acceptable;
exact centering is not required unless explicitly requested. Do not add a confirmation hover or continue
micro-adjustment after this state. If vision only enabled movement, use the
movement objective's completion evidence instead of continuous visibility.

## Recovery

If identity or bearing is ambiguous, hover or use one bounded correction.
Re-check yaw that conflicts with the stated image side. Do not restart or
reverse broad search for one empty view, moderate offset, or incidental
occlusion after a usable cue.
