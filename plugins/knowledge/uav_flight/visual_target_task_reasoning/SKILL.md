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

Classify evidence as target absent, weak candidate, directional cue, or
inspection-ready. A directional cue identifies the target and provides usable
image bearing despite hidden detail. Clipping or imminent loss can invalidate
bearing; incidental occlusion does not. Motion history can reveal oscillation
but cannot create bearing. Current usable bearing is alignment evidence.

When RGB is ambiguous, use available grounding without distance. Confirm
candidates against the instruction rather than confidence alone; request depth
only for a separate metric decision.

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

If a required cue disappears after a maneuver, compare the pre-maneuver and
current views with measured motion. Try one bounded retrace or correction, then
re-observe; do not continue blind motion. Hover only to stabilize or compare.
If no safe evidence-producing recovery remains, stop and cite the limitation.
Do not restart or reverse broad search for moderate offset or incidental
occlusion alone.
