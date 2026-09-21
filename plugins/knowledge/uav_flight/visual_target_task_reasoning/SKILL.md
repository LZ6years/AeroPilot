---
name: visual_target_task_reasoning
description: Use when a UAV task depends on finding, confirming, seeing, facing, observing, centering, or keeping attention on a visual target.
summary: Select for visual target search, confirmation, facing, centering, observation, or attention tasks.
---

# Visual Target Task Reasoning

## Application Boundary

Use for search, facing and inspection. Facing needs identity and bearing;
inspection needs detail. Alignment cannot prove route, distance or landing.

## Evidence Model

Classify evidence as target absent, weak candidate, directional cue, or inspection-ready.
A directional cue identifies the target with usable bearing despite hidden detail.
Clipping or imminent loss can invalidate bearing; incidental occlusion does not.
Motion history reveals oscillation, not bearing. Current bearing supports alignment.

Interpret images first. Supplement only decision-relevant ambiguity. Confirm
candidate identity, not confidence alone; request depth only for a metric decision.
Resolve competing identities that change the decision using another view or
independent category queries. Compare location and appearance; neither a
candidate nor a missing detection proves identity or absence.

## Decision Procedure

1. Classify the latest evidence before acting.
2. With no candidate, sweep consistently into new views. Absence alone does not
   justify reversal. Reverse only after sweeping past a candidate, reaching a
   scene or flight boundary, or exhausting new area. Use
   [search guidance](references/search_under_uncertainty.md) for scale.
3. With a weak candidate, stop broad search; resolve ambiguity while preserving the
   useful view; repeated mention is not confirmation.
4. For facing, correct only when clipping, imminent loss, or ambiguity makes
   bearing unusable. Do not correct merely for lateral offset or alternate
   around an already usable bearing.

## Completion And Stop

For facing, an identifiable forward cue with usable bearing suffices. Moderate
left or right offset is explicitly acceptable; exact centering needs an explicit
request. Do not add a confirmation hover or micro-adjustment. If vision only
enabled movement, judge the movement result, not continuous visibility.

## Recovery

If a required cue disappears after a maneuver, compare the pre-maneuver and
current views with measured motion. Try a bounded retrace or correction, then
re-observe; do not continue blind motion. Hover only to stabilize or compare.
If no safe evidence-producing recovery remains, stop and cite the limitation.
Do not restart or reverse broad search for moderate offset or incidental
occlusion alone.
