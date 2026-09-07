---
name: approach_and_standoff
description: Use when the UAV should finish near, farther from, following, or holding a requested distance or side position relative to a destination target; not when the endpoint is through, past, beyond, across, or around a route reference.
summary: Select only when target-relative proximity, separation, following, or a final held front/back/side position is the terminal concern; exclude route traversal.
---

# Approach And Standoff

## Application Boundary

Use for closeness, separation, following, or a final target-relative stop. A
side relation belongs here only when the UAV must stop or hold there. Exclude
routes ending through, past, beyond, across, or around a reference; preserving
target visibility can conflict with traversal.

## Evidence Model

For qualitative requests, form a credible rough
range from the visible relation, fused depth colors, and change after
translation. For an explicit meter target or threshold, use an available
metric-depth sampler on a tight representative interior region before acting
or claiming completion. Its result is camera-to-visible-surface distance, not
target-center or horizontal distance. UAV displacement from a local origin is
not target distance. Yaw cannot reach a distance or side relation.

## Decision Procedure

1. Establish a usable target cue and rough range before correcting distance.
2. Apply the stop condition below before choosing another correction.
3. Approach only when available cues clearly place the UAV too far; retreat
   only when they clearly place it too near.
4. For qualitative or approximate requests, use a broad coherent range rather
   than waiting for exact measurement. For explicit metric requests, sample a
   small interior ROI and keep the state unknown if it cannot be isolated.
5. After one bounded correction, re-estimate; repeat metric sampling on the new
   observation when the requested relation is numeric. Reverse only when new evidence
   clearly places the relation on the opposite side of the request.

## Completion And Stop

A qualitative standoff is reached when a usable target cue and coherent visual
or depth evidence place the UAV broadly near the requested relation. An explicit
metric standoff additionally requires a fresh small-ROI measurement consistent
with its stated tolerance or threshold. Do not infer a decimal distance from
the color bar. Facing is not the requested distance relation.

## Recovery

If the relation disappears, compare earlier and current views with measured
motion. Try one bounded retrace, then re-observe; do not continue blindly.
Hover only to stabilize. If no safe recovery remains, stop and cite it. Use
smaller changes near people or tight clearance.
