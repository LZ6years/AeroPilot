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

Approach is target-relative. Form a credible rough range from the available
visual relation, depth, and change after translation. Treat the estimate as a
range, not an exact point. A visible target normally spans several depth values;
use the representative target region and overall scene relation instead of
requiring all target pixels to agree. UAV displacement from a local origin is
not target distance. Yaw cannot reach a distance or side relation.

## Decision Procedure

1. Establish a usable target cue and rough range before correcting distance.
2. Apply the stop condition below before choosing another correction.
3. Approach only when available cues clearly place the UAV too far; retreat
   only when they clearly place it too near.
4. For qualitative or approximate requests, use a broad coherent range rather
   than waiting for exact measurement. Improve observation only when the cues
   cannot distinguish the needed direction.
5. After one bounded correction, re-estimate. Reverse only when new evidence
   clearly places the relation on the opposite side of the request.

## Completion And Stop

A requested standoff is sufficiently reached when a usable target cue and
available visual or depth evidence jointly place the UAV broadly near the
requested relation. Minor distance differences and normal depth spread across
the visible target are acceptable; exact equality, point containment, or one
uniform target depth is not required. Stop correcting unless cues clearly show
a substantial near/far mismatch. Instrument-level or decimal certainty is not
required. Facing is not the requested distance relation.

## Recovery

If the target is absent or uncertain, search before translating. If the
relation changes wrongly, stop and reassess. Repeated reversals mean the range
is uncertain; reacquire it. Near people or tight clearance, use smaller changes.
