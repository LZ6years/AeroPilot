---
name: route_traversal_and_passage
description: Use when the UAV should pass through, go through, navigate through, progress past, cross between objects, or follow an explicit passage, gap, corridor, opening, or route cue.
summary: Select for through, past, between, passage, gap, corridor, opening, or explicit route-cue traversal tasks where route progress is the active concern.
---

# Route Traversal And Passage

## Application Boundary

Use this guidance to pass through, past, between, beyond, or along one side of a
passage, gap, corridor, or cue. Do not treat the cue as a destination
when route progress is the concern. Use approach reasoning when the task ends
at a target-relative position.

## Evidence Model

Combine meaningful translation with visual change. A cue that began ahead may
move sideways, behind, outside the blocking path, or out of view. Once its route
relation is established, consistent translation remains evidence while it is
not visible. Cue absence alone is ambiguous: it neither proves completion nor
requires renewed search. The requested side describes the UAV route relative to
the cue, not a final image side. Yaw can inspect a route or change image bearing,
but cannot place the UAV on a physical side or prove traversal.

## Decision Procedure

1. Establish the route, requested side, or gap and usable free space.
2. Prefer bounded translation through or past it over repeated centering.
3. Compare the first useful route view with the latest view and accumulated
   translation; use their agreement as route evidence.
4. Reacquire only when identity, relation, or movement evidence is insufficient
   or contradictory.

Request [route progress examples](references/route_progress_examples.md) when a
complex side passage or disappearing cue needs more detailed interpretation.

## Completion And Stop

Passage completion is a terminal route relation supported by agreeing evidence.
Strong evidence includes meaningful translation while a cue moves from ahead
toward a side, rearward, or beyond the blocking route. Once the route relation
is established, consistent translation still supports passage when the cue
leaves the forward view unless reversal, blockage, or conflict appears. Do not
demand every evidence form, reacquisition, continued visibility, or a particular
final image side after sufficient route progress. Do not wait for an exact
geometric boundary the camera cannot establish.

## Recovery

If route identity, side, or free space was not established, inspect first. If
movement is insufficient, contradictory, or still blocked, gather evidence
before another move. After progress, do not replace
translation evidence with repeated yaw to recover the cue. Shorten movement in
narrow, cluttered, or people-adjacent routes.
