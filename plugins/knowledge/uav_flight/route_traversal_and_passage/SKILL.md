---
name: route_traversal_and_passage
description: Use when the UAV should pass through, go through, navigate through, progress past, cross between objects, or follow an explicit passage, gap, corridor, opening, or route cue.
summary: Select for through, past, between, passage, gap, corridor, opening, or explicit route-cue traversal tasks where route progress is the active concern.
---

# Route Traversal And Passage

## Application Boundary

Use this guidance to pass through, past, between, beyond, or along a requested
side of a passage, gap, opening, corridor, or cue. Do not treat the cue as a
destination when route progress is the concern. Use approach reasoning when the
task ends at a target-relative position.

## Evidence Model

Passage evidence combines meaningful translation with visual change over time.
A cue that began ahead may move sideways, behind, outside the blocking path, or
out of view. Once the cue and route relation are established, consistent
translation remains evidence when the cue is no longer visible. Cue absence
alone is ambiguous: it neither proves completion nor requires renewed search.
Yaw can inspect a route but cannot place the UAV on a side or prove traversal.

## Decision Procedure

1. Establish the route, requested side, or gap and usable free space.
2. Prefer bounded translation through or past it over repeated centering.
3. Compare the first useful route view with the latest view and accumulated
   translation; use their agreement as route evidence.
4. Reacquire the cue only if its identity or relation was not established, or
   movement evidence is insufficient or contradictory.

Request [route progress examples](references/route_progress_examples.md) when a
complex side passage or disappearing cue needs more detailed interpretation.

## Completion And Stop

Either path is independently sufficient: meaningful translation agrees
with the cue moving from ahead to the side, rear, or beyond the blocking route;
or, after establishing the cue and relation, directionally consistent
translation carries the UAV beyond it while the cue leaves the forward view
without reversal, blockage, or conflicting evidence. Do not require both paths,
reacquisition, continued visibility, or a particular final image side after
sufficient route progress. Do not wait for an exact geometric boundary the
camera cannot establish.

## Recovery

If route identity, side, or free space was never established, inspect before
translating. If movement is insufficient, contradictory, or the route remains
ahead and blocking, gather evidence before another substantial move. After
valid passage progress, do not replace translation evidence with repeated yaw
to recover the cue. Shorten movement in narrow, cluttered, or people-adjacent
routes.
