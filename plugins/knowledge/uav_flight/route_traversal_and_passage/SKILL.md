---
name: route_traversal_and_passage
description: Use when the UAV should pass through, go through, navigate through, progress past, cross between objects, or follow an explicit passage, gap, corridor, opening, or route cue.
summary: Select for through, past, between, passage, gap, corridor, opening, or explicit route-cue traversal tasks where route progress is the active concern.
---

# Route Traversal And Passage

## Application Boundary

Use for through, past, between, beyond, or along-side traversal. A route cue
is not necessarily a destination. Use approach reasoning for a target-relative
terminal position.

## Evidence Model

Combine meaningful translation with visual change. A cue that began ahead may
move sideways, behind, outside the blocking path, or out of view. Once its route
relation is established, consistent translation remains evidence while it is
not visible. Cue absence alone is ambiguous: it neither proves completion nor
requires renewed search. The requested side describes the UAV route relative to
the cue, not a final image side. Yaw can inspect a route or change image bearing,
but cannot place the UAV on a physical side or prove traversal.

## Decision Procedure

Choose a bounded action from missing route evidence, not a fixed maneuver
sequence. Establish route, side, or free space before translation.
Once usable, prefer bounded translation through or past it over repeated
centering. Compare the first useful route view with the latest view and
accumulated translation. Reacquire only if that evidence is insufficient or
contradictory.
An upcoming landmark being absent is not loss of the current route. While the
road, heading and free space remain usable, continue bounded progress toward
new evidence rather than declaring failure after a few moves. Reconsider when
route evidence conflicts, a boundary appears, or motion produces no useful view.

Request [route examples](references/route_progress_examples.md) for complex
side passages or disappearing cues.

## Completion And Stop

Passage completion requires a terminal route relation. Meaningful translation
with a cue moving sideways, rearward or beyond the blocking route supports it.
Once established, consistent translation remains evidence after the cue leaves
view unless reversal, blockage or conflict appears. Do not require every source,
reacquisition, continued visibility, a final image side or an exact unseen boundary.

## Recovery

If a required route cue disappears unexpectedly, compare pre-maneuver and
current views with measured motion. Try a bounded retrace and re-observe; do
not continue blind motion. Hover only to compare. If no safe evidence-producing
recovery remains, stop and cite why. Do not replace translation evidence with repeated yaw.
