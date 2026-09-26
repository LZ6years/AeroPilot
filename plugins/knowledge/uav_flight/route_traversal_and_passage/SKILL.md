---
name: route_traversal_and_passage
description: "Applicable needs: following routes, passing landmarks, crossing openings or conditional turns; Decision questions: judge route continuity, junction arrival, road ends and passage progress; Contents: successive views, measured translation, route nodes, turn conditions and recovery; Limits: does not independently prove object identity or precise target distance."
summary: Select for through, past, between, passage, gap, corridor, opening, or explicit route-cue traversal tasks where route progress is the active concern.
---

# Route Traversal And Passage

## Application Boundary

Use to pass through a gap, past, between or beyond route cues. A route cue
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
An absent upcoming landmark is not a lost route. Continue along coherent road
and free-space cues; reconsider at conflict or a boundary.

Request [route examples](references/route_progress_examples.md) for complex
side passages or disappearing cues.

## Completion And Stop

Seeing a route node is not reaching it or entering its continuation. A building
or hidden road alone does not prove an end. Compare road edges, lateral exits and
successive views. At the instructed node, turn toward the supported continuation;
unspecified turns have no fixed angle. Contradictory views require reassessment.

Passage completion requires a terminal route relation. Meaningful translation
with a cue moving sideways, rearward or beyond the blocking route supports it.
Once established, consistent translation remains evidence after the cue leaves
view unless reversal, blockage or conflict appears. Do not require every source,
reacquisition, continued visibility, a final image side or an exact unseen boundary.

## Recovery

Compare earlier views with actual motion when cues conflict. Recover a useful
view if safe; otherwise stop and cite why. Do not replace translation evidence
with repeated yaw.
