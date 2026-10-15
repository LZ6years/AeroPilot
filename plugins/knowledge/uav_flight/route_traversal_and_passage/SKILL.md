---
name: route_traversal_and_passage
description: "Applicable needs: route navigation, passing landmarks, crossing openings or conditional turns; Decision questions: which continuation matches the instruction, when a junction is reached, and whether a required passage occurred; Contents: route continuity, landmark order, before/after relations and turn triggers; Limits: use basic control for action scale, observation for unclear views and abnormal recovery for unexpected outcomes; does not establish precise target distance."
summary: Select for through, past, between, passage, gap, corridor, opening, or explicit route-cue traversal tasks where route progress is the active concern.
---

# Route Traversal And Passage

## Application Boundary

Use for route continuity, passing cues, gaps, junctions and conditional turns.
A route cue is not necessarily the final stopping position.

## Evidence Model

Combine actual translation and successive views. Yaw changes bearing, not physical
side or passage. Seeing a junction is not reaching or entering it. Cue absence
alone is ambiguous; it neither proves passage nor requires renewed search.
A building ahead or hidden road alone does not prove a road end.

## Decision Procedure

1. Preserve instructed order and the next condition: follow, pass, reach, then turn.
2. Match connected road edges, openings and side relations to that condition.
   Seeing the junction leaves arrival unproven. Inspect lateral exits before an end;
   use observation guidance for occlusion.
3. Choose an observation point to check the condition. When the continuation is
   usable, translate through or past it rather than repeatedly centering.
4. Check arrival from opening geometry and actual translation before turning.
   Choose the turn toward the instructed continuation from its direction relative
   to the current heading. If the view is insufficient, turn in stages and inspect
   after each adjustment, then enter the connected road.
   Confirm entry from translation and aligned road boundaries; a turn alone is
   insufficient. Unspecified turns have no fixed angle; preserve explicitly
   instructed angles rather than replacing them with a guessed road alignment.

Worked example: the instruction says to turn right into a road that runs obliquely
to the current heading. Reach its entrance, turn toward that road rather than
assuming a right angle, and inspect the connection. If it remains obscured, adjust
the view in stages; enter only when its connection and usable direction are supported.

## Completion And Stop

Passage needs a supported terminal route relation: translation and a cue moving
sideways, rearward or beyond the obstructing route can support it. Once established,
coherent movement remains evidence after the cue leaves view unless reversal or
conflict appears. Do not demand continued visibility or an exact unseen boundary.
Surface depth alone cannot establish a junction, end or route. Separately check
seeing a junction, reaching it and entering its turn road. Passing an intermediate
cue does not establish the instructed final stopping relation.

## Recovery

Preserve confirmed earlier conditions while correcting unsupported interpretations.
Use observation or abnormal recovery for hidden routes or lost orientation; repeated
yaw is not passage. Load [route examples](references/route_progress_examples.md)
for disappearing cues or complex side passages.
