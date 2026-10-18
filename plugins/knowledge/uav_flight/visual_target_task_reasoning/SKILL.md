---
name: visual_target_task_reasoning
description: "Applicable needs: checking landmark identity or choosing an observation position for unclear, occluded or competing views; Decision questions: which candidate fits the surroundings and spatial relations, and which move can reveal missing information; Contents: identity checks, historical view comparison, ascent/descent, backing, lateral parallax and bearing inspection; Limits: similar appearance is only a candidate; a better view does not establish metric range, passage or arrival."
summary: Select for Observation and situational awareness; select when identity, visibility, occlusion or viewpoint is the active question.
---

# Observation And Situational Awareness

## Application Boundary

Use for identity, search, facing or an unclear view. Basic control handles scale;
abnormal recovery handles lost orientation.

## Evidence Model

Distinguish an outside-view cue, tentative candidate and identifiable target.
Discover candidates from current RGB. Similar appearance is only a candidate;
check attributes, nearby landmarks and position relations before accepting identity.
Surface depth measures a matched visible candidate, not its identity or arrival.
Descriptive surroundings may establish identity across views while target and
scene continuity remain supported. Distinguish these cues from required final
relations, image coverage or continued observation; simultaneous visibility is
needed when the task requires it. Past identity evidence may become stale for
moving targets or changing scenes. Historical range belongs to its source time;
estimating present separation requires supported intervening vehicle and target
motion, compatible geometry and uncertainty. Compare historical and current views
with actual motion to interpret disappearance.
Turning reveals bearings; translation provides parallax. Ascent may expose an
occluded region; descent or backing may reveal low detail. These changes move a
fixed camera's viewpoint, not its pitch, and do not guarantee clearance.

## Decision Procedure

1. Name missing information: identity, hidden route exit, low detail, clipped
   bearing or competing candidates. Respect the direction cue's stated frame;
   starting-heading cues do not become body-relative or prescribe a sequence.
2. Choose a test: rise to expose a foreground-hidden region, descend/back for low detail,
   shift sideways for parallax, or turn for another bearing. Preserve instructed
   constraints and use available clearance. Predict what should become visible.
3. Perform one action, inspect the resulting view and check that predicted change.
   If it fails, revise the occlusion explanation instead of repeating the move.
4. For facing, correct clipping or ambiguous bearing; usable offset is acceptable
   unless exact centering is requested. Identity and arrival remain separate.

Worked example: a canopy hides the road. A short rise may reveal it where clearance
permits. If the same canopy still dominates, backing or sideways parallax tests
a different geometry. Do not assume that another identical rise will help.

## Completion And Stop

Stop when the needed information is usable. An identifiable forward cue with
usable bearing can satisfy facing.

## Recovery

Contradicted attributes reopen identification; temporary occlusion need not.
Seek different useful coverage when no detail appears. Use abnormal recovery for
unusable views, and optional [search](references/search_under_uncertainty.md) or
[viewpoint examples](references/viewpoint_examples.md) for longer investigations.
