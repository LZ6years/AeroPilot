---
name: orbit_and_surround
description: "Applicable needs: circling a target or covering a requested arc; Decision questions: separate heading change, radial repair and actual circulation progress; Contents: tangential movement, target relation, direction and coverage; Limits: yaw alone does not establish an orbit."
summary: Select for orbit, circle-around, surround, clockwise, counterclockwise, or radius-based target-relative loop tasks.
---

# Orbit And Surround

## Application Boundary

Use for orbiting or covering an arc around a target. Facing or holding one side alone is not circulation.

## Evidence Model

Orbit progress requires lateral or tangential translation while maintaining a
plausible target relation. Yaw alone is not orbit progress. Orbit radius is
distance to the target, not displacement from the UAV's local origin. Radius
repair supports circulation but must not replace it.

## Decision Procedure

Choose from alignment, tangential progress, or radial repair according to the
current evidence; do not enforce a fixed maneuver sequence. Tangential movement
requires a usable target relation and plausible circulation direction. Re-align
yaw only when needed to keep the reference usable. Continue tangential progress
while direction remains consistent; if radius becomes clearly unsuitable, use
a brief radial repair without replacing circulation.

Radius judgments need target-relative evidence with precision appropriate to
the requested tolerance and contemplated radial correction. Qualitative range
and metric distance support different levels of precision; vehicle displacement
alone does not measure orbit radius.

Orbit direction controls the shift direction, not the yaw sign. Use the target's
current image side when yaw is specifically for re-alignment.
Use prior actual translation and changing target aspect to scale the next arc
segment. Shorten the observation interval when clearance or the target relation
may change; a previous successful segment does not prove the whole orbit is clear.

## Completion And Stop

Stop orbit movement when the requested arc, circle, or surrounding coverage is
reasonably supported by accumulated translation and changing target relation.
Do not continue yaw, standoff correction, or shifting merely because the target
remains visible. A brief radius uncertainty does not erase valid circulation
progress.

## Recovery

If a required relation disappears after a maneuver, compare pre-maneuver and
current views with measured motion. Try one bounded retrace or correction, then
re-observe; do not continue a blind shift.
If repeated radial corrections add no tangential progress, reassess. If no safe
evidence-producing recovery remains, stop and cite the limitation. Reverse
circulation only for safety, recovery, or clear directional conflict.
