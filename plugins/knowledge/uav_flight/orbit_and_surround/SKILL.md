---
name: orbit_and_surround
description: "Applicable needs: specialist orbit maneuvers, circling a target or covering a requested arc; Decision questions: whether tangential progress, radial correction or bearing inspection is needed; Contents: target-relative geometry, circulation direction, changing aspects and coverage checks; Limits: yaw alone does not establish an orbit; basic control handles motion scale and observation handles target visibility."
summary: Select for orbit, circle-around, surround, clockwise, counterclockwise, or radius-based target-relative loop tasks.
---

# Orbit And Surround

## Application Boundary

Use for specialist circulation around a target or covering an arc. Facing one
side is not circulation. Basic control handles scale; observation handles identity.

## Evidence Model

Orbit needs lateral or tangential translation with a plausible target relation.
Yaw alone is not orbit progress. Radius is target distance, not displacement from
the vehicle's origin. Radial repair supports but does not replace circulation.
Changing target aspects and actual motion help judge direction and coverage.

## Decision Procedure

1. Choose alignment, tangential progress or radial repair from the current relation.
   Tangential progress needs an identifiable reference and supported direction.
2. Maintain useful bearing; realign only when needed. Orbit direction determines
   tangential movement, not yaw sign; yaw correction follows current image bearing.
3. Continue tangential progress while direction remains coherent. Briefly repair
   clearly unsuitable radius, then resume circulation. Use target-relative evidence
   with precision suitable for the requested tolerance, not vehicle displacement.
4. Scale the next segment from prior actual translation and target-aspect change.
   Shorten near changing clearance or relation. A successful segment does not prove
   the entire remaining orbit is clear. Inspect after a turn before translating.

Worked example: sideways movement reveals a new target side while range and
circulation direction remain coherent. This supports arc progress. If bearing
becomes unusable, turn, inspect the new view and select the next tangential move.
That restoring turn adds no orbit coverage.

## Completion And Stop

Stop when actual translation and changing target relation support the requested
arc or coverage. Do not keep yawing or repairing standoff merely because the target
remains visible. Brief radius uncertainty does not erase valid circulation.

## Recovery

Compare pre-maneuver and current views with actual motion when the reference is
lost. Use observation or abnormal recovery for repositioning, then inspect it.
If repeated radial repairs add no circulation, choose a different supported
viewpoint. Correct wrong motion to restore the instructed orbit direction.
Temporary recovery motion does not automatically add arc coverage.
