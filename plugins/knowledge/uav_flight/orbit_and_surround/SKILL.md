---
name: orbit_and_surround
description: Use when the UAV should orbit, circle around, surround, or fly around a target at a requested radius or direction.
summary: Select for orbit, circle-around, surround, clockwise, counterclockwise, or radius-based target-relative loop tasks.
---

# Orbit And Surround

## Application Boundary

Use this guidance for orbiting, circling, surrounding, or covering an arc around
a target or reference. Do not use it for merely facing, approaching, passing,
or holding one side of a target without around-target path progress.

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

Orbit direction controls the shift direction, not the yaw sign. Use the target's
current image side when yaw is specifically for re-alignment.

## Completion And Stop

Stop orbit movement when the requested arc, circle, or surrounding coverage is
reasonably supported by accumulated translation and changing target relation.
Do not continue yaw, standoff correction, or shifting merely because the target
remains visible. A brief radius uncertainty does not erase valid circulation
progress.

## Recovery

If the target is near an image edge, partly lost, or ambiguous, hover or use a
bounded yaw correction before the next shift. If repeated radial corrections
produce no tangential progress, stop repairing radius and reassess the target
relation and circulation direction. Reverse direction only for safety, target
recovery, or clear evidence that the chosen direction is wrong.
