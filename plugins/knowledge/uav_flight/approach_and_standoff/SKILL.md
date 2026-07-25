---
name: approach_and_standoff
description: Use when the UAV should finish near, farther from, following, or holding a requested distance or side position relative to a destination target; not when the endpoint is through, past, beyond, across, or around a route reference.
summary: Select only when target-relative proximity, separation, following, or a final held front/back/side position is the terminal concern; exclude route traversal.
---

# Approach And Standoff

## Application Boundary

Use this guidance when the destination is a target and the active concern is
closeness, separation, following, or stopping at a requested front/back/side
relation. A side relation belongs here only when the final task state is to stop
or hold near the target on that side. Do not use it when the endpoint is to pass
through, past, beyond, across, or around a route reference, even if the route
also names a side; preserving target visibility can conflict with traversal.

## Evidence Model

Approach is target-relative. Use target confirmation, apparent size, familiar
target scale, ground contact, nearby scene geometry, perspective or depth cues,
free space, and change after translation. Combine these cues for a rough range
estimate when calibrated sensing is absent. UAV displacement from a local
origin is not target distance. Yaw cannot reach a distance or requested side.

## Decision Procedure

1. Confirm or refresh the target relation before meaningful translation.
2. Decide whether closing, retreating, holding, or lateral repositioning best
   serves the requested relation.
3. Treat a target that already appears very large, clipped, or near the camera
   as clear near-range evidence; do not move closer merely to improve ranging.
4. Use bounded, verifiable movement when distance or clearance is uncertain.
5. Re-observe and check whether the relation changed in the intended direction
   before continuing or correcting.

## Completion And Stop

Stop when current evidence supports the requested relation with reasonable task
precision. For numeric standoff, broadly consistent visual or depth cues with
no clear evidence of being substantially nearer or farther are enough; do not
demand decimal-level certainty. Do not keep approaching until the target fills
the view, becomes occluded, or leaves too little stopping margin merely to
obtain exact proof. Facing the target is not the same as reaching the requested
relation.

## Recovery

If the target is absent or identity is uncertain, stabilize or search instead
of moving toward an assumed location. If motion changes the relation in the
wrong direction, stop and reassess before correcting. Near people, vehicles, or
limited free space, use smaller changes and preserve stopping margin.
