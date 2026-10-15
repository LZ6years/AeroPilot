---
name: abnormal_situation_recovery
description: "Applicable needs: unexpected motion, dark or unusable views, lost orientation, conflicting route cues, collision or uncertain execution; Decision questions: what remains known, what changed, and which recovery can test the cause; Contents: last-clear-view reconstruction, cautious repositioning, heading-aware retracing, uncertainty and stopping; Limits: does not diagnose hidden geometry, guarantee a clear retreat or treat timeout as no movement."
summary: Select for Abnormal judgment and recovery; select when actual outcomes conflict with expectations or usable orientation is lost.
---

# Abnormal Situation Judgment And Recovery

## Application Boundary

Use for unexpected motion, unusable views, lost orientation or route conflicts.
Recovery restores information while preserving task constraints.

## Evidence Model

Separate unavailable acquisition from an acquired but unusable image. Darkness
alone does not prove collision or camera failure. Action success does not prove a
clear path. Unknown execution may include movement; a pre-execution rejection did
not begin. Combine the last clear view, subsequent actual motion and current
heading. Preserve confirmed passage separately from uncertain landmark identity.

## Decision Procedure

1. Suspend dependent progress and identify the uncertainty: motion, orientation,
   view geometry, acquisition or route interpretation.
2. Compare the last useful view and actual actions. Name a testable cause and
   expected result. Use known geometry to choose a short bearing check, reposition
   or return toward an earlier view. A prior position does not guarantee clear retreat.
3. Account for current heading when retracing. Negating an earlier body-forward
   command after a turn need not return along the same path.
4. Perform one recovery, inspect motion and view, then revise the explanation.
   Repeated ineffective yaw or climb needs new justification. After unknown
   execution, obtain available observations and movement evidence before retrying.

Worked example: darkness follows forward motion and a turn. Compare both with the
last clear scene. A short heading-aware return toward a supported prior position
may test occlusion better than more yaw. Inspect it; darkness alone has no diagnosis.

## Completion And Stop

After the view clears, recheck target identity and route interpretation; clarity
alone does not prove the route correct. Resume with supported continuation.
Stop with the unresolved limitation when no supported
recovery can obtain needed information while preserving constraints. One poor
image is not a mandatory stop; there is no fixed attempt quota.

## Recovery

Retain failed recovery outcomes and correct contradicted interpretations explicitly.
Avoid speculative travel. Persistent communication/acquisition failure preserves
execution uncertainty. See [recovery examples](references/recovery_examples.md).
