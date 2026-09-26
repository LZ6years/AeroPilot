---
name: vertical_landing_and_altitude_control
description: "Applicable needs: ascending, descending, clearing obstacles or landing; Decision questions: judge height progress, clearance, adjustment magnitude and stopping; Contents: actual vertical motion, surface evidence, angled paths and landing relations; Limits: self-height change alone does not prove obstacle clearance or touchdown."
summary: Select for altitude changes, vertical-path motion, landing, or stopping near a low-altitude region, ground, or surface.
---

# Vertical Landing And Altitude Control

## Application Boundary

Use for altitude change, vertical or angled ascent/descent, and landing.
Alignment alone is not height or landing completion. "Land" is not an
open-ended command to keep descending.

## Evidence Model

Use requested height change, progress, clearance and surface cues; no single
source is mandatory. Angled paths deviate from the vertical axis unless the
user specifies yaw, heading, pitch or camera angle. Request
[angled motion details](references/angled_vertical_motion.md) for component ratios.

## Decision Procedure

1. Establish the requested height, direction, path relation, or plausible
   landing region before substantial vertical movement.
2. Use effective bounded progress while far from the target condition. Clear
   space and a substantial remaining height relation can support a larger step;
   do not default to repeated tiny increments or the maximum permitted action.
3. Re-observe after movement and shorten adjustments near the requested height,
   surface, or tolerance band; a smaller adjustment still needs a safe supported direction.
   Compare actual height change and surface appearance with the previous request;
   interrupted or constrained motion is not the requested displacement. Re-estimate
   the next magnitude when the surface or direction of travel changes.
4. For a non-zero angled path, preserve both vertical and horizontal components
   unless a final correction or safety condition requires a simpler movement.

## Completion And Stop

For numeric altitude change, finish within reasonable tolerance, not decimal
correction. Stop descending at the requested drop, near the landing surface,
with little clearance, or when another descent mainly adds risk or error.
Agreeing progress, clearance and region cues need no additional ideal view.

## Recovery

Clearing an obstacle requires being above it along the intended passage, not
merely accumulating a fixed ascent. Use a modest evidence-based clearance margin,
not a guessed typical building height. Reassess after movement; do not assume
intermediate views within an uninterrupted sequence of actions.

If cues conflict or disappear, compare earlier views with actual movement.
Recover a useful view if safe, otherwise stop and cite the limitation. Never
continue a blind descent; correct overshoot gently near a surface.
