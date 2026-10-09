---
name: vertical_landing_and_altitude_control
description: "Applicable needs: ascending, descending, clearing obstacles or landing; Decision questions: judge height progress, clearance, adjustment magnitude and stopping; Contents: actual vertical motion, surface evidence, angled paths and landing relations; Limits: self-height change alone does not prove obstacle clearance or touchdown."
summary: Select for altitude changes, vertical-path motion, landing, or stopping near a low-altitude region, ground, or surface.
---

# Vertical Landing And Altitude Control

## Application Boundary

Use for height changes, vertical or angled ascent/descent and landing. Alignment
alone is not height or landing completion; landing does not mean unlimited descent.

## Evidence Model

Use requested height change, progress, clearance and surface cues. Image position
alone does not prove clearance. Angled paths deviate from the vertical axis unless
specified as yaw, heading, pitch or camera angle. See
[angled motion details](references/angled_vertical_motion.md) for component ratios.

## Decision Procedure

1. Establish the purpose: requested height, clearance, useful observation or a
   landing region. Explicit height requirements and necessary clearance remain binding.
2. While far from the required relation, choose a height change that produces a
   meaningfully different viewpoint or clearance. Where the task permits, choose
   height to support the next observation as well as vertical progress.
3. Compare actual height change and before/after surface appearance. Re-estimate
   remaining need after changed geometry; shorten adjustments near the requested
   height or surface.
4. Preserve vertical and horizontal components of an instructed non-zero angled
   path unless a final correction requires a simpler motion.

## Completion And Stop

Numeric changes use reasonable tolerance, not decimal correction. Stop descending
at the requested drop, near the landing surface, with little clearance, or when
further descent mainly adds risk or error. Agreeing progress, clearance and region
cues need no ideal extra view. Clearing an obstacle requires being above it along
its intended passage, not accumulating a guessed typical height. Use a modest,
evidence-supported clearance margin.

## Recovery

If cues disappear or conflict, compare views and measured motion to distinguish
normal passage, changed angle, occlusion and possible overshoot. Use a different
viewpoint to restore the surface or clearance relation before the next dependent
correction. Near a surface, use shorter corrections and the next real observation.
