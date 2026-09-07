---
name: vertical_landing_and_altitude_control
description: Use when the UAV should ascend, descend, change altitude, follow a vertical-axis angled path, land, or stop near a safe low-altitude/ground surface.
summary: Select for altitude changes, vertical-path motion, landing, or stopping near a low-altitude region, ground, or surface.
---

# Vertical Landing And Altitude Control

## Application Boundary

Use this guidance for altitude change, vertical or angled ascent/descent, and
controlled landing near a surface. Do not treat visual target alignment alone
as altitude or landing completion, and do not interpret "land" as an open-ended
command to keep descending.

## Evidence Model

Use requested height change, vertical progress, available clearance or surface
cues, and region plausibility. No one evidence source is mandatory. For an angled
path, interpret the angle as deviation from the vertical ascent/descent axis
unless the user explicitly identifies yaw, heading, vehicle pitch, or camera
angle. Request [angled motion details](references/angled_vertical_motion.md)
when the component ratio or formula is needed.

## Decision Procedure

1. Establish the requested height, direction, path relation, or plausible
   landing region before substantial vertical movement.
2. Use effective bounded progress while far from the target condition.
3. Re-observe after movement and shorten adjustments near the requested height,
   surface, or tolerance band.
4. For a non-zero angled path, preserve both vertical and horizontal components
   unless a final correction or safety condition requires a simpler movement.

## Completion And Stop

For numeric altitude change, finish inside a reasonable tolerance instead of
making decimal-level corrections. Stop descending when the requested drop is
reached, the landing surface is near enough, little clearance appears to remain,
or another descent would mainly increase risk or vertical error. A coherent
combination of available vertical progress, clearance, and visual region cues
is sufficient without requiring an additional ideal view.

## Recovery

If height or region cues conflict, hover or use a smaller adjustment. If they
disappear after a maneuver, compare pre-maneuver and current views with measured
vertical and horizontal change. Try one bounded retrace, then re-observe; never
continue a blind descent. Hover only to stabilize or compare. If no safe
evidence-producing recovery remains, stop and cite the limitation. Correct
overshoot gently and avoid large downward steps near a surface.
