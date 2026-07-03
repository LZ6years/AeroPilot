---
name: vertical_landing_and_altitude_control
description: Use when the UAV should ascend, descend, change altitude, follow a vertical-axis angled path, land, or stop near a safe low-altitude/ground surface.
summary: Select for altitude changes, vertical-path motion, landing, or stopping near a low-altitude region, ground, or surface.
---

# Vertical Landing And Altitude Control

Use this guidance when the instruction depends on altitude, vertical movement,
controlled descent, angled descent, landing, or stopping near a surface. Treat
vertical flight as a controlled approach to a requested height, surface, or
landing condition, not as an open-ended command to keep moving down.

For numeric altitude changes, make bounded progress toward the requested amount.
Upward movement increases altitude; downward movement decreases altitude. When a
numeric target is inside the reasonable tolerance band, prefer hover, stop, or
completion rather than decimal-level correction. If the UAV overshoots, correct
gently instead of oscillating above and below the target.

Use progress-scaled vertical control. When the remaining requested height change
is large and the scene appears safe, make an efficient bounded vertical step
instead of many tiny corrections. As the UAV gets close to the target height,
ground, surface, or tolerance band, shorten each vertical step and prefer hover
to reassess.

For angled ascent or descent, interpret the angle as the path's deviation from
the vertical ascent/descent direction unless the user explicitly says it is a
yaw, heading, vehicle pitch, or camera angle. Use that angle to choose the ratio
between the sideways/forward components and the vertical component:

```text
angle = atan(sqrt(vx^2 + vy^2) / abs(vz))
```

For descent, `vz` is negative. For ascent, `vz` is positive. A smaller angle is
closer to straight vertical movement; a larger angle allows more sideways or
forward motion while changing altitude.

```text
sqrt(vx^2 + vy^2) = abs(vz) * tan(angle)
```

The angle does not mean the UAV should turn in the horizontal plane or point its
nose/camera by that many degrees. Keep the vertical and horizontal components
consistent with the requested path, but do not continue vertical motion when
doing so would move farther away from the requested height or add near-surface
risk.

When an instruction explicitly asks for an angled ascent or descent, include a
non-zero sideways or forward component from the start unless safety, near-surface
conditions, or a final tolerance correction make a pure vertical step safer.
Starting with pure vertical motion does not satisfy a non-zero path-deviation
angle.

For small ascent or descent angles, most of the movement is still vertical, with
only a modest sideways or forward component. Do not respond by using extremely
short durations or tiny repeated vertical steps when the remaining altitude
change is still large and the scene appears safe. Use an effective bounded step
that keeps the requested ratio, then shorten steps only near the target height,
ground, surface, or tolerance band.

For landing, first make sure the requested landing region or relation is
plausible. If the landing is near a visual target or on a requested side,
identify or approach that region before descending. Once descending, use short
controlled downward moves with observation between windows. Stop descending
when the requested drop is reached, the landing surface is near enough, the view
suggests little remaining clearance, or continued descent would add risk
without improving the landing condition.

For landmark landing, target visibility is only part of the evidence. A landing
near a landmark or on a requested side also needs plausible side/region
positioning and altitude or surface-related progress. Do not treat seeing a
landmark as proof of landing, and do not keep turning only because the landmark
is not perfectly centered if the landing region is already plausible enough to
begin a controlled descent.

Do not keep descending only because the word "land" appears in the instruction.
Do not declare a landmark landing complete from target visibility alone; landing
also needs altitude or surface-related evidence. Do not use stale images as
proof of safe ground contact.

When evidence is uncertain, prefer hover or a smaller vertical adjustment before
continuing. When the UAV is already low, near a surface, or inside a reasonable
tolerance band, avoid large downward steps and stop descending if another step
would mainly increase risk or vertical error.
