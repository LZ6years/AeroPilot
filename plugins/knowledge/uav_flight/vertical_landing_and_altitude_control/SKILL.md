---
name: vertical_landing_and_altitude_control
description: Use when the UAV should ascend, descend, change altitude, follow an angled vertical path, land, or stop near a safe low-altitude/ground surface.
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

For angled ascent or descent, interpret the angle as the vertical flight-path
angle relative to the horizontal plane unless the user explicitly says it is a
yaw, heading, vehicle pitch, or camera angle. Use that angle to choose the ratio
between horizontal speed and vertical speed:

```text
abs(vertical_speed) / horizontal_speed = tan(angle)
```

For descent, vertical speed is negative:

```text
vz = -horizontal_speed * tan(angle)
```

The angle does not mean the UAV should turn in the horizontal plane or point its
nose/camera by that many degrees. Keep the vertical and horizontal components
consistent with the requested path, but do not continue vertical motion when
doing so would move farther away from the requested height or add near-surface
risk.

For shallow ascent or descent angles, the vertical component is naturally
smaller than the horizontal component. Do not respond by using extremely short
durations or tiny repeated vertical steps when the remaining altitude change is
still large and the scene appears safe. Use an effective bounded step that keeps
the requested ratio, then shorten steps only near the target height, ground,
surface, or tolerance band.

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
also needs altitude or surface-related evidence. Do not use hidden benchmark
targets or old images as proof of safe ground contact.

When evidence is uncertain, prefer hover or a smaller vertical adjustment before
continuing. When the UAV is already low, near a surface, or inside a reasonable
tolerance band, avoid large downward steps and stop descending if another step
would mainly increase risk or vertical error.
