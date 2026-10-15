# Angled Vertical Motion

Use this reference when an instruction specifies an ascent or descent angle and
the motion components need explicit interpretation.

Unless the user says otherwise, treat the angle as the path's deviation from
the vertical ascent/descent direction:

```text
angle = atan(sqrt(vx^2 + vy^2) / abs(vz))
sqrt(vx^2 + vy^2) = abs(vz) * tan(angle)
```

For descent, `vz` is negative. For ascent, `vz` is positive. A smaller angle is
closer to straight vertical motion; a larger angle permits more horizontal
motion for the same vertical component.

The angle is not a yaw, horizontal heading, vehicle pitch, or camera angle. A
non-zero path-deviation angle normally needs a non-zero horizontal component
from the start. Near the requested height or a surface, shorten the step or use
a simpler final correction when preserving the exact ratio would add risk or
increase error.

Choose the next observation position from the remaining altitude relation, the
previous action's requested and actual vertical displacement, and the observed
surface change. Preserve the requested component relation while it remains
applicable, then recalibrate as the target condition or visible geometry changes.
