---
name: orbit_and_surround
description: Use when the UAV should orbit, circle around, surround, or fly around a target at a requested radius or direction.
summary: Select for orbit, circle-around, surround, clockwise, counterclockwise, or radius-based target-relative loop tasks.
---

# Orbit And Surround

- Core rule: keep a plausible target or reference relation while making steady
  path progress around it.
- Radius: orbit radius is distance to the target, not UAV local pose origin or
  phase displacement. Treat radius repair as brief support work, not the main
  orbit behavior.
- Progress: yaw alone is not orbit progress. Use lateral or tangential shift to
  move around the target.
- Rhythm: use shift segment -> yaw re-align -> next shift segment.
- Re-align: orbit direction controls shift direction, not yaw sign.
- Image-side yaw: target left -> left/negative yaw; target right ->
  right/positive yaw. Opposite yaw can be search, safety, recovery, or path
  checking, but not re-centering.
- Recovery: if the target is near the image edge, partly lost, or unclear, yaw
  re-align or hover to recover before shifting again.
- Direction: keep a consistent shift direction unless safety, target recovery,
  or a clearly wrong orbit direction requires changing it.
- Radius repair: if too close or too far, use a short radial adjustment, then
  resume the shift-then-yaw rhythm.
- If the target or reference remains usable, continue orbit movement instead of
  repeating standoff correction.
- Stop: stop orbit motions when the requested arc, circle, or surrounding
  coverage is sufficient. Do not continue meaningless yaw or shift just because
  the target remains visible.
