---
name: approach_and_standoff
description: Use when the UAV should approach, move near, follow, or keep a requested distance from a target or landmark.
---

# Approach And Standoff

Use this guidance when the instruction asks the UAV to get closer to a person,
object, vehicle, entrance, building, or landmark, or to maintain a requested
distance from it. Approach tasks are target-relative; they usually require target
confirmation before meaningful movement.

First establish or refresh the target relation. If the target is visible, reason
qualitatively about direction, visual scale, depth evidence when available, free
space, and whether closing distance, increasing distance, or repositioning
laterally better satisfies the requested relation. If the target is not visible,
search or stabilize before moving toward an assumed location.

Prefer short, verifiable moves when distance or path clearance is uncertain.
After moving, use the next observation to update whether the UAV is closer,
farther, safer, or still misaligned. If the task asks for a standoff distance,
avoid overshooting into the target and avoid finishing just because the UAV
turned.

Do not move blindly toward a target that is not visually confirmed or otherwise
well supported by current context. Do not treat hidden benchmark positions as
evidence. Do not confuse "face the target" with "reach the requested distance."

Approach and standoff decisions should consider target visibility and qualitative
bearing, whether there is clear free space between the UAV and target, whether
the requested relation is "near", "at a distance", "follow", or "inspect", and
whether recent movement already changed the relation enough to re-observe.

Side relations such as left side, right side, front side, beside, or adjacent are
spatial relations, not just visual centering requests. Turning can help inspect
the relation, but it does not by itself place the UAV on a requested side. Use
translation evidence and current visual relation to decide whether the UAV is
actually approaching, offsetting, or holding the requested side.

For people and vehicles, keep movements conservative unless the task explicitly
requires close inspection.
