---
name: route_traversal_and_passage
description: Use when the UAV should pass through, go through, navigate through, cross between objects, or travel via a requested side of a passage, gap, obstacle, landmark, or route cue.
---

# Route Traversal And Passage

Use this guidance when the instruction asks the UAV to pass through a region,
go through or between objects, cross a passage, or use a requested side of a
landmark or obstacle as part of a route.

Passage tasks are about route progress, not perfect visual centering. First
identify the relevant passage, gap, side, or route cue well enough to choose a
safe short movement. Once the passage or side is visible and plausible, prefer
bounded translation through, past, between, or along the route over repeated yaw
micro-corrections.

Use visual change over time as route evidence. Compare the first useful view of
the route with the latest view: a cue that was ahead of the UAV may later move
to the side, behind, out of the blocking path, or become less relevant after the
UAV has translated. Those changes can support passage progress when they agree
with recent movement.

Keep the boundaries separate:
- If the target or route cue is not visible or remains ambiguous, use search
  behavior to gather new evidence.
- If a visible target only needs facing or attention, use visual-target
  reasoning.
- If the task is mainly to get near or keep distance from a target, use approach
  and standoff reasoning.
- If the route is narrow, cluttered, occluded, or near people, use obstacle and
  safety reasoning to keep movements conservative.

For side-based passage, remember that "from the left side" or "from the right
side" is a spatial route relation. Yaw can inspect that relation, but it cannot
by itself place the UAV on the requested side or move the UAV through the route.
Side relations usually need lateral or forward movement evidence, not just a
changed camera bearing.

If repeated yaw changes keep the same target or gap in view without improving
the route decision, stop chasing a better-centered image. Prefer a bounded
translation that actually advances through, past, between, or along the
requested side of the route. Hover only when motion safety or route identity is
uncertain.

Do not treat a landmark itself as the destination when the instruction says to
pass through, go around, or use a side of it. The landmark is a route reference;
the UAV should make route progress relative to that reference while preserving
the requested side relation and safety.

Do not treat one small translation or a visually plausible side relation as
enough passage evidence unless recent movement also supports meaningful route
progress.

Do not wait forever for an exact geometric boundary, axis, or entrance line that
cannot be directly observed. If the route cue is clear, motion has been
meaningful, and the latest view suggests the UAV has moved through, past,
between, or beyond the relevant cue, treat that as useful completion evidence
rather than continuing to chase a perfect view.
