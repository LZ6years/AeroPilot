---
name: pixel_depth_measurement
description: "Measure sensor depth at a visible point selected in a current or historical original RGB observation from the active task; pixel placement and target identity remain visual judgments."
---

# Selected Pixel Depth

Use `skill.pixel_depth.measure_pixel_depth` when depth at a visible surface can
inform the next movement. Select a point inside that surface in an original
RGB observation from this task, away from silhouettes, occlusion boundaries,
sky and overlays. Select its exact observation ID; the matching recorded depth
is used for that observation, including a historical view when available.
Coordinates are zero-based original-image pixels: x increases right and y down.
Use the displayed scene dimensions, not a resized preview or a trajectory map.

The returned point marker lets you verify which surface was sampled. A valid
scalar is sensor depth at that exact pixel, rounded to 0.1 m. It is not distance
to an object's center or horizontal distance to a destination. Optical-axis
depth and ray range retain their declared geometry. Pixel selection does not
establish target identity or route completion.
The returned source time and CURRENT/HISTORY label identify when the distance
was observed. Historical range is not present separation and does not create a
new observation. Missing recorded depth cannot be replaced with another frame.

An invalid measurement provides no numeric distance. Select another supported
surface point or obtain a more useful observation rather than treating invalid
depth as zero, object absence, or proof that movement is impossible.
