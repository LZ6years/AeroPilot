---
name: metric_depth_sample_region
description: Use when a flight decision or completion claim depends on an explicit distance in meters in the latest observation.
---

# Metric Depth Region Sampling

Call `skill.metric_depth.sample_region` only when an explicit metric distance or
threshold is required. Use the exact `observation_id` on the CURRENT image and
draw a tight normalized rectangle wholly inside the representative visible
surface. Avoid silhouettes, edges, ground, sky, occluders, and background.
Normalize the box against the complete `display_size` shown in the CURRENT
image label. Keep it wholly inside `scene_bounds_in_display_pixels`; the
appended color bar is part of display coordinates but never part of a valid ROI.
If the tool reports inconsistent depth surfaces, shrink and recenter the box
inside one visible surface before retrying.

Every result may include a confirmation image: left is the complete current
RGB, middle is a magnified neighborhood, right contains only the sampled ROI
pixels, and the red rectangle is the exact sample. Identify what is actually
inside the box before using `distance_m`. If it is not wholly inside the surface
meant by the instruction, retry against the same observation. The confirmation
is tool evidence, not a new observation. A failed sample cannot support an
explicit metric completion claim; keep it unknown if a valid box cannot be
isolated.

The returned `distance_m` is the approximate camera-to-visible-surface depth in
that small rectangle. It is not the target center distance, horizontal ground
distance, or benchmark ground truth. After a physical action, sample the new
CURRENT observation again before making another quantitative claim.

If the relevant surface is not visible or cannot be isolated with a tight box,
keep the quantitative condition unknown.
