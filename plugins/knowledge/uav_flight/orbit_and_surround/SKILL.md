---
name: orbit_and_surround
description: Use when the UAV should orbit, circle around, surround, or fly around a target at a requested radius or direction.
summary: Select for orbit, circle-around, surround, clockwise, counterclockwise, or radius-based target-relative loop tasks.
---

# Orbit And Surround

Use this guidance when the instruction asks the UAV to circle around, orbit,
surround, or fly around a visible target such as a person, vehicle, landmark, or
object. Orbit tasks are target-relative motion tasks: the UAV must manage both
distance from the target and progress around the target.

First establish the target and the requested radius. Seeing the target is not
the same as orbiting it. If the current distance is clearly too small or too
large, use a bounded radial adjustment before starting the orbit. Once the
radius is plausible, prefer tangential movement around the target instead of
more radial correction.

Keep orbit direction stable. Clockwise, right direction, or rightward orbit
should keep moving around the target in one consistent side direction. Counter-
clockwise, left direction, or leftward orbit should keep the opposite side
direction. Do not alternate left and right unless the latest observation shows a
clear safety or target-loss reason to recover.

Use small orbit windows. A useful orbit step usually combines a short tangential
translation with a modest heading correction so the target remains visible. Do
not rely on yaw alone: turning in place can keep the target in view, but it does
not create circular path progress. Do not rely on lateral movement alone if the
target quickly drifts out of view.

Treat radius and orbit progress as separate evidence. Being at the requested
radius only means the UAV is ready to orbit; it does not mean the orbit is
complete. Continuing to correct radius after it is already plausible can prevent
real circular progress.

When the task asks for a full circle, do not claim completion from one or two
small motions. Without explicit orbit-progress evidence, require sustained
target-relative movement around different sides of the target before treating
the orbit as complete. If progress evidence is weak, continue with a bounded
tangential step rather than looping on tiny radius corrections.

If the target is lost, stabilize or make a small recovery turn before continuing
the orbit. If the target is visible but near an image edge, choose a smaller
step and heading correction to preserve visual contact. Keep altitude stable
unless the instruction also asks for vertical motion.
