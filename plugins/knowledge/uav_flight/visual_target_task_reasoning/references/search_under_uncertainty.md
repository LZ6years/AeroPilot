Useful search patterns should be selected by scene context:

- Scale naming: 30-45 degrees is broad search, not a small/local confirmation
  turn. Small confirmation means hover or about 5-15 degrees.
- No-candidate broad sweep: when the current view has no target-like cue, use a
  larger safe yaw increment within planner limits to inspect genuinely new view
  area. Continue the chosen sweep direction while it reveals new area. A
  practical broad sweep is about 30-45 degrees when the tool and safety limits
  allow it.
- Local sweep: when the target may be near the current view boundary, use small
  yaw steps so a possible target is not swept past.
- Peek safely: when an obstacle or corner may hide the target, prefer yaw or
  hover-based stabilization first; use a short reposition only when free space is
  visually clear.
- Landmark chaining: use visible landmarks to decide where to look next, but
  keep confidence low until the target itself is observed.
- Candidate disambiguation: when several similar targets exist, gather another
  view instead of committing early, but keep the promising candidates in view.
- Candidate confirmation: when one plausible candidate appears but is not yet
  reliable, switch from broad sweep to hover or a small 5-15 degree yaw before
  judging so the candidate stays in view. Do not describe a candidate and then
  continue a 30-45 degree broad sweep as if it were confirmation.
- Confirmed or centered target: hover, stop searching, or use the confirmed
  target evidence for the current objective instead of continuing a search
  sweep.

Reverse a broad sweep only when a candidate was plausibly swept past a frame
edge, a scene or flight boundary blocks the current direction, or continuing
would revisit rather than reveal view area. Use a small same-side correction,
not a broad reversal, when the last observation contains a candidate.
