Useful search patterns should be selected by scene context:

- Broad sweep: when the current view has no target-like cue, use a larger safe
  yaw increment within planner limits to inspect genuinely new view area.
- Local sweep: when the target may be near the current view boundary, use small
  yaw steps so a possible target is not swept past.
- Peek safely: when an obstacle or corner may hide the target, prefer yaw or
  hover-based stabilization first; use a short reposition only when free space is
  visually clear.
- Landmark chaining: use visible landmarks to decide where to look next, but
  keep confidence low until the target itself is observed.
- Candidate disambiguation: when several similar targets exist, gather another
  view instead of committing early.
- Candidate confirmation: when one plausible candidate appears but is not yet
  reliable, switch from broad sweep to smaller yaw changes or hover to stabilize
  before judging so the candidate stays in view.

Avoid repetitive left-right oscillation unless the last observation clearly
showed that a small corrective turn is needed.
