Useful search patterns should be selected by scene context:

- Scale naming: 30-45 degrees is broad search, not a small/local confirmation
  turn. A small local confirmation turn is about 5-15 degrees.
- No-candidate broad sweep: when the current view has no target-like cue, use a
  larger safe yaw increment within planner limits to inspect genuinely new view
  area. Continue the chosen sweep direction while it reveals new area. A
  practical broad sweep is about 30-45 degrees when the tool and safety limits
  allow it.
- Local sweep: when the target may be near the current view boundary, use small
  yaw steps so a possible target is not swept past.
- Peek safely: when an obstacle or corner may hide the target, a changed viewing
  angle or a short reposition can expose it. Repositioning needs clear free space;
  choose a view change that preserves useful cues without unsupported movement.
- Landmark chaining: use visible landmarks to decide where to look next, but
  keep confidence low until the target itself is observed.
- Candidate disambiguation: when competing identities change the next action,
  use another view or independent category queries on the same view. Compare
  position and appearance, not confidence rankings alone. Keep promising cues
  in view and reuse results; absence of detection is not proof of absence.
- Candidate confirmation: when one plausible candidate appears but is not yet
  reliable, stop broad sweeping and compare the available evidence. A small
  5-15 degree yaw can expose missing detail while preserving the candidate.
  Do not describe a candidate and then
  continue a 30-45 degree broad sweep as if it were confirmation.
- Confirmed or centered target: stop searching or use the confirmed
  target evidence for the current objective instead of continuing a search
  sweep.

Reverse a broad sweep only when a candidate was plausibly swept past a frame
edge, a scene or flight boundary blocks the current direction, or continuing
would revisit rather than reveal view area. Use a small same-side correction,
not a broad reversal, when the last observation contains a candidate.

A next landmark not yet visible is different from losing the present route.
Continue evidence-producing bounded progress while route and free-space cues
remain coherent. After a recovery, reassess what changed rather than treating
one attempt as a fixed stopping quota. Stop when no safe useful option remains,
not simply because the expected landmark has not appeared after a few actions.
