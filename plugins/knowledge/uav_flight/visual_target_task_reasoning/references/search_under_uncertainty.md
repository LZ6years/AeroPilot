Useful search patterns should be selected by scene context:

- Choose angular scale from view overlap, the candidate's image position and
  actual visual change after previous turns, not universal degree ranges.
- No-candidate broad sweep: when the current view has no target-like cue, use a
  larger safe yaw increment within planner limits to inspect genuinely new view
  area. Continue the chosen sweep direction while it reveals useful new area.
- Local sweep: when the target may be near the current view boundary, use small
  yaw steps so a possible target is not swept past.
- Peek safely: when an obstacle or corner may hide the target, a changed viewing
  angle or a short reposition can expose it. Repositioning needs clear free space;
  choose a view change that preserves useful cues without unsupported movement.
- Landmark chaining: use visible landmarks to decide where to look next, but
  keep confidence low until the target itself is observed.
- Candidate disambiguation: when competing identities change the next action,
  choose a view that reveals the missing distinguishing attribute. Compare
  position and appearance, not confidence rankings alone. Keep promising cues
  in view and reuse results; absence of detection is not proof of absence.
- Candidate confirmation: when one plausible candidate appears but is not yet
  reliable, compare the available evidence and decide whether investigation is
  worthwhile. A closer or lateral view may expose detail; turning can reveal an
  edge but cannot by itself improve range. Select motion for that observation
  purpose rather than automatically approaching or continuing a broad sweep.
- Confirmed or centered target: stop searching or use the confirmed
  target evidence for the current objective instead of continuing a search
  sweep.

Reorient when evidence changes the plausible search region, a boundary prevents
progress or continuing revisits uninformative views. Avoid direction changes
unsupported by new information. A visible candidate may justify local inspection;
if its required attributes are contradicted, reconsider the wider search.

A next landmark not yet visible is different from losing the present route.
Continue evidence-producing bounded progress while route and free-space cues
remain coherent. After a recovery, reassess what changed rather than treating
one attempt as a fixed stopping quota. Stop when no safe useful option remains,
not simply because the expected landmark has not appeared after a few actions.
