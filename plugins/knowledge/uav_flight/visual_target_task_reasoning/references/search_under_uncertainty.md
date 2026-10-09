Useful search patterns should be selected by scene context:

- Anchored direction evidence: when the instruction supplies a direction or yaw
  relative to the starting heading, retain that reference frame while interpreting
  later observations. It identifies relevant bearing evidence, not a prescribed
  motion sequence or corridor to follow.
- Choose the next bearing from view overlap, the candidate's image position, actual
  visual change after previous turns and relevant area not yet covered. Do not infer
  an action scale from the available parameter range.
- No-candidate view: when the current view has no target-like cue, choose a bearing
  that exposes relevant uncovered area while accounting for any anchored direction.
- Boundary candidate: when a possible target lies near the current view boundary,
  choose a view that preserves enough overlap to compare identity and position.
- Occlusion view: when an obstacle or corner may hide the target, ascent, lateral
  movement or a short reposition can expose the hidden region while retaining useful
  route cues.
- Landmark chaining: use visible landmarks to decide where to look next, but
  keep confidence low until the target itself is observed.
- Candidate disambiguation: when competing identities change the next action,
  choose a view that reveals the distinguishing attribute. Compare position and
  appearance, and retain promising cues across successive views.
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

A next landmark outside the current view is different from losing the present route.
Continue along coherent route and free-space cues to obtain new coverage. After a
viewpoint change, use what changed to choose the next action. A persistent lack of
new coverage across distinct useful viewpoints can establish no progress; one
unsuccessful observation cannot.
