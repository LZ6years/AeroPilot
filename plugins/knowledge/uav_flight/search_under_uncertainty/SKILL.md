---
name: search_under_uncertainty
description: Use when the target, route cue, landmark, entrance, sign, person, or object is not yet visible or is ambiguous.
---

# Search Under Uncertainty

Use this guidance when the task cannot be solved from the current observation
because the target is absent, ambiguous, partially occluded, or confused with
similar candidates. Search is a controlled information-gathering behavior, not a
blind repetition of the same motion.

Separate search from confirmation. During search, prefer actions that create new
visual evidence while preserving safety. After each observation, reassess whether
the target is absent, only a weak candidate, a plausible candidate needing
confirmation, or visually confirmed.

Action scale must match the evidence you describe:
- If your message or reason says no target-like cue is visible, broad search may
  use about 30-45 degrees.
- If your message or reason says possible, suspected, candidate, target-like,
  person-like, silhouette, outline, patch, region, target side, or any similar
  uncertain cue, do not use 30-45 degrees. That is candidate confirmation:
  hover or use a small 5-15 degree yaw that keeps the cue in view.
- If the target is clear or roughly centered, hover/stop searching and let the
  current phase decision handle completion.

Do not call a 30 degree yaw "small" or "local" confirmation. In this skill,
small confirmation means about 5-15 degrees or hover.

Use recent yaw and movement history to avoid oscillation and repeated scanning of
the same view. Look for instruction-relevant cues instead of generic objects that
the user did not ask for.

Use a coarse-to-fine rhythm instead of one fixed yaw size. Decide the action
scale from the latest visual evidence before choosing the yaw amount:
- No-candidate broad search: use broad search only when no target-like evidence
  is visible: no target, no plausible candidate, no relevant silhouette, color
  patch, landmark cue, object outline, or target-side hint. In this case, use a
  larger safe yaw increment within the current planner limits to expose new view
  area. A practical broad-search preference is about 30-45 degrees when the tool
  limits and safety review allow it, rather than many repeated 5-10 degree
  turns.
- Local sweep: when the target may be near the current view boundary or behind a
  nearby occluder but no stable candidate is in view, use smaller yaw changes
  that preserve continuity.
- Candidate confirmation: when one plausible but uncertain candidate appears,
  broad search is over. Prefer hover, or use a small 5-15 degree yaw that keeps
  the candidate in view while gathering a clearer observation. Do not use a
  30-45 degree broad-search turn as "further confirmation" after describing a
  candidate, possible target, target-like outline, or target on one image side.
  Use the visual-target skill for detailed image-side centering behavior.
- Confirmed target handling: when the target is visually confirmed or already
  roughly centered, stop broad search. Hover or let the current phase decision
  decide whether to align, continue, advance, or complete.

Action-scale self-check:
- no current target-like cue -> broad search may use 30-45 degrees;
- uncertain candidate in view -> hover or small 5-15 degree confirmation;
- clear or roughly centered target -> hover/stop searching and let the current
  phase decision handle completion.

Do not approach an uncertain target as if it were confirmed. Do not treat map
memory hints, old observations, or artifact paths as current visual confirmation.
Do not treat weak candidate evidence as task completion.
