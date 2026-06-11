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

Use recent yaw and movement history to avoid oscillation and repeated scanning of
the same view. Look for instruction-relevant cues instead of generic objects that
the user did not ask for.

Use a coarse-to-fine rhythm instead of one fixed yaw size:
- Broad search: when no target-like evidence is visible, use a larger safe yaw
  increment within the current planner limits to expose new view area.
- Local sweep: when the target may be near the current view boundary or behind a
  nearby occluder, use smaller yaw changes that preserve continuity.
- Candidate confirmation: when one plausible but uncertain candidate appears,
  reduce the yaw step or hover briefly to stabilize before judging it.
- Confirmed target handling: when the target is visually confirmed, stop broad
  search and let the current phase and phase judge decide whether to align,
  continue, advance, or complete.

Do not approach an uncertain target as if it were confirmed. Do not treat map
memory hints, old observations, or artifact paths as current visual confirmation.
Do not treat weak candidate evidence as task completion.
