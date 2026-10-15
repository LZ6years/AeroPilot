---
name: save_observed_place
description: "Actively remember key landmarks, junctions, informative revisits and search findings at a numbered visited UAV site. Saves task-local notes and photo references without moving."
---

# Save observed place

Use the automatic map's place IDs to retain useful spatial memory:

- Save recognizable landmarks at the start or along the route.
- At an important junction or observation transition, retain useful findings
  before leaving the place.
- Update a revisited place when new evidence adds information or corrects a note.
- After an unproductive search, record what was inspected and what remains
  uncertain, so the same search need not be repeated without a reason.

Ordinary movement and unchanged views do not require a save. Check the latest
map and successful save result before writing again; older checkpoints describe
the state before that save. Saving is one separate tool turn, without movement.

Keep the summary to one or two sentences: observed appearance, useful findings
and uncertainty. Actual actions are already logged. Landmarks are recognition
clues attached to a UAV observation site, not measured building coordinates.
Reference supporting observations or environmental actions, and select at most
two original photos from that site's visits. Do not invent features to fill an
empty field. Saved notes do not certify route completion or replace the plan.

Save useful spatial memory at an existing visited-place ID. Proactively annotate recognizable landmarks at the start or along the route, important junctions or viewpoint transitions before leaving, revisits that add or correct information, and unsuccessful searches worth remembering. Record observed features, findings and uncertainty so you can recognize the place and avoid repeating searches. Ordinary movement or unchanged views do not need another save. Consult the latest map and successful save result: an older checkpoint may still show the pre-save state. Update a saved place only when useful information changes. Returns the saved notes, annotation status and updated map immediately; no flight motion. Notes are model interpretations, not task-completion evidence or a replacement for the current plan. First save adds the place to list_places. Subsequent saves replace the entire summary, landmarks and evidence_refs; include still-useful earlier information. Empty photo_observation_ids retains existing photos. Use list_places to recover saved IDs and inspect_place for all legal photo choices.
