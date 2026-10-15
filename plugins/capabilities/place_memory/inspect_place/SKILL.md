---
name: inspect_observed_place
description: "Inspect a numbered visited site from the automatic map, including its saved landmarks, actual actions and up to two historical photos."
---

# Inspect observed place

Historical photos can help recognize revisits. They do not show the present
camera view or establish current clearance. Use current observations for motion.

Retrieve old original photos, complete notes and historical UAV poses no longer retained in the current context, to compare earlier views with the current judgment. Use a known P-ID when short map labels and available history are insufficient for a revisit, search review or correction. Use list_places only when a saved ID is unknown. Returns notes, annotation status, original evidence references, recent measured actions, all available photo observation IDs and up to two original HISTORICAL photos with timestamps, historical map-relative positions and UAV headings relative to the initial heading (not camera optical axes). Unknown historical pose fields are null. Known unsaved map sites may also be inspected and are marked unannotated. Task-local only; no movement or new environmental observation. Use current views for present clearance.
