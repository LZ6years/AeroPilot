---
name: visual_grounding_measure_visible_depth
description: "Applicable needs: measuring an observed object surface; Decision questions: whether valid range supports the current decision; Contents: candidate locations, evidence images and distance validity; Limits: does not establish identity, route relations or completion."
---

# Visible Surface Measurement

Use `skill.visual_grounding.measure_visible_depth(observation_id, query)` when distance to a
candidate already observed in the current RGB can inform a decision. Supply the
CURRENT observation ID and a concise visible category and appearance. Candidate
discovery and identity remain visual judgments; this capability supplies locations,
an annotated image and attempted visible-surface depth for matching candidates.
Spatial words in query do not constrain an image region. Compare returned boxes
and appearance with the intended candidate; a location conflict means identity
is unresolved, even if a returned distance is valid.

The service locates candidates and attempts metric depth for each. Use the associated
annotated evidence to determine which candidate was measured. Multiple candidates
remain separate model choices rather than being merged by confidence or range.

`distance_m` is robust camera-to-visible-surface depth, not horizontal distance,
target-center distance or distance to a route node. A numeric claim requires a
valid measurement with geometry appropriate to the claim. Neither approaching
nor finishing a task imposes a mandatory measurement call.

Two independent queries may share a read-only turn within the declared quota.
Wait for results before dependent decisions and reuse applicable results.

An empty candidate list means that no measurement was produced. `metric_depth_unavailable`
means metric depth could not be used; `inconsistent_depth_surfaces` means the
candidate region cannot support a reliable scalar depth, including extended or
sloped surfaces. Other invalid-distance statuses retain their stated reason.
Execution success and measurement validity remain separate result fields.
`measurement_status` is `no_candidates`, `no_valid_distance` or `distance_available`;
`valid_distance_count` counts only candidates with valid scalar depth. A zero count
does not establish object absence or justify a numeric distance claim.
