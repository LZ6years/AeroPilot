---
name: visual_grounding_measure_visible_depth
description: "Applicable needs: measuring an observed object surface; Decision questions: whether valid range supports the current decision; Contents: candidate locations, evidence images and distance validity; Limits: does not establish identity, route relations or completion."
---

# Visible Surface Measurement

Use `skill.visual_grounding.measure_visible_depth(observation_id, query)` when distance to a
candidate already observed in the current RGB can inform a decision. Supply the
CURRENT observation ID and a concise category and observed appearance. Do not
copy an unseen target or a hoped-for task relation into the query. The image must
already provide direct evidence of the candidate. Never call this capability to
discover, search for, or test whether an instruction object is present.

The service locates candidates and attempts metric depth for each. Inspect the
associated annotated evidence to determine which candidate was measured. A label
or confidence score does not confirm task identity. Multiple candidates remain
separate; do not select a task target solely by confidence or available range.

`distance_m` is robust camera-to-visible-surface depth, not horizontal distance,
target-center distance or distance to a route node. A numeric claim requires a
valid measurement with geometry appropriate to the claim. Neither approaching
nor finishing a task imposes a mandatory measurement call.

Two independent queries may share a read-only turn within the declared quota.
Wait for results before dependent decisions and reuse applicable results.

An empty candidate list is a failed measurement and leaves presence uncertain. `metric_depth_unavailable`
means metric depth could not be used; `inconsistent_depth_surfaces` means the
candidate region cannot support a reliable scalar depth, including extended or
sloped surfaces. Other invalid-distance statuses retain their stated reason.
Execution success does not imply measurement validity. A failed call supplies no
negative scene evidence. Do not repeat or reword a query merely to obtain a
preferred answer.
