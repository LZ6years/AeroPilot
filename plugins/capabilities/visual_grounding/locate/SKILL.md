---
name: visual_grounding_locate
description: Locate a natural-language visual target in the latest RGB observation and optionally estimate metric distance to each candidate.
---

# Visual Grounding

First interpret the current image directly. Use `skill.visual_grounding.locate`
only when decision-relevant localization or identity remains ambiguous, or a
metric target-relative distance is genuinely decision-critical. Do not outsource
general street/building layout or already clear visibility to detection. These
are examples of reasoning boundaries, not forbidden object categories.
Pass the exact CURRENT observation ID and a concise visual query. Use
`include_distance=false` for unresolved localization. Use `include_distance=true`
only near a metric threshold, when action
magnitude depends on that threshold, when the user asks for a measured distance,
or before supporting a quantitative completion claim.

Two independent queries may be submitted in the same read-only turn, within
the per-observation quota. Runtime may execute them sequentially and return
both results before the next decision. A query that depends on another result
must wait for it. Reuse existing results instead of repeating identical calls.

Candidates are detector evidence, not guaranteed task identity. Inspect the
annotated tool-evidence image and keep the relation unknown when multiple
candidates cannot be distinguished. `distance_m` is robust camera-to-visible-
surface depth for that candidate, not target-center or horizontal distance.
