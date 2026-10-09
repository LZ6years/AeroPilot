---
name: visual_target_task_reasoning
description: "Applicable needs: finding, identifying or facing a visual target; Decision questions: choose search direction, disambiguate candidates and decide when visibility is adequate; Contents: directional cues, viewpoint changes, candidate confirmation and alignment; Limits: does not establish passage, metric range or arrival."
summary: Select for visual target search, confirmation, facing, centering, observation, or attention tasks.
---

# Visual Target Task Reasoning

## Application Boundary

Use for search, identification, facing and inspection. Facing needs identity and
bearing; inspection needs detail. Alignment does not establish passage or arrival.

## Evidence Model

Distinguish a candidate outside the current view, a tentative candidate, a directional
cue and an identifiable target. Choose viewpoints that separate competing identities.
Discover and identify candidates from CURRENT RGB; surface depth measures a selected
visible candidate. Current bearing supports alignment, and metric claims use matching
geometry.

## Decision Procedure

1. Choose a viewpoint that expands useful coverage. Turning changes bearing;
   translation, ascent or lateral movement can reveal areas hidden from a stationary
   view. See [search guidance](references/search_under_uncertainty.md).
2. Interpret directional cues in their stated reference frame. A direction anchored
   to the starting heading remains tied to that frame rather than the current heading,
   but does not prescribe a later motion sequence. Choose the next viewpoint from
   current coverage, scene cues and accumulated measured motion.
3. Investigate when a new view can resolve identity. Approach, shift, clear an
   occlusion or turn according to the viewpoint needed. Inspection and arrival remain
   separate outcomes.
4. For facing, correct when clipping, imminent loss or ambiguity prevents usable
   bearing. Do not correct merely for lateral offset or alternate around an already
   usable bearing. Preserve explicit angles; otherwise scale turns to visual purpose.

## Completion And Stop

For facing, an identifiable forward cue with usable bearing suffices. Moderate
left or right offset is explicitly acceptable; exact centering needs an explicit
request. Identification alone does not require approach; judge any separate
movement relation independently.

## Recovery

Contradicted attributes can exclude a candidate and reopen search. If no detail
appears, compare actual motion, viewpoint and occlusion, then choose a pattern that
adds different coverage. Temporary occlusion need not invalidate identity.
