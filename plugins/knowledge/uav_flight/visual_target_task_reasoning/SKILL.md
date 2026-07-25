---
name: visual_target_task_reasoning
description: Use when a UAV task depends on finding, confirming, seeing, facing, observing, centering, or keeping attention on a visual target.
summary: Select for visual target search, confirmation, facing, centering, observation, or attention tasks.
---

# Visual Target Task Reasoning

## Application Boundary

Use this guidance for finding, confirming, facing, inspecting, or attending to a
visual target. Match evidence to the objective: facing needs identity and a
usable bearing; inspection may need requested detail. Visual alignment cannot
prove route, distance, orbit, or landing progress. In a compound task, target
confirmation may only establish a cue for later movement.

## Evidence Model

Classify the latest evidence as target absent, weak candidate, directional
target cue, or inspection-ready. A directional cue identifies the target and
gives a usable image bearing even when nonessential detail is hidden.
Frame-boundary clipping can make bearing unreliable; incidental occlusion does
not when identity and bearing remain clear. Motion history can reveal
oscillation but cannot create visibility or target-relative bearing.

## Decision Procedure

1. Classify the latest evidence before acting.
2. For target absent, search within a bound. For a weak candidate, hover or make
   a small confirming adjustment. Request
   [search guidance](references/search_under_uncertainty.md) when search scale
   needs more detail.
3. For facing, correct only when the cue is near the frame boundary or too
   ambiguous for a usable bearing. Do not demand inspection quality.
4. Re-observe after correction; stop alternating or unhelpful corrections.

## Completion And Stop

For facing, an identifiable cue inside the forward view with usable bearing is
sufficient. Modest offset, incidental occlusion, and missing nonessential
detail are acceptable. Exact centering, full visibility, or confirmation hover
apply only when explicitly requested. When vision only enabled movement, use
the movement objective's completion evidence instead of requiring continuous
visibility.

## Recovery

If identity or bearing is ambiguous, hover or use a bounded correction. Re-check
yaw that conflicts with the stated image side. Do not restart broad search for
modest offset or incidental occlusion after establishing a usable cue.
