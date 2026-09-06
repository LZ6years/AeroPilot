---
name: person_detector
description: Use when the UAV needs to confirm whether a person is visible in the latest RGB observation before moving or aligning.
---

# Person Detector

Use this callable skill when the current instruction depends on a visible person and the available image context is uncertain.

The skill provides a controlled external capability named `skill.person_detector.detect`.
Call it before moving toward a person when the planner needs explicit target confirmation.
Call it only when `skill.person_detector.detect` is present in the provider's
available tools, and pass the exact `observation_id` shown beside the image.

Do not treat this skill text as a detection result. Only the returned capability result is runtime evidence.
