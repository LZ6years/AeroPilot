---
name: task_completion_evidence
description: Use when deciding whether a UAV task or semantic phase is truly complete from observations, actions, and confidence.
---

# Task Completion Evidence

Use this guidance when the agent must decide whether to finish a phase, continue,
or collect more evidence. Completion should be grounded in the task semantics and
latest observations, not merely in the fact that an action was executed.

For visual tasks, completion usually requires current evidence that the target is
visible, aligned, observed, reached, or safely related to the UAV as requested.
For search tasks, completion requires target confirmation or a clear reason that
search cannot progress. For approach and standoff tasks, completion requires
evidence about relative position or distance, not just a turn.

Use recent actions as supporting context: they can explain how the state may have
changed, but they are not proof by themselves. When evidence is weak, prefer
continuing with a targeted observation or conservative action rather than
declaring success.

Do not let hidden benchmark fields, reference paths, raw file names, or old
observations count as completion evidence. Do not require perfect certainty; use
task-appropriate confidence and explain uncertainty when it remains.
