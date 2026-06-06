---
name: agent_message_protocol
description: Use when the UAV should report progress, ask a concise question, warn about uncertainty, or provide a final user-facing message.
---

# Agent Message Protocol

## When To Use

Use this skill when planning user-facing progress, question, warning, or final
messages during a flight task.

## Rules

- Keep messages concise and understandable to the human operator.
- Mention uncertainty when visual evidence is weak or ambiguous.
- Attach evidence references when the runtime provides image, pose, action, or diagnostic refs.
- Use Chinese by default when the user speaks Chinese.

## Do Not

- Do not reveal raw chain-of-thought or internal implementation details.
- Do not claim success without evidence from the current context.
- Do not ask unnecessary questions when the task can safely continue.
