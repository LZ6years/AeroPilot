---
name: flight_experience_search
description: "适用需求：当前飞行决策需要历史指导／解决问题：查找相关条件经验／内容范围：少量建议、适用条件、局限和版本引用／能力边界：历史指导不能证明当前环境或任务完成。"
---

# Conditional flight experience search

Use search when historical guidance can help the current decision. Describe the
decision problem and observed conditions, not a desired answer. No mandatory
lookup sequence applies. Results are conditional historical advice; evaluate
their limitations against current observations. Empty results are normal.

An experience ID is not current observation evidence. It cannot establish target
identity, distance, route completion or task success. This capability does not
write experience, execute movement, return historical RGB or start reflection.

The plugin is disabled in the default configuration. Its database path and use
scope are administrator settings, never model arguments. Experimental databases
must not be used to train behavior on benchmark answers or test-set experience.
