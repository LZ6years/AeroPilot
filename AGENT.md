# UAV-Agent Development Boundaries

This file records the project-level working agreement for future Codex work in
this repository. Keep changes aligned with these boundaries before adding new
logic.

## Responsibility Boundaries

- `src/memory/context/modules/*.py` builds compact model-facing context from
  runtime memory. It owns field names, compression, role labels, and the split
  between model-facing summaries and debug metadata. An enabled module may
  declare a short `semantic_evidence` capability list through
  `describe_output()`. These declarations describe configured evidence, not
  current observations.
- `src/agent/planning/prompts/*.txt` defines general planner and judge
  contracts. Prompts may describe how to use available evidence, but should not
  duplicate module internals or become a case-specific controller.
- `coupled_phase_strategy_system.txt` owns the compact, revisable maneuver
  strategy between immutable phase semantics and one-window tool actions. It
  must not redefine completion or contain benchmark-specific routes.
- `memory/FLYAGENT.md` owns only the shared embodied Agent identity, general
  work cycle, direction consistency, and safety/reporting baseline. It must not
  enumerate configured sensors, temporal image roles, task-specific completion
  rules, or project-private fields. `memory/USER.md` owns ordinary human spatial
  language and precision preferences. Load both as separate model messages
  instead of copying their guidance into parser prompts.
- `src/agent/semantic/prompts/task_parser_structure_system.txt` owns only task
  intent, task goal, and immutable phase structure. Phase count, order, name,
  and goal come from the user instruction before any skill is selected.
- `src/agent/semantic/prompts/done_when_generation_system.txt` owns the general
  quality rules for a stable, minimal, judgeable completed state. It contains no
  task-domain tutorial. `task_parser_phase_criteria_system.txt` only assembles
  the per-phase `done_when` and `continue_when` JSON contract from
  that shared rule, selected domain knowledge, and configured evidence.
- `plugins/knowledge/*/SKILL.md` contains reusable domain knowledge. Skills must
  not depend on local context field names, debug schemas, benchmark cases, or
  project-private runtime details.
- `src/runtime/*` coordinates execution, observations, phase anchors, and debug
  artifacts. Runtime should not silently encode semantic task fixes that belong
  in context contracts, prompts, skills, validators, or future semantic update
  middleware.
- `evaluation/*` measures behavior and renders review/debug reports. Evaluation
  code may expose raw audit data for people, but benchmark-specific discoveries
  should become generic runtime/context/prompt improvements before affecting
  normal behavior.

## Benchmark Generalization Boundary

Benchmarks are diagnostic probes, not product specifications. A benchmark case
may reveal a weakness in the general UAV agent, but it must not directly decide
parser, planner, runtime, context, safety, skill, or prompt behavior.

Do not put benchmark case names, benchmark classes, reference paths, SR judge
results, nDTW scores, hidden truth, leaderboard heuristics, or test-set
experience into model-facing prompts, reusable skills, context contracts, or
runtime behavior. Evaluation-only facts must stay in `evaluation/*`, reports,
and human-facing debug artifacts.

Before changing normal behavior based on a benchmark result, restate the issue
as a general UAV capability gap. Examples include target-relative spatial
relations, around-target progress evidence, uncertainty-aware visual movement,
history summarization, or safe completion judgment. If a proposed change only
helps one benchmark and cannot be justified as a general user-facing UAV
capability, do not make it in the shared runtime, prompts, or knowledge base.

## Model-Facing vs Human-Facing Data

Model-facing context should be small, current, and decision useful:
- current instruction and semantic phase;
- one current observation image, preferring the aligned RGB-depth overlay when
  configured and valid;
- one raw phase-start reference image when distinct;
- one compact motion history summary containing phase-start-relative UAV motion
  and a short oldest-to-newest successful-action sequence;
- compact phase completion evidence.

Human/debug/review data may be larger and more detailed:
- raw planner context and raw model outputs;
- full action logs and tool sequences;
- image paths, artifact refs, dimensions, and source metadata;
- safety internals, budgets, loop guards, and diagnostics;
- provenance notes explaining how model-facing summaries were built.

Do not move debug-only fields into model-facing context just because they are
useful for diagnosis. Convert them into compact, task-relevant summaries first.

## Prompt And Skill Writing Rules

Reusable skills are flight-domain knowledge, not runtime documentation. Do not
reference project-private fields such as `current_phase`, `phase_decision`,
`phase_progress_evidence`, `model_inputs`, debug metadata, report fields, or
benchmark artifacts from `plugins/knowledge/*/SKILL.md`. Phrase skill guidance
as general UAV behavior, visual reasoning, movement, route, safety, or
completion knowledge.

Semantic completion uses one shared contract across criteria generation,
semantic updates, and phase decisions. The shared done_when prompt owns
minimum sufficiency, stable completed-state wording, condition quality, and
grouping. Criteria and updater prompts own only their output shape and numeric
provenance in natural-language conditions; they must not duplicate generic done_when rules or contain visual,
route, orbit, landing, or other task-domain tutorials. Knowledge skills own
task evidence and behavior semantics, but must not teach `done_when`,
`continue_when`, schema fields, or list organization. The semantic updater
reuses the triggering planner window's selected core skill guidance; it does
not run another selection pass or load skill references.
Every explicit terminal result in an immutable phase goal must remain an
already-achieved result in `done_when`; criteria and updater may not weaken it
into proximity, feasibility, preparation, or readiness for a later action.

Semantic evidence capabilities are rediscovered from enabled context modules
before each model task parse. The structure parser must not receive them because they do
not decide task or phase structure. Per-phase criteria generation and semantic
updates may receive the same bounded capability list after selected skill
guidance and before their output contract. They must treat it as potential
evidence availability, not observed truth; must not turn RGB, depth, pose, map,
or action-history source names into terminal obligations; and must not require
hidden geometry or unavailable exact target-relative measurements. This
capability metadata does not add fields to `SemanticPlan`, and validators must
not parse modality names from natural-language conditions.

Validators enforce structure, numeric provenance, and explicit contracts. They
must not infer AND/OR semantics from natural-language connective words such as
`or` or `alternatively`. Tests should lock responsibility boundaries and
observable contracts rather than exact long prompt sentences that duplicate
knowledge or preserve contradictory wording.

Phase-start-relative cumulative motion describes the whole phase; a short
recent-action tail must not override it. The coupled model classifies every
`done_when` item as `supported`, `unknown`, or `contradicted`; those judgments
alone derive the semantic phase transition. Numeric requirements stay in the
natural-language condition and are judged from the evidence available at the
requested precision. Free-text evidence is explanatory and must not be parsed
as control logic. The phase-decision prompt must not duplicate separate visual,
route, orbit, or landing tutorials.

Within coupled planning, the phase-decision prompt owns only completion-evidence
classification. The phase-strategy prompt owns the current phase's short
ordered maneuver stages and active stage. The action prompt owns tool choice,
signed argument use, action-window consistency, and runtime-control output.
Stable primitive argument
semantics belong to the canonical UAV tool contract and enter model context
through action schemas; prompts and knowledge skills must not maintain duplicate
sign conventions.

Phase strategy is mutable Runtime state, not part of `SemanticPlan` and not a
phase-completion authority. Runtime owns its revision and update classification;
the model sees only the latest compact snapshot through the action context.
An intermediate strategy stage may advance while phase completion remains
unknown. Strategy updates must assess the active stage's own observable cue
before selecting the next action; they must not use the final `done_when` result
as the stage-advance gate.

`last_window_uav_motion` means measured UAV self-motion during the immediately
preceding completed action window. A planning-time refresh with no new window
measurement preserves that snapshot; only an explicitly measured zero-motion
window replaces it with zeros.

Knowledge skills under `plugins/knowledge/*/SKILL.md` use one compact authoring
contract in this order: `Application Boundary`, `Evidence Model`, `Decision
Procedure`, `Completion And Stop`, and `Recovery`. This convention applies to
knowledge guidance only; callable skills may use the structure appropriate to
their external capability. Runtime treats these headings as ordinary Markdown
and does not parse or route sections independently.

Keep each core knowledge body within the current 2400-character guidance
budget. The core must retain the evidence and completion knowledge needed by
semantic criteria generation. Move formulas, long examples, search scales, and
optional action detail to one-level `references/*.md` files, and state clearly
when a planner should request each reference. Do not move essential completion
conditions into references because parser-time criteria selection does not load
them.

When two knowledge files describe one continuous reasoning chain, prefer one
primary skill plus optional references over multiple overlapping skills. For
example, visual target search, candidate confirmation, and target alignment
belong to one visual-target skill; detailed search patterns can live in that
skill's references and load only when selected.

Planner and parser prompts may reference the model contract they own, but they
should not reintroduce obsolete fields or teach reusable skills about internal
schemas. Keep old debug, evaluation, and report concepts out of model-facing
prompt examples unless that prompt explicitly owns the field.

Model task parsing uses two distinct stages. The structure stage sees no skill
body. After structure is validated, each phase independently selects all and
only the knowledge skills relevant to its completion, continuation, or evidence
criteria. There is no runtime hard limit on the number selected, but
the selector should choose the smallest genuinely relevant set. Parser-time
selection loads bounded `SKILL.md` knowledge only: no references, source paths,
or callable capabilities.

Parser-time and planner-time skill selection are independent. Parser selection
helps define a phase's semantic criteria; planner selection helps choose the
next action from current runtime evidence. Do not carry one stage's selected
skill list into the other as an implicit decision or let a skill determine
phase count, order, name, or goal.

Do not restore model-facing `direction` shorthand for movement. UAV movement
plans use numeric velocity arguments, while semantic direction and distance
requirements stay in phase goals and natural-language completion conditions.

Do not add prompt or skill warnings about data that should never be provided to
the model, such as hidden benchmark truth. Enforce that boundary in context and
evaluation plumbing instead of asking the model to ignore unavailable fields.

## Change Discipline

Before changing behavior, identify which layer owns the behavior. Do not put a
runtime/context field name into a reusable plugin. Do not put benchmark-specific
repairs into general planner prompts. Do not make SafetySupervisor solve semantic
or visual reasoning problems unless the issue is truly safety policy.

When replacing old behavior, clean the whole affected path:
- remove or update stale prompt text;
- update context schema descriptions;
- update plugin guidance that teaches the old behavior;
- update tests that preserve obsolete assumptions;
- check review/debug rendering so it shows the true model-facing context;
- document any compatibility path that must remain.

Every fix should report what changed, what old logic was removed or kept, how it
was tested, and what remains for later semantic update or image-memory work.
