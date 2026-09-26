# UAV-Agent Development Boundaries

This file records the project-level working agreement for future Codex work in
this repository. Keep changes aligned with these boundaries before adding new
logic.

## Responsibility Boundaries

- `src/llm/` owns stateless model communication, wire adapters, HTTP retries
  and generic in-memory request audits. It must not import Agent, Runtime,
  Context or plugins. ModelConfig converts only transport settings into
  ProviderSettings; business loop limits and prompt paths stay upstream.
- Model callers own prompt loading, message order, generation parameters and
  response semantics. MessageBuilder is optional ordered-copy tooling with
  caller-defined keys, not a recipe/profile registry. It never inserts prompts,
  samples history, merges system messages or edits durable Transcript events.
- Communication audits separate image bytes without interpreting evidence.
  CURRENT/HISTORY labels and observation associations remain Agent-owned.
  Runtime/report callers own persistence. No hidden provider reasoning enters
  audit snapshots. The legacy Agent HTTP implementation has been removed;
  all production model callers use the shared client/injection contract.

- `src/memory/context/modules/*.py` builds compact model-facing context from
  runtime memory. It owns field names, compression, role labels, and the split
  between model-facing summaries and debug metadata. An enabled module may
  declare a short `semantic_evidence` capability list through
  `describe_output()`. These declarations describe configured evidence, not
  current observations.
- `src/agent/planning/prompts/*.txt` defines general planner and judge
  contracts. Prompts may describe how to use available evidence, but should not
  duplicate module internals or become a case-specific controller.
- `src/agent/transcript.py` owns chronological model intent, native tool calls,
  tool results, protocol feedback, and state checkpoints for the active
  instruction. Context modules must not duplicate that history.
- `agent_tool_loop_system.txt` owns task goals, observation interpretation and
  the feedback loop. `tool_user_system.txt` explains generic parameter, execution
  and result semantics without naming tools or prescribing a selection order.
  `tool_use_case_studies.txt` contains illustrative syntax/dependency examples,
  not executable routes. Tool descriptions own applicability; complex tasks
  should normally start with a plan, while simple tasks may act directly.
- Model-visible descriptions follow purpose, applicable need, returned content,
  and capability limits. Parameter prose explains meaning, units and effects,
  not tool-selection priority. The complex-task planning preference is explicit;
  other descriptions must not introduce an unrelated first/last-resort ordering.
  Shared call rules belong in ToolUser; per-tool execution attributes come from
  ToolRuntimePolicy. Knowledge explains evidence and flight methods without
  prescribing hidden actions or duplicating a capability's selection policy.
- `src/tools/schemas.py` owns ToolParameter, ToolRuntimePolicy, ToolSpec and
  ToolCall. ToolSpec generates input_schema from parameters and object options;
  plugin JSON Schema is adapted at registration and examples are validated there.
  model_specs always intersects allowed names with runtime.model_exposed.
  Hover remains Runtime-only, including safety substitution and stabilization.
  ContextBuilder emits state/evidence, never tool declarations. Provider tools
  is the only declaration path; HistoryProjector removes legacy provider-only
  state fields without deleting historical calls/results or changing storage.
- `memory/FLYAGENT.md` owns only the shared embodied Agent identity, general
  work cycle, direction consistency, and safety/reporting baseline. It must not
  enumerate configured sensors, temporal image roles, task-specific completion
  rules, or project-private fields. `memory/USER.md` owns ordinary human spatial
  language and precision preferences. Load both as separate model messages
  instead of copying their guidance into parser prompts.
- `plugins/knowledge/*/SKILL.md` contains reusable domain knowledge. Skills must
  not depend on local context field names, debug schemas, benchmark cases, or
  project-private runtime details.
- `src/runtime/*` coordinates execution, observations, instruction anchors, and debug
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

Model-facing state checkpoints should be current and decision useful:
- original instruction, caller-owned protocol_context and optional task plan;
- one current original RGB observation image, preserving scene colors;
- one compact motion summary containing only measured instruction-start-relative and
  last-window UAV self-motion;
- knowledge catalog and loaded index, never duplicate knowledge bodies.

The provider receives every checkpoint's state text together with the complete
active-task assistant/tool transcript in chronological order. Observation images
may be sampled by an explicit `context_history` policy in `configs/plugins.json`.
The default experiment caps observation images at 20 and retains the latest 3
consecutively, with log-spaced older frames including the first. Tool evidence
images are additional; this is not a total image or token budget.
Each selected observation image is sent once at its original checkpoint;
earlier labels say `HISTORY`, and only the newest says `CURRENT`.
Labels carry `observation_id`, `frame_id`, and scene dimensions outside the
image pixels. Tool results pair to calls through
`tool_call_id`. Tool availability, argument schemas, runtime effects, result
timing, and batching come from `ToolSpec` plus the provider `tools` array, not
from duplicate context fields or name-prefix inference. Version 1 performs no
application-level token budgeting, text summarization, or silent history truncation.
The transcript does not replay phase-start or before-window images. Runtime may
retain those observation IDs as audit/state anchors, while full image history
comes from the chronological checkpoints and remains fully stored even when
the request view omits images.

`src/agent/history_projection.py` owns the request view, not durable events.
Transcript snapshots are deep-copy isolated. Context modules may provide pure
history-image selectors through the base interface: candidates contain only
observation IDs, order and frame IDs. The Image module owns `recency_log`;
Registry validates JSON and binds the actual enabled instance; Runtime only
injects these bindings into ModelActionPlanner and persists selection audit.
Do not import agent/transcript types in module selectors, access files there,
or let selectors mutate state or execute tools. Unknown configured modes fail
initialization. Missing history configuration means `all`. Legacy
`AgentTranscript.provider_messages()` delegates to the same full-history
projector; standalone planners likewise default to full history unless injected.
Historical state must never be overwritten with future state. Reports separate
complete history, captured requests and explicitly labeled offline simulations.

`context_options` in `configs/plugins.json` is routed generically to enabled
modules through `configure(options)`. Modules validate their own options;
unknown or disabled module options fail initialization. Image defaults to
`include_current_depth=false`; explicit true retains the experimental aligned
overlay and meter color bar. This option does not disable Depth acquisition or
grounding distance tools. Historical captured inputs must never be regenerated
as if a new image mode had been used in an old request.

The optional fused color bar supports qualitative range only. Precise target-relative
evidence comes from an enabled callable capability rather than model-authored
pixel coordinates. The default `skill.visual_grounding.locate` capability is
selected through `configs/plugins.json`; its ToolSpec and private detector,
segmenter, depth, and threshold settings live in the capability's own JSON.
Runtime and plugin discovery must not hardcode that plugin ID or its model.

Visual grounding accepts the latest observation plus a natural-language query.
Detection-only calls establish candidates without loading metric depth. Calls
with distance enabled compute rounded robust camera-to-visible-surface depth
inside a detector/segmenter-derived candidate region. Candidate masks, raw
medians, MAD/filter details, backend settings, and model paths remain durable
audit data and are not replayed to the provider. Candidate tool evidence is not
a new observation and does not change `CURRENT`; confidence alone is not proof
that a candidate matches the instruction.

Capabilities may declare a generic per-observation call limit through ToolSpec.
Past distance-tool use does not impose a final-observation re-query.
Evidence precision and geometric meaning must support the requested judgment;
this does not impose an order between direct visual interpretation and callable
localization. Image labels explain the sensor representation, not tool choice.
ToolSpec projects public execution constraints into provider tool descriptions;
private plugin settings remain handler-only. `parallel_safe` permits independent
read-only calls in the same turn, without requiring concurrent backend execution.
Rejected turns identify the source call when known, distinguish affected calls,
and retain terminal rejected results for every submitted call.
Review pages may repeat links to an observation image without changing provider
image deduplication. Requested, submitted, and actually applied strategies must
remain separate; missing records must never be reconstructed from future state.

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

There is one ToolRegistry for builtins, UAV tools and JSON-discovered capabilities.
TaskRuntime.run_task accepts raw instructions and optional protocol_context.
No parser, automatic skill selector, fixed Phase, commit or semantic updater runs.
MissionState owns one optional plan with stable IDs and revisions. Plans are not
environmental evidence and cannot automatically complete an instruction.

KnowledgeService snapshots enabled knowledge per instruction. knowledge.load
returns full body and requested declared references incrementally as ToolResults.
Context contains only catalog/loading indexes. Loading does not execute text,
register capabilities, or extend permissions. Invalid paths fail without mutation.

HistoryProjector owns the request view, not durable history. Context modules
build current state; Transcript retains full append-only intent/results. No
transport, task semantics or history selection policy belongs in knowledge.

Physical turns contain 1–3 ordered calls, read-only turns follow ToolSpec batching,
and each control call is exclusive. Safety reviews actual arguments. Each physical
batch has one automatic after observation; terminal refreshes are separate.
The instruction limit is 300 physical windows, with 10 nonphysical rounds and
6 consecutive protocol rejections by default. Plan changes cannot reset limits.

runtime.report_completed and runtime.report_blocked require environmental
evidence references, never plan or knowledge as proof. The Runtime closes results,
performs terminal stabilization/final observation, and then caller-owned cleanup.
Infrastructure errors remain separate from the model's terminal state.

Legacy Phase compatibility exists only at the report-reading boundary. Formal
Campaigns cannot merge loop versions. Reports show task, optional plan and
ID-linked Decisions; requested, submitted and applied plans stay distinct.

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

Keep core knowledge concise; long examples and optional detail belong in declared
references. KnowledgeService returns complete requested text without truncation.

When two knowledge files describe one continuous reasoning chain, prefer one
primary skill plus optional references over multiple overlapping skills. For
example, visual target search, candidate confirmation, and target alignment
belong to one visual-target skill; detailed search patterns can live in that
skill's references and load only when selected.

Planner and parser prompts may reference the model contract they own, but they
should not reintroduce obsolete fields or teach reusable skills about internal
schemas. Keep old debug, evaluation, and report concepts out of model-facing
prompt examples unless that prompt explicitly owns the field.

Do not restore model-facing `direction` shorthand for movement. UAV movement
plans use numeric velocity arguments, while semantic direction and distance
requirements stay in the original instruction and optional plan.

Do not add prompt or skill warnings about data that should never be provided to
the model, such as hidden benchmark truth. Enforce that boundary in context and
evaluation plumbing instead of asking the model to ignore unavailable fields.

## Change Discipline

Offline flight-experience experiments live in plugins/research/flight_experience.
They read saved evidence through their own adapters and use only src/llm; they
must not import Agent/Runtime or register flight tools. Experimental summaries
are guidance, never current environmental evidence.

Optional plans preserve prior work using stable IDs. Evidence, not previous intent,
supports progress; Runtime validates structure and references, not natural-language
route correctness. No additional semantic judge is introduced.

Visual grounding uses the capability JSON box threshold (currently 0.4). Empty
candidates mean uncertain detection, not confirmed absence; empty evidence images
are not emitted. Competing identity queries reuse the same observation quota.
Route and vertical knowledge favor evidence-scaled progress, not fixed small
steps or a fixed recovery-attempt limit. Safety limits are unchanged.

Review defaults to a compact plan/decision timeline with human-only trajectory
comparisons. Full requests and duplicate histories are available only through
explicit detailed-audit generation; raw audit files are preserved. The AerialVLN
landing protocol text is passed unchanged as caller-owned protocol_context;
this migration does not claim to solve its interpretation by the model.

Scene lifecycle ownership belongs to `src/sim_server`, not benchmark or model
policy. Register launch ownership before readiness; roll back backend clients,
owned processes and transports independently. AirSim wrapper/inner RPC release
is shared at the adapter boundary. Never infer process cleanup from an empty
error list or a dead launch-script parent: verify the owned group and endpoint.
Scene lifecycle timing and SSH logs are human diagnostics, never model context.
Longer startup ceilings require measured readiness/cleanup evidence, not a
benchmark-specific retry or silent relaunch.

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
