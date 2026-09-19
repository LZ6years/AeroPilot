# FlyAgent: Execution-Grounded Agent Loop

> CVPR paper blueprint and writing source of truth

## 1. Paper identity

### Working title

**FlyAgent: An Execution-Grounded Agent Loop for Training-Free Aerial Vision-and-Language Navigation**

### One-sentence claim

FlyAgent is a training-free aerial embodied agent that grounds every new
decision in a chronological record of observations, requested actions, actual
execution outcomes, tool evidence, and explicit task-completion conditions.

### Scope of generality

The paper claims reuse across heterogeneous aerial navigation tasks and
benchmarks. It does **not** claim that FlyAgent understands every possible
natural-language instruction. The same frozen MLLM, agent loop, primitive action
interface, and safety layer must be used across benchmarks; benchmark adapters
may translate data, coordinates, scene initialization, and metrics, but may not
inject task answers or benchmark-specific policies into the agent.

### Central research question

Can a frozen MLLM control an aerial robot more reliably across heterogeneous
navigation tasks when its decisions are grounded not only in the latest visual
observation, but also in what it attempted, what actually happened, which task
conditions remain unresolved, and which optional capability can reduce the
current uncertainty?

### Definition of reflection

In this paper, reflection is an externally auditable behavior rather than a
hidden chain of thought. A reflective transition occurs when new execution
evidence causes the agent to do at least one of the following:

1. revise or advance its current maneuver strategy;
2. select a different capability or issue a new evidence query;
3. change the sign, magnitude, or type of its next action;
4. revise a completion judgment from `unknown` or `contradicted` to
   `supported`, or refuse premature completion;
5. terminate as blocked when no safe, evidence-supported action remains.

## 2. Storyline

The paper follows one claim from motivation through experiments:

```text
Fixed or loosely coupled perception--planning pipelines
                         ↓
do not jointly represent intended action, actual effect,
tool evidence, and explicit task-completion state
                         ↓
errors can accumulate and the agent may repeat a failed strategy,
invoke unnecessary modules, or stop without sufficient evidence
                         ↓
Execution-Grounded Agent Loop
                         ↓
query evidence → commit a short action window → observe actual effect
→ assess completion → keep, advance, or revise the strategy
                         ↓
better task success, recovery behavior, and capability efficiency
from short-horizon UAV-Flow to long-horizon AerialVLN
```

The paper must not use benchmark integration as a standalone algorithmic
contribution. Cross-benchmark evaluation is evidence that the loop is reusable,
not the novelty itself.

## 3. Draft abstract

> Aerial vision-and-language navigation requires an embodied agent to jointly
> interpret natural-language instructions, perceive large-scale three-dimensional
> environments, and execute reliable flight actions. Existing aerial navigation
> agents commonly rely on fixed perception--planning components or repeatedly
> predict actions from the latest observation, making it difficult to jointly
> account for execution deviations, failed tool calls, unresolved completion
> conditions, and task-dependent capability use. We present **FlyAgent**, a
> training-free framework built around an **Execution-Grounded Agent Loop** that
> treats tool results, measured flight outcomes, and semantic completion evidence
> as first-class inputs to subsequent decisions. FlyAgent combines an explicit
> semantic task contract, a dual loop that separates evidence acquisition from
> short-horizon physical execution, and phase-adaptive routing of pluggable
> knowledge and perception capabilities. Experiments on UAV-Flow and AerialVLN,
> together with a cross-benchmark study on TravelUAV, show **[INSERT VERIFIED
> RESULTS]**, while recovery and efficiency analyses demonstrate **[INSERT
> VERIFIED RECOVERY AND LATENCY RESULTS]**.

Do not fill the bracketed claims until the corresponding result artifact and
evaluation protocol have been verified.

## 4. Introduction

### Paragraph 1: task and importance

Aerial vision-and-language navigation extends embodied language following to
continuous three-dimensional environments. Compared with ground navigation, an
aerial agent must reason about altitude, viewpoint changes, long travel
distances, ambiguous urban landmarks, and the physical consequences of every
control command. These properties make aerial navigation a useful setting for
studying whether multimodal foundation models can move from passive scene
understanding to reliable embodied interaction.

### Paragraph 2: progress in existing approaches

Prior work has progressed along three main directions. Learning-based methods
predict actions or waypoints from language and visual observations. Structured
methods such as STMR and CityNavAgent introduce semantic maps, hierarchical
planning, and global memory. More recent MLLM agents, including LookasideVLN,
improve zero-shot planning by aligning observations with structured spatial and
directional cues. These methods substantially improve what an aerial agent can
perceive or where it plans to move.

### Paragraph 3: common limitation

However, strong perception and planning do not by themselves close the
embodied execution loop. A flight command can be clipped, under-executed, or
produce an unexpected viewpoint; a perception tool can return ambiguous or
empty evidence; and a visually plausible state may still fail an instruction's
terminal relation. Representative aerial pipelines either employ a fixed set of
modules or replan from a new observation without explicitly coupling the
agent's previous intent, actual outcome, outstanding completion conditions, and
next capability choice. Consequently, an agent can repeat an ineffective
strategy, call expensive modules unnecessarily, or terminate based on intent
rather than achieved state.

### Paragraph 4: key insight and Figure 1

Our key insight is that history becomes useful for embodied control only when
it is organized as execution evidence. Instead of treating prior actions as a
textual summary, the agent should receive a chronological and attributable
record of what it requested, which tool or controller executed it, what result
was returned, how the UAV actually moved, and which semantic conditions remain
unsupported. This representation lets the same model decide whether to gather
evidence, continue a maneuver, revise its strategy, or stop.

### Paragraph 5: method overview

Based on this insight, we introduce FlyAgent, a training-free aerial agent
centered on an Execution-Grounded Agent Loop. FlyAgent first converts an
instruction into a semantic task contract with explicit, evidence-judgeable
completion conditions. Within each phase, an inner loop permits immediate
read-only capability calls and returns their results to the same chronological
transcript. An outer loop commits a bounded physical action window, applies
safety constraints, measures the realized motion, and captures a new
observation. The agent then reassesses every completion condition and keeps,
advances, or replans a compact maneuver strategy. Phase-conditioned capability
routing supplies only the knowledge and tools relevant to the current task
state.

### Paragraph 6: experimental claim

We evaluate FlyAgent from short-horizon language-conditioned flight to
long-horizon aerial navigation. UAV-Flow tests fine-grained motion semantics and
trajectory following across diverse action classes, while AerialVLN tests
sustained instruction grounding in seen and unseen city-scale scenes. A
TravelUAV study evaluates transfer to another task protocol. In addition to
official navigation metrics, we report controlled baselines, component
ablations, naturally occurring recovery transitions, tool use, latency, and
token cost. Replace this paragraph's qualitative claims with quantitative ones
only after all tables are frozen.

### Contributions

1. We introduce the **Execution-Grounded Agent Loop**, which represents tool
   results, realized flight effects, and explicit task-completion judgments as
   evidence for subsequent aerial navigation decisions.
2. We develop **FlyAgent**, a training-free framework that combines a semantic
   completion contract, a dual evidence--execution loop, and phase-adaptive
   capability routing to support auditable online correction with a frozen
   MLLM.
3. We provide a controlled evaluation from short-horizon UAV-Flow tasks to
   long-horizon AerialVLN, supplemented by TravelUAV transfer, and jointly
   measure navigation quality, recovery behavior, and runtime efficiency.

## 5. Related work

### 5.1 Aerial vision-and-language navigation

Organize the discussion by the source of structure introduced into the
navigation problem:

- **Action and waypoint learning:** AerialVLN and later VLA-style methods learn
  a direct mapping from observations and instructions to actions or waypoints.
- **Semantic maps and memory:** STMR represents a scene as an LLM-readable
  semantic-topo-metric map; CityNavAgent combines open-vocabulary perception,
  hierarchical semantic planning, and global topological memory.
- **Instruction-aware spatial reasoning:** LookasideVLN constructs a compact
  direction-aware graph from instruction-relevant landmarks and uses an MLLM
  to select paths.
- **Platforms and scale:** TravelUAV emphasizes realistic UAV control and
  target-oriented navigation, while OpenFly increases environment and renderer
  diversity.

Close the subsection with the precise gap:

> These methods improve action prediction, spatial representation, memory, or
> instruction alignment. FlyAgent instead studies how a frozen aerial agent
> should organize and act on execution feedback, explicit completion evidence,
> and optional capabilities within one reusable runtime loop.

Do not claim that these methods have no feedback. LookasideVLN, for example,
summarizes navigation progress at every step. The distinction is the joint,
auditable treatment of action intent, realized outcome, tool result, completion
state, and capability choice.

### 5.2 Feedback and reflective navigation agents

- **RILA** uses a reflective planner to discount unreliable perceptual
  descriptions in semantic audio-visual navigation, together with a fixed
  imaginative assistant and perception stack.
- **EvoNav** combines history experience with predicted future thoughts for
  zero-shot VLN decision making.
- **Slow4Fast-VLN** extracts generalized experience from a learned fast policy's
  memory using a slow reflective module.

Position FlyAgent as same-episode, execution-grounded online correction with a
frozen MLLM. Do not claim to have invented reflection or history-aware
navigation.

### 5.3 Tool-using embodied agents

ReAct interleaves language-model reasoning and actions; Inner Monologue feeds
environment and success feedback to embodied planners; SayCan selects among a
fixed repertoire of skills using language and affordance scores. These works
establish the general value of action--feedback loops and skill selection.
FlyAgent's contribution is their aerial instantiation as an explicit dual loop
with continuous flight measurements, semantic completion contracts, and
phase-adaptive capability routing, followed by evaluation on heterogeneous UAV
benchmarks.

### Comparison boundary

| Family | Primary mechanism | Feedback granularity | Capability use | FlyAgent distinction |
|---|---|---|---|---|
| STMR / CityNavAgent | Map, hierarchy, global memory | Iterative navigation state | Fixed method stack | Explicit execution and completion evidence |
| LookasideVLN | Direction-aware candidate paths | Observation and progress updates | Fixed ELG/SLKB/agent pipeline | Result-attributed tool and action loop |
| RILA | Reflective perception assessment | Per-step semantic exploration | Fixed perception and assistant modules | General completion contract and UAV tools |
| EvoNav | Historical experience and future thought | History/future reasoning | Fixed reasoning design | Same-episode realized action effects |
| Slow4Fast-VLN | Slow reflection improves fast policy | Cross-module memory interaction | Learned fast/slow modules | Frozen MLLM and no policy training |
| ReAct / Inner Monologue / SayCan | Reason--act feedback or skill selection | General embodied interaction | Tools or fixed skill repertoire | Continuous aerial execution plus benchmark validation |

## 6. Method

### 6.1 Problem formulation

Let (I) denote a natural-language instruction and
(o_t=(v_t,d_t,p_t)) the current RGB, depth, and UAV-pose observation. The
instruction is represented by a semantic task contract

\[
G=\{G_k\}_{k=1}^{K},\qquad G_k=(g_k,D_k,C_k),
\]

where (g_k) is the goal of phase (k), (D_k=\{d_k^j\}) is its set of
completion conditions, and (C_k) describes conditions under which safe
progress remains possible.

FlyAgent has access to a capability catalog

\[
\mathcal{C}=\{c_i\}_{i=1}^{N},\qquad
c_i=(n_i,\ell_i,\mathcal{X}_i,\rho_i,\mathcal{Y}_i),
\]

where (n_i) and (ell_i) are the capability name and natural-language
description, (mathcal{X}_i) is its input schema, (ho_i) specifies its
runtime effect and result timing, and (mathcal{Y}_i) is its returned evidence.

The chronological execution trace is

\[
\tau_t=[I,G,o_0,z_1,z_2,\ldots,z_t],
\]

where each (z_i) is an attributed observation checkpoint, capability call,
tool result, requested flight action, measured action outcome, or protocol
feedback event. At each reasoning round, the frozen MLLM selects

\[
\pi(I,G_k,\tau_t,\mathcal{C})\rightarrow
\{\operatorname{Query},\operatorname{CommitAction},
\operatorname{Advance},\operatorname{Blocked}\}.
\]

No benchmark identifier, reference trajectory, goal coordinate, or evaluator
decision is included in (I), (G), (	au_t), or (mathcal{C}).

### 6.2 Semantic task and completion contract

FlyAgent first decomposes (I) into ordered semantic phases. Phase structure is
fixed by user intent; optional domain knowledge may clarify evidence and
completion semantics but cannot add, remove, or reorder user-requested goals.

For every condition (d_k^j), the agent produces an evidence state

\[
e_t^j\in\{\texttt{supported},\texttt{unknown},
\texttt{contradicted}\}.
\]

A phase advances only when all entries are `supported`. An intended action is
never completion evidence. A quantitative condition requires measured evidence
at the requested precision; a visual relation may use credible visual or tool
evidence when the task does not demand an exact value. `unknown` keeps the phase
active while a safe information-gathering or physical action remains available.
`contradicted` requires strategy revision rather than silent phase completion.

### 6.3 Execution-grounded dual loop

#### Inner evidence loop

For a stationary observation (o_t), the model may issue one or more
parallel-safe, read-only capability calls. Each result is appended to
(	au_t) using a stable call identifier before the next model request. The
inner loop does not consume a physical flight window. Invalid schemas,
unavailable tools, repeated queries, and empty results are preserved as explicit
feedback so that the model can repair its next call or choose a different
source of evidence.

#### Phase commit

The inner loop ends when the model commits a phase turn. A valid commit contains
the complete evidence assessment (\{e_t^j\}), a compact mutable maneuver
strategy, and either no physical action when the phase is complete or a bounded
action window when it remains incomplete.

#### Outer physical loop

An incomplete phase may emit one to three ordered UAV primitives. A safety layer
validates and clips the requested motion before dispatch. After the batch, the
runtime waits for stabilization, records success or failure for every primitive,
measures realized translation and rotation, and captures (o_{t+1}). These
events are appended to (	au_{t+1}) before the next decision.

#### Strategy revision

Let (s_t) be the compact maneuver strategy active before an action window.
Given new evidence, the next strategy update is classified as

\[
u_{t+1}\in\{\texttt{keep},\texttt{advance},\texttt{replan}\}.
\]

This classification is runtime metadata derived from successive strategy
snapshots, not hidden model reasoning. A `replan` combined with an observed
change in capability or physical action after a failure is the primary unit in
the natural recovery analysis.

### 6.4 Phase-adaptive capability routing

Capability adaptation occurs at two levels:

1. **Knowledge routing.** At the start of a semantic phase, a model selector
   receives the phase goal, completion conditions, compact progress, and the
   available skill descriptions. It loads the smallest relevant set and caches
   it for that phase.
2. **Callable capability use.** During the inner loop, the main agent decides
   whether a tool result is needed and invokes only an advertised capability
   with schema-valid arguments.

The paper's public capability interface is:

```text
name: stable capability identifier
description: task-facing natural-language purpose
input_schema: typed model-call arguments
effect: read_only | physical | runtime_control
result_timing: immediate | after_batch_observation | async | user_reply
returned_evidence: compact model-facing result with durable audit provenance
```

The main capability-routing claim requires at least two independently optional
capabilities in the final evaluation. The default pair is visual grounding and
a semantic map/spatial representation capability. If only visual grounding is
available at submission time, rename this mechanism to **skill-conditioned tool
use** and remove all claims of multi-capability routing.

### 6.5 Benchmark-agnostic execution

Benchmark adapters are outside the agent policy. They may initialize scenes,
normalize coordinate frames, convert native observations into the common
RGB/depth/pose contract, translate canonical primitives to backend commands, and
compute official metrics. Evaluation truth remains private to the adapter and
metric suite. This separation is verified by an audit that rejects benchmark
case identifiers, reference paths, destinations, and evaluator outputs from
model-facing artifacts.

## 7. Figures

### Figure 1: paradigm comparison

**Left panel -- representative fixed-stack or direct-replanning agent**

```text
Instruction + Current Observation
              ↓
     Fixed Perception / Map
              ↓
          Planner
              ↓
           Action
              ↓
        Next Observation
```

Annotate a failed approach whose target leaves the field of view. Show that the
next prediction has a new image but no unified representation of requested
motion, realized motion, tool failure, and remaining completion evidence.

**Right panel -- FlyAgent**

```text
Instruction → Semantic Completion Contract
                         ↓
Current Observation → Evidence Transcript → Capability Choice
                         ↑                    ↓
Measured Motion ← Short Action Window ← Commit + Safety
       ↓
Completion Assessment → Keep / Advance / Replan
       └─────────────────────────────── loop
```

Show the same failed approach followed by an explicit `unknown` assessment, a
visual-grounding query or bounded search turn, target reacquisition, and
evidence-supported completion.

**Caption draft**

> Comparison between a representative fixed-stack/direct-replanning paradigm
> and FlyAgent. Existing aerial agents commonly predict a new action from the
> latest observation using a predefined perception and planning stack. FlyAgent
> instead records requested actions, realized flight effects, tool results, and
> completion evidence in an attributed transcript. This execution-grounded
> state allows the frozen MLLM to acquire missing evidence, revise a maneuver,
> or terminate based on achieved rather than intended state.

### Figure 2: method overview

Use three horizontal regions:

1. **Task contract:** instruction, semantic phases, completion conditions.
2. **Execution-grounded agent core:** chronological transcript, phase skill
   router, read-only tool loop, completion assessment, maneuver strategy.
3. **Embodied runtime:** safety, one-to-three primitive action window, simulator,
   stabilization, new RGB/depth/pose and measured motion.

Draw benchmark adapters as an outer boundary, not as inputs to the model.

### Figure 3: natural recovery trace

Use four chronological columns:

1. observation and intended action;
2. tool/action outcome exposing failure;
3. changed capability, strategy, or action;
4. final completion evidence.

Every claim in the figure must cite an observation ID, tool-call ID, or action
result from the exported audit trace.

## 8. Experiments

### 8.1 Evaluation progression

#### Main benchmark 1: UAV-Flow

Purpose: short-horizon, fine-grained instruction semantics and trajectory
execution.

Report Fixed and Open splits separately. Preserve the official category order:
Approach, Retreat, Pass, Land, Turn, Move, Shift, Rotate, Surround, and A/D.
Report official nDTW-compatible trajectory scores and semantic success rate as
separate measures. Do not merge them into an undocumented single score.

#### Main benchmark 2: AerialVLN / AerialVLN-S

Purpose: long-horizon instruction following, landmark tracking, accumulated
execution error, and seen/unseen generalization.

Report SR, OSR, SDTW, and NE for every supported Seen and Unseen split. State
whether the comparison uses AerialVLN or AerialVLN-S, because their published
results are not interchangeable.

#### Supplementary benchmark: TravelUAV

Purpose: transfer to a target-oriented benchmark and another simulator/task
protocol without modifying the agent core.

Report the official NE, SR, OSR, and SPL. Keep the complete result in the
supplementary material and include one compact transfer row or paragraph in the
main paper.

### 8.2 Controlled baselines

All controlled baselines use the same frozen `qwen3-vl-plus`, temperature,
episode set, observation modalities, primitive limits, safety rules, and maximum
decision windows.

| Baseline | Latest observation | Chronological action/tool results | Completion contract | Adaptive knowledge/capability use |
|---|---:|---:|---:|---:|
| Direct Action | Yes | No | No | No |
| Rolling Replan | Yes | Compact action summary only | No | No |
| ReAct-Style | Yes | Yes | No | Callable tools only |
| FlyAgent | Yes | Yes | Yes | Yes |

Published SOTA numbers belong in separate tables or visually separated table
blocks because backbone models, action spaces, inputs, and episode protocols may
differ.

### 8.3 Required ablations

| Variant | Removed mechanism | Question answered |
|---|---|---|
| No result feedback | Hide prior tool/action outcomes and retain the newest observation | Does execution evidence improve decisions? |
| No completion contract | Replace per-condition assessment with free-form stop prediction | Does the contract prevent premature or incorrect completion? |
| No strategy revision | Freeze the initial phase strategy | Does online revision recover from changed evidence? |
| Fixed-all capabilities | Load every available skill/capability every phase | Does routing improve success/cost over a large fixed context? |
| No inner tool loop | Permit no read-only query before the physical commit | Does evidence acquisition improve action choice? |

Do not combine several removals into one ablation row. Each row must modify only
the named mechanism.

### 8.4 Main result table shells

#### Table 1: UAV-Flow

| Method | Split | Approach | Retreat | Pass | Land | Turn | Move | Shift | Rotate | Surround | A/D | Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Published baseline(s) | Fixed | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Direct Action | Fixed | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Rolling Replan | Fixed | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ReAct-Style | Fixed | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FlyAgent | Fixed | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FlyAgent | Open | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Use separate table blocks or supplementary tables if both nDTW and semantic SR
cannot fit legibly.

#### Table 2: AerialVLN

| Method | Training | Backbone | Seen SR | Seen OSR | Seen SDTW | Seen NE | Unseen SR | Unseen OSR | Unseen SDTW | Unseen NE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Published baseline(s) | As published | As published | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Direct Action | None | qwen3-vl-plus | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Rolling Replan | None | qwen3-vl-plus | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ReAct-Style | None | qwen3-vl-plus | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FlyAgent | None | qwen3-vl-plus | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

#### Table 3: mechanism and efficiency analysis

| Variant | Success metric | Premature-stop rate | Strategy-change rate | Recovery success | Calls/episode | P50/P95 decision latency | Tokens/episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full FlyAgent | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| No result feedback | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| No completion contract | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| No strategy revision | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Fixed-all capabilities | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| No inner tool loop | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 8.5 Natural failure and recovery analysis

This is a diagnostic analysis, not a causal robustness experiment.

A failure trigger is one of:

- a tool call is rejected or returns no usable evidence;
- a completed physical window produces no meaningful task progress;
- a previously visible required target is absent from the current observation;
- new evidence contradicts the active maneuver strategy.

A run is counted as behaviorally responsive when, within the next two decision
windows, the agent changes its selected capability, strategy stage, physical
action type, or action direction. It is counted as recovered only if the episode
subsequently succeeds under the benchmark's official criterion.

Report trigger count, responsive-transition rate, recovery success, additional
action windows, additional tool calls, and additional wall-clock time. Analyze
all failure-containing episodes automatically. For the qualitative taxonomy,
manually inspect a stratified sample of 100 episodes, or all episodes if fewer
than 100 exist.

### 8.6 Efficiency protocol

Record the following timestamps separately:

- model provider request and response;
- read-only capability execution;
- safety validation and command dispatch;
- simulator action execution and stabilization;
- observation capture and context construction.

Report model-round and physical-window P50/P95 latency, end-to-end episode time,
model rounds per episode, capability calls per episode, input/output tokens, and
estimated API cost. Never describe simulator acceleration as model response
speed.

For deterministic evaluation, run every selected episode once and compute
paired bootstrap 95% confidence intervals over episodes. If any configuration
contains sampling or nondeterministic action selection, repeat that controlled
experiment with three independent seeds and report mean and standard deviation.

## 9. Implementation-to-paper evidence map

These code paths currently support the method claims and should be cited in the
internal reproducibility checklist, not in the anonymous paper text:

| Paper mechanism | Current implementation source |
|---|---|
| Outer phase loop and physical windows | `src/runtime/runners/task_runtime.py` |
| Inner read-only native tool loop and protocol feedback | `src/runtime/runners/task_runtime.py` |
| Chronological intent/call/result/checkpoint trace | `src/agent/transcript.py` |
| Phase state, measured progress, and strategy revision | `src/agent/mission_state.py` |
| Semantic completion contract | `src/agent/semantic/` |
| Phase-conditioned knowledge selection | `src/plugins/skills/selection.py` and `src/memory/context/modules/skills.py` |
| Unified tool and capability policy | `src/tools/registry.py` and `src/plugins/contracts/` |
| Benchmark truth isolation | `evaluation/benchmarks/uav_flow/` and `evaluation/benchmarks/aerial_vln/` |

Before submission, archive one model-facing transcript and one evaluator-only
artifact for the same episode and verify that no hidden goal coordinate,
reference trajectory, benchmark success label, or test answer appears in the
model-facing transcript.

## 10. Figure and table order

Main-paper order:

1. Figure 1: fixed-stack/direct-replanning versus execution-grounded loop.
2. Figure 2: FlyAgent architecture and the two nested loops.
3. Table 1: UAV-Flow Fixed/Open results.
4. Table 2: AerialVLN Seen/Unseen results.
5. Table 3: controlled baselines, mechanism ablations, recovery, and efficiency.
6. Figure 3: one fully attributable natural recovery trace.

Supplementary material:

- complete TravelUAV results;
- per-category UAV-Flow SR and nDTW tables;
- complete prompts and capability schemas;
- additional recovery and failure examples;
- benchmark adapter and truth-isolation audit;
- latency distributions and API cost details.

Recommended main-paper budget:

| Section | Pages |
|---|---:|
| Abstract + Introduction | 1.25 |
| Related Work | 0.75 |
| Method | 2.5--3.0 |
| Experiments | 2.5--3.0 |
| Conclusion | 0.25 |

## 11. Claim checklist

The paper is ready to freeze only when all applicable statements are true:

- [ ] The abstract, introduction, and conclusion identify the
      Execution-Grounded Agent Loop as the central contribution.
- [ ] The paper never claims to understand all instructions or to be the first
      reflective/tool-using embodied agent.
- [ ] Direct Action, Rolling Replan, ReAct-Style, and full FlyAgent use the same
      frozen backbone and execution budget.
- [ ] Full FlyAgent is evaluated on complete, protocol-valid UAV-Flow and
      AerialVLN splits.
- [ ] TravelUAV is reported as supplementary transfer evidence, not silently
      presented as a third equally complete main benchmark.
- [ ] Every reported result can be traced to a frozen config, dataset identity,
      run manifest, and raw episode outputs.
- [ ] Completion-contract, result-feedback, strategy-revision, capability-routing,
      and inner-loop ablations each isolate one mechanism.
- [ ] Recovery claims use the behavioral definition in Section 8.5 and are not
      presented as causal disturbance robustness.
- [ ] Latency separates model, capability, simulator, and observation time.
- [ ] At least two independently optional capabilities are evaluated; otherwise
      multi-capability routing claims are replaced by skill-conditioned tool use.
- [ ] No benchmark truth enters model-facing context or transcripts.
- [ ] The sentence about “finding a place that sells water” appears only in a
      demo/future-work paragraph unless a reproducible open-instruction task set
      is added later.

## 12. References to lock in the BibTeX database

The bibliography should include, at minimum:

- AerialVLN;
- STMR;
- CityNavAgent;
- LookasideVLN;
- TravelUAV / TRAVEL;
- OpenFly;
- RILA;
- EvoNav;
- Slow4Fast-VLN;
- ReAct;
- Inner Monologue;
- SayCan.

Retrieve BibTeX from the official proceedings, OpenReview, CVF, PMLR, or the
authors' official repository. Do not create guessed BibTeX fields from memory.

