# STMR Research Module

This package is a standalone implementation of a Semantic-Topo-Metric-like
research pipeline for UAV-Flow recordings. It does not participate in the
default planner context or flight runtime. UAV-Flow can explicitly install its
stateful online adapter in the existing `map` context slot for controlled
`shadow` or planner-visible canary runs.

## Online UAV-Flow Context Integration

The experimental provider keeps the normal fragment order and contributes one
map image through `map.model_inputs`. The shared model-input adapter therefore
sends existing RGB images first, the STMR image next, and the compact JSON
context last. The STMR image is labelled as semantic history rather than
verified traversability.

Modes are `off` (default), `shadow` (build artifacts without changing the
model-facing map fragment), and `planner` (attach one STMR PNG). UAV-Flow also
enables its existing real depth capture only for the two STMR modes. Run the
preflight before launching UE; the STMR modes require the project environment
with Torch and Transformers:

```powershell
F:\conda_env\flyagent\python.exe evaluation/benchmarks/uav_flow/scripts/run_manual_eval.py `
  --preflight-only --stmr-mode shadow
```

The staged five-class canary is available at
`tests/manual/uav_flow_eval/run_stmr_canary.py`. Models are warmed before UE is
started so missing dependencies or an incomplete model download fail early
instead of wasting a benchmark run.

## Processing Blocks

```text
Instruction
  -> task landmark extraction
  -> query prior assembly
  -> GroundingDINO boxes
  -> TF-IDF phrase filter
  -> SAM masks
  -> depth-aware dynamic target policy
  -> open-vocabulary semantic mask artifact
  -> depth + mask + pose projection
  -> semantic 3D points
  -> metric voxel pooling
  -> top-height / sub-goal-priority column projection
  -> 20 x 20 UAV-centered semantic matrix
  -> path stamp (-1) + ordered START/ENTRY -> NOW direction overlay
  -> PNG / lightweight 3D HTML / JSON
```

Each block has one responsibility:

- **Task landmark extraction** finds noun phrases mentioned by the instruction.
- **Query prior assembly** adds stable concepts worth checking in an urban UAV
  scene. Task concepts always come first.
- **GroundingDINO** detects text-conditioned boxes in each RGB frame.
- **TF-IDF filter** rejects detector phrases unrelated to the requested query.
- **SAM** converts retained boxes into pixel masks.
- **Dynamic target policy** suppresses far, tiny, or low-confidence dynamic
  detections and selects a primary track when appropriate.
- **3D projection** combines the mask with the paired depth and UAV pose.
- **Incremental voxel pooling** projects each observation once and permanently
  merges its evidence into process-local world-coordinate voxels.
- **Column projection** selects a top-down semantic label per metric column.
- **Path stamping** marks visited cells as `-1` after semantics. The PNG also
  overlays the ordered actual pose trace: orange arrowheads point from older to
  newer poses, the green `NOW` marker shows current heading, and map top is the
  current UAV forward direction. The full fractional trace stays in debug JSON;
  planner context receives only a compact numeric trajectory summary.

## Urban Query Priors

`default_uav_prior` contains the small historical query set. The
`urban_uav_prior_v1` variant adds stable urban structures such as rooftop,
entrance, sidewalk, traffic fixtures, facade, tower, and bridge.

A prior is a **detector query, not a map fact**. For example, adding `bridge` to
the query list only asks GroundingDINO to look for a bridge. A bridge enters the
STMR matrix only after RGB detection, SAM masking, text filtering, valid depth,
pose projection, and voxel pooling all succeed.

Query assembly rules are:

1. Preserve instruction/manual landmark order.
2. Normalize exact synonyms such as `car -> vehicle`.
3. Append the selected fixed prior in its declared order.
4. Remove duplicates.
5. Stop at 16 queries.

## Profiling Stages

`StageProfiler` records wall time, process RSS, CUDA allocation/peak memory,
input/output counts, and artifact sizes. CUDA is synchronized around measured
GPU stages so inference timings are meaningful.

Cold-start stages:

- `grounding_model_load`
- `sam_model_load`

Before measured cases, the profiling probe also runs one unmeasured forward
pass for each prior query shape on the same RGB frame. Those records use the
`model_warmup` scope in `model_warmup_timings.jsonl`. This keeps CUDA kernel and
input-shape initialization out of the per-prior case comparison.

Warm per-frame perception stages:

- `grounding_preprocess`
- `grounding_inference`
- `grounding_postprocess`
- `tfidf_and_semantic_filter`
- `sam_preprocess`
- `sam_inference`
- `sam_mask_postprocess`
- `depth_and_dynamic_candidate_stats`
- `mask_overlay_and_memory_write`

Incremental mapping and rendering stages:

- `variant_memory_prepare`
- `landmark_query_assembly`
- `dynamic_track_policy`
- `memory_record_load`
- `stmr_incremental_projection`
- `stmr_incremental_merge`
- `stmr_incremental_snapshot`
- `incremental_png_render`
- `incremental_compact_json_render`
- `incremental_frame_total`
- `batch_rebuild_reference`
- `png_render`
- `html_render`
- `json_render`
- `complete_pipeline_total`

`online_frame_component_total` is derived from the measured per-frame
GroundingDINO, TF-IDF, SAM, dynamic-candidate, artifact-writing, and
`incremental_frame_total` records. `batch_rebuild_reference` rebuilds the same
history through the compatibility API and must match the incremental matrix,
voxel counts, semantic cells, source observations, and center pose.

## Incremental State And History

`IncrementalStmrMapState` owns the process-local map state. `update(records)`
ignores observation IDs already committed, keeps incomplete observations
pending for retry, and commits a frame only after its complete projection
succeeds. `snapshot()` regenerates the UAV-centered matrix from accumulated
world voxels without rereading historical depth or mask files.

Per-frame geometry uses NumPy arrays for sampled pixel selection, camera/world
coordinate transforms, and voxel grouping. Python only merges the aggregated
voxel evidence and materializes retained debug records; this preserves the
original rounding and semantic-vote behavior while removing the previous
per-pixel Python projection loop.

All observations are retained by default. There is no frame window, TTL,
evidence decay, or eviction. Raw JSONL/depth/mask files remain the durable
source of truth; process restart recovery creates a new state and imports all
complete history once. Voxel checkpoints are intentionally outside this phase.

Historic debug points keep canonical world coordinates. Lightweight snapshots
recenter only their configured point budget against the latest UAV pose; full
debug snapshots can still recenter and emit the entire retained point history.
The original `build_stmr_map()` API remains available as a one-shot wrapper
over the same incremental engine.

## A/B Profile Run

The default probe reuses Turn, Shift, and Approach recordings from
`tests/manual_outputs/stmr_light_regen_v2`. It runs five frames per case, so
both priors see the same 15 RGB/depth/pose frame bundles.

```powershell
Set-Location F:\UAV-Agent
conda activate flyagent

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

python tests/manual/stmr_urban_prior_profile_probe.py `
  --source-root tests/manual_outputs/stmr_light_regen_v2 `
  --output-dir tests/manual_outputs/stmr_incremental_profile `
  --grounding-model F:\hf_models\grounding-dino-tiny `
  --sam-model F:\hf_models\sam-vit-base `
  --seg-device cuda
```

Important outputs:

```text
tests/manual_outputs/stmr_incremental_profile/
  profile_summary.json
  profile_summary.html
  model_warmup_timings.jsonl
  <case>/
    default_uav_prior/
      latest_stmr_map.png
      latest_stmr_map.html
      latest_stmr_map.json
      incremental_frames/
        <frame>/latest_stmr_map.png
        <frame>/latest_stmr_map.json
      stmr_stage_timings.jsonl
      stmr_timing_summary.json
    urban_uav_prior_v1/
      ...
```

The HTML report compares query lists, detector/SAM previews, map artifacts,
incremental map and online-component P50/P95 latency, batch rebuild speedup,
batch parity, CUDA peak memory, CPU RSS, and output counts. Full point arrays
remain outside the report so the browser viewer stays responsive.

## Boundary

This phase does not modify `TaskRuntime`, memory context composition, plugin
enablement, benchmark configuration, or planner prompts. Runtime integration
should happen only after the A/B quality and resource-cost report is accepted.
