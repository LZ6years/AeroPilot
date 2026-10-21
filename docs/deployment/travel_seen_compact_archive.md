# TravelUAV closed-attempt archive continuation

The `intent_tools_images_v2` retention profile changes post-attempt storage.
The existing frozen Runtime, planner, provider, prompts, plugins, image budget,
historical depth reader and scoring code remain authoritative. The live logging
adapter also stays unchanged. No archive operation runs in the flight host.

The controller first finishes the attempt, verifies scene cleanup and writes
the ordinary public review. It then calls `campaign/artifacts.py`. Public memory
JSONL, request receipts and their media hashes, images, final topology, scores,
settings and process diagnostics remain at their original paths. The explicit
archive allowlist contains repeated context checkpoints, historical topology
JSON snapshots, duplicate audit JSON and retained diagnostic depth arrays.
An HTML-linked file is always protected from archival.

`diagnostic_archive.zip` retains exact original bytes. Its JSON manifest lists
each original path, byte count and SHA256. ZIP contents are read back and
verified before any original is unlinked. Failed archives preserve recoverable
data and stop further dispatch after other active flights drain. An archive
failure never changes a valid navigation score or grants another attempt.
While a cleanup worker is pending, the controller stops publishing running
heartbeats to that attempt's process record. This transfers record ownership
to the worker and preserves its terminal proof through report generation and
archiving. A concurrency regression holds the real archive worker at a barrier
while the polling writer ticks, then verifies a successful unchanged score.

The original full audit cohort and existing attempts keep their original
retention profile. Newly claimed compact attempts carry the new profile, so
old files already included in measured disk usage do not inflate the new
format's projected per-case size. The capacity guard retains its factor of two,
three-lane temporary working allowance and 10 GiB minimum.

## Offline verification before activation

Use a separate worktree based on the deployed controller revision, never the
uncommitted Windows workspace or an active frozen source directory. On Windows:

```powershell
& 'F:\conda_env\flyagent\python.exe' -X utf8 -m pytest -q tests/diagnostics/test_campaign_artifacts.py tests/diagnostics/test_campaign_retention.py tests/diagnostics/test_linux_campaign.py tests/diagnostics/test_multi_lane_campaign.py tests/test_aerial_vln_parallel_campaign.py tests/test_shared_campaign.py tests/test_model_compaction_audit.py tests/test_historical_pixel_depth.py tests/diagnostics/test_guarded_proxy_campaign.py
```

On Linux, explicitly export `DATADIR=/home/Newdisk/wangzequan` and use
`/home/wangzequan/miniconda3/envs/flyagent/bin/python`. Place the candidate
worktree, logs and probe copies under `$DATADIR/flyagent/diagnostics`.

Run `tests/manual/campaign_archive_probe.py --output CAMPAIGN --destination NEW_DIAGNOSTICS_DIRECTORY`.
This inspects closed compact attempts, copies the largest records plus median
and small examples, compares every retained file, verifies all HTML references,
and restores every archived member into a separate investigation tree. It
never invokes a model, issues a simulator action or rewrites the original run.
The report records the maximum measured retained size and temporary working
size. Any unprobed original larger than the measured seed is also tested.

The continuation retention policy adds these explicitly measured fields:

```json
{
  "compact_profile": "intent_tools_images_v2",
  "compact_seed_bytes": 41817305,
  "compact_peak_bytes": 77452490
}
```

These values are observations of the 2026-10-20 probe, not defaults or fixed
runtime thresholds. Use a new probe's values for another campaign. The
controller raises the estimate if new attempts retain more data.

## Activation and recovery

Verify the controller and host are stopped, no leases are active, owned scene
cleanup is complete, endpoints are closed, and the configured proxy works.
Back up the queue, overlay and supervision records. Install changed evaluation
helpers in a new hashed controller directory and preserve the original frozen
source and manifest. Revalidate the original technical trial before switching
the overlay. Restart the watchdog through the existing guarded launcher with
the same GPU UUIDs and ports. Remove only the identified guard-created pause
after all preflight checks pass. Historical scores and attempts remain intact.

For forensic restoration, copy the retained attempt to a separate destination,
then call `restore_archive(original_attempt, copied_attempt)` from
`evaluation.benchmarks.core.campaign.artifacts`. The helper verifies the
archive and refuses traversal, in-place restoration and changed-file
overwrites. This reconstructs the original diagnostic layout without touching
a live task. Archived raw depth is historical evidence, not a current range.
