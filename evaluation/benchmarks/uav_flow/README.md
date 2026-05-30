# UAV-Flow Benchmark Harness

This package evaluates UAV-Agent against UAV-Flow tasks while keeping UAV-Agent's native runtime contract intact.

## Boundary

- UAV-Flow provides task JSON, UnrealCV scene assumptions, reference paths, and evaluation metrics.
- UAV-Agent still owns observation, context construction, planner calls, tool actions, phase transitions, and shutdown.
- `target_pos` and `reference_path_preprocessed` are evaluation-only. They are written to benchmark artifacts, not planner context.
- The harness does not implement UAV-Flow `/predict`, HTTP policy server mode, or `action_tokens`.
- Case completion uses UAV-Agent's native runtime: semantic completion comes from `complete_task`, while `runtime_loop_guard_reached` is reported as stopped/incomplete and never advances a phase.

## Dry Smoke

The default example config uses dry UnrealCV mode so pure Python tests and local smoke checks do not require the simulator.

Environment-only smoke, no model calls:

```powershell
python tests/manual/uav_flow_eval/run_env_smoke.py --config evaluation/benchmarks/uav_flow/configs/uav_flow_eval.example.json --class Turn
```

Runtime smoke:

```powershell
python tests/manual/uav_flow_eval/run_eval_smoke.py --config evaluation/benchmarks/uav_flow/configs/uav_flow_eval.example.json --classes Turn,Move --limit-per-class 1
```

## Real UnrealCV

Install and configure UAV-Flow-Eval separately, then enable real UnrealCV:

```powershell
python tests/manual/uav_flow_eval/run_eval_smoke.py --config evaluation/benchmarks/uav_flow/configs/uav_flow_eval.example.json --classes Turn --limit-per-class 1 --real-unrealcv
```

Outputs are written under `evaluation/benchmarks/uav_flow/outputs/<eval_id>/`.

Each case review includes `completed`, `stopped`, `stopped_reason`, action displacement diagnostics, and plots under `cases/<case_id>/plots/`. The trajectory plots draw both the UAV-Flow reference path and UAV-Agent's actual path so deviations are visible without reading raw JSON.

For real UnrealCV runs, the adapter prepends `uav_flow_repo_path` to `sys.path`, then imports `gym_unrealcv` lazily. If startup fails, check that the UAV-Flow-Eval repo exists, the env id is registered, and the UnrealCV package path in UAV-Flow's `DowntownWest.json` points to the local UnrealZoo binary.
