# UAV-Flow Benchmark Manual Entrypoints

这些脚本用于手动运行 UAV-Flow 外部 benchmark。默认配置使用 dry-run UnrealCV scenario；真实环境需要在配置或命令行里启用 `--real-unrealcv`，并确保 UAV-Flow-Eval、gym_unrealcv 和 UnrealCV 环境可用。

## Smoke

Environment-only smoke, no model calls:

```powershell
python tests/manual/uav_flow_eval/run_env_smoke.py --config evaluation/benchmarks/uav_flow/configs/uav_flow_eval.example.json --class Turn
```

```powershell
python tests/manual/uav_flow_eval/run_eval_smoke.py --config evaluation/benchmarks/uav_flow/configs/uav_flow_eval.example.json --classes Turn,Move,Approach --limit-per-class 1
```

## Batch

```powershell
python tests/manual/uav_flow_eval/run_eval_batch.py --config evaluation/benchmarks/uav_flow/configs/uav_flow_eval.example.json --classes Turn,Move --limit-per-class 3
```

## SR Judge

```powershell
python tests/manual/uav_flow_eval/judge_sr_outputs.py --eval-dir evaluation/benchmarks/uav_flow/outputs/<eval_id>
```
