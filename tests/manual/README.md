# Manual scripts

These scripts are never collected by the default pytest suite. Start Python scripts with the required interpreter:

```powershell
& 'F:\\conda_env\\flyagent\\python.exe' tests/manual/<script>.py --help
```

Every run must use a new `runs/experiments/<probe>/<timestamp>/` directory. Do not write to `tests/manual_outputs` and do not copy a complete source snapshot. Record the Git commit, dirty diff hash, effective configuration, `sys.executable`, and—when relevant—PyTorch/CUDA diagnostics.

## Inventory

| Script | Required environment | Default output root | Purpose |
|---|---|---|---|
| `aerial_vln_case_gate.py` | flyagent + configured simulator/network | `runs/experiments/aerial_vln_case_gate/<timestamp>/` | Explicit manual/diagnostic workflow |
| `aerial_vln_real_engine_probe.py` | flyagent + configured simulator/network | `runs/experiments/aerial_vln_real_engine_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `aerial_vln_remote_difficulty_batch.py` | flyagent + configured simulator/network | `runs/experiments/aerial_vln_remote_difficulty_batch/<timestamp>/` | Explicit manual/diagnostic workflow |
| `aerial_vln_remote_readiness.py` | flyagent + configured simulator/network | `runs/experiments/aerial_vln_remote_readiness/<timestamp>/` | Explicit manual/diagnostic workflow |
| `aerial_vln_selected_case_probe.py` | flyagent + configured simulator/network | `runs/experiments/aerial_vln_selected_case_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `aerial_vln_stmr_replay_probe.py` | flyagent + configured simulator/network | `runs/experiments/aerial_vln_stmr_replay_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `agent_guidance_probe.py` | flyagent | `runs/experiments/agent_guidance_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `airsim_action_regression_probe.py` | flyagent | `runs/experiments/airsim_action_regression_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `airsim_capture_matrix_probe.py` | flyagent | `runs/experiments/airsim_capture_matrix_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `airsim_phase2_probe.py` | flyagent | `runs/experiments/airsim_phase2_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `airsim_phase2_report.py` | flyagent | `runs/experiments/airsim_phase2_report/<timestamp>/` | Explicit manual/diagnostic workflow |
| `airsim_task_profile_probe.py` | flyagent | `runs/experiments/airsim_task_profile_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `airsim_timing_report.py` | flyagent | `runs/experiments/airsim_timing_report/<timestamp>/` | Explicit manual/diagnostic workflow |
| `benchmark_20_case_probe.py` | flyagent | `runs/experiments/benchmark_20_case_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `benchmark_probe_support.py` | flyagent | `runs/experiments/benchmark_probe_support/<timestamp>/` | Explicit manual/diagnostic workflow |
| `body_frame_projection_calibration_probe.py` | flyagent | `runs/experiments/body_frame_projection_calibration_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `cdp_websocket.ps1` | PowerShell/Windows | `runs/experiments/cdp_websocket/<timestamp>/` | Explicit manual/diagnostic workflow |
| `context_assembly_probe.py` | flyagent | `runs/experiments/context_assembly_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `cuda_event_probe.py` | flyagent + CUDA/GPU | `runs/experiments/cuda_event_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `cuda_timeline.py` | flyagent + CUDA/GPU | `runs/experiments/cuda_timeline/<timestamp>/` | Explicit manual/diagnostic workflow |
| `custom_map_real_collection_probe.py` | flyagent | `runs/experiments/custom_map_real_collection_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `custom_map_segformer_probe.py` | flyagent | `runs/experiments/custom_map_segformer_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `custom_map_segmentation_compare_probe.py` | flyagent | `runs/experiments/custom_map_segmentation_compare_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `done_when_flow_compare.py` | flyagent | `runs/experiments/done_when_flow_compare/<timestamp>/` | Explicit manual/diagnostic workflow |
| `experiment_sources.py` | flyagent | `runs/experiments/experiment_sources/<timestamp>/` | Explicit manual/diagnostic workflow |
| `export_benchmark_review.py` | flyagent | `runs/experiments/export_benchmark_review/<timestamp>/` | Explicit manual/diagnostic workflow |
| `feedback_recovery_probe.py` | flyagent | `runs/experiments/feedback_recovery_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `flight_experience_batch_probe.py` | flyagent | `runs/experiments/flight_experience_batch_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `flight_experience_library_probe.py` | flyagent | `runs/experiments/flight_experience_library_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `flight_experience_library.py` | flyagent | `runs/experiments/flight_experience_library/<timestamp>/` | Explicit manual/diagnostic workflow |
| `flight_experience_probe.py` | flyagent | `runs/experiments/flight_experience_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `flight_experience_v2_probe.py` | flyagent | `runs/experiments/flight_experience_v2_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `flight_experience_v2_report.py` | flyagent | `runs/experiments/flight_experience_v2_report/<timestamp>/` | Explicit manual/diagnostic workflow |
| `gpu_probe_environment.py` | flyagent + CUDA/GPU | `runs/experiments/gpu_probe_environment/<timestamp>/` | Explicit manual/diagnostic workflow |
| `gpu_resource_monitor.py` | flyagent + CUDA/GPU | `runs/experiments/gpu_resource_monitor/<timestamp>/` | Explicit manual/diagnostic workflow |
| `gpu_variance_probe.py` | flyagent + CUDA/GPU | `runs/experiments/gpu_variance_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `history_compaction_probe.py` | flyagent + saved transcript; Provider only with `--invoke-model` | `runs/experiments/history_compaction/<timestamp>/` | Select the largest image-free complete-window prefix under 32k, audit its binary-free prefix-media index, and optionally validate one non-blocking model compaction per sample. |
| `model_admission_probe.py` | flyagent | `runs/experiments/model_admission_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `model_multimodal_smoke.py` | flyagent | `runs/experiments/model_multimodal_smoke/<timestamp>/` | Explicit manual/diagnostic workflow |
| `model_native_tool_loop_smoke.py` | flyagent | `runs/experiments/model_native_tool_loop_smoke/<timestamp>/` | Explicit manual/diagnostic workflow |
| `native_tool_loop_real_audit.py` | flyagent | `runs/experiments/native_tool_loop_real_audit/<timestamp>/` | Explicit manual/diagnostic workflow |
| `native_tool_transcript_probe.py` | flyagent | `runs/experiments/native_tool_transcript_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `navigation_decision_probe.py` | flyagent | `runs/experiments/navigation_decision_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `nsight_stage_probe.py` | flyagent + CUDA/GPU | `runs/experiments/nsight_stage_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `process_diagnostics.py` | flyagent | `runs/experiments/process_diagnostics/<timestamp>/` | Explicit manual/diagnostic workflow |
| `read_only_concurrency_probe.py` | flyagent | `runs/experiments/read_only_concurrency_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `real_uav_primitive_probe.py` | flyagent | `runs/experiments/real_uav_primitive_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `remote_ssh_scene_probe.py` | flyagent + configured simulator/network | `runs/experiments/remote_ssh_scene_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `review_browser_probe.py` | flyagent | `runs/experiments/review_browser_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `rgb_phase_review_probe.py` | flyagent | `runs/experiments/rgb_phase_review_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `sam_coordinate_probe.py` | flyagent | `runs/experiments/sam_coordinate_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `stmr_map_probe.py` | flyagent | `runs/experiments/stmr_map_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `stmr_urban_prior_profile_probe.py` | flyagent | `runs/experiments/stmr_urban_prior_profile_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `supervised_flight_probe.py` | flyagent | `runs/experiments/supervised_flight_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `task_profile.py` | flyagent | `runs/experiments/task_profile/<timestamp>/` | Explicit manual/diagnostic workflow |
| `task_progress_guidance_probe.py` | flyagent | `runs/experiments/task_progress_guidance_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `task_progress_simulation_probe.py` | flyagent | `runs/experiments/task_progress_simulation_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `task_runtime_episode_options_probe.py` | flyagent | `runs/experiments/task_runtime_episode_options_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `tool_completion_probe.py` | flyagent | `runs/experiments/tool_completion_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `tool_contract_behavior_probe.py` | flyagent | `runs/experiments/tool_contract_behavior_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `tool_dispatcher_forward_stability_probe.py` | flyagent | `runs/experiments/tool_dispatcher_forward_stability_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `tool_selection_ablation_probe.py` | flyagent | `runs/experiments/tool_selection_ablation_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `travel_uav_real_engine_probe.py` | flyagent + configured simulator/network | `runs/experiments/travel_uav_real_engine_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `travel_uav_remote_readiness.py` | flyagent + configured simulator/network | `runs/experiments/travel_uav_remote_readiness/<timestamp>/` | Explicit manual/diagnostic workflow |
| `uav_flow_full_campaign_audit.py` | flyagent | `runs/experiments/uav_flow_full_campaign_audit/<timestamp>/` | Explicit manual/diagnostic workflow |
| `uav_flow_skill_selection_cases.py` | flyagent | `runs/experiments/uav_flow_skill_selection_cases/<timestamp>/` | Explicit manual/diagnostic workflow |
| `unified_tool_loop_replay_probe.py` | flyagent | `runs/experiments/unified_tool_loop_replay_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `unrealcv_local_actor_probe.py` | flyagent | `runs/experiments/unrealcv_local_actor_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `visual_grounding_capability_probe.py` | flyagent | `runs/experiments/visual_grounding_capability_probe/<timestamp>/` | Explicit manual/diagnostic workflow |
| `visual_instruction_eval.py` | flyagent | `runs/experiments/visual_instruction_eval/<timestamp>/` | Explicit manual/diagnostic workflow |
| `windows_gpu_memory.ps1` | flyagent + CUDA/GPU | `runs/experiments/windows_gpu_memory/<timestamp>/` | Explicit manual/diagnostic workflow |
| `windows_gpu_memory.py` | flyagent + CUDA/GPU | `runs/experiments/windows_gpu_memory/<timestamp>/` | Explicit manual/diagnostic workflow |

## Retired probes

- `image_context_probe.py` and its note were removed with the V1 RGB encoding cache.
- `context_build_probe.py` was replaced by `context_assembly_probe.py` and the product tests in `tests/test_model_context_v3.py`.
- Product-contract checks belong in `tests/`; profiler, GPU, simulator, and network assertions belong in `tests/diagnostics/`.
