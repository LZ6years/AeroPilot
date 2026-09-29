"""Offline evaluator for real-engine FlyAgent runs.

The evaluator does not start UE, call a model, or mutate flight state. It reads
completed run artifacts and writes a compact metrics report under each run's
``eval/`` directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from src.runtime.context.checkpoints import (
    UnsupportedRunSchema,
    load_state_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FORBIDDEN_TOOLS = {
    "uav.connect",
    "uav.takeoff",
    "uav.land",
    "uav.set_pose",
    "uav.get_pose",
    "uav.get_frames",
    "env.observe",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved real-engine FlyAgent runs")
    parser.add_argument("--run-dir", action="append", help="Run directory to evaluate; can be passed multiple times")
    parser.add_argument("--run-root", default="runs", help="Root used when discovering runs")
    parser.add_argument("--task-suite", default="configs/eval/standard_tasks.json", help="Standard task suite JSON")
    parser.add_argument("--task-id", help="Force a specific task id from the suite")
    parser.add_argument("--latest", type=int, default=0, help="Evaluate the latest N run directories under --run-root")
    parser.add_argument("--write", action="store_true", help="Write eval/metrics.json, eval/report.md, and eval/summary.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    suite = load_task_suite(Path(args.task_suite))
    run_dirs = resolve_run_dirs(args)
    if not run_dirs:
        raise SystemExit("No run directories selected. Use --run-dir or --latest.")
    results = [evaluate_run(run_dir, suite, task_id=args.task_id) for run_dir in run_dirs]
    if args.write:
        for result in results:
            write_eval_outputs(result)
    print(json.dumps({"evaluated": [compact_result(result) for result in results]}, indent=2, ensure_ascii=False))


def resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    explicit = [Path(item) for item in args.run_dir or []]
    if explicit:
        return explicit
    latest = int(args.latest or 0)
    if latest <= 0:
        return []
    root = Path(args.run_root)
    if not root.exists():
        return []
    return [path for path in sorted(root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True) if path.is_dir()][:latest]


def load_task_suite(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"defaults": {}, "single_instruction_tasks": [], "continuous_instruction_tasks": []}
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_run(run_dir: Path, suite: dict[str, Any], task_id: str | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    events = load_events(run_dir / "events.jsonl")
    debug_events = load_events(run_dir / "debug" / "debug_events.jsonl")
    all_events = events_in_time_order(events + debug_events)
    latest_state = load_first_json(
        run_dir / "state" / "latest_state.json",
        run_dir / "latest_state.json",
        run_dir / "mission_state.json",
    ) or {}
    checkpoint_path = run_dir / "contexts" / "latest_checkpoint.json"
    legacy_context_path = run_dir / "contexts" / "latest_context.json"
    if checkpoint_path.is_file():
        latest_checkpoint = load_state_checkpoint(checkpoint_path)
    elif legacy_context_path.is_file():
        raise UnsupportedRunSchema(
            f"V1 context artifact is unsupported: {legacy_context_path}"
        )
    else:
        latest_checkpoint = {}
    latest_action_context = load_json(run_dir / "contexts" / "latest_action_planner_model_context.json") or {}
    demo_summary = load_json(run_dir / "demo" / "summary.json") or {}
    run_config = load_json(run_dir / "config" / "model_config.json") or {}
    task = select_task_for_run(demo_summary, all_events, suite, task_id, run_config=run_config)
    action_sequence = extract_action_sequence(all_events, demo_summary)
    phase_contexts = extract_phase_contexts(events, latest_action_context, run_dir)
    observations = extract_observations(all_events, latest_state, run_dir)
    artifact_inventory = build_artifact_inventory(observations, run_dir)
    expected = dict(task.get("expected", {})) if task else {}

    metrics = {
        "run_dir": str(run_dir),
        "task": task_summary(task),
        "basic": evaluate_basic(latest_state, latest_checkpoint, phase_contexts, demo_summary),
        "actions": evaluate_actions(action_sequence),
        "action_windows": evaluate_action_windows(phase_contexts, all_events),
        "artifacts": evaluate_artifacts(artifact_inventory, expected, suite),
        "phases": evaluate_phases(phase_contexts, expected, suite),
        "motion": evaluate_motion(phase_contexts, expected, suite),
        "continuous": evaluate_continuous(demo_summary, all_events, expected, suite),
        "batch": evaluate_batch_instructions(demo_summary, all_events, suite, run_dir, task),
        "files": {
            "latest_state": (run_dir / "state" / "latest_state.json").exists(),
            "legacy_latest_state": (run_dir / "latest_state.json").exists(),
            "legacy_mission_state": (run_dir / "mission_state.json").exists(),
            "events": (run_dir / "events.jsonl").exists(),
            "debug_events": (run_dir / "debug" / "debug_events.jsonl").exists(),
            "latest_checkpoint": checkpoint_path.exists(),
            "latest_action_planner_model_context": (run_dir / "contexts" / "latest_action_planner_model_context.json").exists(),
        },
    }
    metrics["overall"] = overall_status(metrics)
    return {"run_dir": run_dir, "metrics": metrics, "report": build_markdown_report(metrics)}


def select_task_for_run(
    demo_summary: dict[str, Any],
    events: list[dict[str, Any]],
    suite: dict[str, Any],
    task_id: str | None = None,
    run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    all_tasks = list(suite.get("single_instruction_tasks", [])) + list(suite.get("continuous_instruction_tasks", []))
    if task_id:
        return next((task for task in all_tasks if task.get("id") == task_id), {"id": task_id, "expected": {}})
    instructions = instructions_from_summary_or_events(demo_summary, events)
    for task in suite.get("continuous_instruction_tasks", []):
        if normalize_instruction_list(task.get("instructions", [])) == normalize_instruction_list(instructions):
            return task
    if instructions:
        if len(instructions) > 1:
            suite_group = str((run_config or {}).get("suite_group", "")).lower()
            batch_kind = "continuous" if suite_group == "continuous" else "single"
            return {
                "id": f"batch_{batch_kind}",
                "instructions": instructions,
                "expected": {},
                "batch_kind": batch_kind,
            }
        first = normalize_instruction(instructions[0])
        for task in suite.get("single_instruction_tasks", []):
            if normalize_instruction(task.get("instruction", "")) == first:
                return task
    return {"id": "ad_hoc", "instruction": instructions[0] if instructions else "", "expected": {}}


def instructions_from_summary_or_events(demo_summary: dict[str, Any], events: list[dict[str, Any]]) -> list[str]:
    instructions = []
    for item in demo_summary.get("instructions", []) if isinstance(demo_summary.get("instructions", []), list) else []:
        if isinstance(item, dict) and item.get("instruction"):
            instructions.append(str(item["instruction"]))
    if instructions:
        return instructions
    for event in events:
        if event.get("event") in {"task_parsed", "instruction_started"}:
            payload = event.get("payload", {})
            if isinstance(payload, dict) and payload.get("instruction"):
                instructions.append(str(payload["instruction"]))
    return instructions


def events_in_time_order(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return public and debug events in timestamp order for segmentation."""

    def sort_key(indexed_event: tuple[int, dict[str, Any]]) -> tuple[int, str, int]:
        index, event = indexed_event
        timestamp = str(event.get("timestamp", ""))
        # Synthetic tests and some legacy logs have no timestamps; keep their
        # original order instead of sorting alphabetically by event name.
        return (0 if timestamp else 1, timestamp, index)

    return [event for _, event in sorted(enumerate(events), key=sort_key)]


def split_events_by_instruction(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split a run into instruction-scoped event segments.

    Public events may contain ``instruction_started`` while debug events contain
    ``task_parsed``. Prefer the parser boundary when available so the two event
    streams do not create duplicate instruction segments.
    """

    ordered = events_in_time_order(events)
    start_names = {"task_parsed"} if any(event.get("event") == "task_parsed" for event in ordered) else {"instruction_started"}
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for event in ordered:
        name = event.get("event")
        payload = event.get("payload", {}) if isinstance(event.get("payload", {}), dict) else {}
        if name in start_names:
            if current is not None:
                segments.append(current)
            current = {
                "instruction": str(payload.get("instruction", "")),
                "events": [event],
            }
            continue
        if current is not None:
            current["events"].append(event)
    if current is not None:
        segments.append(current)
    return segments


def single_task_by_instruction(suite: dict[str, Any], instruction: str) -> dict[str, Any] | None:
    normalized = normalize_instruction(instruction)
    for task in suite.get("single_instruction_tasks", []):
        if normalize_instruction(task.get("instruction", "")) == normalized:
            return task
    return None


def extract_action_sequence(events: list[dict[str, Any]], demo_summary: dict[str, Any]) -> list[dict[str, Any]]:
    # Prefer detailed debug tool-call records when present. Compact
    # tool_step_completed events are still written for live UI progress, and
    # counting both would double-count the same physical action.
    actions = _actions_from_event_name(events, _preferred_action_event_name(events))
    if actions:
        return actions
    return list(demo_summary.get("action_sequence", [])) if isinstance(demo_summary.get("action_sequence", []), list) else []


def _actions_from_event_name(events: list[dict[str, Any]], event_name: str) -> list[dict[str, Any]]:
    actions = []
    for event in events:
        if event.get("event") != event_name:
            continue
        payload = event.get("payload", {}) if isinstance(event.get("payload", {}), dict) else {}
        step = payload.get("step", {}) if isinstance(payload.get("step", {}), dict) else {}
        result = payload.get("result", {}) if isinstance(payload.get("result", {}), dict) else {}
        actions.append(
            {
                "phase": payload.get("phase"),
                "tool": step.get("tool") or payload.get("tool"),
                "args": step.get("args", payload.get("args", {})),
                "ok": result.get("ok") if isinstance(result, dict) and "ok" in result else payload.get("ok"),
            }
        )
    return actions


def _preferred_action_event_name(events: list[dict[str, Any]]) -> str:
    for name in ("tool_call_completed", "step_completed", "tool_step_completed"):
        if any(event.get("event") == name for event in events):
            return name
    return "tool_step_completed"


def extract_phase_contexts(events: list[dict[str, Any]], latest_context: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    contexts = []
    for event in events:
        if event.get("event") != "phase_window_recorded":
            continue
        payload = event.get("payload")
        if isinstance(payload, dict):
            ref = payload.get("debug_ref")
            context = load_json(run_dir / str(ref)) if ref else None
            contexts.append(normalize_phase_window_context(context or {}, payload))
    if not contexts and latest_context:
        contexts.append(latest_context)
    return contexts


def normalize_phase_window_context(context: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Convert coupled runtime phase-window records into evaluator phase contexts."""

    merged = dict(context)
    for key, value in payload.items():
        merged.setdefault(key, value)
    phase = str(merged.get("phase") or payload.get("phase") or "")
    summary = merged.get("phase_execution_summary")
    if not isinstance(summary, dict):
        measurements = merged.get("measurements")
        if not isinstance(measurements, dict):
            measurements = payload.get("measurements", {})
        if not isinstance(measurements, dict):
            measurements = {}
        summary = {
            "phase": phase,
            "window_count": payload.get("phase_window_count") or ((payload.get("window_index") or 0) + 1),
            "max_phase_window_count": payload.get("max_phase_window_count"),
            "phase_local_measurements": measurements,
            "last_window_measurements": measurements,
            "cumulative_measurements": measurements,
            "cumulative_tool_facts": {},
        }
    elif not summary.get("phase") and phase:
        summary = {**summary, "phase": phase}
    facts = payload.get("phase_execution_summary", {}).get("cumulative_tool_facts") if isinstance(payload.get("phase_execution_summary"), dict) else None
    if isinstance(facts, dict) and "cumulative_tool_facts" not in summary:
        summary = {**summary, "cumulative_tool_facts": facts}
    merged["phase_execution_summary"] = summary
    return merged


def extract_observations(events: list[dict[str, Any]], latest_state: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    observations = []
    for event in events:
        if event.get("event") not in {"initial_observation_updated", "instruction_start_observation_updated", "automatic_observation_updated"}:
            continue
        payload = event.get("payload")
        if isinstance(payload, dict):
            observations.append(payload)
    latest_observation = latest_state.get("latest_observation")
    if not observations and isinstance(latest_observation, dict):
        observations.append(latest_observation)
    if not observations:
        latest_debug = load_first_json(
            run_dir / "debug" / "latest_observation.json",
            run_dir / "observations" / "latest_observation.json",
        )
        if isinstance(latest_debug, dict):
            observations.append(latest_debug)
    return observations


def evaluate_basic(
    latest_state: dict[str, Any],
    latest_checkpoint: dict[str, Any],
    phase_contexts: list[dict[str, Any]],
    demo_summary: dict[str, Any],
) -> dict[str, Any]:
    instructions = demo_summary.get("instructions", [])
    completed_values = [item.get("completed") for item in instructions if isinstance(item, dict)]
    completed = bool(latest_state.get("completed")) or bool(completed_values and all(value is True for value in completed_values))
    return {
        "passed": completed and bool(latest_checkpoint),
        "completed": completed,
        "connected": latest_state.get("connected"),
        "airborne": latest_state.get("airborne"),
        "has_latest_checkpoint": bool(latest_checkpoint),
        "has_phase_context": bool(phase_contexts),
        "instruction_count": len(instructions) if isinstance(instructions, list) else 0,
    }


def evaluate_actions(action_sequence: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(action_sequence)
    ok_count = sum(1 for action in action_sequence if action.get("ok") is True)
    forbidden = [action for action in action_sequence if action.get("tool") in FORBIDDEN_TOOLS]
    return {
        "passed": total > 0 and ok_count == total and not forbidden,
        "total": total,
        "ok_count": ok_count,
        "success_rate": round(ok_count / total, 3) if total else 0.0,
        "forbidden_tools": forbidden,
        "sequence": action_sequence,
    }


def evaluate_action_windows(phase_contexts: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    windows = extract_action_windows(phase_contexts, events)
    invalid_scopes = [
        item for item in windows
        if item.get("scope") and item.get("scope") != "current_phase_only"
    ]
    observation_mismatches = [
        item for item in windows
        if item.get("requires_intermediate_observation") is True and int(item.get("step_count", 0) or 0) > 1
    ]
    missing_descriptions = [
        item for item in windows
        if not str(item.get("intent", "")).strip() or not str(item.get("expected_effect", "")).strip()
    ]
    return {
        "passed": not invalid_scopes and not observation_mismatches and not missing_descriptions,
        "total": len(windows),
        "windows": windows,
        "invalid_scopes": invalid_scopes,
        "observation_mismatches": observation_mismatches,
        "missing_descriptions": missing_descriptions,
    }


def extract_action_windows(phase_contexts: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    windows = []
    for context in phase_contexts:
        if not isinstance(context, dict):
            continue
        window = context.get("action_window")
        if isinstance(window, dict) and window:
            windows.append(_action_window_record(window, context))
    if not windows:
        for event in events:
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            window = payload.get("action_window")
            if isinstance(window, dict) and window:
                windows.append(_action_window_record(window, payload))
    return _dedupe_action_window_records(windows)


def _action_window_record(window: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    steps = source.get("executed_steps", []) if isinstance(source.get("executed_steps", []), list) else []
    return {
        "phase": phase_name_from_context(source),
        "intent": window.get("intent"),
        "scope": window.get("scope"),
        "requires_intermediate_observation": window.get("requires_intermediate_observation"),
        "expected_effect": window.get("expected_effect"),
        "step_count": len(steps) if steps else source.get("executed_step_count"),
    }


def _dedupe_action_window_records(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen = set()
    for window in windows:
        signature = json.dumps(window, sort_keys=True, ensure_ascii=False)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(window)
    return deduped


def evaluate_artifacts(artifact_inventory: dict[str, Any], expected: dict[str, Any], suite: dict[str, Any]) -> dict[str, Any]:
    default_min = dict(suite.get("defaults", {}).get("min_artifacts", {}))
    expected_min = {**default_min, **dict(expected.get("min_artifacts", {}))}
    counts = artifact_inventory.get("counts", {})
    checks = {}
    for modality, minimum in expected_min.items():
        checks[modality] = {"actual": int(counts.get(modality, 0) or 0), "minimum": int(minimum or 0)}
        checks[modality]["passed"] = checks[modality]["actual"] >= checks[modality]["minimum"]
    missing_paths = [
        item
        for bucket in ("rgb", "depth", "other")
        for item in artifact_inventory.get(bucket, [])
        if item.get("exists") is False
    ]
    path_counts: dict[str, int] = {}
    for bucket in ("rgb", "depth", "other"):
        for item in artifact_inventory.get(bucket, []):
            path = str(item.get("path", ""))
            if path:
                path_counts[path] = path_counts.get(path, 0) + 1
    duplicate_paths = [
        {"path": path, "count": count}
        for path, count in sorted(path_counts.items())
        if count > 1
    ]
    return {
        "passed": all(item["passed"] for item in checks.values()) and not missing_paths and not duplicate_paths,
        "counts": counts,
        "checks": checks,
        "missing_paths": missing_paths,
        "duplicate_paths": duplicate_paths,
    }


def evaluate_phases(phase_contexts: list[dict[str, Any]], expected: dict[str, Any], suite: dict[str, Any]) -> dict[str, Any]:
    expected_phases = list(expected.get("phases", []))
    observed = [phase_name_from_context(context) for context in phase_contexts]
    observed = [phase for phase in observed if phase]
    missing = [phase for phase in expected_phases if not any(phase_alias_matches(phase, item) for item in observed)]
    loop_guard_count = sum(1 for context in phase_contexts if int(context.get("phase_window_count", 0) or 0) >= int(suite.get("defaults", {}).get("max_phase_windows", 5)))
    return {
        "passed": bool(observed) or not expected_phases,
        "expected": expected_phases,
        "observed": observed,
        "missing": missing,
        "note": "Phase names are free-form labels; missing names are advisory when motion criteria can be matched by order.",
        "loop_guard_like_contexts": loop_guard_count,
        "phase_count": len(observed),
    }


def evaluate_motion(phase_contexts: list[dict[str, Any]], expected: dict[str, Any], suite: dict[str, Any]) -> dict[str, Any]:
    metrics = []
    cursor = 0
    for metric in expected.get("phase_metrics", []):
        phase = metric.get("phase")
        context, cursor, match_status = context_for_metric(phase_contexts, metric, cursor)
        summary = context.get("phase_execution_summary", {}) if isinstance(context, dict) else {}
        actual = metric_actual(summary, metric)
        target = float(metric.get("target", 0.0) or 0.0)
        tolerance = float(metric.get("tolerance", default_tolerance(metric.get("field"), suite)) or 0.0)
        error = None if actual is None else round(float(actual) - target, 3)
        passed = actual is not None and abs(float(error)) <= tolerance
        metrics.append(
            {
                "phase": phase,
                "field": metric.get("field"),
                "target": target,
                "actual": actual,
                "error": error,
                "tolerance": tolerance,
                "passed": passed,
                "window_count": summary.get("window_count"),
                "observed_phase": phase_name_from_context(context) if context else None,
                "match_status": match_status,
                "status": phase_timing_status(summary, actual, target, tolerance),
            }
        )
    return {"passed": all(item["passed"] for item in metrics) if metrics else True, "metrics": metrics}


def evaluate_continuous(
    demo_summary: dict[str, Any],
    events: list[dict[str, Any]],
    expected: dict[str, Any],
    suite: dict[str, Any],
) -> dict[str, Any]:
    instructions = demo_summary.get("instructions", []) if isinstance(demo_summary.get("instructions", []), list) else []
    context_updates = sum(1 for event in events if event.get("event") == "phase_window_recorded")
    action_event_name = _preferred_action_event_name(events)
    action_count = sum(1 for event in events if event.get("event") == action_event_name)
    checks = {
        "instruction_count": {"actual": len(instructions), "minimum": int(expected.get("min_instruction_count", 0) or 0)},
        "context_updates": {"actual": context_updates, "minimum": int(expected.get("min_context_updates", 0) or 0)},
        "action_count": {"actual": action_count, "minimum": int(expected.get("min_action_count", 0) or 0)},
    }
    for item in checks.values():
        item["passed"] = item["actual"] >= item["minimum"]
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "instruction_results": continuous_instruction_results(events),
    }


def evaluate_batch_instructions(
    demo_summary: dict[str, Any],
    events: list[dict[str, Any]],
    suite: dict[str, Any],
    run_dir: Path,
    task: dict[str, Any],
) -> dict[str, Any]:
    instructions = instructions_from_summary_or_events(demo_summary, events)
    is_batch = bool(task.get("batch_kind")) or len(instructions) > 1
    if not is_batch:
        return {"is_batch": False, "passed": True, "instruction_results": []}

    results = []
    for index, segment in enumerate(split_events_by_instruction(events), start=1):
        instruction = str(segment.get("instruction", ""))
        segment_events = list(segment.get("events", []))
        matched_task = single_task_by_instruction(suite, instruction)
        expected = dict(matched_task.get("expected", {})) if matched_task else {}
        actions = evaluate_actions(extract_action_sequence(segment_events, {}))
        phase_contexts = extract_phase_contexts(segment_events, {}, run_dir)
        action_windows = evaluate_action_windows(phase_contexts, segment_events)
        observations = extract_observations(segment_events, {}, run_dir)
        artifact_inventory = build_artifact_inventory(observations, run_dir)
        artifacts = evaluate_artifacts(artifact_inventory, expected, suite)
        phases = evaluate_phases(phase_contexts, expected, suite)
        motion = evaluate_motion(phase_contexts, expected, suite)
        completed = any(
            event.get("event") == "phase_transition_applied"
            and isinstance(event.get("payload"), dict)
            and event["payload"].get("decision") == "complete_task"
            for event in segment_events
        )
        passed = (
            bool(completed)
            and actions["passed"]
            and action_windows["passed"]
            and artifacts["passed"]
            and phases["passed"]
            and motion["passed"]
        )
        results.append(
            {
                "index": index,
                "instruction": instruction,
                "task_id": matched_task.get("id") if matched_task else None,
                "completed": completed,
                "passed": passed,
                "actions": {
                    "total": actions["total"],
                    "success_rate": actions["success_rate"],
                    "forbidden_tools": len(actions["forbidden_tools"]),
                },
                "action_windows": {
                    "total": action_windows["total"],
                    "invalid_scopes": len(action_windows["invalid_scopes"]),
                    "observation_mismatches": len(action_windows["observation_mismatches"]),
                },
                "artifacts": {
                    "counts": artifacts["counts"],
                    "missing_paths": len(artifacts["missing_paths"]),
                    "duplicate_paths": len(artifacts.get("duplicate_paths", [])),
                },
                "phases": {
                    "observed": phases["observed"],
                    "missing": phases["missing"],
                },
                "motion": motion["metrics"],
            }
        )

    return {
        "is_batch": True,
        "kind": task.get("batch_kind", "unknown"),
        "passed": bool(results) and all(item["passed"] for item in results),
        "instruction_count": len(results),
        "instruction_results": results,
    }


def continuous_instruction_results(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split a continuous run by task_parsed events for per-instruction reporting."""

    results: list[dict[str, Any]] = []
    action_event_name = _preferred_action_event_name(events)
    for segment in split_events_by_instruction(events):
        current = {
            "instruction": segment.get("instruction", ""),
            "completed": False,
            "action_total": 0,
            "action_ok": 0,
            "context_updates": 0,
            "rgb_observations": 0,
            "depth_observations": 0,
            "observed_phases": [],
        }
        for event in segment.get("events", []):
            name = event.get("event")
            payload = event.get("payload", {}) if isinstance(event.get("payload", {}), dict) else {}
            if name == action_event_name:
                current["action_total"] += 1
                result = payload.get("result", {}) if isinstance(payload.get("result", {}), dict) else {}
                if result.get("ok") is True or payload.get("ok") is True:
                    current["action_ok"] += 1
            elif name == "phase_window_recorded":
                current["context_updates"] += 1
                phase = phase_name_from_context(payload)
                if phase:
                    current["observed_phases"].append(phase)
            elif name in {"automatic_observation_updated", "instruction_start_observation_updated", "initial_observation_updated"}:
                frames = payload.get("frames", {}) if isinstance(payload.get("frames", {}), dict) else {}
                if "rgb" in frames:
                    current["rgb_observations"] += 1
                if "depth" in frames:
                    current["depth_observations"] += 1
            elif name == "phase_transition_applied":
                if payload.get("decision") == "complete_task":
                    current["completed"] = True
        results.append(current)
    for item in results:
        total = int(item.get("action_total", 0) or 0)
        ok = int(item.get("action_ok", 0) or 0)
        item["action_success_rate"] = round(ok / total, 3) if total else 0.0
        item["passed"] = bool(item.get("completed")) and total > 0 and ok == total
    return results


def metric_actual(summary: dict[str, Any], metric: dict[str, Any]) -> float | None:
    tool_fact = metric.get("prefer_tool_fact")
    facts = summary.get("cumulative_tool_facts", {}) if isinstance(summary, dict) else {}
    if tool_fact and tool_fact in facts:
        return safe_float(facts.get(tool_fact))
    field = metric.get("field")
    for section in ("phase_local_measurements", "cumulative_measurements", "last_window_measurements"):
        values = summary.get(section, {}) if isinstance(summary.get(section, {}), dict) else {}
        if field in values:
            return safe_float(values.get(field))
    return None


def phase_timing_status(summary: dict[str, Any], actual: float | None, target: float, tolerance: float) -> str:
    if actual is None:
        return "missing_measurement"
    window_count = int(summary.get("window_count", 0) or 0)
    if abs(float(actual) - target) <= tolerance:
        return "within_tolerance"
    if window_count >= int(summary.get("max_phase_window_count", 5) or 5):
        return "late_or_loop_guard_risk"
    if abs(float(actual)) < abs(target) - tolerance:
        return "advanced_or_incomplete_below_target"
    return "overshoot"


def context_for_metric(
    phase_contexts: list[dict[str, Any]],
    metric: dict[str, Any],
    cursor: int,
) -> tuple[dict[str, Any], int, str]:
    """Find the best phase context for a metric without requiring stable phase names."""

    expected_phase = str(metric.get("phase", ""))
    for index in range(cursor, len(phase_contexts)):
        if phase_alias_matches(expected_phase, phase_name_from_context(phase_contexts[index])):
            end = index
            while end + 1 < len(phase_contexts) and phase_alias_matches(
                expected_phase,
                phase_name_from_context(phase_contexts[end + 1]),
            ):
                end += 1
            return phase_contexts[end], end + 1, "phase_name_or_alias"
    if cursor < len(phase_contexts):
        return phase_contexts[cursor], cursor + 1, "sequential_fallback"
    return {}, cursor, "missing_context"


def phase_alias_matches(expected: str, observed: str) -> bool:
    if not expected or not observed:
        return False
    if expected == observed:
        return True
    if observed.startswith(f"{expected}_"):
        return True
    aliases = {
        "move_left": {"strafe_left"},
        "move_right": {"strafe_right"},
        "move_up": {"adjust_altitude_up", "climb", "ascend"},
        "move_down": {"adjust_altitude_down", "adjust_altitude_down_and_stabilize", "descend"},
        "move_forward_again": {"move_forward", "move_forward_then_hover"},
        "rotate_left": {"rotate_left_and_stabilize"},
        "rotate_right": {"rotate_right_and_stabilize"},
        "hover": {"stabilize_hover", "stop", "orient_and_hover"},
        "scan_right": {"scan_right_slowly"},
        "orient_to_landmark": {"orient_and_hover", "align_to_landmark"},
    }
    return any(observed == alias or observed.startswith(f"{alias}_") for alias in aliases.get(expected, set()))


def phase_name_from_context(context: dict[str, Any]) -> str:
    summary = context.get("phase_execution_summary", {}) if isinstance(context.get("phase_execution_summary", {}), dict) else {}
    if summary.get("phase"):
        return str(summary["phase"])
    current_phase = context.get("current_phase", {}) if isinstance(context.get("current_phase", {}), dict) else {}
    return str(current_phase.get("name", ""))


def build_artifact_inventory(observations: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    inventory: dict[str, list[dict[str, Any]]] = {"rgb": [], "depth": [], "other": []}
    for obs_index, observation in enumerate(observations):
        frames = observation.get("frames", {}) if isinstance(observation, dict) else {}
        if not isinstance(frames, dict):
            continue
        for modality, metadata in frames.items():
            if not isinstance(metadata, dict) or "artifact_path" not in metadata:
                continue
            normalized = "rgb" if str(modality).lower() in {"rgb", "scene", "color"} else str(modality).lower()
            bucket = normalized if normalized in {"rgb", "depth"} else "other"
            path = Path(str(metadata["artifact_path"]))
            if not path.is_absolute():
                path = REPO_ROOT / path
            inventory[bucket].append(
                {
                    "observation_index": obs_index,
                    "modality": str(modality),
                    "path": str(path),
                    "exists": path.exists(),
                    "width": metadata.get("width"),
                    "height": metadata.get("height"),
                }
            )
    return {**inventory, "counts": {name: len(items) for name, items in inventory.items()}}


def overall_status(metrics: dict[str, Any]) -> dict[str, Any]:
    sections = ["basic", "actions", "action_windows", "artifacts", "phases", "motion", "continuous"]
    if metrics.get("batch", {}).get("is_batch"):
        sections.append("batch")
    failed = [section for section in sections if metrics.get(section, {}).get("passed") is False]
    return {"passed": not failed, "failed_sections": failed}


def build_markdown_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Real Run Evaluation Report",
        "",
        f"- Run: `{metrics['run_dir']}`",
        f"- Task: `{metrics['task'].get('id', 'unknown')}`",
        f"- Overall: {'PASS' if metrics['overall']['passed'] else 'FAIL'}",
        "",
        "## Basic",
        "",
        f"- completed: `{metrics['basic'].get('completed')}`",
        f"- connected: `{metrics['basic'].get('connected')}`",
        f"- airborne: `{metrics['basic'].get('airborne')}`",
        f"- has latest checkpoint: `{metrics['basic'].get('has_latest_checkpoint')}`",
        f"- has phase context: `{metrics['basic'].get('has_phase_context')}`",
        "",
        "## Actions",
        "",
        f"- total: `{metrics['actions']['total']}`",
        f"- ok: `{metrics['actions']['ok_count']}`",
        f"- success rate: `{metrics['actions']['success_rate']}`",
        f"- forbidden tools: `{len(metrics['actions']['forbidden_tools'])}`",
        "",
        "## Action Windows",
        "",
        f"- total: `{metrics['action_windows']['total']}`",
        f"- invalid scopes: `{len(metrics['action_windows']['invalid_scopes'])}`",
        f"- observation mismatches: `{len(metrics['action_windows']['observation_mismatches'])}`",
        f"- missing descriptions: `{len(metrics['action_windows']['missing_descriptions'])}`",
        "",
        "## Phases",
        "",
        f"- expected: `{metrics['phases']['expected']}`",
        f"- observed: `{metrics['phases']['observed']}`",
        f"- missing: `{metrics['phases']['missing']}`",
        "",
        "## Motion Metrics",
        "",
    ]
    if metrics["motion"]["metrics"]:
        lines.append("| phase | field | target | actual | error | tolerance | status | pass |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
        for item in metrics["motion"]["metrics"]:
            lines.append(
                f"| {item['phase']} | {item['field']} | {item['target']} | {item['actual']} | {item['error']} | {item['tolerance']} | {item['status']} | {item['passed']} |"
            )
    else:
        lines.append("- No motion metrics configured for this task.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- counts: `{metrics['artifacts']['counts']}`",
            f"- missing artifact paths: `{len(metrics['artifacts']['missing_paths'])}`",
            f"- duplicate artifact paths: `{len(metrics['artifacts'].get('duplicate_paths', []))}`",
            "",
            "## Continuous Checks",
            "",
            f"- checks: `{metrics['continuous']['checks']}`",
        ]
    )
    if metrics.get("batch", {}).get("is_batch"):
        lines.extend(["", "## Batch Instructions", ""])
        lines.append("| # | task | completed | actions | action windows | duplicate artifacts | pass |")
        lines.append("| ---: | --- | --- | ---: | ---: | ---: | --- |")
        for item in metrics["batch"].get("instruction_results", []):
            lines.append(
                f"| {item['index']} | {item.get('task_id')} | {item['completed']} | {item['actions']['total']} | {item.get('action_windows', {}).get('total', 0)} | {item['artifacts']['duplicate_paths']} | {item['passed']} |"
            )
    return "\n".join(lines) + "\n"


def write_eval_outputs(result: dict[str, Any]) -> None:
    run_dir = Path(result["run_dir"])
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (eval_dir / "report.md").write_text(result["report"], encoding="utf-8")
    with (eval_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_dir", "task_id", "passed", "failed_sections", "action_success_rate", "rgb_count", "depth_count"])
        writer.writeheader()
        writer.writerow(
            {
                "run_dir": metrics["run_dir"],
                "task_id": metrics["task"].get("id", ""),
                "passed": metrics["overall"]["passed"],
                "failed_sections": ";".join(metrics["overall"]["failed_sections"]),
                "action_success_rate": metrics["actions"]["success_rate"],
                "rgb_count": metrics["artifacts"]["counts"].get("rgb", 0),
                "depth_count": metrics["artifacts"]["counts"].get("depth", 0),
            }
        )


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    return {
        "run_dir": metrics["run_dir"],
        "task_id": metrics["task"].get("id"),
        "passed": metrics["overall"]["passed"],
        "failed_sections": metrics["overall"]["failed_sections"],
        "actions": {"total": metrics["actions"]["total"], "success_rate": metrics["actions"]["success_rate"]},
        "artifacts": metrics["artifacts"]["counts"],
        "duplicate_artifact_paths": len(metrics["artifacts"].get("duplicate_paths", [])),
        "batch": {
            "is_batch": metrics.get("batch", {}).get("is_batch", False),
            "instruction_count": metrics.get("batch", {}).get("instruction_count", 0),
            "passed": metrics.get("batch", {}).get("passed", True),
        },
    }


def task_summary(task: dict[str, Any]) -> dict[str, Any]:
    return {"id": task.get("id", "unknown"), "instruction": task.get("instruction"), "instructions": task.get("instructions", [])}


def default_tolerance(field: str, suite: dict[str, Any]) -> float:
    tolerances = suite.get("defaults", {}).get("tolerances", {})
    return float(tolerances.get(field, 0.0) or 0.0)


def normalize_instruction(value: str) -> str:
    return "".join(str(value).split())


def normalize_instruction_list(values: list[Any]) -> list[str]:
    return [normalize_instruction(str(value)) for value in values]


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_first_json(*paths: Path) -> dict[str, Any] | None:
    for path in paths:
        data = load_json(path)
        if data is not None:
            return data
    return None


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


if __name__ == "__main__":
    main()
