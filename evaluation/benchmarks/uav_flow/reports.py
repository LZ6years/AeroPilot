"""Static report writers for UAV-Flow benchmark outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from evaluation.benchmarks.uav_flow.metrics import metric_interpretation_for_class, summarize_metrics
from evaluation.benchmarks.uav_flow.trajectory import target_local_from_case
from src.runtime.configs.action_config import UavActionConfig, load_uav_action_config
from src.uav import body_velocity_to_world_delta, move_body_velocity_from_args, pose_delta


def create_eval_dir(output_root: str | Path, eval_id: str | None = None) -> Path:
    run_id = eval_id or f"uav_flow_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    path = Path(output_root) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_case_outputs(case_dir: str | Path, payloads: dict[str, Any]) -> dict[str, str]:
    path = Path(case_dir)
    path.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, payload in payloads.items():
        file_path = path / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        written[name] = str(file_path)
    return written


def run_artifact_summary(run_dir: str | Path, action_config: UavActionConfig | None = None) -> dict[str, Any]:
    """Collect lightweight run details for human review reports."""

    run_path = Path(run_dir)
    rgb_frames = sorted((run_path / "artifacts" / "rgb").glob("*.png"))
    action_config = action_config or load_uav_action_config()
    return {
        "first_rgb": str(rgb_frames[0]) if rgb_frames else None,
        "last_rgb": str(rgb_frames[-1]) if rgb_frames else None,
        "rgb_count": len(rgb_frames),
        "actions": _actions_from_run(run_path, action_config),
        "agent_messages": _agent_messages_from_events(run_path / "events.jsonl"),
    }


def write_case_review(case_dir: str | Path, case_payload: dict[str, Any]) -> str:
    """Write a compact human-readable per-case review file."""

    path = Path(case_dir) / "review.md"
    case = case_payload.get("case", {})
    artifacts = case_payload.get("run_artifacts", {})
    visual = case_payload.get("visual_diagnostics", {})
    ndtw = case_payload.get("ndtw", {})
    metric_interpretation = ndtw.get("metric_interpretation") or metric_interpretation_for_class(case.get("class_name"))
    sr = case_payload.get("sr_judge", {})
    lines = [
        f"# UAV-Flow Case Review: {case.get('case_id')}",
        "",
        f"- Instruction: {case.get('selected_instruction')}",
        f"- Class: {case.get('class_name')}",
        f"- Status: {case_payload.get('status')}",
        f"- Completed: {case_payload.get('completed')}",
        f"- Stopped: {case_payload.get('stopped')}",
        f"- Stopped reason: {case_payload.get('stopped_reason')}",
        f"- Stop source: {case_payload.get('stop_source')}",
        f"- Run dir: `{case_payload.get('run_dir')}`",
        f"- First RGB: `{artifacts.get('first_rgb')}`",
        f"- Last RGB: `{artifacts.get('last_rgb')}`",
        f"- RGB count: {artifacts.get('rgb_count', 0)}",
        f"- Visual diagnostics: {visual.get('status')} enabled={visual.get('enabled')} judged={visual.get('judged_count', 0)}/{visual.get('frame_count', 0)} visible={visual.get('person_visible_count', 0)}",
        f"- Visual diagnostics file: `{visual.get('path')}`",
        f"- nDTW: {ndtw.get('ndtw')} ({ndtw.get('status')})",
        f"- Metric mode: {metric_interpretation.get('mode')} position_zeroed={metric_interpretation.get('position_zeroed')} sample_step={metric_interpretation.get('sample_step')}",
        f"- Metric interpretation: {metric_interpretation.get('summary')}",
        f"- SR judge: {sr.get('status')}",
        "",
        "## Actions",
        "",
    ]
    if artifacts.get("actions"):
        lines.extend(
            [
                "| # | Tool | Phase | Args | Move? | Expected delta m | Actual delta m | Planar m | OK |",
                "|---:|---|---|---|---|---|---|---:|---|",
            ]
        )
    for action in artifacts.get("actions", []):
        lines.append(
            "| {index} | `{tool}` | {phase} | `{args}` | {is_move} | `{expected}` | `{actual}` | {planar} | {ok} |".format(
                index=action.get("step_index"),
                tool=action.get("tool"),
                phase=action.get("phase"),
                args=action.get("args"),
                is_move=action.get("is_move"),
                expected=action.get("expected_delta_m"),
                actual=action.get("actual_delta_m"),
                planar=action.get("actual_planar_m"),
                ok=action.get("ok"),
            )
        )
    if not artifacts.get("actions"):
        lines.append("- No tool step events recorded.")
    lines.extend(["", "## Visual Diagnostics", ""])
    lines.append(
        f"- status={visual.get('status')} enabled={visual.get('enabled')} "
        f"truth_leak={visual.get('truth_leak_check', {}).get('status')}"
    )
    lines.append(f"- path: `{visual.get('path')}`")
    lines.extend(["", "## Agent Messages", ""])
    for message in artifacts.get("agent_messages", []):
        body = message.get("message", {})
        lines.append(f"- {body.get('type')}: {body.get('content')}")
    if not artifacts.get("agent_messages"):
        lines.append("- No AgentMessage events recorded.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def write_trajectory_plots(case_dir: str | Path, trajectory_path: str | Path, case: UavFlowCase) -> dict[str, Any]:
    """Write simple 2D/3D trajectory plots when matplotlib is available."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "skipped", "reason": "matplotlib_unavailable", "error": str(exc)}

    try:
        data = json.loads(Path(trajectory_path).read_text(encoding="utf-8"))
        points = [item["state"][0] for item in data if isinstance(item, dict) and "state" in item]
        if not points:
            return {"status": "skipped", "reason": "empty_trajectory"}
        plot_dir = Path(case_dir) / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        target = target_local_from_case(case)
        reference_points = _reference_points_from_case(case)
        path_2d = plot_dir / "trajectory_2d.png"
        path_3d = plot_dir / "trajectory_3d.png"
        _plot_2d(plt, points, target, path_2d, reference_points)
        _plot_3d(plt, points, target, path_3d, reference_points)
        return {
            "status": "ok",
            "trajectory_2d": str(path_2d),
            "trajectory_3d": str(path_3d),
            "agent_point_count": len(points),
            "reference_point_count": len(reference_points),
            "has_reference_path": bool(reference_points),
        }
    except Exception as exc:
        return {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}


def write_eval_summary(eval_dir: str | Path, results: list[dict[str, Any]], console_lines: list[str]) -> dict[str, str]:
    path = Path(eval_dir)
    summary = {
        "metrics": summarize_metrics(results),
        "failure_counts": failure_counts(results),
        "cases": results,
    }
    summary_json = path / "summary.json"
    summary_md = path / "summary.md"
    console_txt = path / "console.txt"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_md.write_text(_summary_markdown(summary), encoding="utf-8")
    console_txt.write_text("\n".join(console_lines) + ("\n" if console_lines else ""), encoding="utf-8")
    return {"summary_json": str(summary_json), "summary_md": str(summary_md), "console_txt": str(console_txt)}


def _summary_markdown(summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics", {})
    failure_counts_data = summary.get("failure_counts", {})
    lines = [
        "# UAV-Flow Benchmark Summary",
        "",
        f"- Cases: {metrics.get('case_count', 0)}",
        f"- Evaluated nDTW: {metrics.get('evaluated_count', 0)}",
        f"- Overall mean nDTW: {metrics.get('overall_mean_ndtw')}",
        "",
        "## Failure Counts",
        "",
        "| Type | Count |",
        "|---|---:|",
    ]
    for name, count in sorted(failure_counts_data.items()):
        lines.append(f"| {name} | {count} |")
    lines.extend([
        "",
        "## Metric Notes",
        "",
        "- Turn/Rotate use the original UAV-Flow rotation-sequence nDTW mode: xyz position is zeroed before DTW, so low scores should be read as sampled orientation-sequence mismatch, not planar movement failure.",
        "- Move and other translation classes use position-and-rotation nDTW, so translation path differences directly affect the score.",
        "",
        "## By Class",
        "",
        "| Class | Evaluated | Mean nDTW |",
        "|---|---:|---:|",
    ])
    for class_name, item in metrics.get("by_class", {}).items():
        lines.append(f"| {class_name} | {item.get('evaluated_count', 0)} | {item.get('mean_ndtw')} |")
    lines.extend(["", "## Cases", ""])
    for result in summary.get("cases", []):
        ndtw = result.get("ndtw", {})
        lines.append(
            f"- `{result.get('case_id')}` ({result.get('class_name')}) "
            f"status={result.get('status')} ndtw={ndtw.get('ndtw')}"
        )
    return "\n".join(lines) + "\n"


def failure_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "env_start_failed": 0,
        "model_failed": 0,
        "runtime_failed": 0,
        "metric_skipped": 0,
        "sr_skipped": 0,
        "runtime_loop_guard_reached": 0,
    }
    for result in results:
        error = result.get("error") or {}
        error_text = f"{error.get('error_type', '')} {error.get('error', '')}".lower()
        if result.get("stopped_reason") == "runtime_loop_guard_reached":
            counts["runtime_loop_guard_reached"] += 1
        if result.get("status") == "error":
            if any(term in error_text for term in ("unrealcv", "gym", "env", "scene", "object placement")):
                counts["env_start_failed"] += 1
            elif any(term in error_text for term in ("model", "api", "http", "urlerror")):
                counts["model_failed"] += 1
            else:
                counts["runtime_failed"] += 1
        if (result.get("ndtw") or {}).get("status") == "skipped":
            counts["metric_skipped"] += 1
        if (result.get("sr_judge") or {}).get("status") == "skipped":
            counts["sr_skipped"] += 1
    return counts


def _actions_from_run(run_path: Path, action_config: UavActionConfig) -> list[dict[str, Any]]:
    step_events = _tool_step_events(run_path / "events.jsonl")
    action_records = _read_jsonl(run_path / "memory" / "actions.jsonl")
    pose_records = _read_jsonl(run_path / "memory" / "poses.jsonl")
    if not action_records:
        return [
            {
                "phase": step.get("phase"),
                "step_index": step.get("step_index"),
                "tool": step.get("tool"),
                "ok": step.get("ok"),
                "args": {},
                "is_move": step.get("tool") == "uav.move_body",
                "actual_delta_m": None,
                "expected_delta_m": None,
                "actual_planar_m": None,
            }
            for step in step_events
        ]
    matched = _match_action_records(step_events, action_records)
    summaries = []
    for step, record in matched:
        before_pose, after_pose = _poses_around_action(record, pose_records)
        actual = pose_delta(before_pose or {}, after_pose or {}) if before_pose and after_pose else {}
        expected = _expected_delta(record.get("tool"), record.get("args", {}), before_pose or {}, action_config)
        planar = _planar_distance(actual)
        summaries.append(
            {
                "phase": step.get("phase"),
                "step_index": step.get("step_index"),
                "tool": step.get("tool"),
                "ok": step.get("ok"),
                "action_id": record.get("id"),
                "args": record.get("args", {}),
                "is_move": step.get("tool") == "uav.move_body",
                "actual_delta_m": actual or None,
                "expected_delta_m": expected,
                "actual_planar_m": planar,
            }
        )
    return summaries


def _tool_step_events(path: Path) -> list[dict[str, Any]]:
    actions = []
    for event in _read_jsonl(path):
        if event.get("event") != "tool_step_completed":
            continue
        payload = event.get("payload", {})
        if isinstance(payload, dict):
            actions.append(dict(payload))
    return actions


def _match_action_records(
    step_events: list[dict[str, Any]],
    action_records: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    matches = []
    cursor = 0
    for step in step_events:
        tool = step.get("tool")
        for index in range(cursor, len(action_records)):
            record = action_records[index]
            if record.get("tool") == tool:
                matches.append((step, record))
                cursor = index + 1
                break
    return matches


def _poses_around_action(
    action_record: dict[str, Any],
    pose_records: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    action_id = action_record.get("id")
    after_index = next(
        (
            index
            for index, pose in enumerate(pose_records)
            if pose.get("source") == "action_result" and pose.get("action_id") == action_id
        ),
        None,
    )
    if after_index is None:
        return None, None
    before = pose_records[after_index - 1].get("pose") if after_index > 0 else None
    after = pose_records[after_index].get("pose")
    return (
        dict(before) if isinstance(before, dict) else None,
        dict(after) if isinstance(after, dict) else None,
    )


def _expected_delta(
    tool: Any,
    args: dict[str, Any],
    before_pose: dict[str, Any],
    action_config: UavActionConfig,
) -> dict[str, float | None]:
    if tool == "uav.turn_relative":
        return {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw_deg": round(float(args.get("yaw_deg", 0.0) or 0.0), 3)}
    if tool != "uav.move_body":
        return {"dx": 0.0, "dy": 0.0, "dz": 0.0, "dyaw_deg": 0.0}
    vx, vy, vz, duration = move_body_velocity_from_args(args, action_config, vertical_sign=1.0)
    delta = body_velocity_to_world_delta(vx, vy, vz, duration, float(before_pose.get("yaw_deg", 0.0) or 0.0))
    return {
        "dx": round(delta["dx"], 3),
        "dy": round(delta["dy"], 3),
        "dz": round(delta["dz"], 3),
        "dyaw_deg": 0.0,
    }


def _planar_distance(delta: dict[str, Any]) -> float | None:
    try:
        dx = float(delta.get("dx") or 0.0)
        dy = float(delta.get("dy") or 0.0)
    except (TypeError, ValueError):
        return None
    return round((dx * dx + dy * dy) ** 0.5, 3)


def _agent_messages_from_events(path: Path) -> list[dict[str, Any]]:
    messages = []
    for event in _read_jsonl(path):
        if event.get("event") == "agent_message":
            payload = event.get("payload", {})
            if isinstance(payload, dict):
                messages.append(payload)
    return messages


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _reference_points_from_case(case: UavFlowCase) -> list[list[float]]:
    points = []
    for item in case.reference_path_preprocessed:
        if isinstance(item, list) and len(item) >= 3:
            points.append([float(item[0]), float(item[1]), float(item[2])])
    return points


def _plot_2d(
    plt: Any,
    points: list[list[float]],
    target: list[float] | None,
    path: Path,
    reference_points: list[list[float]] | None = None,
) -> None:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    plt.figure(figsize=(8, 8))
    if reference_points:
        ref_xs = [point[0] for point in reference_points]
        ref_ys = [point[1] for point in reference_points]
        plt.plot(ref_ys, ref_xs, marker="x", linestyle="--", color="gray", label="reference_path")
    plt.plot(ys, xs, marker="o", color="blue", label="uav_agent")
    plt.scatter([ys[0]], [xs[0]], color="green", label="agent_start")
    plt.scatter([ys[-1]], [xs[-1]], color="blue", marker="s", label="agent_end")
    if target:
        plt.scatter([target[1]], [target[0]], color="red", label="target")
    plt.xlabel("Y local cm")
    plt.ylabel("X local cm")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_3d(
    plt: Any,
    points: list[list[float]],
    target: list[float] | None,
    path: Path,
    reference_points: list[list[float]] | None = None,
) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    if reference_points:
        ref_xs = [point[0] for point in reference_points]
        ref_ys = [point[1] for point in reference_points]
        ref_zs = [point[2] for point in reference_points]
        ax.plot(ref_ys, ref_xs, ref_zs, marker="x", linestyle="--", color="gray", label="reference_path")
    ax.plot(ys, xs, zs, marker="o", color="blue", label="uav_agent")
    ax.scatter([ys[0]], [xs[0]], [zs[0]], color="green", label="agent_start")
    ax.scatter([ys[-1]], [xs[-1]], [zs[-1]], color="blue", marker="s", label="agent_end")
    if target:
        ax.scatter([target[1]], [target[0]], [target[2]], color="red", label="target")
    ax.set_xlabel("Y local cm")
    ax.set_ylabel("X local cm")
    ax.set_zlabel("Z local cm")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
