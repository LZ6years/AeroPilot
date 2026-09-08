"""Static report writers for UAV-Flow benchmark outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from evaluation.benchmarks.uav_flow.metrics import summarize_metrics
from evaluation.benchmarks.uav_flow.trajectory import target_local_from_case
from evaluation.benchmarks.core.report_io import atomic_write_text


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


def write_eval_summary(
    eval_dir: str | Path,
    results: list[dict[str, Any]],
    console_lines: list[str],
    *,
    report_metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write an atomic evaluation snapshot for completed cases.

    ``report_metadata`` describes the batch lifecycle only.  It never changes
    per-case metrics, and its defaults preserve the historical meaning of a
    normally completed run.
    """

    path = Path(eval_dir)
    metadata = dict(report_metadata or {})
    metadata.setdefault("run_status", "completed")
    metadata.setdefault("expected_case_count", len(results))
    metadata["finalized_case_count"] = len(results)
    metadata.setdefault("incomplete_case_ids", [])
    metadata.setdefault("report_generated_from", "live_runner")
    summary = {
        **metadata,
        "metrics": summarize_metrics(results),
        "failure_counts": failure_counts(results),
        "backend_action_tracking": _summarize_backend_action_tracking(
            results
        ),
        "cases": results,
    }
    summary_json = path / "summary.json"
    summary_md = path / "summary.md"
    console_txt = path / "console.txt"
    atomic_write_text(
        summary_json,
        json.dumps(summary, indent=2, ensure_ascii=False),
    )
    atomic_write_text(summary_md, _summary_markdown(summary))
    atomic_write_text(
        console_txt,
        "\n".join(console_lines) + ("\n" if console_lines else ""),
    )
    return {"summary_json": str(summary_json), "summary_md": str(summary_md), "console_txt": str(console_txt)}


def _summary_markdown(summary: dict[str, Any]) -> str:
    metrics = summary.get("metrics", {})
    failure_counts_data = summary.get("failure_counts", {})
    action_tracking = summary.get("backend_action_tracking", {})
    lines = [
        "# UAV-Flow Benchmark Summary",
        "",
        f"- Run status: {summary.get('run_status', 'completed')}",
        f"- Finalized cases: {summary.get('finalized_case_count', metrics.get('case_count', 0))}",
        f"- Expected cases: {summary.get('expected_case_count', metrics.get('case_count', 0))}",
        f"- Incomplete cases: {summary.get('incomplete_case_ids', [])}",
        f"- Cases: {metrics.get('case_count', 0)}",
        f"- Evaluated nDTW: {metrics.get('evaluated_count', 0)}",
        f"- Overall mean nDTW: {metrics.get('overall_mean_ndtw')}",
        f"- Zero-motion completions: "
        f"{action_tracking.get('zero_motion_completion_count', 0)}",
        f"- Max translation tracking error (m): "
        f"{action_tracking.get('max_position_error_m')}",
        f"- Max yaw tracking error (deg): "
        f"{action_tracking.get('max_abs_yaw_error_deg')}",
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
        turn = result.get("turn_diagnostics") or {}
        tracking = result.get("backend_action_tracking") or {}
        lines.append(
            f"- `{result.get('case_id')}` ({result.get('class_name')}) "
            f"status={result.get('status')} ndtw={ndtw.get('ndtw')} "
            f"reference_yaw_delta={turn.get('reference_yaw_delta_deg')} "
            f"actual_yaw_delta={turn.get('actual_yaw_delta_deg')} "
            f"final_yaw_error={turn.get('final_yaw_error_deg')} "
            f"translation_drift_m={turn.get('translation_drift_m')} "
            f"translation_tracking_max_error_m="
            f"{tracking.get('max_position_error_m')} "
            f"zero_motion_completion="
            f"{tracking.get('zero_motion_completion')}"
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


def _summarize_backend_action_tracking(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    position_errors = []
    yaw_errors = []
    zero_motion_count = 0
    trace_invalid_count = 0
    tracking_invalid_count = 0
    for result in results:
        tracking = result.get("backend_action_tracking") or {}
        if tracking.get("zero_motion_completion") is True:
            zero_motion_count += 1
        if tracking.get("trace_valid") is not True:
            trace_invalid_count += 1
        if tracking.get("tracking_valid") is not True:
            tracking_invalid_count += 1
        position_error = tracking.get("max_position_error_m")
        yaw_error = tracking.get("max_abs_yaw_error_deg")
        if isinstance(position_error, (int, float)):
            position_errors.append(float(position_error))
        if isinstance(yaw_error, (int, float)):
            yaw_errors.append(float(yaw_error))
    return {
        "zero_motion_completion_count": zero_motion_count,
        "trace_invalid_count": trace_invalid_count,
        "tracking_invalid_count": tracking_invalid_count,
        "max_position_error_m": (
            max(position_errors) if position_errors else None
        ),
        "max_abs_yaw_error_deg": (
            max(yaw_errors) if yaw_errors else None
        ),
    }


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
