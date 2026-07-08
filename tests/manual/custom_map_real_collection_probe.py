"""Collect real UAV-Flow depth/mask/pose samples for custom_map debugging.

This is a standalone research probe, not a pytest test and not part of the
default planner context. It enables UAV-Flow real UnrealCV frame capture, runs a
short deterministic action script, writes run-local FlightMemory records, and
renders the custom_map PNG/JSON/HTML artifacts for inspection.
"""

from __future__ import annotations

import argparse
import os
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.uav_flow.cases import UavFlowCase, load_cases
from evaluation.benchmarks.uav_flow.config import UavFlowEvalConfig
from evaluation.benchmarks.uav_flow.unrealcv_scenario import UavFlowBenchmarkEnvClient
from plugins.research.custom_map.context_modules.custom_map import PluginContextModule
from plugins.research.custom_map.perception import (
    ClosedSetSegmentationConfig,
    DEFAULT_SEGFORMER_MODEL,
    SegFormerBackend,
    run_closed_set_segmentation,
)
from plugins.research.custom_map.perception.segmentation_writer import append_segmentation_result
from src.memory.context.schemas import MemorySourceBundle, StageContextInput
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.perception.raw_observation import RawObservation
from src.runtime.configs.action_config import UavActionConfig, load_uav_action_config


DEFAULT_CLASSES = ["Turn", "Pass", "Surround"]
AGENT_ID = "uav_flow_drone"
MAP_ID = "uav_flow_downtownwest"
ACTION_SCRIPT_CHOICES = ("short_debug_v1", "explore_20_v1")


def main() -> None:
    args = _parser().parse_args()
    config = _load_config(args)
    cases = load_cases(
        config.uav_flow_repo_path,
        classes=args.classes or config.classes or DEFAULT_CLASSES,
        limit_per_class=args.limit_per_class,
        instruction_field=config.instruction_field,
        test_jsons_path=config.test_jsons_path,
        classified_instr_path=config.classified_instr_path,
    )
    if not cases:
        raise SystemExit("No UAV-Flow cases found for custom_map real collection")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for case in cases[: args.max_cases if args.max_cases is not None else len(cases)]:
        results.append(_collect_case(case, config, output_root, args=args))

    index_path = output_root / "custom_map_real_collection_index.json"
    index_path.write_text(json.dumps({"runs": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    sweep_summary = _write_sweep_summary(output_root, results)
    print(f"index: {index_path}")
    print(f"sweep summary json: {sweep_summary['summary_json']}")
    print(f"sweep summary html: {sweep_summary['summary_html']}")
    for result in results:
        print(f"run_dir:          {result['run_dir']}")
        print(f"custom_map image: {result['custom_map_image']}")
        print(f"custom_map json:  {result['custom_map_json']}")
        print(f"custom_map html:  {result['custom_map_html']}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect real UAV-Flow frames and render custom_map artifacts")
    parser.add_argument("--config", help="Optional UAV-Flow eval config JSON")
    parser.add_argument("--output-dir", default="tests/manual_outputs/custom_map_real")
    parser.add_argument("--classes", nargs="*", help="UAV-Flow classes to sample")
    parser.add_argument("--limit-per-class", type=int, default=1)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--uav-flow-repo-path", help="Override UAV-Flow-Eval checkout path")
    parser.add_argument("--env-id", help="Override gym UnrealCV env id")
    parser.add_argument("--semantic-source", choices=["none", "segformer"], default="none")
    parser.add_argument("--skip-unrealcv-mask", action="store_true", help="Collect rgb/depth/pose only and use semantic segmentation masks")
    parser.add_argument("--segmentation-model", default=DEFAULT_SEGFORMER_MODEL)
    parser.add_argument("--seg-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--action-script", choices=ACTION_SCRIPT_CHOICES, default="short_debug_v1")
    return parser


def _load_config(args: argparse.Namespace) -> UavFlowEvalConfig:
    config = UavFlowEvalConfig.from_file(args.config) if args.config else UavFlowEvalConfig()
    data = config.to_dict()
    data["real_unrealcv"] = True
    if args.uav_flow_repo_path:
        data["uav_flow_repo_path"] = args.uav_flow_repo_path
    if args.env_id:
        data["env_id"] = args.env_id
    return UavFlowEvalConfig.from_dict(data)


def _collect_case(
    case: UavFlowCase,
    config: UavFlowEvalConfig,
    output_root: Path,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_class = _safe_path_token(str(case.class_name or "unknown"))
    run_dir = output_root / f"{safe_class}_{case.case_id}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    action_config = _action_config(config)
    client = UavFlowBenchmarkEnvClient(
        case,
        run_dir=run_dir,
        env_id=config.env_id,
        uav_flow_repo_path=config.uav_flow_repo_path,
        real_unrealcv=True,
        time_dilation=config.time_dilation,
        resolution=config.resolution,
        seed=config.seed,
        action_config=action_config,
        enable_research_frame_capture=True,
    )
    writer = MemoryEventWriter(run_dir)
    segmenter = _segmenter(args)
    scene = None
    try:
        scene = client.start_scene(MAP_ID, case.case_id)
        writer.write_instruction(case.selected_instruction, semantic_plan=case.planner_safe_metadata())
        _write_observation(client, writer, scene.scene_instance_id, "initial", args=args, segmenter=segmenter)
        for action, tool, action_args in _action_script(args.action_script):
            result = client.execute_action(scene.scene_instance_id, AGENT_ID, action, action_args)
            writer.write_action(tool, action_args, result.to_dict())
            _write_observation(client, writer, scene.scene_instance_id, f"after_{action}", args=args, segmenter=segmenter)
        if not args.skip_unrealcv_mask:
            _write_mask_label_map(run_dir, client)
        fragment = _render_custom_map(run_dir, case, scene.scene_instance_id)
        summary = _summary(run_dir, case, fragment)
        (run_dir / "collection_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary
    finally:
        if scene is not None:
            client.stop_scene(scene.scene_instance_id)


def _action_config(config: UavFlowEvalConfig) -> UavActionConfig:
    data = load_uav_action_config().to_dict()
    data.update(config.action_config_overrides)
    return UavActionConfig.from_dict(data)


def _action_script(name: str = "short_debug_v1") -> tuple[tuple[str, str, dict[str, Any]], ...]:
    if name == "short_debug_v1":
        return (
            ("turn_relative", "uav.turn_relative", {"yaw_deg": 15.0}),
            ("move_body", "uav.move_body", {"vx": 1.0, "vy": 0.0, "vz": 0.0, "duration": 1.5}),
            ("turn_relative", "uav.turn_relative", {"yaw_deg": -20.0}),
            ("move_body", "uav.move_body", {"vx": 1.0, "vy": 0.3, "vz": 0.0, "duration": 1.5}),
        )
    if name == "explore_20_v1":
        return (
            ("turn_relative", "uav.turn_relative", {"yaw_deg": 12.0}),
            ("move_body", "uav.move_body", {"vx": 0.9, "vy": 0.0, "vz": 0.0, "duration": 1.2}),
            ("move_body", "uav.move_body", {"vx": 0.0, "vy": 0.45, "vz": 0.0, "duration": 1.0}),
            ("move_body", "uav.move_body", {"vx": 0.75, "vy": -0.25, "vz": 0.0, "duration": 1.2}),
            ("turn_relative", "uav.turn_relative", {"yaw_deg": -18.0}),
            ("move_body", "uav.move_body", {"vx": 0.9, "vy": 0.0, "vz": 0.0, "duration": 1.2}),
            ("move_body", "uav.move_body", {"vx": 0.0, "vy": -0.45, "vz": 0.0, "duration": 1.0}),
            ("move_body", "uav.move_body", {"vx": 0.65, "vy": 0.25, "vz": 0.12, "duration": 1.1}),
            ("turn_relative", "uav.turn_relative", {"yaw_deg": 20.0}),
            ("move_body", "uav.move_body", {"vx": 0.9, "vy": 0.0, "vz": 0.0, "duration": 1.2}),
            ("move_body", "uav.move_body", {"vx": 0.0, "vy": 0.5, "vz": 0.0, "duration": 1.0}),
            ("move_body", "uav.move_body", {"vx": 0.7, "vy": -0.3, "vz": -0.12, "duration": 1.1}),
            ("turn_relative", "uav.turn_relative", {"yaw_deg": -12.0}),
            ("move_body", "uav.move_body", {"vx": 1.0, "vy": 0.0, "vz": 0.0, "duration": 1.2}),
            ("move_body", "uav.move_body", {"vx": 0.0, "vy": -0.5, "vz": 0.0, "duration": 1.0}),
            ("move_body", "uav.move_body", {"vx": 0.65, "vy": 0.3, "vz": 0.0, "duration": 1.2}),
            ("turn_relative", "uav.turn_relative", {"yaw_deg": 15.0}),
            ("move_body", "uav.move_body", {"vx": 0.85, "vy": 0.0, "vz": 0.0, "duration": 1.2}),
            ("move_body", "uav.move_body", {"vx": 0.0, "vy": 0.4, "vz": 0.0, "duration": 1.0}),
            ("move_body", "uav.move_body", {"vx": 0.55, "vy": -0.2, "vz": 0.0, "duration": 1.1}),
        )
    raise ValueError(f"Unknown custom_map action script: {name}")


def _write_observation(
    client: UavFlowBenchmarkEnvClient,
    writer: MemoryEventWriter,
    scene_instance_id: str,
    label: str,
    *,
    args: argparse.Namespace,
    segmenter: SegFormerBackend | None,
) -> None:
    modalities = ["rgb", "depth", "pose"] if args.skip_unrealcv_mask else ["rgb", "depth", "mask", "pose"]
    observation = client.get_observation(scene_instance_id, AGENT_ID, modalities)
    raw = RawObservation.from_env_observation(observation, source=f"custom_map_real_collection:{label}").to_dict()
    write_result = writer.write_observation(raw)
    if args.semantic_source == "segformer":
        _write_segformer_observation(writer.run_dir, raw, write_result["observation_ref"]["id"], args, segmenter)


def _segmenter(args: argparse.Namespace) -> SegFormerBackend | None:
    if args.semantic_source != "segformer":
        return None
    return SegFormerBackend(args.segmentation_model, args.seg_device)


def _write_segformer_observation(
    run_dir: Path,
    raw_observation: dict[str, Any],
    observation_id: str,
    args: argparse.Namespace,
    segmenter: SegFormerBackend | None,
) -> None:
    frames = raw_observation.get("frames", {})
    rgb = frames.get("rgb") if isinstance(frames, dict) else None
    rgb_path = rgb.get("path") if isinstance(rgb, dict) else None
    if not rgb_path:
        return
    config = ClosedSetSegmentationConfig(
        model_name=args.segmentation_model,
        device=args.seg_device,
    )
    result = run_closed_set_segmentation(
        rgb_path,
        run_dir / "artifacts",
        observation_id=observation_id,
        config=config,
        backend=segmenter,
    )
    append_segmentation_result(run_dir, result, force=True)


def _write_mask_label_map(run_dir: Path, client: UavFlowBenchmarkEnvClient) -> None:
    label_map = client.collect_research_mask_label_map(_visible_mask_colors(run_dir))
    if not label_map.get("colors") and not label_map.get("unmapped_objects"):
        return
    label_dir = run_dir / "memory" / "map"
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "mask_label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _visible_mask_colors(run_dir: Path) -> set[tuple[int, int, int]]:
    colors: set[tuple[int, int, int]] = set()
    for path in sorted((run_dir / "artifacts" / "mask").glob("*.png")):
        try:
            image = Image.open(path).convert("RGB")
        except OSError:
            continue
        image_colors = image.getcolors(maxcolors=image.width * image.height)
        for _count, color in image_colors or []:
            rgb = tuple(int(channel) for channel in color)
            if rgb != (0, 0, 0):
                colors.add(rgb)
    return colors


def _render_custom_map(run_dir: Path, case: UavFlowCase, scene_instance_id: str) -> dict[str, Any]:
    memory = FlightMemory(
        run_id=run_dir.name,
        map_id=MAP_ID,
        scene_instance_id=scene_instance_id,
        agent_id=AGENT_ID,
        active_instruction=case.selected_instruction,
        memory_reader=MemoryRecordReader(run_dir),
    )
    stage_context = StageContextInput(
        active_instruction=case.selected_instruction,
        semantic_goal={"class_name": case.class_name, "case_id": case.case_id},
    )
    fragment = PluginContextModule().build(MemorySourceBundle(memory), stage_context).to_dict()
    sanitized = dict(fragment)
    sanitized["model_inputs"] = [
        {key: value for key, value in item.items() if key != "data"}
        for item in fragment.get("model_inputs", [])
    ]
    output_dir = run_dir / "contexts" / "custom_map"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest_custom_map.json").write_text(
        json.dumps(sanitized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return fragment


def _summary(run_dir: Path, case: UavFlowCase, fragment: dict[str, Any]) -> dict[str, Any]:
    content = fragment.get("content", {})
    projection_debug = content.get("projection_debug", {})
    artifact_refs = fragment.get("artifact_refs", [])
    semantic_legend = content.get("semantic_legend", {})
    semantic_point_counts_by_label = _semantic_point_counts_by_label(projection_debug, semantic_legend)
    semantic_matrix_cell_counts_by_label = _semantic_matrix_cell_counts_by_label(content, semantic_legend)
    segmentation_pixel_counts = _segmentation_pixel_counts(run_dir)
    return {
        "run_dir": str(run_dir),
        "case_id": case.case_id,
        "class_name": case.class_name,
        "instruction": case.selected_instruction,
        "semantic_projection": content.get("map_config", {}).get("semantic_projection"),
        "projection_point_count": projection_debug.get("point_count", 0),
        "sampled_point_count": len(projection_debug.get("sampled_points", [])) if isinstance(projection_debug, dict) else 0,
        "priority_override_count": projection_debug.get("priority_override_count", 0),
        "source_observation_count": projection_debug.get("source_observation_count", 0),
        "semantic_class_count": len(projection_debug.get("semantic_classes", [])) if isinstance(projection_debug, dict) else 0,
        "semantic_point_counts_by_label": semantic_point_counts_by_label,
        "segmentation_pixel_counts": segmentation_pixel_counts,
        "semantic_matrix_cell_counts_by_label": semantic_matrix_cell_counts_by_label,
        "unknown_projection_ratio": _ratio(semantic_point_counts_by_label.get("unknown", 0), projection_debug.get("point_count", 0)),
        "unknown_pixel_ratio": _ratio(segmentation_pixel_counts.get("unknown", 0), sum(segmentation_pixel_counts.values())),
        "unknown_matrix_ratio": _ratio(semantic_matrix_cell_counts_by_label.get("unknown", 0), sum(semantic_matrix_cell_counts_by_label.values())),
        "action_count": _jsonl_count(run_dir / "memory" / "actions.jsonl"),
        "observation_count": _jsonl_count(run_dir / "memory" / "observations.jsonl"),
        "frame_counts": _frame_counts(run_dir),
        "custom_map_image": artifact_refs[0] if artifact_refs else "",
        "custom_map_html": artifact_refs[1] if len(artifact_refs) > 1 else "",
        "custom_map_json": str(run_dir / "contexts" / "custom_map" / "latest_custom_map.json"),
    }


def _write_sweep_summary(output_root: Path, runs: list[dict[str, Any]]) -> dict[str, str]:
    summary = _build_sweep_summary(runs)
    json_path = output_root / "custom_map_semantic_sweep_summary.json"
    html_path = output_root / "custom_map_semantic_sweep_summary.html"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(_sweep_summary_html(summary), encoding="utf-8")
    return {"summary_json": str(json_path), "summary_html": str(html_path)}


def _build_sweep_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    point_counts: Counter[str] = Counter()
    pixel_counts: Counter[str] = Counter()
    matrix_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    total_points = 0
    total_sampled_points = 0
    total_observations = 0
    total_actions = 0
    total_priority_overrides = 0
    by_class: Counter[str] = Counter()
    for run in runs:
        by_class[str(run.get("class_name") or "unknown")] += 1
        total_points += int(run.get("projection_point_count") or 0)
        total_sampled_points += int(run.get("sampled_point_count") or 0)
        total_observations += int(run.get("observation_count") or 0)
        total_actions += int(run.get("action_count") or 0)
        total_priority_overrides += int(run.get("priority_override_count") or 0)
        point_counts.update(_counter_from_dict(run.get("semantic_point_counts_by_label")))
        pixel_counts.update(_counter_from_dict(run.get("segmentation_pixel_counts")))
        matrix_counts.update(_counter_from_dict(run.get("semantic_matrix_cell_counts_by_label")))
        class_counts.update(str(label) for label in (run.get("semantic_point_counts_by_label") or {}).keys())
    return {
        "run_count": len(runs),
        "runs_by_class": dict(sorted(by_class.items())),
        "totals": {
            "projection_point_count": total_points,
            "sampled_point_count": total_sampled_points,
            "action_count": total_actions,
            "observation_count": total_observations,
            "priority_override_count": total_priority_overrides,
        },
        "semantic_point_counts_by_label": dict(sorted(point_counts.items())),
        "segmentation_pixel_counts": dict(sorted(pixel_counts.items())),
        "semantic_matrix_cell_counts_by_label": dict(sorted(matrix_counts.items())),
        "semantic_class_run_presence": dict(sorted(class_counts.items())),
        "ratios": {
            "unknown_projection_ratio": _ratio(point_counts.get("unknown", 0), total_points),
            "unknown_pixel_ratio": _ratio(pixel_counts.get("unknown", 0), sum(pixel_counts.values())),
            "unknown_matrix_ratio": _ratio(matrix_counts.get("unknown", 0), sum(matrix_counts.values())),
            "sampled_to_projected_ratio": _ratio(total_sampled_points, total_points),
        },
        "runs": runs,
    }


def _sweep_summary_html(summary: dict[str, Any]) -> str:
    rows = []
    for run in summary.get("runs", []):
        html_path = str(run.get("custom_map_html") or "")
        html_link = _summary_link(html_path)
        rows.append(
            "<tr>"
            f"<td>{_escape(run.get('class_name'))}</td>"
            f"<td>{_escape(run.get('case_id'))}</td>"
            f"<td>{int(run.get('action_count') or 0)}</td>"
            f"<td>{int(run.get('observation_count') or 0)}</td>"
            f"<td>{int(run.get('projection_point_count') or 0)}</td>"
            f"<td>{int(run.get('sampled_point_count') or 0)}</td>"
            f"<td>{int(run.get('priority_override_count') or 0)}</td>"
            f"<td>{_escape(run.get('unknown_projection_ratio'))}</td>"
            f"<td><a href=\"{_escape(html_link)}\">custom map</a></td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Custom Map Semantic Sweep Summary</title>
  <style>
    body {{ font: 14px/1.45 system-ui, sans-serif; margin: 24px; color: #0f172a; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; }}
    th {{ background: #f1f5f9; }}
    pre {{ background: #f8fafc; border: 1px solid #cbd5e1; padding: 10px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>Custom Map Semantic Sweep Summary</h1>
  <p>Standalone research artifact for comparing closed-set semantic classes across custom_map runs.</p>
  <h2>Totals</h2>
  <pre>{_escape(json.dumps(summary.get("totals", {}), ensure_ascii=False, indent=2))}</pre>
  <h2>Ratios</h2>
  <pre>{_escape(json.dumps(summary.get("ratios", {}), ensure_ascii=False, indent=2))}</pre>
  <h2>Semantic Point Counts</h2>
  <pre>{_escape(json.dumps(summary.get("semantic_point_counts_by_label", {}), ensure_ascii=False, indent=2))}</pre>
  <h2>Runs</h2>
  <table>
    <thead><tr><th>class</th><th>case</th><th>actions</th><th>observations</th><th>points</th><th>sampled</th><th>overrides</th><th>unknown ratio</th><th>html</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</body>
</html>
"""


def _semantic_point_counts_by_label(projection_debug: dict[str, Any], legend: dict[Any, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if not isinstance(projection_debug, dict):
        return {}
    for semantic_id, count in (projection_debug.get("semantic_point_counts") or {}).items():
        counts[_semantic_label(legend, semantic_id)] += int(count or 0)
    return dict(sorted(counts.items()))


def _semantic_matrix_cell_counts_by_label(content: dict[str, Any], legend: dict[Any, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    matrix = content.get("semantic_matrix", [])
    if not isinstance(matrix, list):
        return {}
    for row in matrix:
        if not isinstance(row, list):
            continue
        for semantic_id in row:
            counts[_semantic_label(legend, semantic_id)] += 1
    return dict(sorted(counts.items()))


def _segmentation_pixel_counts(run_dir: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for frame in _read_jsonl(run_dir / "memory" / "frames.jsonl"):
        if str(frame.get("modality", "")).lower() != "segmentation":
            continue
        metadata = frame.get("metadata", {}) if isinstance(frame.get("metadata"), dict) else {}
        stats = metadata.get("segmentation_stats", {}) if isinstance(metadata.get("segmentation_stats"), dict) else {}
        for label, count in (stats.get("custom_class_pixel_counts") or {}).items():
            counts[str(label)] += int(count or 0)
    return dict(sorted(counts.items()))


def _frame_counts(run_dir: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for frame in _read_jsonl(run_dir / "memory" / "frames.jsonl"):
        counts[str(frame.get("modality") or "unknown").lower()] += 1
    return dict(sorted(counts.items()))


def _jsonl_count(path: Path) -> int:
    return len(_read_jsonl(path))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def _counter_from_dict(value: Any) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(value, dict):
        return counter
    for key, count in value.items():
        counter[str(key)] += int(count or 0)
    return counter


def _semantic_label(legend: dict[Any, Any], semantic_id: Any) -> str:
    for key in (semantic_id, str(semantic_id)):
        item = legend.get(key) if isinstance(legend, dict) else None
        if isinstance(item, dict) and item.get("label"):
            return str(item["label"])
    if str(semantic_id) == "-2":
        return "unknown"
    if str(semantic_id) == "-1":
        return "visited_path"
    if str(semantic_id) == "0":
        return "free"
    return f"semantic_{semantic_id}"


def _ratio(numerator: Any, denominator: Any) -> float:
    denominator = int(denominator or 0)
    if denominator <= 0:
        return 0.0
    return round(float(numerator or 0) / float(denominator), 6)


def _escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _safe_path_token(value: str) -> str:
    return "".join("_" if char in {'/', '\\', ':', '*', '?', '"', '<', '>', '|', ' '} else char for char in value)


def _summary_link(path: str) -> str:
    if not path:
        return ""
    try:
        relative = os.path.relpath(Path(path), Path(path).parents[3])
    except (IndexError, ValueError):
        return path.replace("\\", "/")
    return relative.replace("\\", "/")


if __name__ == "__main__":
    main()
