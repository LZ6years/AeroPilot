"""Compare multiple closed-set segmentation models on existing custom_map runs.

This probe is offline: it reuses already collected RGB/depth/pose memory,
creates model-specific segmentation frame overlays, rerenders custom_map into a
model-specific debug directory, and writes an HTML summary for human review.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plugins.research.custom_map.context_modules.custom_map.grid_builder import DEFAULT_GRID_SIZE, DEFAULT_METERS_PER_CELL, build_grid_map
from plugins.research.custom_map.context_modules.custom_map.html_report import render_custom_map_html
from plugins.research.custom_map.context_modules.custom_map.mask_semantics import build_semantic_layer
from plugins.research.custom_map.context_modules.custom_map.projection import build_projected_semantic_layer
from plugins.research.custom_map.context_modules.custom_map.records import read_custom_map_records
from plugins.research.custom_map.context_modules.custom_map.renderer import render_custom_map
from plugins.research.custom_map.context_modules.custom_map.trajectory import build_trajectory
from plugins.research.custom_map.perception import (
    ADE_MASK2FORMER_LARGE,
    ADE_ONEFORMER_DINAT_LARGE,
    ADE_ONEFORMER_SWIN_LARGE,
    ADE_SEGFORMER_B5,
    CITYSCAPES_MASK2FORMER_LARGE,
    CITYSCAPES_ONEFORMER_DINAT_LARGE,
    CITYSCAPES_ONEFORMER_SWIN_LARGE,
    CITYSCAPES_SEGFORMER_B2,
    DEFAULT_SEGFORMER_MODEL,
    MAPILLARY_MASK2FORMER_LARGE,
    ClosedSetSegmentationConfig,
    HuggingFaceSegmentationBackend,
    run_closed_set_segmentation,
)
from plugins.research.custom_map.perception.segmentation_writer import append_segmentation_result
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryRecordReader


MODEL_PRESETS = {
    "segformer_ade_b0": DEFAULT_SEGFORMER_MODEL,
    "segformer_cityscapes_b2": CITYSCAPES_SEGFORMER_B2,
    "mask2former_cityscapes_large": CITYSCAPES_MASK2FORMER_LARGE,
    "mask2former_mapillary_large": MAPILLARY_MASK2FORMER_LARGE,
    "segformer_ade_b5": ADE_SEGFORMER_B5,
    "mask2former_ade_large": ADE_MASK2FORMER_LARGE,
    "oneformer_ade_dinat_large": ADE_ONEFORMER_DINAT_LARGE,
    "oneformer_cityscapes_dinat_large": CITYSCAPES_ONEFORMER_DINAT_LARGE,
    "oneformer_ade_swin_large": ADE_ONEFORMER_SWIN_LARGE,
    "oneformer_cityscapes_swin_large": CITYSCAPES_ONEFORMER_SWIN_LARGE,
}
PAPER_CANDIDATE_MODELS = (
    "segformer_ade_b5",
    "mask2former_ade_large",
    "oneformer_ade_swin_large",
    "oneformer_cityscapes_swin_large",
)
STABLE_PAPER_CANDIDATE_MODELS = (
    "segformer_ade_b5",
    "mask2former_ade_large",
)
REPRESENTATIVE_CLASSES = ("Move", "Shift", "Approach", "Pass", "Retreat")


def main() -> None:
    args = _parser().parse_args()
    sweep_dir = Path(args.sweep_dir)
    runs = _select_runs(sweep_dir, args)
    if not runs:
        raise SystemExit(f"No custom_map runs selected under {sweep_dir}")
    models = _selected_models(args.models)
    results = []
    for run in runs:
        for model_slug, model_name in models.items():
            results.append(_compare_model_on_run(run, model_slug, model_name, args))
    summary = _build_compare_summary(sweep_dir, results)
    json_path = sweep_dir / "model_compare_summary.json"
    html_path = sweep_dir / "model_compare_summary.html"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(_compare_summary_html(summary, html_path.parent), encoding="utf-8")
    print(f"compare summary json: {json_path}")
    print(f"compare summary html: {html_path}")
    for item in results:
        print(f"{item['class_name']} {item['case_id']} {item['model_slug']}: {item['custom_map_html']}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare segmentation models on existing custom_map sweep runs")
    parser.add_argument("--sweep-dir", default="runs/experiments/custom_map_semantic_sweep")
    parser.add_argument("--run-dir", nargs="*", help="Explicit run directories; overrides preset selection")
    parser.add_argument("--preset", choices=["representative_5"], default="representative_5")
    parser.add_argument("--all-runs", action="store_true")
    parser.add_argument(
        "--models",
        nargs="*",
        choices=sorted(MODEL_PRESETS) + ["paper_candidates_2", "paper_candidates_4"],
        default=["paper_candidates_2"],
    )
    parser.add_argument("--seg-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--resize-long-edge", type=int, default=768)
    parser.add_argument("--oom-retry-long-edge", type=int, default=512)
    return parser


def _selected_models(names: list[str]) -> dict[str, str]:
    if not names:
        names = ["paper_candidates_2"]
    selected: list[str] = []
    for name in names:
        if name == "paper_candidates_2":
            selected.extend(STABLE_PAPER_CANDIDATE_MODELS)
        elif name == "paper_candidates_4":
            selected.extend(PAPER_CANDIDATE_MODELS)
        else:
            selected.append(name)
    deduped = list(dict.fromkeys(selected))
    return {name: MODEL_PRESETS[name] for name in deduped}


def _select_runs(sweep_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.run_dir:
        return [_run_item_from_dir(Path(path)) for path in args.run_dir]
    summary_path = sweep_dir / "custom_map_semantic_sweep_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runs = [item for item in summary.get("runs", []) if isinstance(item, dict)]
    else:
        runs = [_run_item_from_dir(path.parent.parent.parent) for path in sorted(sweep_dir.rglob("contexts/custom_map/latest_custom_map.json"))]
    if args.all_runs:
        return runs
    return _representative_runs(runs)


def _representative_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for class_name in REPRESENTATIVE_CLASSES:
        candidates = [run for run in runs if run.get("class_name") == class_name]
        if not candidates:
            continue
        if class_name in {"Approach", "Pass", "Retreat"}:
            selected.append(max(candidates, key=lambda item: float(item.get("unknown_projection_ratio") or 0.0)))
        else:
            selected.append(candidates[0])
    return selected


def _run_item_from_dir(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "collection_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "run_dir": str(run_dir),
        "case_id": run_dir.name,
        "class_name": run_dir.name.split("_", 1)[0],
    }


def _compare_model_on_run(run: dict[str, Any], model_slug: str, model_name: str, args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(str(run["run_dir"]))
    output_dir = run_dir / "contexts" / "custom_map_compare" / model_slug
    variant_dir = output_dir / "_variant_run"
    _prepare_variant_memory(run_dir, variant_dir)
    backend = HuggingFaceSegmentationBackend(model_name, args.seg_device, resize_long_edge=args.resize_long_edge)
    resized_after_oom = False
    written = []
    for frame in _rgb_frames(run_dir):
        try:
            result = _segment_frame(run_dir, frame, model_slug, model_name, backend, args)
        except RuntimeError as exc:
            if not _is_oom(exc) or args.oom_retry_long_edge <= 0 or args.oom_retry_long_edge == args.resize_long_edge:
                raise
            _empty_cuda_cache()
            resized_after_oom = True
            backend = HuggingFaceSegmentationBackend(model_name, args.seg_device, resize_long_edge=args.oom_retry_long_edge)
            result = _segment_frame(run_dir, frame, model_slug, model_name, backend, args)
        append_segmentation_result(variant_dir, result, force=True)
        written.append(result)
    content = _render_variant_map(variant_dir, output_dir, run)
    first_result = written[0] if written else None
    return {
        "run_dir": str(run_dir),
        "class_name": run.get("class_name"),
        "case_id": run.get("case_id"),
        "model_slug": model_slug,
        "model_name": model_name,
        "resize_long_edge": args.oom_retry_long_edge if resized_after_oom else args.resize_long_edge,
        "resized_after_oom": resized_after_oom,
        "rgb_sample": _first_rgb_path(run_dir),
        "overlay_sample": str(first_result.overlay_path) if first_result else "",
        "semantic_mask_sample": str(first_result.mask_path) if first_result else "",
        "custom_map_image": content["rendered_map"]["artifact_path"],
        "custom_map_html": content["rendered_html"]["artifact_path"],
        "custom_map_json": str(output_dir / "latest_custom_map.json"),
        "projection_point_count": content["projection_debug"].get("point_count", 0),
        "sampled_point_count": len(content["projection_debug"].get("sampled_points", [])),
        "unknown_projection_ratio": _ratio(
            _semantic_point_counts_by_label(content).get("unknown", 0),
            content["projection_debug"].get("point_count", 0),
        ),
        "priority_override_count": content["projection_debug"].get("priority_override_count", 0),
        "semantic_point_counts_by_label": _semantic_point_counts_by_label(content),
        "semantic_matrix_cell_counts_by_label": _semantic_matrix_cell_counts_by_label(content),
        "segmentation_pixel_counts": _segmentation_pixel_counts(written),
    }


def _segment_frame(
    run_dir: Path,
    frame: dict[str, Any],
    model_slug: str,
    model_name: str,
    backend: HuggingFaceSegmentationBackend,
    args: argparse.Namespace,
) -> Any:
    config = ClosedSetSegmentationConfig(
        model_name=model_name,
        device=args.seg_device,
        source=model_slug,
        mask_dir_name=f"semantic_mask_{model_slug}",
        overlay_dir_name=f"semantic_overlay_{model_slug}",
    )
    return run_closed_set_segmentation(
        frame["path"],
        run_dir / "artifacts",
        observation_id=frame["observation_id"],
        config=config,
        backend=backend,
    )


def _prepare_variant_memory(run_dir: Path, variant_dir: Path) -> None:
    memory_dir = variant_dir / "memory"
    if memory_dir.exists():
        shutil.rmtree(memory_dir)
    memory_dir.mkdir(parents=True, exist_ok=True)
    source_memory = run_dir / "memory"
    for name in ("actions.jsonl", "observations.jsonl", "poses.jsonl"):
        source = source_memory / name
        if source.exists():
            shutil.copyfile(source, memory_dir / name)
    kept_frames = []
    for frame in _read_jsonl(source_memory / "frames.jsonl"):
        modality = str(frame.get("modality") or "").lower()
        if modality in {"segmentation", "mask", "object_mask"}:
            continue
        raw_path = frame.get("path")
        if isinstance(raw_path, str) and raw_path:
            frame = dict(frame)
            frame["path"] = str(_resolve_path(raw_path, run_dir))
            metadata = dict(frame.get("metadata", {})) if isinstance(frame.get("metadata"), dict) else {}
            if isinstance(metadata.get("path"), str):
                metadata["path"] = str(_resolve_path(str(metadata["path"]), run_dir))
            frame["metadata"] = metadata
        kept_frames.append(frame)
    with (memory_dir / "frames.jsonl").open("w", encoding="utf-8") as handle:
        for frame in kept_frames:
            handle.write(json.dumps(frame, ensure_ascii=False) + "\n")


def _render_variant_map(variant_dir: Path, output_dir: Path, run: dict[str, Any]) -> dict[str, Any]:
    memory = FlightMemory(
        run_id=variant_dir.name,
        map_id="compare",
        scene_instance_id="compare",
        agent_id="uav_flow_drone",
        active_instruction=str(run.get("instruction") or "custom_map segmentation compare"),
        memory_reader=MemoryRecordReader(variant_dir),
    )
    records = read_custom_map_records(memory)
    trajectory = build_trajectory(records)
    center_pose = trajectory.path[-1] if trajectory.path else None
    semantic_layer = build_projected_semantic_layer(
        records,
        center_pose,
        grid_size=DEFAULT_GRID_SIZE,
        meters_per_cell=DEFAULT_METERS_PER_CELL,
    ) or build_semantic_layer(records, grid_size=DEFAULT_GRID_SIZE)
    grid_map = build_grid_map(
        trajectory,
        semantic_layer,
        grid_size=DEFAULT_GRID_SIZE,
        meters_per_cell=DEFAULT_METERS_PER_CELL,
    )
    rendered_path = render_custom_map(grid_map, output_dir)
    html_path = render_custom_map_html(grid_map, rendered_path, output_dir)
    content = {
        "semantic_matrix": grid_map.semantic_matrix,
        "semantic_legend": {str(key): dict(value) for key, value in sorted(grid_map.semantic_legend.items())},
        "path_summary": grid_map.path_summary,
        "center_pose": grid_map.center_pose,
        "map_config": grid_map.map_config,
        "rendered_map": {"artifact_path": str(rendered_path), "role": "custom_map"},
        "rendered_html": {"artifact_path": str(html_path), "role": "custom_map_debug_viewer"},
        "projection_debug": grid_map.projection_debug,
        "path_cells": grid_map.path_cells,
        "failed_cells": grid_map.failed_cells,
        "current_cell": grid_map.current_cell,
    }
    (output_dir / "latest_custom_map.json").write_text(json.dumps({"content": content}, ensure_ascii=False, indent=2), encoding="utf-8")
    return content


def _build_compare_summary(sweep_dir: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    by_model = {}
    for model_slug in sorted({item["model_slug"] for item in results}):
        model_items = [item for item in results if item["model_slug"] == model_slug]
        total_points = sum(int(item.get("projection_point_count") or 0) for item in model_items)
        counts = Counter()
        for item in model_items:
            counts.update(_counter_from_dict(item.get("semantic_point_counts_by_label")))
        by_model[model_slug] = {
            "run_count": len(model_items),
            "projection_point_count": total_points,
            "unknown_projection_ratio": _ratio(counts.get("unknown", 0), total_points),
            "semantic_point_counts_by_label": dict(sorted(counts.items())),
        }
    return {
        "sweep_dir": str(sweep_dir),
        "model_count": len(by_model),
        "run_count": len({item["run_dir"] for item in results}),
        "by_model": by_model,
        "results": results,
    }


def _compare_summary_html(summary: dict[str, Any], output_dir: Path) -> str:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in summary.get("results", []):
        key = f"{item.get('class_name')}::{item.get('case_id')}"
        grouped.setdefault(key, []).append(item)
    sections = []
    for key, items in grouped.items():
        first = items[0]
        cards = "".join(_model_card(item, output_dir) for item in sorted(items, key=lambda value: value["model_slug"]))
        sections.append(
            f"<section class=\"case\"><h2>{_escape(first.get('class_name'))} / {_escape(first.get('case_id'))}</h2>"
            f"<div class=\"rgb\"><img src=\"{_rel(first.get('rgb_sample'), output_dir)}\" alt=\"rgb\"></div>"
            f"<div class=\"models\">{cards}</div></section>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Custom Map Segmentation Model Compare</title>
  <style>
    body {{ margin: 24px; font: 14px/1.45 system-ui, sans-serif; color: #0f172a; background: #f8fafc; }}
    h1 {{ margin-bottom: 4px; }}
    .summary, .case {{ background: white; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px; margin: 16px 0; }}
    .models {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px; background: #fff; }}
    img {{ width: 100%; max-height: 260px; object-fit: contain; border: 1px solid #e2e8f0; background: white; }}
    .rgb {{ max-width: 420px; margin-bottom: 12px; }}
    code, pre {{ background: #f1f5f9; padding: 2px 4px; border-radius: 4px; }}
    dl {{ display: grid; grid-template-columns: 150px 1fr; gap: 4px 10px; }}
    dd {{ margin: 0; word-break: break-word; }}
  </style>
</head>
<body>
  <h1>Custom Map Segmentation Model Compare</h1>
  <div class="summary"><pre>{_escape(json.dumps(summary.get("by_model", {}), ensure_ascii=False, indent=2))}</pre></div>
  {''.join(sections)}
</body>
</html>
"""


def _model_card(item: dict[str, Any], output_dir: Path) -> str:
    counts = item.get("semantic_point_counts_by_label", {})
    return f"""<article class="card">
  <h3>{_escape(item.get("model_slug"))}</h3>
  <a href="{_rel(item.get("custom_map_html"), output_dir)}">open 3D custom_map viewer</a>
  <h4>Overlay</h4>
  <img src="{_rel(item.get("overlay_sample"), output_dir)}" alt="overlay">
  <h4>Top-down</h4>
  <img src="{_rel(item.get("custom_map_image"), output_dir)}" alt="custom map">
  <dl>
    <dt>points</dt><dd>{int(item.get("projection_point_count") or 0)}</dd>
    <dt>unknown</dt><dd>{item.get("unknown_projection_ratio")}</dd>
    <dt>route</dt><dd>{int(counts.get("route_surface", 0))}</dd>
    <dt>building</dt><dd>{int(counts.get("building_structure", 0))}</dd>
    <dt>vegetation</dt><dd>{int(counts.get("vegetation", 0))}</dd>
    <dt>person</dt><dd>{int(counts.get("person", 0))}</dd>
    <dt>vehicle</dt><dd>{int(counts.get("vehicle", 0))}</dd>
  </dl>
</article>"""


def _rgb_frames(run_dir: Path) -> list[dict[str, Any]]:
    frames = []
    for item in _read_jsonl(run_dir / "memory" / "frames.jsonl"):
        if str(item.get("modality", "")).lower() != "rgb":
            continue
        path = _resolve_path(str(item.get("path") or ""), run_dir)
        if path.exists() and item.get("observation_id"):
            frames.append({"observation_id": str(item["observation_id"]), "path": path})
    return frames


def _first_rgb_path(run_dir: Path) -> str:
    frames = _rgb_frames(run_dir)
    return str(frames[0]["path"]) if frames else ""


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


def _resolve_path(raw_path: str, run_dir: Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    run_path = run_dir / path
    if run_path.exists():
        return run_path
    return Path.cwd() / path


def _semantic_point_counts_by_label(content: dict[str, Any]) -> dict[str, int]:
    debug = content.get("projection_debug", {})
    legend = content.get("semantic_legend", {})
    counts = Counter()
    for semantic_id, count in (debug.get("semantic_point_counts") or {}).items():
        counts[_semantic_label(legend, semantic_id)] += int(count or 0)
    return dict(sorted(counts.items()))


def _semantic_matrix_cell_counts_by_label(content: dict[str, Any]) -> dict[str, int]:
    legend = content.get("semantic_legend", {})
    counts = Counter()
    for row in content.get("semantic_matrix", []):
        for semantic_id in row:
            counts[_semantic_label(legend, semantic_id)] += 1
    return dict(sorted(counts.items()))


def _segmentation_pixel_counts(results: list[Any]) -> dict[str, int]:
    counts = Counter()
    for result in results:
        counts.update(_counter_from_dict(result.stats.get("custom_class_pixel_counts")))
    return dict(sorted(counts.items()))


def _counter_from_dict(value: Any) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(value, dict):
        return counter
    for key, count in value.items():
        counter[str(key)] += int(count or 0)
    return counter


def _semantic_label(legend: dict[str, Any], semantic_id: Any) -> str:
    item = legend.get(str(semantic_id)) if isinstance(legend, dict) else None
    if isinstance(item, dict) and item.get("label"):
        return str(item["label"])
    return {"-2": "unknown", "-1": "visited_path", "0": "free"}.get(str(semantic_id), f"semantic_{semantic_id}")


def _ratio(numerator: Any, denominator: Any) -> float:
    denominator = int(denominator or 0)
    if denominator <= 0:
        return 0.0
    return round(float(numerator or 0) / float(denominator), 6)


def _rel(path: Any, output_dir: Path) -> str:
    if not path:
        return ""
    try:
        return os.path.relpath(Path(str(path)), output_dir).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _is_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error" in message and "memory" in message


def _empty_cuda_cache() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
