"""Add SegFormer semantic masks to an existing custom_map run.

This manual probe does not rerun UnrealCV. It reads existing RGB/depth/pose
memory, appends segmentation frame records, merges a color label map, and then
rerenders the custom_map PNG/JSON/HTML artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plugins.research.custom_map.context_modules.custom_map import PluginContextModule
from plugins.research.custom_map.perception import (
    ClosedSetSegmentationConfig,
    DEFAULT_SEGFORMER_MODEL,
    SegFormerBackend,
    run_closed_set_segmentation,
)
from plugins.research.custom_map.perception.segmentation_writer import append_segmentation_result
from src.memory.context.schemas import MemorySourceBundle, TaskContextInput
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryRecordReader


def main() -> None:
    args = _parser().parse_args()
    run_dir = Path(args.run_dir)
    if not (run_dir / "memory" / "frames.jsonl").exists():
        raise SystemExit(f"Run directory does not contain memory/frames.jsonl: {run_dir}")

    before_fragment = _render_custom_map(run_dir)
    before_map = _map_summary(before_fragment)
    backend = SegFormerBackend(args.segmentation_model, args.seg_device)
    config = ClosedSetSegmentationConfig(model_name=args.segmentation_model, device=args.seg_device)
    written = []
    skipped = []
    for frame in _rgb_frames(run_dir):
        if not args.force and _has_existing_segmentation(run_dir, frame["observation_id"]):
            skipped.append(frame["observation_id"])
            continue
        result = run_closed_set_segmentation(
            frame["path"],
            run_dir / "artifacts",
            observation_id=frame["observation_id"],
            config=config,
            backend=backend,
        )
        ref = append_segmentation_result(run_dir, result, force=args.force)
        written.append(
            {
                "observation_id": frame["observation_id"],
                "mask_path": str(result.mask_path),
                "overlay_path": str(result.overlay_path),
                "frame_ref": ref,
                "stats": result.stats,
            }
        )

    fragment = _render_custom_map(run_dir)
    after_map = _map_summary(fragment)
    pixel_counts = _aggregate_segmentation_pixel_counts(written)
    summary = {
        "run_dir": str(run_dir),
        "segmentation_written": len(written),
        "segmentation_skipped_existing": len(skipped),
        "before_map": before_map,
        "after_map": after_map,
        "comparison": {
            "projection_point_count_delta": after_map["projection_point_count"] - before_map["projection_point_count"],
            "semantic_class_count_delta": after_map["semantic_class_count"] - before_map["semantic_class_count"],
        },
        "segmentation_pixel_counts": pixel_counts,
        "written": written,
        "skipped_observation_ids": skipped,
        "custom_map_image": fragment.get("artifact_refs", [""])[0],
        "custom_map_html": fragment.get("artifact_refs", ["", ""])[1] if len(fragment.get("artifact_refs", [])) > 1 else "",
        "custom_map_json": str(run_dir / "contexts" / "custom_map" / "latest_custom_map.json"),
    }
    summary_path = run_dir / "contexts" / "custom_map" / "segformer_probe_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"run_dir:               {run_dir}")
    print(f"segmentation written:  {len(written)}")
    print(f"segmentation skipped:  {len(skipped)}")
    print(f"points before/after:   {before_map['projection_point_count']} -> {after_map['projection_point_count']}")
    print(f"class count before/after: {before_map['semantic_class_count']} -> {after_map['semantic_class_count']}")
    print(f"custom_map image:      {summary['custom_map_image']}")
    print(f"custom_map json:       {summary['custom_map_json']}")
    print(f"custom_map html:       {summary['custom_map_html']}")
    print(f"segformer summary:     {summary_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SegFormer on an existing custom_map run and rerender artifacts")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--segmentation-model", default=DEFAULT_SEGFORMER_MODEL)
    parser.add_argument("--seg-device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--force", action="store_true", help="Append new segmentation frames even if they already exist")
    return parser


def _rgb_frames(run_dir: Path) -> list[dict[str, Any]]:
    frames = []
    for item in _read_jsonl(run_dir / "memory" / "frames.jsonl"):
        if str(item.get("modality", "")).lower() != "rgb":
            continue
        path = _resolve_path(str(item.get("path") or ""), run_dir)
        if path.exists() and item.get("observation_id"):
            frames.append({"observation_id": str(item["observation_id"]), "path": path})
    return frames


def _has_existing_segmentation(run_dir: Path, observation_id: str) -> bool:
    for item in _read_jsonl(run_dir / "memory" / "frames.jsonl"):
        if item.get("observation_id") == observation_id and str(item.get("modality", "")).lower() == "segmentation":
            return True
    return False


def _render_custom_map(run_dir: Path) -> dict[str, Any]:
    memory = FlightMemory(
        run_id=run_dir.name,
        map_id="manual",
        scene_instance_id="manual",
        agent_id="uav_1",
        active_instruction="manual SegFormer custom_map probe",
        memory_reader=MemoryRecordReader(run_dir),
    )
    fragment = PluginContextModule().build(MemorySourceBundle(memory), TaskContextInput(active_instruction=memory.active_instruction)).to_dict()
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


def _map_summary(fragment: dict[str, Any]) -> dict[str, Any]:
    content = fragment.get("content", {})
    debug = content.get("projection_debug", {})
    semantic_classes = debug.get("semantic_classes", []) if isinstance(debug, dict) else []
    by_label = {}
    if isinstance(semantic_classes, list):
        for item in semantic_classes:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("semantic_id") or "unknown")
            by_label[label] = by_label.get(label, 0) + int(item.get("point_count") or 0)
    return {
        "semantic_projection": content.get("map_config", {}).get("semantic_projection"),
        "projection_point_count": int(debug.get("point_count") or 0) if isinstance(debug, dict) else 0,
        "source_observation_count": int(debug.get("source_observation_count") or 0) if isinstance(debug, dict) else 0,
        "semantic_class_count": len(semantic_classes) if isinstance(semantic_classes, list) else 0,
        "semantic_point_counts_by_label": dict(sorted(by_label.items())),
    }


def _aggregate_segmentation_pixel_counts(written: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in written:
        stats = item.get("stats", {}) if isinstance(item, dict) else {}
        pixel_counts = stats.get("custom_class_pixel_counts", {}) if isinstance(stats, dict) else {}
        if not isinstance(pixel_counts, dict):
            continue
        for label, count in pixel_counts.items():
            counts[str(label)] += int(count or 0)
    return dict(sorted(counts.items()))


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


if __name__ == "__main__":
    main()
