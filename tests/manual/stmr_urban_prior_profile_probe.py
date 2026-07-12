"""Profile default and urban STMR query priors on the same recorded frames."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plugins.research.stmr_map.perception import (
    DEFAULT_UAV_PRIOR_NAME,
    URBAN_UAV_PRIOR_V1_NAME,
    StmrPerceptionSession,
    assemble_landmark_queries,
)
from plugins.research.stmr_map.profiling import StageProfiler
from plugins.research.stmr_map.profiling_report import write_profile_report


DEFAULT_SOURCE_ROOT = Path("tests/manual_outputs/stmr_light_regen_v2")
DEFAULT_OUTPUT_ROOT = Path("tests/manual_outputs/stmr_incremental_profile")
DEFAULT_CLASSES = ("Turn", "Shift", "Approach")


def main() -> None:
    args = _parser().parse_args()
    probe = _load_stmr_probe_module()
    source_runs = _select_source_runs(Path(args.source_root), tuple(args.classes), probe)
    if not source_runs:
        raise SystemExit("No source runs with RGB/depth/pose memory were found")

    local_models = Path(args.grounding_model).exists() and Path(args.sam_model).exists()
    if local_models:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    cold_profiler = StageProfiler(
        device=args.seg_device,
        context={"experiment": "stmr_urban_prior_profile", "scope": "model_warmup"},
    )
    session = StmrPerceptionSession(
        grounding_model=args.grounding_model,
        sam_model=args.sam_model,
        device=args.seg_device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        tfidf_threshold=args.tfidf_threshold,
    )
    warmup_frames = probe._rgb_frames(Path(str(source_runs[0]["run_dir"])))
    if not warmup_frames:
        raise SystemExit("The selected source runs do not contain an RGB frame for model warmup")
    warmup_landmark_sets = [
        assemble_landmark_queries([], prior=prior_name, max_queries=args.max_queries)
        for prior_name in args.priors
    ]
    session.warmup(
        profiler=cold_profiler,
        rgb_path=warmup_frames[0]["path"],
        landmark_sets=warmup_landmark_sets,
    )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for source_run in source_runs:
        case_folder = Path(str(source_run["run_dir"])).name
        for prior_name in args.priors:
            variant_dir = output_root / case_folder / prior_name
            profiler = StageProfiler(
                device=args.seg_device,
                context={
                    "experiment": "stmr_urban_prior_profile",
                    "case_id": str(source_run.get("case_id") or case_folder),
                    "class_name": str(source_run.get("class_name") or case_folder.split("_", 1)[0]),
                    "variant": prior_name,
                },
            )
            run_args = _probe_args(args)
            summary = probe.run_stmr_probe(
                source_run,
                run_args,
                session=session,
                profiler=profiler,
                output_dir_override=variant_dir,
                prior_name=prior_name,
            )
            timing_jsonl = profiler.write_jsonl(variant_dir / "stmr_stage_timings.jsonl")
            timing_json = profiler.write_summary(variant_dir / "stmr_timing_summary.json")
            summary.update(
                _evidence_summary(
                    variant_dir / "latest_stmr_map.json",
                    max_frames=args.frames_per_case,
                )
            )
            summary.update(
                {
                    "variant": prior_name,
                    "source_run_dir": str(source_run["run_dir"]),
                    "output_dir": str(variant_dir),
                    "timing_jsonl": str(timing_jsonl),
                    "timing_json": str(timing_json),
                    "timing_records": list(profiler.records),
                }
            )
            runs.append(summary)
            print(
                f"profiled {summary['class_name']} / {prior_name}: "
                f"{summary['frame_count']} frames, {summary['detection_count']} detections, "
                f"{summary['mask_count']} masks",
                flush=True,
            )

    json_path, html_path, _ = write_profile_report(
        runs,
        output_root,
        cold_start_records=cold_profiler.records,
    )
    cold_profiler.write_jsonl(output_root / "model_warmup_timings.jsonl")
    print(f"profile summary json: {json_path}")
    print(f"profile summary html: {html_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A/B profile STMR default and urban UAV detector-query priors")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument(
        "--priors",
        nargs="+",
        choices=[DEFAULT_UAV_PRIOR_NAME, URBAN_UAV_PRIOR_V1_NAME],
        default=[DEFAULT_UAV_PRIOR_NAME, URBAN_UAV_PRIOR_V1_NAME],
    )
    parser.add_argument("--frames-per-case", type=int, default=5)
    parser.add_argument("--grounding-model", default=r"F:\hf_models\grounding-dino-tiny")
    parser.add_argument("--sam-model", default=r"F:\hf_models\sam-vit-base")
    parser.add_argument("--seg-device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--tfidf-threshold", type=float, default=0.8)
    parser.add_argument("--max-queries", type=int, default=16)
    parser.add_argument("--dynamic-target-filter", choices=["on", "off"], default="on")
    parser.add_argument("--write-full-debug", action="store_true")
    return parser


def _probe_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the compatibility namespace consumed by ``stmr_map_probe``."""

    return argparse.Namespace(
        perception_source="grounding_dino_sam",
        grounding_model=args.grounding_model,
        sam_model=args.sam_model,
        seg_device=args.seg_device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        tfidf_threshold=args.tfidf_threshold,
        landmarks=[],
        landmark_source="rule",
        max_landmarks=args.max_queries,
        dump_landmark_prompt=False,
        allow_rule_fallback=False,
        max_frames=max(1, int(args.frames_per_case)),
        dynamic_target_filter=args.dynamic_target_filter,
        dynamic_min_score=0.45,
        dynamic_min_mask_pixels=100,
        dynamic_max_depth_p95=35.0,
        dynamic_track_radius_m=8.0,
        html_stmr_point_budget=8000,
        html_voxel_budget=4000,
        html_custom_point_budget=0,
        write_full_debug=bool(args.write_full_debug),
    )


def _select_source_runs(source_root: Path, classes: tuple[str, ...], probe: Any) -> list[dict[str, Any]]:
    selected = []
    for class_name in classes:
        candidates = sorted(
            path
            for path in source_root.glob(f"{class_name}_*")
            if (path / "memory" / "frames.jsonl").exists() and (path / "memory" / "poses.jsonl").exists()
        )
        if not candidates:
            continue
        run = probe._run_item_from_dir(candidates[0])
        run["run_dir"] = str(candidates[0].resolve())
        run.setdefault("class_name", class_name)
        selected.append(run)
    return selected


def _evidence_summary(path: Path, *, max_frames: int) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    content = raw.get("content", raw)
    results = list(content.get("perception_results", []))[:max_frames]
    kept_masks = [mask for item in results for mask in item.get("kept_masks", [])]
    matches = [match for item in results for match in item.get("tfidf_matches", []) if match.get("keep")]
    first = results[0] if results else {}
    return {
        "frame_count": len(results),
        "queries": [item.get("label") for item in content.get("landmarks", []) if item.get("label")],
        "detection_count": len(matches),
        "mask_count": len(kept_masks),
        "boxes_path": first.get("boxes_path"),
        "overlay_path": first.get("overlay_path"),
        "semantic_cell_count": sum(
            1
            for row in content.get("semantic_matrix", [])
            for value in row
            if int(value) > 0
        ),
    }


def _load_stmr_probe_module() -> Any:
    path = Path(__file__).with_name("stmr_map_probe.py")
    spec = importlib.util.spec_from_file_location("flyagent_stmr_map_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load STMR probe module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
