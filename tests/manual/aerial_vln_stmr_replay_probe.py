"""Replay one completed AerialVLN run through STMR without a simulator.

The probe copies append-only memory into a new manual-output directory before
running GroundingDINO/SAM, so the source benchmark run remains untouched. It is
intentionally outside pytest because model loading is slow and may require a
CUDA-capable environment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plugins.research.stmr_map.online import StmrOnlineMapBuilder
from plugins.research.stmr_map.perception import (
    DEFAULT_UAV_PRIOR_NAME,
    MAX_LANDMARK_QUERIES,
    StmrPerceptionSession,
)
from plugins.research.stmr_map.perception.grounding_dino_backend import (
    DEFAULT_GROUNDING_DINO_MODEL,
)
from plugins.research.stmr_map.perception.sam_backend import DEFAULT_SAM_MODEL
from src.memory.context.contracts import TaskContextInput
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryRecordReader


def main() -> None:
    args = _parser().parse_args()
    source_run = Path(args.run_dir).resolve()
    if not (source_run / "memory" / "frames.jsonl").is_file():
        raise SystemExit(f"AerialVLN run memory not found: {source_run}")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else REPO_ROOT
        / "runs"
        / "experiments"
        / (
            "aerial_vln_stmr_replay_"
            + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        )
    )
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    shutil.copytree(source_run / "memory", output_dir / "memory")

    metadata = _read_json(source_run / "state" / "flight_memory.json")
    instruction = str(metadata.get("active_instruction") or "").strip()
    if not instruction:
        raise SystemExit("Source run does not contain an active instruction")
    memory = FlightMemory(
        run_id=str(metadata.get("run_id") or source_run.name),
        map_id=str(metadata.get("map_id") or "unknown"),
        scene_instance_id=str(metadata.get("scene_instance_id") or "unknown"),
        agent_id=str(metadata.get("agent_id") or "unknown"),
        active_instruction=instruction,
        memory_reader=MemoryRecordReader(output_dir),
    )
    session = StmrPerceptionSession(
        grounding_model=args.grounding_model,
        sam_model=args.sam_model,
        device=args.device,
    )
    session.warmup()
    builder = StmrOnlineMapBuilder(
        session=session,
        prior=args.prior,
        max_queries=args.max_queries,
        device=args.device,
    )
    result = builder.build(
        memory,
        TaskContextInput(active_instruction=instruction),
    )
    if result.png_path is None or result.json_path is None:
        raise SystemExit(
            "STMR replay did not produce a map: "
            + json.dumps(builder.debug_summary(), ensure_ascii=False)
        )

    png_path = output_dir / "latest_stmr_map.png"
    json_path = output_dir / "latest_stmr_map.json"
    shutil.copy2(result.png_path, png_path)
    shutil.copy2(result.json_path, json_path)
    payload = _read_json(json_path)
    coordinate_frame = (
        payload.get("stmr_map", {}).get("coordinate_frame", {})
        if isinstance(payload.get("stmr_map"), dict)
        else {}
    )
    summary = {
        "probe": "aerial_vln_stmr_replay",
        "source_run": str(source_run),
        "instruction": instruction,
        "status": result.status,
        "processed_observation_count": result.processed_observation_count,
        "point_count": result.point_count,
        "voxel_count": result.voxel_count,
        "center_pose": result.center_pose,
        "coordinate_frame": coordinate_frame,
        "trajectory_summary": result.trajectory_summary,
        "warnings": list(result.warnings),
        "debug": builder.debug_summary(),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    review_path = output_dir / "review.html"
    review_path.write_text(_review_html(summary), encoding="utf-8")
    print(f"STMR replay summary: {summary_path}")
    print(f"STMR replay review: {review_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay one AerialVLN Runtime run through STMR"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--grounding-model", default=DEFAULT_GROUNDING_DINO_MODEL)
    parser.add_argument("--sam-model", default=DEFAULT_SAM_MODEL)
    parser.add_argument("--prior", default=DEFAULT_UAV_PRIOR_NAME)
    parser.add_argument("--max-queries", type=int, default=MAX_LANDMARK_QUERIES)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _review_html(summary: dict[str, Any]) -> str:
    instruction = html.escape(str(summary.get("instruction") or ""))
    payload = html.escape(json.dumps(summary, indent=2, ensure_ascii=False))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AerialVLN STMR Replay</title>
  <style>
    body {{ max-width: 1180px; margin: 32px auto; padding: 0 20px; font: 15px/1.5 system-ui; color: #18212b; }}
    img {{ width: 100%; max-width: 900px; border: 1px solid #ccd5df; border-radius: 8px; }}
    pre {{ overflow: auto; padding: 16px; background: #f4f6f8; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>AerialVLN STMR Replay</h1>
  <p><strong>Parser diagnostic only:</strong> offline spatial-memory replay; no simulator or Planner actions were executed.</p>
  <h2>Instruction</h2><p>{instruction}</p>
  <h2>Latest STMR Map</h2><img src="latest_stmr_map.png" alt="Latest STMR map">
  <h2>Summary</h2><pre>{payload}</pre>
</body>
</html>
"""


if __name__ == "__main__":
    main()
