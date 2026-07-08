"""Manual probe for rendering custom_map artifacts from an existing run.

Usage:
    python tests/manual/custom_map_render_probe.py --run-dir outputs/runs/example
    python tests/manual/custom_map_render_probe.py --generate-uav-flow-dry --output-dir tmp/custom_map_probe

The script writes latest_custom_map.png through the plugin and stores a sanitized
JSON copy of the fragment next to it for inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from evaluation.benchmarks.uav_flow.unrealcv_scenario import UavFlowBenchmarkEnvClient
from plugins.research.custom_map.context_modules.custom_map import PluginContextModule
from src.memory.context.schemas import MemorySourceBundle, StageContextInput
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.perception.raw_observation import RawObservation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", help="Run directory containing memory/*.jsonl")
    parser.add_argument("--generate-uav-flow-dry", action="store_true", help="Generate a dry UAV-Flow run before rendering")
    parser.add_argument("--output-dir", help="Output directory for --generate-uav-flow-dry")
    parser.add_argument(
        "--with-synthetic-mask",
        action="store_true",
        help="Add a synthetic mask and mask_label_map.json for semantic parser validation",
    )
    args = parser.parse_args()
    if args.generate_uav_flow_dry:
        if not args.output_dir:
            raise SystemExit("--output-dir is required with --generate-uav-flow-dry")
        run_dir = _generate_uav_flow_dry_run(Path(args.output_dir), with_synthetic_mask=args.with_synthetic_mask)
    elif args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        raise SystemExit("Provide --run-dir or --generate-uav-flow-dry --output-dir")
    fragment = _render_custom_map(run_dir)
    output_dir = run_dir / "contexts" / "custom_map"
    json_path = output_dir / "latest_custom_map.json"
    print(f"run_dir:          {run_dir}")
    print(f"custom_map image: {fragment['artifact_refs'][0]}")
    print(f"custom_map json:  {json_path}")
    if len(fragment.get("artifact_refs", [])) > 1:
        print(f"custom_map html:  {fragment['artifact_refs'][1]}")


def _render_custom_map(run_dir: Path) -> dict:
    memory = FlightMemory(
        run_id=run_dir.name,
        map_id="manual",
        scene_instance_id="manual",
        agent_id="uav_1",
        active_instruction="manual custom_map render probe",
        memory_reader=MemoryRecordReader(run_dir),
    )
    stage_context = StageContextInput(active_instruction=memory.active_instruction)
    fragment = PluginContextModule().build(MemorySourceBundle(memory), stage_context).to_dict()
    sanitized = dict(fragment)
    sanitized["model_inputs"] = [
        {key: value for key, value in item.items() if key != "data"}
        for item in fragment.get("model_inputs", [])
    ]
    output_dir = run_dir / "contexts" / "custom_map"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "latest_custom_map.json"
    json_path.write_text(json.dumps(sanitized, ensure_ascii=False, indent=2), encoding="utf-8")
    return fragment


def _generate_uav_flow_dry_run(output_dir: Path, *, with_synthetic_mask: bool) -> Path:
    run_id = f"custom_map_dry_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    case = UavFlowCase(
        case_id="custom_map_dry",
        file_name="custom_map_dry.json",
        source_path="manual",
        class_name="Turn",
        instruction="Turn and move forward for custom map probing.",
        instruction_unified="Turn and move forward for custom map probing.",
        selected_instruction="Turn and move forward for custom map probing.",
        initial_pos=[0, 0, 100, 0, 0, 0],
        reference_path_preprocessed=[[0, 0, 0, 0, 0, 0]],
    )
    client = UavFlowBenchmarkEnvClient(case, run_dir=run_dir, env_id="dry", real_unrealcv=False)
    scene = client.start_scene("uav_flow_downtownwest", case.case_id)
    writer = MemoryEventWriter(run_dir)
    _write_observation(client, writer, scene.scene_instance_id)
    for action, tool, args_dict in (
        ("turn_relative", "uav.turn_relative", {"yaw_deg": 20.0}),
        ("move_body", "uav.move_body", {"vx": 1.0, "vy": 0.0, "vz": 0.0, "duration": 2.0}),
        ("turn_relative", "uav.turn_relative", {"yaw_deg": -10.0}),
        ("move_body", "uav.move_body", {"vx": 1.0, "vy": 0.4, "vz": 0.0, "duration": 1.5}),
    ):
        result = client.execute_action(scene.scene_instance_id, "uav_flow_drone", action, args_dict)
        writer.write_action(tool, args_dict, result.to_dict())
        _write_observation(client, writer, scene.scene_instance_id)
    if with_synthetic_mask:
        _write_synthetic_mask(run_dir, writer, pose=client._model_pose())
    return run_dir


def _write_observation(client: UavFlowBenchmarkEnvClient, writer: MemoryEventWriter, scene_instance_id: str) -> None:
    observation = client.get_observation(scene_instance_id, "uav_flow_drone", ["rgb", "pose"])
    raw = RawObservation.from_env_observation(observation, source="uav_flow_dry_custom_map_probe").to_dict()
    writer.write_observation(raw)


def _write_synthetic_mask(run_dir: Path, writer: MemoryEventWriter, *, pose: dict) -> None:
    mask_path = run_dir / "artifacts" / "mask" / "synthetic_mask.png"
    depth_path = run_dir / "artifacts" / "depth" / "synthetic_depth.npy"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (80, 80), (0, 0, 0))
    depth = np.zeros((80, 80), dtype=np.float32)
    for y in range(8, 42):
        for x in range(8, 72):
            image.putpixel((x, y), (255, 0, 0))
            depth[y, x] = 8.0
    image.save(mask_path)
    np.save(depth_path, depth)
    label_dir = run_dir / "memory" / "map"
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "mask_label_map.json").write_text(
        json.dumps({"colors": {"255,0,0": {"id": 2, "label": "synthetic_building", "category": "debug"}}}),
        encoding="utf-8",
    )
    writer.write_observation(
        {
            "scene_instance_id": "custom_map_dry",
            "agent_id": "uav_flow_drone",
            "modalities": ["depth", "mask", "pose"],
            "pose": pose,
            "frames": {
                "depth": {
                    "path": str(depth_path),
                    "mock": False,
                    "source": "synthetic_debug_depth",
                    "fov_degrees": 90,
                },
                "mask": {
                    "path": str(mask_path),
                    "mock": False,
                    "source": "synthetic_debug_mask",
                    "fov_degrees": 90,
                },
            },
            "source": "synthetic_mask_probe",
        }
    )


if __name__ == "__main__":
    main()
