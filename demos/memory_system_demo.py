"""Inspect static FlightMemory records and V2 context without simulating flight.

Run: python demos/memory_system_demo.py --run-root runs
Writes memory JSONL, flight_memory.json and latest_checkpoint.json. These are
educational fixtures, not benchmark trajectories or completed flight results.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.mission_state import MissionState
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.runtime.factory import RuntimeFactory
from src.uav.config import UavSafetyConfig


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs")
    return parser


def main():
    args = build_parser().parse_args()
    run_dir = Path(args.run_root) / ("memory_system_demo_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    writer = MemoryEventWriter(run_dir)
    memory = FlightMemory("fixture", "fixture_map", "offline", "uav", memory_reader=MemoryRecordReader(run_dir))
    for instruction in ("Inspect the road to the right.", "Inspect the entrance to the left."):
        memory.update_instruction_ref(writer.write_instruction(instruction), instruction)
        observation = {
            "source": "static_fixture", "modalities": ["pose"],
            "pose": {"x": 0.0, "y": 0.0, "z": 3.0, "yaw_deg": 0.0, "coordinate_frame": "canonical_world_m"},
        }
        refs = writer.write_observation(observation)
        memory.append_observation_ref(refs["observation_ref"])
        memory.append_pose_ref(refs["pose_ref"])
    state = MissionState("fixture", memory.active_instruction)
    state.record_observation(observation)
    builder = RuntimeFactory().context_builder(run_dir, UavSafetyConfig())
    envelope = builder.build_instruction_envelope(state, memory)
    checkpoint, _ = builder.build_checkpoint(
        state, memory, checkpoint_id="checkpoint_000001", reason="instruction_start"
    )
    artifacts = (
        ("state/flight_memory.json", memory.to_dict()),
        ("contexts/instruction_envelope.json", envelope.to_dict()),
        ("contexts/latest_checkpoint.json", checkpoint.to_dict()),
    )
    for relative, payload in artifacts:
        path = run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"mode": "static_memory_fixture_no_execution", "run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
