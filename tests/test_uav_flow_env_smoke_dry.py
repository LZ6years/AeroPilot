"""Dry tests for the UAV-Flow environment-only smoke path."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.benchmarks.uav_flow.config import UavFlowEvalConfig
from evaluation.benchmarks.uav_flow.env_smoke import run_env_smoke


def write_case(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "instruction": "Turn to the direction of the person.",
                "instruction_unified": "Turn to the direction of the person.",
                "initial_pos": [0, 0, 100, 0, 0, 0],
                "target_pos": [100, 0, 100, 0, 0, 0],
                "obj_id": 19,
                "use_obj": 1,
                "reference_path_preprocessed": [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 10, 0]],
            }
        ),
        encoding="utf-8",
    )


def test_env_smoke_dry_generates_rgb_pose_and_action_outputs(tmp_path):
    repo = tmp_path / "uav_flow"
    write_case(repo / "test_jsons" / "turn_a.json")
    (repo / "classified_instr.json").write_text(json.dumps({"Turn": ["turn_a.json"]}), encoding="utf-8")
    config = UavFlowEvalConfig.from_dict(
        {
            "uav_flow_repo_path": str(repo),
            "classes": ["Turn"],
            "real_unrealcv": False,
            "limit_per_class": 1,
            "action_config_overrides": {"linear_velocity_mps": 1.0, "yaw_step_deg": 10.0},
        }
    )

    result = run_env_smoke(config, class_name="Turn", output_root=tmp_path / "out", real_unrealcv=False)
    summary = result["summary"]

    assert summary["turn_result"]["ok"] is True
    assert summary["move_result"]["ok"] is True
    before_rgb = summary["before_observation"]["data"]["frames"]["rgb"]["path"]
    after_rgb = summary["after_observation"]["data"]["frames"]["rgb"]["path"]
    assert Path(before_rgb).exists()
    assert Path(after_rgb).exists()
    assert summary["before_observation"]["data"]["pose"] != summary["after_observation"]["data"]["pose"]
    assert summary["action_config"]["linear_velocity_mps"] == 1.0
    assert (Path(result["case_dir"]) / "env_smoke.json").exists()
