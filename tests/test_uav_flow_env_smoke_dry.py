"""Dry tests for the UAV-Flow environment-only smoke path."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from evaluation.benchmarks.uav_flow.config import UavFlowEvalConfig
from evaluation.benchmarks.uav_flow.env_smoke import run_env_smoke
from evaluation.benchmarks.uav_flow.unrealcv_scenario import UavFlowBenchmarkEnvClient


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


def test_research_frame_capture_is_disabled_by_default(tmp_path):
    client = UavFlowBenchmarkEnvClient(
        _case(),
        run_dir=tmp_path / "run",
        env_id="dry",
        real_unrealcv=False,
    )
    scene = client.start_scene("uav_flow_downtownwest", "case")

    observation = client.get_observation(scene.scene_instance_id, "uav_flow_drone", ["depth", "mask", "pose"])
    frames = observation.to_dict()["data"]["frames"]

    assert client.enable_research_frame_capture is False
    assert frames["depth"] == {"mock": True, "reason": "uav_flow_v1_rgb_pose_only"}
    assert frames["mask"]["mock"] is True
    assert frames["mask"]["research_frame_capture"] == "disabled"
    assert "path" not in frames["depth"]
    assert "path" not in frames["mask"]
    assert not (tmp_path / "run" / "artifacts" / "depth").exists()
    assert not (tmp_path / "run" / "artifacts" / "mask").exists()


def test_research_frame_capture_dry_mode_does_not_create_fake_depth_or_mask(tmp_path):
    client = UavFlowBenchmarkEnvClient(
        _case(),
        run_dir=tmp_path / "run",
        env_id="dry",
        real_unrealcv=False,
        enable_research_frame_capture=True,
    )
    scene = client.start_scene("uav_flow_downtownwest", "case")

    observation = client.get_observation(scene.scene_instance_id, "uav_flow_drone", ["depth", "mask", "pose"])
    frames = observation.to_dict()["data"]["frames"]

    assert frames["depth"]["mock"] is True
    assert frames["depth"]["research_frame_capture"] == "unavailable"
    assert frames["mask"]["mock"] is True
    assert frames["mask"]["research_frame_capture"] == "unavailable"
    assert "path" not in frames["depth"]
    assert "path" not in frames["mask"]
    assert not (tmp_path / "run" / "artifacts" / "depth").exists()
    assert not (tmp_path / "run" / "artifacts" / "mask").exists()


def test_research_frame_capture_real_mode_writes_depth_and_mask_artifacts(tmp_path):
    client = UavFlowBenchmarkEnvClient(
        _case(),
        run_dir=tmp_path / "run",
        env_id="fake_real",
        real_unrealcv=True,
        enable_research_frame_capture=True,
    )
    client.env = _fake_unrealcv_env()

    observation = client.get_observation("scene_1", "uav_flow_drone", ["depth", "mask", "pose"])
    frames = observation.to_dict()["data"]["frames"]

    depth_path = Path(frames["depth"]["path"])
    mask_path = Path(frames["mask"]["path"])
    assert depth_path.suffix == ".npy"
    assert mask_path.suffix == ".png"
    depth = np.load(depth_path)
    assert depth.shape == (4, 4)
    assert abs(float(depth[0, 0]) - 0.01) < 1e-6
    assert Image.open(mask_path).size == (4, 4)
    assert frames["depth"]["mock"] is False
    assert frames["depth"]["depth_unit"] == "meter"
    assert frames["depth"]["raw_depth_unit"] == "centimeter"
    assert frames["mask"]["mock"] is False


def test_research_mask_label_map_uses_visible_object_inventory_only(tmp_path):
    client = UavFlowBenchmarkEnvClient(
        _case(),
        run_dir=tmp_path / "run",
        env_id="fake_real",
        real_unrealcv=True,
        enable_research_frame_capture=True,
    )
    client.env = _fake_unrealcv_env(
        objects=["BP_Character_21", "BP_BaseCar_22", "MysteryAsset_7"],
        colors={
            "BP_Character_21": (255, 0, 0),
            "BP_BaseCar_22": (0, 255, 0),
            "MysteryAsset_7": (0, 0, 255),
        },
    )

    label_map = client.collect_research_mask_label_map({(255, 0, 0), (0, 0, 255)})

    assert label_map["colors"]["255,0,0"]["label"] == "person"
    assert label_map["colors"]["255,0,0"]["object_name"] == "BP_Character_21"
    assert "0,255,0" not in label_map["colors"]
    assert "0,0,255" not in label_map["colors"]
    assert label_map["unmapped_objects"] == [
        {
            "object_name": "MysteryAsset_7",
            "color": [0, 0, 255],
            "reason": "object_name_not_classified",
        }
    ]


def _case() -> UavFlowCase:
    return UavFlowCase(
        case_id="case",
        file_name="case.json",
        source_path="test",
        class_name="Turn",
        instruction="Turn toward the target.",
        instruction_unified="Turn toward the target.",
        selected_instruction="Turn toward the target.",
        initial_pos=[0, 0, 100, 0, 0, 0],
        target_pos=[100, 0, 100, 0, 0, 0],
        use_obj=1,
        obj_id=19,
        reference_path_preprocessed=[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 10, 0]],
    )


def _fake_unrealcv_env(
    objects: list[str] | None = None,
    colors: dict[str, tuple[int, int, int]] | None = None,
) -> Any:
    class FakeRequestClient:
        def request(self, command: str) -> bytes:
            if "depth npy" in command:
                buffer = BytesIO()
                np.save(buffer, np.ones((4, 4), dtype=np.float32))
                return buffer.getvalue()
            if "object_mask png" in command:
                buffer = BytesIO()
                Image.new("RGB", (4, 4), (255, 0, 0)).save(buffer, format="PNG")
                return buffer.getvalue()
            raise AssertionError(f"unexpected command: {command}")

    object_names = list(objects or [])
    object_colors = dict(colors or {})
    unrealcv = SimpleNamespace(
        client=FakeRequestClient(),
        get_obj_location=lambda name: (0.0, 0.0, 100.0),
        get_obj_rotation=lambda name: (0.0, 0.0, 0.0),
        set_cam=lambda camera_id, location, rotation: None,
        get_objects=lambda: object_names,
        build_color_dict=lambda names: {name: object_colors[name] for name in names if name in object_colors},
    )
    return SimpleNamespace(unwrapped=SimpleNamespace(unrealcv=unrealcv, player_list=["uav"]))
