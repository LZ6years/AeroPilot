"""Tests for UAV-Flow real UnrealCV adapter setup calls.

These tests use a fake UnrealCV object so they can verify parity with the
upstream UAV-Flow runner without launching the simulator.
"""

from __future__ import annotations

from pathlib import Path

from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from evaluation.benchmarks.uav_flow.unrealcv_scenario import UavFlowBenchmarkEnvClient


def make_case(*, use_obj: int, target_pos: list[float] | None = None) -> UavFlowCase:
    return UavFlowCase(
        case_id="case",
        file_name="case.json",
        source_path="case.json",
        class_name="Turn",
        instruction="Turn to the person",
        instruction_unified="Turn to the direction of the person",
        selected_instruction="Turn to the direction of the person",
        initial_pos=[10.0, 20.0, 30.0, 0.0, 45.0, 0.0],
        use_obj=use_obj,
        obj_id=19,
        target_pos=target_pos or [100.0, 200.0, 300.0, 1.0, 2.0, 3.0],
        reference_path_preprocessed=[[0, 0, 0, 0, 0, 0]],
    )


def test_real_env_initializes_uav_flow_case_objects_before_placement(tmp_path):
    client, unrealcv = client_with_fake_env(make_case(use_obj=1), tmp_path)

    client._init_case_objects_in_real_env()
    client._place_case_objects()

    assert unrealcv.calls[:7] == [
        ("new_obj", "bp_character_C", "BP_Character_21", [0, 0, 0]),
        ("set_appearance", "BP_Character_21", 0),
        ("set_obj_rotation", "BP_Character_21", [0, 0, 0]),
        ("new_obj", "BP_BaseCar_C", "BP_Character_22", [1000, 0, 0]),
        ("set_appearance", "BP_Character_22", 2),
        ("set_obj_rotation", "BP_Character_22", [0, 0, 0]),
        ("set_phy", "BP_Character_22", 0),
    ]
    assert ("set_appearance", "BP_Character_21", 19) in unrealcv.calls
    assert ("set_obj_location", "BP_Character_21", [100.0, 200.0, 300.0]) in unrealcv.calls
    assert ("set_obj_rotation", "BP_Character_21", [1.0, 2.0, 3.0]) in unrealcv.calls
    assert ("set_obj_location", "BP_Character_22", [0, 0, -1000]) in unrealcv.calls


def test_real_env_places_vehicle_case_and_hides_person(tmp_path):
    client, unrealcv = client_with_fake_env(make_case(use_obj=2), tmp_path)

    client._init_case_objects_in_real_env()
    client._place_case_objects()

    assert ("set_appearance", "BP_Character_22", 2) in unrealcv.calls
    assert ("set_obj_location", "BP_Character_22", [100.0, 200.0, 0]) in unrealcv.calls
    assert ("set_obj_rotation", "BP_Character_22", [1.0, 2.0, 3.0]) in unrealcv.calls
    assert ("set_phy", "BP_Character_22", 0) in unrealcv.calls
    assert ("set_obj_location", "BP_Character_21", [0, 0, -1000]) in unrealcv.calls


def test_real_env_pose_updates_sync_camera_to_player(tmp_path):
    client, unrealcv = client_with_fake_env(make_case(use_obj=1), tmp_path)

    client._apply_pose_to_real_env()

    assert ("set_obj_location", "Drone_1", [10.0, 20.0, 30.0]) in unrealcv.calls
    assert ("set_rotation", "Drone_1", -135.0) in unrealcv.calls
    assert unrealcv.calls[-1] == ("set_cam", 0, [10.0, 20.0, 30.0], [0.0, 0.0, 45.0])


def test_real_rgb_capture_syncs_camera_before_get_image(tmp_path):
    client, unrealcv = client_with_fake_env(make_case(use_obj=1), tmp_path)

    frame = client._rgb_frame()

    assert Path(frame["path"]).exists()
    set_cam_index = unrealcv.calls.index(("set_cam", 0, [10.0, 20.0, 30.0], [0.0, 0.0, 45.0]))
    get_image_index = unrealcv.calls.index(("get_image", 0, "lit"))
    assert set_cam_index < get_image_index


def client_with_fake_env(case: UavFlowCase, tmp_path: Path) -> tuple[UavFlowBenchmarkEnvClient, "FakeUnrealCv"]:
    client = UavFlowBenchmarkEnvClient(case, run_dir=tmp_path, env_id="fake", real_unrealcv=True)
    unrealcv = FakeUnrealCv(case.initial_pos)
    client.env = FakeEnv(unrealcv)
    return client, unrealcv


class FakeEnv:
    def __init__(self, unrealcv: "FakeUnrealCv") -> None:
        self.unwrapped = FakeUnwrapped(unrealcv)


class FakeUnwrapped:
    def __init__(self, unrealcv: "FakeUnrealCv") -> None:
        self.unrealcv = unrealcv
        self.player_list = ["Drone_1"]


class FakeUnrealCv:
    def __init__(self, initial_pose: list[float]) -> None:
        self.calls: list[tuple] = []
        self.locations = {"Drone_1": list(initial_pose[:3])}
        self.rotations = {"Drone_1": [0.0, initial_pose[4], 0.0]}

    def new_obj(self, obj_class_name: str, obj_name: str, loc: list[float]) -> None:
        self.calls.append(("new_obj", obj_class_name, obj_name, list(loc)))

    def set_appearance(self, name: str, value: int) -> None:
        self.calls.append(("set_appearance", name, value))

    def set_obj_rotation(self, name: str, rot: list[float]) -> None:
        self.rotations[name] = list(rot)
        self.calls.append(("set_obj_rotation", name, list(rot)))

    def set_obj_location(self, name: str, loc: list[float]) -> None:
        self.locations[name] = list(loc)
        self.calls.append(("set_obj_location", name, list(loc)))

    def set_phy(self, name: str, value: int) -> None:
        self.calls.append(("set_phy", name, value))

    def set_rotation(self, name: str, yaw: float) -> None:
        current = self.rotations.get(name, [0.0, 0.0, 0.0])
        self.rotations[name] = [current[0], yaw + 180.0, current[2]]
        self.calls.append(("set_rotation", name, yaw))

    def get_obj_location(self, name: str) -> list[float]:
        return list(self.locations[name])

    def get_obj_rotation(self, name: str) -> list[float]:
        return list(self.rotations[name])

    def set_cam(self, cam_id: int, loc: list[float], rot: list[float]) -> None:
        self.calls.append(("set_cam", cam_id, list(loc), list(rot)))

    def get_image(self, cam_id: int, mode: str):
        self.calls.append(("get_image", cam_id, mode))
        import numpy as np

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, :] = [12, 34, 56]
        return image
