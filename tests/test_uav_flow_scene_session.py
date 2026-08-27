"""Contract tests for UAV-Flow's batch-owned simulator scene."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from evaluation.benchmarks.core.models import PreparedBenchmarkEpisode
from evaluation.benchmarks.uav_flow.scene_session import (
    UavFlowBatchSceneSession,
)
from src.env_client import ObservationProfile
from src.env_client.schemas import ActionResult, Observation, SceneCapabilities, SceneInstance
from src.runtime.configs import InitialPose, TaskExecutionOptions


class FakeService:
    def __init__(self) -> None:
        self.release_calls: list[str] = []

    def release_scene_client(self, scene_instance_id: str) -> None:
        self.release_calls.append(scene_instance_id)


class FakeClient:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls: list[str] = []
        self.actions: list[tuple[str, dict]] = []
        self.pose = InitialPose(1.0, 2.0, 3.0, yaw_deg=15.0).to_dict()

    def get_or_start_scene(self, map_id: str, scene_id: str | None):
        self.start_calls += 1
        return SceneInstance(
            "scene_shared",
            map_id,
            scene_id or "scene",
            "local",
            "127.0.0.1",
            9068,
            None,
            123,
            "running",
            "local",
            capabilities=SceneCapabilities(
                "unrealcv",
                "actor_pose",
                ("connect", "hover", "set_pose"),
                ("rgb", "depth", "pose"),
                False,
                False,
            ),
        )

    def execute_action(self, scene_id, agent_id, action, args):
        self.actions.append((action, dict(args)))
        if action == "set_pose":
            self.pose = dict(args)
        return ActionResult(True, action, "ok")

    def get_observation(self, scene_id, agent_id, modalities):
        return Observation(
            scene_id,
            agent_id,
            list(modalities),
            {"pose": dict(self.pose), "frames": {}},
        )

    def stop_scene(self, scene_instance_id: str) -> None:
        self.stop_calls.append(scene_instance_id)


class FakeRuntime:
    shutdown_policy = "hover"

    def __init__(self) -> None:
        self.hover_calls = 0

    def safe_hover_reused_scene(self, *args, **kwargs) -> None:
        self.hover_calls += 1


def make_episode() -> PreparedBenchmarkEpisode:
    profile = ObservationProfile()
    return PreparedBenchmarkEpisode(
        benchmark_name="UAV-Flow",
        case_id="case_1",
        protocol_variant="uav-flow-test",
        instruction="move forward",
        map_id="map",
        scene_id="scene",
        agent_id="drone",
        observation_profile=profile,
        execution_options=TaskExecutionOptions.from_observation_profile(
            profile,
            initial_pose=InitialPose(1.0, 2.0, 3.0, yaw_deg=15.0),
        ),
        planner_safe_metadata={"case_id": "case_1"},
    )


def test_batch_session_starts_once_prepares_each_episode_and_stops_once(tmp_path):
    service = FakeService()
    client = FakeClient()
    session = UavFlowBatchSceneSession(
        service, client, map_id="map", scene_id="scene", eval_dir=tmp_path
    )

    first = session.prepare_episode(make_episode())
    second = session.prepare_episode(make_episode())
    session.close()
    session.close()

    assert first.scene_instance_id == second.scene_instance_id == "scene_shared"
    assert client.start_calls == 1
    assert client.stop_calls == ["scene_shared"]
    assert [action for action, _ in client.actions] == [
        "connect",
        "set_pose",
        "connect",
        "set_pose",
    ]
    assert service.release_calls == ["scene_shared", "scene_shared"]
    manifest = json.loads(
        (tmp_path / "scene_session.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "closed"
    assert manifest["start_count"] == 1
    assert manifest["stop_count"] == 1


def test_episode_runtime_hovers_and_releases_without_stopping_scene(tmp_path):
    service = FakeService()
    client = FakeClient()
    session = UavFlowBatchSceneSession(
        service, client, map_id="map", scene_id="scene", eval_dir=tmp_path
    )
    session.ensure_scene()
    runtime = FakeRuntime()

    session.runtime_view(runtime).finish_reused_scene(
        "map", SimpleNamespace(), agent_id="drone", scene_id="scene"
    )

    assert runtime.hover_calls == 1
    assert service.release_calls == ["scene_shared"]
    assert client.stop_calls == []
    session.close()


def test_restart_replaces_scene_and_is_audited(tmp_path):
    service = FakeService()
    client = FakeClient()
    session = UavFlowBatchSceneSession(
        service, client, map_id="map", scene_id="scene", eval_dir=tmp_path
    )
    session.ensure_scene()

    session.restart_scene(reason="UnrealCV connection lost")
    session.close()

    assert client.start_calls == 2
    assert client.stop_calls == ["scene_shared", "scene_shared"]
    manifest = json.loads(
        (tmp_path / "scene_session.json").read_text(encoding="utf-8")
    )
    assert manifest["restart_count"] == 1
    assert manifest["start_count"] == 2
