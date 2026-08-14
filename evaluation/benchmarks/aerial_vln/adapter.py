"""Convert AerialVLN NED/WXYZ cases into canonical Runtime episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluation.benchmarks.aerial_vln.cases import AerialVlnCase
from evaluation.benchmarks.aerial_vln.protocol import AERIAL_VLN_PROTOCOL
from evaluation.benchmarks.core import (
    EpisodeTrace,
    PreparedBenchmarkEpisode,
    ned_to_canonical_position,
    quaternion_wxyz_to_euler_deg,
)
from src.env_client import ObservationProfile
from src.runtime.configs import InitialPose, TaskExecutionOptions


@dataclass(frozen=True)
class AerialVlnEvaluationPayload:
    """Official AerialVLN truth retained outside Runtime artifacts."""

    episode_id: str
    trajectory_id: str
    scene_id: str
    goal_position_ned: tuple[float, float, float]
    reference_path_ned: tuple[tuple[float, ...], ...]
    official_actions: tuple[Any, ...]


class AerialVlnCaseAdapter:
    """Build one native-continuous-action episode from official data."""

    def __init__(
        self,
        *,
        scene_map: dict[str, str],
        observation_profile: ObservationProfile,
        agent_id: str = "Drone_1",
    ) -> None:
        self.scene_map = {str(key): value for key, value in scene_map.items()}
        self.observation_profile = observation_profile
        self.agent_id = agent_id

    def prepare_case(
        self, case: AerialVlnCase
    ) -> PreparedBenchmarkEpisode[None, AerialVlnEvaluationPayload]:
        try:
            map_id = self.scene_map[case.scene_id]
        except KeyError as exc:
            raise ValueError(
                f"AerialVLN scene_id {case.scene_id!r} has no scene_map entry"
            ) from exc
        x, y, z = ned_to_canonical_position(case.start_position)
        yaw, pitch, roll = quaternion_wxyz_to_euler_deg(
            case.start_rotation_wxyz
        )
        options = TaskExecutionOptions.from_observation_profile(
            self.observation_profile,
            initial_pose=InitialPose(
                x=x,
                y=y,
                z=z,
                yaw_deg=yaw,
                pitch_deg=pitch,
                roll_deg=roll,
                ignore_collision=True,
            ),
        )
        return PreparedBenchmarkEpisode(
            benchmark_name="aerial_vln",
            case_id=case.episode_id,
            protocol_variant=AERIAL_VLN_PROTOCOL,
            instruction=case.instruction,
            map_id=map_id,
            scene_id=case.scene_id,
            agent_id=self.agent_id,
            observation_profile=self.observation_profile,
            execution_options=options,
            planner_safe_metadata=case.planner_safe_metadata(),
            scenario_payload=None,
            evaluation_payload=AerialVlnEvaluationPayload(
                episode_id=case.episode_id,
                trajectory_id=case.trajectory_id,
                scene_id=case.scene_id,
                goal_position_ned=case.goal_position,
                reference_path_ned=case.reference_path,
                official_actions=case.official_actions,
            ),
        )


class AerialVlnCoordinateAdapter:
    """Convert canonical positive-up Runtime samples back to AirSim NED."""

    accepted_runtime_frames = {
        "canonical_positive_up_m",
        "airsim_canonical_m",
        # Current EnvClient memory fallback. Accepting it is an explicit v1
        # adapter decision based on Runtime's canonical pose contract.
        "canonical_unspecified",
    }

    def trace_to_ned(
        self, trace: EpisodeTrace
    ) -> list[tuple[float, float, float]]:
        unexpected = sorted(
            set(trace.coordinate_frames).difference(
                self.accepted_runtime_frames
            )
        )
        if unexpected:
            raise ValueError(
                "AerialVLN cannot convert Runtime coordinate frames: "
                + ", ".join(unexpected)
            )
        return [self.pose_to_ned(pose) for pose in trace.poses]

    @staticmethod
    def pose_to_ned(
        pose: dict[str, Any],
    ) -> tuple[float, float, float]:
        position = pose.get("position")
        nested = position if isinstance(position, dict) else {}
        x = float(pose.get("x", nested.get("x", 0.0)))
        y = float(pose.get("y", nested.get("y", 0.0)))
        z = float(pose.get("z", nested.get("z", 0.0)))
        return x, y, -z
