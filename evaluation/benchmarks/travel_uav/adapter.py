"""Adapt TravelUAV NED/XYZW cases to the shared Runtime contract."""

from __future__ import annotations
from evaluation.benchmarks.travel_uav.protocol import TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT

from dataclasses import dataclass

from evaluation.benchmarks.core import (
    PreparedBenchmarkEpisode,
    ned_to_canonical_position,
    quaternion_wxyz_to_euler_deg,
)
from evaluation.benchmarks.travel_uav.cases import TravelUavCase, TravelUavTargetObject
from evaluation.benchmarks.travel_uav.protocol import TRAVEL_UAV_PROTOCOL
from src.env_client import ObservationProfile
from src.runtime.configs import InitialPose, TaskExecutionOptions


@dataclass(frozen=True)
class TravelUavScenarioPayload:
    """Privileged target placement consumed only by the scenario controller."""

    target: TravelUavTargetObject


@dataclass(frozen=True)
class TravelUavEvaluationPayload:
    """Hidden metric truth retained outside Runtime and Planner artifacts."""

    target_position_ned: tuple[float, float, float]
    reference_path_ned: tuple[tuple[float, float, float], ...]
    reference_path_length_m: float


class TravelUavCaseAdapter:
    """Create one front-view native-action TravelUAV episode."""

    def __init__(
        self,
        *,
        scene_map: dict[str, str],
        observation_profile: ObservationProfile,
        agent_id: str = "Drone_1",
    ) -> None:
        self.scene_map = {str(key): str(value) for key, value in scene_map.items()}
        self.observation_profile = observation_profile
        self.agent_id = agent_id

    def prepare_case(
        self, case: TravelUavCase
    ) -> PreparedBenchmarkEpisode[TravelUavScenarioPayload, TravelUavEvaluationPayload]:
        try:
            map_id = self.scene_map[case.map_name]
        except KeyError as exc:
            raise ValueError(
                f"TravelUAV map {case.map_name!r} has no scene_map entry"
            ) from exc
        x, y, z = ned_to_canonical_position(case.start_position_ned)
        qx, qy, qz, qw = case.start_rotation_xyzw
        yaw, pitch, roll = quaternion_wxyz_to_euler_deg((qw, qx, qy, qz))
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
            protocol_context=TRAVEL_UAV_PARSER_PROTOCOL_CONTEXT,
            benchmark_name="travel_uav",
            case_id=case.episode_id,
            protocol_variant=TRAVEL_UAV_PROTOCOL,
            instruction=case.instruction,
            map_id=map_id,
            scene_id=case.map_name,
            agent_id=self.agent_id,
            observation_profile=self.observation_profile,
            execution_options=options,
            planner_safe_metadata=case.planner_safe_metadata(),
            scenario_payload=TravelUavScenarioPayload(target=case.target),
            evaluation_payload=TravelUavEvaluationPayload(
                target_position_ned=case.target.position_ned,
                reference_path_ned=case.reference_path_ned,
                reference_path_length_m=case.reference_path_length_m,
            ),
        )
