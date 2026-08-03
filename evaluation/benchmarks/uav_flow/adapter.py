"""Translate UAV-Flow dataset cases into canonical benchmark episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluation.benchmarks.core import PreparedBenchmarkEpisode
from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from src.env_client import ObservationProfile
from src.runtime.configs import InitialPose, TaskExecutionOptions


UAV_FLOW_PROTOCOL = "uav_flow.uav_agent_native_actions.v1"


@dataclass(frozen=True)
class UavFlowScenarioPayload:
    """Evaluation scene objects placed through a privileged UnrealCV client."""

    case_id: str
    use_obj: int | None
    obj_id: int | None
    target_pos: tuple[float, ...] | None


@dataclass(frozen=True)
class UavFlowEvaluationPayload:
    """UAV-Flow truth that must never enter Runtime artifacts."""

    case_id: str
    class_name: str | None
    initial_pos: tuple[float, ...]
    end_pos: tuple[float, ...]
    target_pos: tuple[float, ...] | None
    reference_path_raw: tuple[tuple[float, ...], ...]
    reference_path_preprocessed: tuple[tuple[float, ...], ...]


class UavFlowCaseAdapter:
    """Convert centimeters and benchmark fields at the dataset boundary."""

    def __init__(
        self,
        *,
        map_id: str,
        scene_id: str | None,
        agent_id: str,
        observation_profile: ObservationProfile,
    ) -> None:
        self.map_id = map_id
        self.scene_id = scene_id
        self.agent_id = agent_id
        self.observation_profile = observation_profile

    def prepare_case(
        self, case: UavFlowCase
    ) -> PreparedBenchmarkEpisode[
        UavFlowScenarioPayload, UavFlowEvaluationPayload
    ]:
        initial = case.initial_pos
        if len(initial) < 6:
            raise ValueError(
                f"UAV-Flow case {case.case_id} requires six-value initial_pos"
            )
        initial_pose = InitialPose(
            x=initial[0] / 100.0,
            y=initial[1] / 100.0,
            z=initial[2] / 100.0,
            # The original UAV-Flow runner places its drone with location and
            # Blueprint yaw only. Dataset slots 3/5 can contain reference-path
            # orientation for angled translation cases, but were never applied
            # as controllable UAV pitch/roll. Preserve those values only in the
            # evaluation payload so the migrated runner keeps the old scene
            # initialization contract.
            pitch_deg=0.0,
            yaw_deg=initial[4],
            roll_deg=0.0,
            ignore_collision=True,
        )
        options = TaskExecutionOptions.from_observation_profile(
            self.observation_profile,
            initial_pose=initial_pose,
        )
        return PreparedBenchmarkEpisode(
            benchmark_name="uav_flow",
            case_id=case.case_id,
            protocol_variant=UAV_FLOW_PROTOCOL,
            instruction=case.selected_instruction,
            map_id=self.map_id,
            scene_id=self.scene_id,
            agent_id=self.agent_id,
            observation_profile=self.observation_profile,
            execution_options=options,
            planner_safe_metadata=case.planner_safe_metadata(),
            scenario_payload=UavFlowScenarioPayload(
                case_id=case.case_id,
                use_obj=case.use_obj,
                obj_id=case.obj_id,
                target_pos=(
                    tuple(float(value) for value in case.target_pos)
                    if case.target_pos
                    else None
                ),
            ),
            evaluation_payload=UavFlowEvaluationPayload(
                case_id=case.case_id,
                class_name=case.class_name,
                initial_pos=tuple(float(value) for value in case.initial_pos),
                end_pos=tuple(float(value) for value in case.end_pos),
                target_pos=(
                    tuple(float(value) for value in case.target_pos)
                    if case.target_pos
                    else None
                ),
                reference_path_raw=_freeze_path(case.reference_path_raw),
                reference_path_preprocessed=_freeze_path(
                    case.reference_path_preprocessed
                ),
            ),
        )


def validate_uav_flow_profile(
    profile: ObservationProfile, *, stmr_mode: str
) -> None:
    """Fail before scene launch when perception requirements cannot be met."""

    required = {"rgb", "depth", "pose"}
    if stmr_mode != "off":
        required.add("segmentation")
    missing = sorted(required.difference(profile.modalities))
    if missing:
        mode = "STMR" if stmr_mode != "off" else "UAV-Flow"
        raise ValueError(
            f"{mode} observation profile is missing modalities: "
            + ", ".join(missing)
        )


def _freeze_path(path: list[list[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in path)
