"""Dataset-boundary tests for the migrated UAV-Flow adapter."""

from __future__ import annotations

import pytest

from evaluation.benchmarks.core import (
    BenchmarkExecutionOutcome,
    EpisodePhysicalStep,
    EpisodePhysicalWindow,
    EpisodeTrace,
    EpisodeTraceSample,
)
from evaluation.benchmarks.uav_flow.adapter import (
    UAV_FLOW_PROTOCOL,
    UavFlowCaseAdapter,
    UavFlowEvaluationPayload,
    validate_uav_flow_profile,
)
from evaluation.benchmarks.uav_flow.cases import UavFlowCase
from evaluation.benchmarks.uav_flow.metric_suite import UavFlowMetricSuite
from evaluation.benchmarks.uav_flow.trajectory import UavFlowCoordinateAdapter
from src.env_client import ObservationProfile


def test_case_adapter_converts_centimeters_to_canonical_initial_pose():
    case = make_case()
    profile = ObservationProfile()

    episode = UavFlowCaseAdapter(
        map_id="downtownwest_local_unrealcv",
        scene_id="DowntownWest",
        agent_id="drone",
        observation_profile=profile,
    ).prepare_case(case)

    pose = episode.execution_options.initial_pose
    assert pose is not None
    assert (pose.x, pose.y, pose.z) == (1.0, 2.0, 0.5)
    assert pose.yaw_deg == 90.0
    assert pose.ignore_collision is True
    assert episode.protocol_variant == UAV_FLOW_PROTOCOL
    assert episode.evaluation_payload.reference_path_preprocessed
    assert "reference_path_preprocessed" not in episode.planner_manifest()


def test_case_adapter_preserves_old_yaw_only_drone_initialization():
    case = UavFlowCase(
        **{
            **make_case().__dict__,
            "initial_pos": [
                100.0,
                200.0,
                50.0,
                -32.475,
                74.0,
                -54.717,
            ],
        }
    )

    episode = UavFlowCaseAdapter(
        map_id="downtownwest_local_unrealcv",
        scene_id="DowntownWest",
        agent_id="drone",
        observation_profile=ObservationProfile(),
    ).prepare_case(case)

    pose = episode.execution_options.initial_pose
    assert pose is not None
    assert pose.pitch_deg == 0.0
    assert pose.yaw_deg == 74.0
    assert pose.roll_deg == 0.0
    assert episode.evaluation_payload.initial_pos[3] == -32.475
    assert episode.evaluation_payload.initial_pos[5] == -54.717


def test_coordinate_adapter_exports_initial_local_origin():
    trace = EpisodeTrace(
        samples=(
            sample(1.0, 2.0, 0.5, 90.0),
            sample(1.0, 3.0, 0.6, 120.0),
        ),
        action_count=1,
        coordinate_frames=("unrealcv_world_m",),
    )

    poses = UavFlowCoordinateAdapter().trace_poses(
        trace, make_case().initial_pos
    )

    assert poses[0]["x"] == 0.0
    assert poses[0]["y"] == 0.0
    assert poses[0]["z"] == 0.0
    assert poses[1]["x"] == 1.0
    assert poses[1]["z"] == 0.1
    assert poses[1]["yaw_deg"] == 30.0


def test_coordinate_adapter_fails_on_wrong_runtime_frame():
    trace = EpisodeTrace(
        samples=(sample(0.0, 0.0, 0.0, 0.0, frame="airsim_ned"),),
        action_count=0,
        coordinate_frames=("airsim_ned",),
    )

    with pytest.raises(ValueError, match="does not match"):
        UavFlowCoordinateAdapter().trace_poses(trace, make_case().initial_pos)


def test_stmr_profile_requires_segmentation_before_scene_launch():
    with pytest.raises(ValueError, match="segmentation"):
        validate_uav_flow_profile(
            ObservationProfile(), stmr_mode="shadow"
        )
    validate_uav_flow_profile(
        ObservationProfile.from_dict(
            {
                "modalities": [
                    "rgb",
                    "depth",
                    "segmentation",
                    "pose",
                ]
            }
        ),
        stmr_mode="planner",
    )


def test_turn_metric_suite_reports_real_pose_and_yaw_diagnostics(tmp_path):
    case = UavFlowCase(
        case_id="turn",
        file_name="turn.json",
        source_path="turn.json",
        class_name="Turn",
        instruction="Turn",
        instruction_unified="Turn",
        selected_instruction="Turn",
        initial_pos=[100.0, 200.0, 50.0, 0.0, 90.0, 0.0],
        end_pos=[100.0, 200.0, 50.0, 0.0, 120.0, 0.0],
        reference_path_preprocessed=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 30.0, 0.0],
        ],
    )
    trace = EpisodeTrace(
        samples=(
            sample(1.0, 2.0, 0.5, 90.0),
            sample(
                1.06,
                2.0,
                0.5,
                115.0,
                tool="uav.turn_relative",
                action_id="act_turn",
                action_args={"yaw_deg": 25.0},
                post_yaw=115.4,
            ),
        ),
        action_count=1,
        coordinate_frames=("unrealcv_world_m",),
        physical_windows=(
            physical_window(
                1,
                pose(1.0, 2.0, 0.5, 90.0),
                pose(1.06, 2.0, 0.5, 115.4),
                [("uav.turn_relative", {"yaw_deg": 25.0})],
            ),
        ),
        physical_call_count=1,
        physical_window_count=1,
        motion_call_count=1,
        trace_schema_version="physical_window_v2",
        trace_integrity="ok",
    )
    payload = UavFlowEvaluationPayload(
        case_id=case.case_id,
        class_name=case.class_name,
        initial_pos=tuple(case.initial_pos),
        end_pos=tuple(case.end_pos),
        target_pos=None,
        reference_path_raw=(),
        reference_path_preprocessed=tuple(
            tuple(row) for row in case.reference_path_preprocessed
        ),
    )

    result = UavFlowMetricSuite(case, tmp_path).evaluate(
        payload,
        trace,
        BenchmarkExecutionOutcome(completed=True),
    )

    diagnostics = result["turn_diagnostics"]
    assert diagnostics["reference_yaw_delta_deg"] == 30.0
    assert diagnostics["actual_yaw_delta_deg"] == 25.0
    assert diagnostics["final_yaw_error_deg"] == 5.0
    assert diagnostics["translation_drift_m"] == pytest.approx(0.06)
    assert diagnostics["unexpected_translation"] is True
    assert diagnostics["initial_position_error_m"] == 0.0
    assert diagnostics["initial_yaw_error_deg"] == 0.0
    assert diagnostics["initial_pose_within_tolerance"] is True
    tracking = diagnostics["backend_turn_tracking"]
    assert tracking["turn_window_count"] == 1
    assert tracking["observed_count"] == 1
    assert tracking["max_abs_error_deg"] == pytest.approx(0.4)
    assert tracking["all_observed_within_tolerance"] is True


def test_turn_tracking_handles_signed_wrap_and_missing_observation(tmp_path):
    case = UavFlowCase(
        case_id="turn_tracking",
        file_name="turn_tracking.json",
        source_path="turn_tracking.json",
        class_name="Turn",
        instruction="Turn",
        instruction_unified="Turn",
        selected_instruction="Turn",
        initial_pos=[0.0, 0.0, 0.0, 0.0, 170.0, 0.0],
        end_pos=[0.0, 0.0, 0.0, 0.0, -175.0, 0.0],
        reference_path_preprocessed=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 15.0, 0.0],
        ],
    )
    trace = EpisodeTrace(
        samples=(
            sample(0.0, 0.0, 0.0, 170.0),
            sample(
                0.0,
                0.0,
                0.0,
                -160.0,
                tool="uav.turn_relative",
                action_id="act_positive",
                action_args={"yaw_deg": 30.0},
                post_yaw=-159.7,
            ),
            sample(
                0.0,
                0.0,
                0.0,
                -175.0,
                tool="uav.turn_relative",
                action_id="act_negative",
                action_args={"yaw_deg": -15.0},
                post_yaw=-174.9,
            ),
            sample(
                0.0,
                0.0,
                0.0,
                -164.9,
                tool="uav.turn_relative",
                action_id="act_missing",
                action_args={"yaw_deg": 10.0},
            ),
        ),
        action_count=3,
        coordinate_frames=("unrealcv_world_m",),
        physical_windows=(
            physical_window(
                1,
                pose(0.0, 0.0, 0.0, 170.0),
                pose(0.0, 0.0, 0.0, -159.7),
                [("uav.turn_relative", {"yaw_deg": 30.0})],
            ),
            physical_window(
                2,
                pose(0.0, 0.0, 0.0, -159.7),
                pose(0.0, 0.0, 0.0, -174.9),
                [("uav.turn_relative", {"yaw_deg": -15.0})],
            ),
            physical_window(
                3,
                pose(0.0, 0.0, 0.0, -174.9),
                None,
                [("uav.turn_relative", {"yaw_deg": 10.0})],
                unavailable_reason="post_window_observation_unavailable",
            ),
        ),
        physical_call_count=3,
        physical_window_count=3,
        motion_call_count=3,
        trace_schema_version="physical_window_v2",
        trace_integrity="ok",
    )
    payload = UavFlowEvaluationPayload(
        case_id=case.case_id,
        class_name=case.class_name,
        initial_pos=tuple(case.initial_pos),
        end_pos=tuple(case.end_pos),
        target_pos=None,
        reference_path_raw=(),
        reference_path_preprocessed=tuple(
            tuple(row) for row in case.reference_path_preprocessed
        ),
    )

    result = UavFlowMetricSuite(case, tmp_path).evaluate(
        payload,
        trace,
        BenchmarkExecutionOutcome(completed=False),
    )

    tracking = result["turn_diagnostics"]["backend_turn_tracking"]
    assert tracking["turn_window_count"] == 3
    assert tracking["observed_count"] == 2
    assert tracking["unavailable_count"] == 1
    assert tracking["mean_abs_error_deg"] == pytest.approx(0.25)
    assert tracking["max_abs_error_deg"] == pytest.approx(0.3)
    assert tracking["samples"][0]["observed_yaw_delta_deg"] == (
        pytest.approx(30.3)
    )
    assert tracking["samples"][1]["observed_yaw_delta_deg"] == (
        pytest.approx(-15.2)
    )
    assert tracking["samples"][2]["unavailable_reason"] == (
        "post_window_observation_unavailable"
    )


def test_backend_action_tracking_reports_translation_and_zero_action(tmp_path):
    case = make_case()
    payload = UavFlowEvaluationPayload(
        case_id=case.case_id,
        class_name=case.class_name,
        initial_pos=tuple(case.initial_pos),
        end_pos=(),
        target_pos=tuple(case.target_pos or ()),
        reference_path_raw=(),
        reference_path_preprocessed=tuple(
            tuple(row) for row in case.reference_path_preprocessed
        ),
    )
    moving_trace = EpisodeTrace(
        samples=(
            sample(1.0, 2.0, 0.5, 90.0),
            sample(
                1.0,
                4.0,
                0.5,
                90.0,
                tool="uav.move_body",
                action_id="move",
                action_args={
                    "vx": 1.0,
                    "vy": 0.0,
                    "vz": 0.0,
                    "duration": 2.0,
                },
                post_yaw=90.0,
                post_xyz=(1.0, 4.0, 0.5),
            ),
        ),
        action_count=1,
        coordinate_frames=("unrealcv_world_m",),
        physical_windows=(
            physical_window(
                1,
                pose(1.0, 2.0, 0.5, 90.0),
                pose(1.0, 4.0, 0.5, 90.0),
                [
                    (
                        "uav.move_body",
                        {
                            "vx": 1.0,
                            "vy": 0.0,
                            "vz": 0.0,
                            "duration": 2.0,
                        },
                    )
                ],
            ),
        ),
        physical_call_count=1,
        physical_window_count=1,
        motion_call_count=1,
        trace_schema_version="physical_window_v2",
        trace_integrity="ok",
    )

    result = UavFlowMetricSuite(case, tmp_path).evaluate(
        payload,
        moving_trace,
        BenchmarkExecutionOutcome(completed=True),
    )

    tracking = result["backend_action_tracking"]
    assert tracking["zero_motion_completion"] is False
    assert tracking["max_position_error_m"] == pytest.approx(0.0)
    assert tracking["all_observed_within_tolerance"] is True
    assert tracking["windows"][0]["expected_delta_m"] == pytest.approx(
        {"dx": 0.0, "dy": 2.0, "dz": 0.0}
    )
    assert tracking["windows"][0]["observed_delta_m"] == pytest.approx(
        {"dx": 0.0, "dy": 2.0, "dz": 0.0}
    )

    zero_result = UavFlowMetricSuite(case, tmp_path / "zero").evaluate(
        payload,
        EpisodeTrace(
            samples=(sample(1.0, 2.0, 0.5, 90.0),),
            action_count=0,
            coordinate_frames=("unrealcv_world_m",),
            trace_schema_version="physical_window_v2",
            trace_integrity="ok",
        ),
        BenchmarkExecutionOutcome(completed=True),
    )
    assert (
        zero_result["backend_action_tracking"]["zero_motion_completion"]
        is True
    )
    assert (
        zero_result["backend_action_tracking"][
            "all_observed_within_tolerance"
        ]
        is None
    )


def test_window_tracking_composes_turn_then_move_in_one_batch(tmp_path):
    before = pose(0.0, 0.0, 1.0, 0.0)
    after = pose(0.984808, 0.173648, 1.0, 10.0)
    window = physical_window(
        1,
        before,
        after,
        [
            ("uav.turn_relative", {"yaw_deg": 10.0}),
            (
                "uav.move_body",
                {"vx": 1.0, "vy": 0.0, "vz": 0.0, "duration": 1.0},
            ),
        ],
    )
    trace = EpisodeTrace(
        samples=(sample(0.0, 0.0, 1.0, 0.0),),
        action_count=2,
        coordinate_frames=("unrealcv_world_m",),
        physical_windows=(window,),
        physical_call_count=2,
        physical_window_count=1,
        motion_call_count=2,
        trace_schema_version="physical_window_v2",
        trace_integrity="ok",
    )

    tracking = metric_result(tmp_path, trace)["backend_action_tracking"]

    assert tracking["tracked_window_count"] == 1
    assert tracking["windows"][0]["expected_delta_m"] == pytest.approx(
        {"dx": 0.984807753, "dy": 0.173648178, "dz": 0.0}
    )
    assert tracking["windows"][0]["expected_yaw_delta_deg"] == 10.0
    assert tracking["all_observed_within_tolerance"] is True


def test_window_tracking_uses_each_window_boundary_after_hover(tmp_path):
    hover = physical_window(
        1,
        pose(0.0, 0.0, 1.0, 0.0),
        pose(0.0, 0.0, 1.0, 0.0),
        [("uav.hover", {})],
    )
    move = physical_window(
        2,
        pose(0.0, 0.0, 1.0, 0.0),
        pose(2.0, 0.0, 1.0, 0.0),
        [
            (
                "uav.move_body",
                {"vx": 1.0, "vy": 0.0, "vz": 0.0, "duration": 2.0},
            )
        ],
    )
    trace = EpisodeTrace(
        samples=(sample(0.0, 0.0, 1.0, 0.0),),
        action_count=2,
        coordinate_frames=("unrealcv_world_m",),
        physical_windows=(hover, move),
        physical_call_count=2,
        physical_window_count=2,
        motion_call_count=1,
        trace_schema_version="physical_window_v2",
        trace_integrity="ok",
    )

    tracking = metric_result(tmp_path, trace)["backend_action_tracking"]

    assert [item["position_tracking_error_m"] for item in tracking["windows"]] == [
        pytest.approx(0.0),
        pytest.approx(0.0),
    ]
    assert tracking["zero_motion_completion"] is False


def test_window_tracking_regression_depth_calls_cannot_consume_move_observation(
    tmp_path,
):
    before = pose(-111.01492, 1.85048, 1.86642, -105.257)
    after = pose(-111.53660, -0.08028, 1.86642, -104.957)
    move = physical_window(
        1,
        before,
        after,
        [
            (
                "uav.move_body",
                {"vx": 1.0, "vy": 0.0, "vz": 0.0, "duration": 2.0},
            )
        ],
    )
    trace = EpisodeTrace(
        samples=(
            sample(
                before["x"],
                before["y"],
                before["z"],
                before["yaw_deg"],
            ),
        ),
        action_count=1,
        coordinate_frames=("unrealcv_world_m",),
        physical_windows=(move,),
        physical_call_count=1,
        physical_window_count=1,
        motion_call_count=1,
        trace_schema_version="physical_window_v2",
        trace_integrity="ok",
    )

    tracking = metric_result(tmp_path, trace)["backend_action_tracking"]
    row = tracking["windows"][0]

    assert row["observed_delta_m"] == pytest.approx(
        {"dx": -0.52168, "dy": -1.93076, "dz": 0.0}
    )
    assert row["position_tracking_error_m"] < 0.01
    assert row["position_tracking_error_m"] != pytest.approx(2.0)


def test_window_tracking_uses_safety_executed_args(tmp_path):
    step = EpisodePhysicalStep(
            step_index=0,
            tool_call_id="call_move",
            requested_tool="uav.move_body",
            tool="uav.move_body",
        effect="physical",
        status="ok",
        action_ref="memory/actions.jsonl#1",
        result_pose_ref="memory/poses.jsonl#1",
        action_id="act_0001",
        requested_args={"vx": 1.0, "vy": 0.0, "vz": 0.0, "duration": 2.0},
        executed_args={"vx": 1.0, "vy": 0.0, "vz": 0.0, "duration": 1.0},
        safety_interventions=({"field": "duration", "to": 1.0},),
    )
    window = EpisodePhysicalWindow(
        decision_id="decision_0001",
        phase="test",
        phase_key="1:test",
        window_index=1,
        status="ok",
        observation_before_id="obs_0001",
        observation_after_id="obs_0002",
        observation_before_pose=pose(0.0, 0.0, 1.0, 0.0),
        observation_after_pose=pose(1.0, 0.0, 1.0, 0.0),
        observation_after_timestamp="2026-01-01T00:00:01+00:00",
        ordered_steps=(step,),
    )
    trace = EpisodeTrace(
        samples=(sample(0.0, 0.0, 1.0, 0.0),),
        action_count=1,
        coordinate_frames=("unrealcv_world_m",),
        physical_windows=(window,),
        physical_call_count=1,
        physical_window_count=1,
        motion_call_count=1,
        trace_schema_version="physical_window_v2",
        trace_integrity="ok",
    )

    row = metric_result(tmp_path, trace)["backend_action_tracking"]["windows"][0]

    assert row["expected_delta_m"] == pytest.approx(
        {"dx": 1.0, "dy": 0.0, "dz": 0.0}
    )
    assert row["position_tracking_error_m"] == pytest.approx(0.0)


def metric_result(tmp_path, trace: EpisodeTrace) -> dict:
    case = make_case()
    payload = UavFlowEvaluationPayload(
        case_id=case.case_id,
        class_name=case.class_name,
        initial_pos=tuple(case.initial_pos),
        end_pos=(),
        target_pos=tuple(case.target_pos or ()),
        reference_path_raw=(),
        reference_path_preprocessed=tuple(
            tuple(row) for row in case.reference_path_preprocessed
        ),
    )
    return UavFlowMetricSuite(case, tmp_path).evaluate(
        payload,
        trace,
        BenchmarkExecutionOutcome(completed=True),
    )


def make_case() -> UavFlowCase:
    return UavFlowCase(
        case_id="case",
        file_name="case.json",
        source_path="case.json",
        class_name="Move",
        instruction="Move",
        instruction_unified="Move",
        selected_instruction="Move",
        initial_pos=[100.0, 200.0, 50.0, 0.0, 90.0, 0.0],
        target_pos=[300.0, 400.0, 50.0],
        reference_path_preprocessed=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ],
    )


def pose(x: float, y: float, z: float, yaw: float) -> dict:
    return {
        "x": x,
        "y": y,
        "z": z,
        "yaw_deg": yaw,
        "coordinate_frame": "unrealcv_world_m",
    }


def physical_window(
    index: int,
    before: dict,
    after: dict | None,
    steps: list[tuple[str, dict]],
    *,
    unavailable_reason: str | None = None,
) -> EpisodePhysicalWindow:
    physical_steps = tuple(
        EpisodePhysicalStep(
            step_index=step_index,
            tool_call_id=f"call_{index}_{step_index}",
            requested_tool=tool,
            tool=tool,
            effect="physical",
            status="ok",
            action_ref=f"memory/actions.jsonl#{index + step_index}",
            result_pose_ref=None,
            action_id=f"act_{index}_{step_index}",
            requested_args=dict(args),
            executed_args=dict(args),
        )
        for step_index, (tool, args) in enumerate(steps)
    )
    return EpisodePhysicalWindow(
        decision_id=f"decision_{index:04d}",
        phase="test",
        phase_key="1:test",
        window_index=index,
        status="ok",
        observation_before_id=f"obs_{index:04d}",
        observation_after_id=(f"obs_{index + 1:04d}" if after else None),
        observation_before_pose=dict(before),
        observation_after_pose=(dict(after) if after else None),
        observation_after_timestamp="2026-01-01T00:00:00+00:00" if after else None,
        ordered_steps=physical_steps,
        unavailable_reason=unavailable_reason,
    )


def sample(
    x: float,
    y: float,
    z: float,
    yaw: float,
    *,
    frame: str = "unrealcv_world_m",
    tool: str | None = None,
    action_id: str | None = None,
    action_args: dict | None = None,
    post_yaw: float | None = None,
    post_xyz: tuple[float, float, float] | None = None,
) -> EpisodeTraceSample:
    return EpisodeTraceSample(
        pose={
            "x": x,
            "y": y,
            "z": z,
            "yaw_deg": yaw,
            "coordinate_frame": frame,
        },
        source="task_initial" if tool is None else "action_result",
        tool=tool,
        action_id=action_id,
        timestamp=None,
        coordinate_frame=frame,
        action_args=dict(action_args or {}),
        post_action_observation_pose=(
            {
                "x": post_xyz[0] if post_xyz is not None else x,
                "y": post_xyz[1] if post_xyz is not None else y,
                "z": post_xyz[2] if post_xyz is not None else z,
                "yaw_deg": post_yaw,
                "coordinate_frame": frame,
            }
            if post_yaw is not None
            else None
        ),
        post_action_observation_timestamp=(
            "2026-01-01T00:00:00+00:00"
            if post_yaw is not None
            else None
        ),
    )
