"""Tests for the benchmark-owned UAV-Flow ScenarioController."""

from __future__ import annotations

from evaluation.benchmarks.uav_flow.adapter import UavFlowScenarioPayload
from evaluation.benchmarks.uav_flow.scenario import (
    MASK_COLORS,
    PERSON_NAME,
    VEHICLE_NAME,
    UavFlowScenarioController,
)
from src.env_client.schemas import SceneInstance


def test_scenario_creates_objects_places_person_and_collects_labels():
    connection = FakeConnection()
    controller = UavFlowScenarioController(
        connection_factory=lambda host, port: connection
    )

    controller.prepare_episode(
        scene(),
        payload(use_obj=1, obj_id=19),
    )

    batch = [
        command
        for item in connection.requests
        if isinstance(item, list)
        for command in item
    ]
    assert f"vset /objects/spawn bp_character_C {PERSON_NAME}" in batch
    assert f"vset /objects/spawn BP_BaseCar_C {VEHICLE_NAME}" in batch
    assert f"vbp {PERSON_NAME} set_app 19" in connection.requests
    assert not any(
        "set_mixamoapp" in str(command)
        for command in connection.requests
    )
    for name, color in MASK_COLORS.items():
        assert (
            f"vset /object/{name}/color {color[0]} {color[1]} {color[2]}"
            in connection.requests
        )
    assert (
        f"vset /object/{PERSON_NAME}/location 100.0 200.0 300.0"
        in connection.requests
    )
    assert (
        f"vset /object/{VEHICLE_NAME}/location 0.0 0.0 -1000.0"
        in connection.requests
    )
    assert (
        f"vset /object/{PERSON_NAME}/rotation 3.0 2.0 1.0"
        in connection.requests
    )
    labels = controller.label_map_for("case")
    assert labels["colors"]["10,20,30"]["label"] == "person"
    assert labels["colors"]["40,50,60"]["label"] == "vehicle"
    assert all(timeout is None for timeout in connection.timeouts)
    assert connection.disconnect_count == 1
    diagnostics = controller.diagnostics_for("case")
    assert diagnostics["contract"]["appearance_method"] == "set_app"
    assert diagnostics["contract"]["rotation_input_order"] == (
        "roll_yaw_pitch"
    )
    assert diagnostics["target"]["appearance_id"] == 19
    assert diagnostics["target"]["observed_location"] == [
        100.0,
        200.0,
        300.0,
    ]
    assert diagnostics["target"]["observed_rotation_roll_yaw_pitch"] == [
        1.0,
        2.0,
        3.0,
    ]


def test_scenario_places_vehicle_and_hides_person():
    connection = FakeConnection(existing=True)
    controller = UavFlowScenarioController(
        connection_factory=lambda host, port: connection
    )

    controller.prepare_episode(scene(), payload(use_obj=2))

    assert not any(
        "/objects/spawn" in command
        for item in connection.requests
        if isinstance(item, list)
        for command in item
    )
    assert (
        f"vset /object/{VEHICLE_NAME}/location 100.0 200.0 0.0"
        in connection.requests
    )
    assert (
        f"vset /object/{PERSON_NAME}/location 0.0 0.0 -1000.0"
        in connection.requests
    )
    assert f"vbp {VEHICLE_NAME} set_phy 0" in connection.requests
    assert f"vbp {VEHICLE_NAME} set_app 2" in connection.requests
    assert controller.diagnostics_for("case")["target"][
        "observed_location"
    ] == [100.0, 200.0, 0.0]


def test_scenario_cleanup_hides_both_objects():
    connection = FakeConnection(existing=True)
    controller = UavFlowScenarioController(
        connection_factory=lambda host, port: connection
    )

    controller.cleanup_episode(
        scene(process_ownership="external"), payload(use_obj=1)
    )

    assert connection.requests == [
        f"vset /object/{PERSON_NAME}/location 0.0 0.0 -1000.0",
        f"vset /object/{VEHICLE_NAME}/location 0.0 0.0 -1000.0",
    ]
    assert connection.disconnect_count == 1


def test_managed_scenario_cleanup_relies_on_owned_process_shutdown():
    called = False

    def factory(host, port):
        nonlocal called
        called = True
        return FakeConnection(existing=True)

    controller = UavFlowScenarioController(connection_factory=factory)

    controller.cleanup_episode(
        scene(process_ownership="local"), payload(use_obj=1)
    )

    assert called is False


def test_disabled_dry_scenario_does_not_open_connection():
    called = False

    def factory(host, port):
        nonlocal called
        called = True
        return FakeConnection()

    controller = UavFlowScenarioController(
        enabled=False, connection_factory=factory
    )

    controller.prepare_episode(scene(), payload(use_obj=1))
    controller.cleanup_episode(scene(), payload(use_obj=1))

    assert called is False


def payload(
    *, use_obj: int, obj_id: int | None = None
) -> UavFlowScenarioPayload:
    return UavFlowScenarioPayload(
        case_id="case",
        use_obj=use_obj,
        obj_id=obj_id,
        target_pos=(100.0, 200.0, 300.0, 1.0, 2.0, 3.0),
    )


def scene(*, process_ownership: str = "local") -> SceneInstance:
    return SceneInstance(
        "scene",
        "map",
        "DowntownWest",
        "local",
        "127.0.0.1",
        9068,
        None,
        None,
        "running",
        process_ownership,
    )


class FakeConnection:
    def __init__(self, existing: bool = False) -> None:
        self.objects = (
            {PERSON_NAME, VEHICLE_NAME} if existing else set()
        )
        self.requests: list[str | list[str]] = []
        self.timeouts: list[int | None] = []
        self.disconnect_count = 0
        self.locations = {
            PERSON_NAME: [0.0, 0.0, 0.0],
            VEHICLE_NAME: [1000.0, 0.0, 0.0],
        }
        self.rotations = {
            PERSON_NAME: [0.0, 0.0, 0.0],
            VEHICLE_NAME: [0.0, 0.0, 0.0],
        }

    def request(self, command, timeout=None):
        self.timeouts.append(timeout)
        if isinstance(command, list):
            self.requests.append(list(command))
            responses = []
            for item in command:
                if "/objects/spawn" in item:
                    self.objects.add(item.split()[-1])
                if item == f"vget /object/{PERSON_NAME}/color":
                    responses.append("10 20 30")
                elif item == f"vget /object/{VEHICLE_NAME}/color":
                    responses.append("40 50 60")
                elif item.endswith("/location"):
                    name = item.split("/")[2]
                    responses.append(
                        " ".join(str(value) for value in self.locations[name])
                    )
                elif item.endswith("/rotation"):
                    name = item.split("/")[2]
                    responses.append(
                        " ".join(str(value) for value in self.rotations[name])
                    )
                else:
                    responses.append("ok")
            return responses
        if command == "vget /objects":
            return " ".join(sorted(self.objects))
        if command == f"vget /object/{PERSON_NAME}/color":
            return "10 20 30"
        if command == f"vget /object/{VEHICLE_NAME}/color":
            return "40 50 60"
        if command.startswith("vset /object/") and "/location " in command:
            name = command.split("/")[2]
            self.locations[name] = [
                float(value) for value in command.split()[-3:]
            ]
        if command.startswith("vset /object/") and "/rotation " in command:
            name = command.split("/")[2]
            pitch, yaw, roll = [
                float(value) for value in command.split()[-3:]
            ]
            self.rotations[name] = [roll, yaw, pitch]
        self.requests.append(command)
        return "ok"

    def disconnect(self):
        self.disconnect_count += 1
