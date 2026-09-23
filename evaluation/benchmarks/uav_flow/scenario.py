"""Benchmark-only UAV-Flow object placement over a privileged connection."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable

from evaluation.benchmarks.uav_flow.adapter import UavFlowScenarioPayload
from src.env_client import SceneInstance


PERSON_NAME = "BP_Character_21"
VEHICLE_NAME = "BP_Character_22"
HIDDEN_LOCATION = (0.0, 0.0, -1000.0)
MASK_COLORS = {
    PERSON_NAME: (10, 20, 30),
    VEHICLE_NAME: (40, 50, 60),
}


@dataclass
class UavFlowScenarioController:
    """Place benchmark targets without owning flight or camera capture."""

    connection_factory: Callable[[str, int], Any] | None = None

    def __post_init__(self) -> None:
        self.label_maps: dict[str, dict[str, Any]] = {}
        self.scenario_diagnostics: dict[str, dict[str, Any]] = {}

    def prepare_episode(
        self,
        scene: SceneInstance,
        payload: UavFlowScenarioPayload | None,
    ) -> None:
        if payload is None:
            return
        connection = self._connect(scene)
        try:
            objects = _object_names(connection)
            if PERSON_NAME not in objects:
                _request_many(
                    connection,
                    [
                        f"vset /objects/spawn bp_character_C {PERSON_NAME}",
                        f"vset /object/{PERSON_NAME}/location 0 0 0",
                        f"vset /object/{PERSON_NAME}/rotation 0 0 0",
                        f"vbp {PERSON_NAME} set_phy 0",
                    ],
                )
            if VEHICLE_NAME not in objects:
                _request_many(
                    connection,
                    [
                        f"vset /objects/spawn BP_BaseCar_C {VEHICLE_NAME}",
                        f"vset /object/{VEHICLE_NAME}/location 1000 0 0",
                        f"vset /object/{VEHICLE_NAME}/rotation 0 0 0",
                        f"vbp {VEHICLE_NAME} set_phy 0",
                    ],
                )
            _set_appearance(connection, PERSON_NAME, 0)
            _set_appearance(connection, VEHICLE_NAME, 2)
            _request(connection, f"vbp {VEHICLE_NAME} set_phy 0")
            for name, color in MASK_COLORS.items():
                _request(
                    connection,
                    f"vset /object/{name}/color "
                    f"{color[0]} {color[1]} {color[2]}",
                )
            self._place(connection, payload)
            self.label_maps[payload.case_id] = _mask_label_map(connection)
            self.scenario_diagnostics[payload.case_id] = (
                _scenario_diagnostics(connection, payload)
            )
        except Exception as exc:
            raise RuntimeError(
                f"UAV-Flow scenario prepare failed: {exc}"
            ) from exc
        finally:
            _disconnect(connection)

    def cleanup_episode(
        self,
        scene: SceneInstance,
        payload: UavFlowScenarioPayload | None,
    ) -> None:
        if payload is None:
            return
        connection = self._connect(scene)
        try:
            _set_location(connection, PERSON_NAME, HIDDEN_LOCATION)
            _set_location(connection, VEHICLE_NAME, HIDDEN_LOCATION)
        finally:
            _disconnect(connection)

    def label_map_for(self, case_id: str) -> dict[str, Any]:
        return dict(self.label_maps.get(case_id, {}))

    def diagnostics_for(self, case_id: str) -> dict[str, Any]:
        """Return evaluation-only scene contract evidence for one episode."""

        return dict(self.scenario_diagnostics.get(case_id, {}))

    def _connect(self, scene: SceneInstance) -> Any:
        if self.connection_factory is not None:
            return self.connection_factory(
                scene.control_host, scene.control_port
            )
        from unrealcv import Client

        client = Client((scene.control_host, int(scene.control_port)))
        client.connect()
        if not client.isconnected():
            raise ConnectionError(
                f"Could not open privileged UnrealCV connection at "
                f"{scene.control_host}:{scene.control_port}"
            )
        return client

    @staticmethod
    def _place(connection: Any, payload: UavFlowScenarioPayload) -> None:
        target = payload.target_pos
        if not target or payload.use_obj not in {1, 2}:
            _set_location(connection, PERSON_NAME, HIDDEN_LOCATION)
            _set_location(connection, VEHICLE_NAME, HIDDEN_LOCATION)
            return
        rotation = target[3:6] if len(target) >= 6 else (0.0, 0.0, 0.0)
        if payload.use_obj == 1:
            if payload.obj_id is not None:
                _set_appearance(connection, PERSON_NAME, payload.obj_id)
            _set_location(connection, PERSON_NAME, target[:3])
            _set_rotation(connection, PERSON_NAME, rotation)
            _set_location(connection, VEHICLE_NAME, HIDDEN_LOCATION)
            _set_location(connection, PERSON_NAME, target[:3])
            return
        _set_appearance(connection, VEHICLE_NAME, 2)
        vehicle_location = (target[0], target[1], 0.0)
        _set_location(connection, VEHICLE_NAME, vehicle_location)
        _set_rotation(connection, VEHICLE_NAME, rotation)
        _request(connection, f"vbp {VEHICLE_NAME} set_phy 0")
        _set_location(connection, PERSON_NAME, HIDDEN_LOCATION)
        _set_location(connection, VEHICLE_NAME, vehicle_location)


def _object_names(connection: Any) -> set[str]:
    response = _request(connection, "vget /objects")
    if isinstance(response, bytes):
        response = response.decode("utf-8", errors="replace")
    return {item for item in str(response or "").split() if item}


def _mask_label_map(connection: Any) -> dict[str, Any]:
    colors: dict[str, dict[str, Any]] = {}
    unmapped: list[dict[str, Any]] = []
    classified = [
        (name, _semantic_label_for_object_name(name))
        for name in (PERSON_NAME, VEHICLE_NAME)
    ]
    responses = _request_many(
        connection,
        [f"vget /object/{name}/color" for name, _ in classified],
    )
    if not isinstance(responses, (list, tuple)):
        responses = [responses]
    for index, (name, semantic) in enumerate(classified):
        if semantic is None:
            continue
        response = responses[index] if index < len(responses) else None
        color = _parse_color(response)
        if color is None:
            unmapped.append({"object_name": name, "reason": "invalid_color"})
            continue
        colors[",".join(str(value) for value in color)] = {
            "object_name": name,
            **semantic,
        }
    return {
        "colors": colors,
        "unmapped_objects": unmapped,
        "object_count": len(classified),
        "source": "uav_flow_controlled_objects",
    }


def _semantic_label_for_object_name(name: str) -> dict[str, Any] | None:
    normalized = name.lower()
    if name == VEHICLE_NAME:
        return {"id": 2, "label": "vehicle", "category": "vehicle"}
    if name == PERSON_NAME:
        return {"id": 1, "label": "person", "category": "person"}
    categories = (
        (1, "person", "person", ("character", "person", "human", "pedestrian")),
        (2, "vehicle", "vehicle", ("car", "vehicle", "truck", "bus")),
        (3, "tree", "vegetation", ("tree", "plant", "bush")),
        (4, "building", "structure", ("building", "house")),
    )
    for semantic_id, label, category, keywords in categories:
        if any(keyword in normalized for keyword in keywords):
            return {
                "id": semantic_id,
                "label": label,
                "category": category,
            }
    return None


def _parse_color(value: Any) -> tuple[int, int, int] | None:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    parts = re.findall(r"-?\d+(?:\.\d+)?", str(value or ""))
    if len(parts) < 3:
        return None
    try:
        return tuple(int(float(item)) for item in parts[:3])
    except (TypeError, ValueError):
        return None


def _set_location(connection: Any, name: str, values: Any) -> None:
    x, y, z = values
    _request(connection, f"vset /object/{name}/location {x} {y} {z}")


def _set_rotation(connection: Any, name: str, values: Any) -> None:
    # UAV-Flow's Character_API accepts [roll, yaw, pitch], while the raw
    # UnrealCV command is ordered [pitch, yaw, roll].
    roll, yaw, pitch = values
    _request(connection, f"vset /object/{name}/rotation {pitch} {yaw} {roll}")


def _set_appearance(
    connection: Any,
    name: str,
    appearance_id: int,
) -> None:
    """Use the Character Blueprint appearance contract from upstream UAV-Flow."""

    _request(connection, f"vbp {name} set_app {int(appearance_id)}")


def _scenario_diagnostics(
    connection: Any,
    payload: UavFlowScenarioPayload,
) -> dict[str, Any]:
    """Read back controlled-object pose after final benchmark placement."""

    target = payload.target_pos
    if not target or payload.use_obj not in {1, 2}:
        return {
            "contract": {
                "appearance_method": "set_app",
                "rotation_input_order": "roll_yaw_pitch",
                "native_rotation_order": "pitch_yaw_roll",
            },
            "target": None,
        }

    object_name = PERSON_NAME if payload.use_obj == 1 else VEHICLE_NAME
    expected_location = (
        tuple(float(value) for value in target[:3])
        if payload.use_obj == 1
        else (float(target[0]), float(target[1]), 0.0)
    )
    expected_rotation = tuple(
        float(value) for value in (
            target[3:6] if len(target) >= 6 else (0.0, 0.0, 0.0)
        )
    )
    responses = _request_many(
        connection,
        [
            f"vget /object/{object_name}/location",
            f"vget /object/{object_name}/rotation",
        ],
    )
    if not isinstance(responses, (list, tuple)):
        responses = [responses]
    if len(responses) < 2:
        raise RuntimeError(
            f"UAV-Flow target pose readback incomplete for {object_name}"
        )
    observed_location = _parse_vector(
        responses[0], label=f"{object_name} location"
    )
    observed_rotation = _parse_vector(
        responses[1], label=f"{object_name} rotation"
    )
    return {
        "contract": {
            "appearance_method": "set_app",
            "rotation_input_order": "roll_yaw_pitch",
            "native_rotation_order": "pitch_yaw_roll",
        },
        "target": {
            "object_name": object_name,
            "appearance_id": (
                payload.obj_id if payload.use_obj == 1 else 2
            ),
            "expected_location": list(expected_location),
            "observed_location": list(observed_location),
            "expected_rotation_roll_yaw_pitch": list(expected_rotation),
            "observed_rotation_roll_yaw_pitch": list(observed_rotation),
        },
    }


def _parse_vector(value: Any, *, label: str) -> tuple[float, float, float]:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    parts = re.findall(r"-?\d+(?:\.\d+)?", str(value or ""))
    if len(parts) < 3:
        raise RuntimeError(f"Invalid UnrealCV {label} readback: {value!r}")
    try:
        return tuple(float(item) for item in parts[:3])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid UnrealCV {label} readback: {value!r}"
        ) from exc


def _request_many(connection: Any, commands: list[str]) -> Any:
    # Use synchronous requests: unrealcv.Client interprets timeout=-1 as
    # fire-and-forget and therefore cannot return object inventories/colors.
    try:
        return connection.request(commands)
    except TypeError:
        return [_request(connection, command) for command in commands]


def _request(connection: Any, command: str) -> Any:
    # Reads and placement writes must complete before Runtime starts using the
    # scene through its own UnrealCV connection.
    try:
        return connection.request(command)
    except TypeError:
        return connection.request(command, 5)


def _disconnect(connection: Any) -> None:
    disconnect = getattr(connection, "disconnect", None)
    if callable(disconnect):
        disconnect()
