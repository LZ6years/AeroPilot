"""Lock the public EnvClient v1 surface used by Benchmark integrations.

These tests are intentionally stricter than ordinary delegation tests.  A
failure means a public integration contract changed and therefore requires an
explicit API-version decision, compatibility migration, and documentation
update rather than a silent test adjustment.
"""

from __future__ import annotations

from dataclasses import fields
from inspect import Parameter, signature

from src.env_client import (
    ENV_CLIENT_API_VERSION,
    ENV_CLIENT_PUBLIC_METHODS,
    ActionResult,
    EnvClient,
    Observation,
    ObservationProfile,
    SceneCapabilities,
    SceneInstance,
    SimulatorEnvClient,
    load_observation_profile,
)
from src.env_client import __all__ as public_exports


EXPECTED_METHOD_PARAMETERS = {
    "start_scene": (("self", False, None), ("map_id", False, None), ("scene_id", True, None)),
    "get_or_start_scene": (
        ("self", False, None),
        ("map_id", False, None),
        ("scene_id", True, None),
    ),
    "stop_scene": (("self", False, None), ("scene_instance_id", False, None)),
    "reset_scene": (("self", False, None), ("scene_instance_id", False, None)),
    "health_check": (("self", False, None),),
    "get_observation": (
        ("self", False, None),
        ("scene_instance_id", False, None),
        ("agent_id", False, None),
        ("modalities", False, None),
    ),
    "execute_action": (
        ("self", False, None),
        ("scene_instance_id", False, None),
        ("agent_id", False, None),
        ("action", False, None),
        ("args", False, None),
    ),
}


def _parameter_contract(callable_object) -> tuple[tuple[str, bool, object], ...]:
    """Return the compatibility-relevant part of a Python call signature."""

    result = []
    for parameter in signature(callable_object).parameters.values():
        assert parameter.kind is Parameter.POSITIONAL_OR_KEYWORD
        has_default = parameter.default is not Parameter.empty
        default = parameter.default if has_default else None
        result.append((parameter.name, has_default, default))
    return tuple(result)


def test_public_package_exports_are_frozen_to_env_client_v1():
    """Benchmark code gets one deliberate import surface, not internals."""

    assert ENV_CLIENT_API_VERSION == "1.0"
    assert SimulatorEnvClient.API_VERSION == ENV_CLIENT_API_VERSION
    assert tuple(public_exports) == (
        "ENV_CLIENT_API_VERSION",
        "ENV_CLIENT_PUBLIC_METHODS",
        "EnvClient",
        "SimulatorEnvClient",
        "ObservationProfile",
        "load_observation_profile",
        "SceneInstance",
        "SceneCapabilities",
        "Observation",
        "ActionResult",
    )


def test_protocol_and_concrete_client_keep_exact_v1_method_signatures():
    """Prevent accidental argument additions, removals, or default changes."""

    assert ENV_CLIENT_PUBLIC_METHODS == tuple(EXPECTED_METHOD_PARAMETERS)
    for method_name, expected in EXPECTED_METHOD_PARAMETERS.items():
        assert _parameter_contract(getattr(EnvClient, method_name)) == expected
        assert (
            _parameter_contract(getattr(SimulatorEnvClient, method_name))
            == expected
        )

    assert _parameter_contract(SimulatorEnvClient.__init__) == (
        ("self", False, None),
        ("service", False, None),
        ("observation_profile", True, None),
    )


def test_simulator_client_structurally_implements_env_client_protocol():
    """The public contract remains structural rather than inheritance-based."""

    class NoIoService:
        pass

    assert isinstance(SimulatorEnvClient(NoIoService()), EnvClient)


def test_v1_response_schema_field_names_are_frozen():
    """Protect fields consumed by Runtime and future Benchmark adapters."""

    assert tuple(field.name for field in fields(SceneCapabilities)) == (
        "protocol",
        "control_mode",
        "supported_actions",
        "observation_modalities",
        "requires_takeoff",
        "supports_soft_reset",
    )
    assert tuple(field.name for field in fields(SceneInstance)) == (
        "scene_instance_id",
        "map_id",
        "scene_id",
        "machine_id",
        "control_host",
        "control_port",
        "settings_path",
        "process_id",
        "status",
        "process_ownership",
        "remote_process_id",
        "settings_sha256",
        "capabilities",
    )
    assert tuple(field.name for field in fields(Observation)) == (
        "scene_instance_id",
        "agent_id",
        "modalities",
        "data",
    )
    assert tuple(field.name for field in fields(ActionResult)) == (
        "ok",
        "action",
        "message",
        "data",
    )
