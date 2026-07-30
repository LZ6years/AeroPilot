"""Stable public entry point for EnvClient API v1.

Benchmark and Runtime integrations should import the symbols listed in
``__all__`` from this package.  Modules such as ``sim_server``, concrete
Backends, adapters, process control, and SSH transport are implementation
details and are intentionally not re-exported here.
"""

from .base import (
    ENV_CLIENT_API_VERSION,
    ENV_CLIENT_PUBLIC_METHODS,
    EnvClient,
)
from .observation_profile import (
    ObservationProfile,
    load_observation_profile,
)
from .schemas import (
    ActionResult,
    Observation,
    SceneCapabilities,
    SceneInstance,
)
from .simulator_client import SimulatorEnvClient

__all__ = [
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
]
