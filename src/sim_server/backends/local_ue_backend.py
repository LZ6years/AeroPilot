"""Provide the local UE backend placeholder for Windows or script launches."""

from __future__ import annotations

from pathlib import Path

from src.runtime.action_config import UavActionConfig

from .airvln_backend import AirVlnBackend


class LocalUeBackend(AirVlnBackend):
    """Local UE backend.

    It currently shares the primitive AirSim/UAV control skeleton with the
    AirVLN-style backend. Process launch policy is owned by SimulationServer.
    """

    def __init__(
        self,
        use_real_uav: bool = False,
        action_config: UavActionConfig | None = None,
        artifact_dir: str | Path | None = None,
        save_artifacts: bool = False,
    ) -> None:
        # Share the AirSim primitive backend while keeping the local-UE backend name.
        super().__init__(
            use_real_uav=use_real_uav,
            action_config=action_config,
            artifact_dir=artifact_dir,
            save_artifacts=save_artifacts,
        )
