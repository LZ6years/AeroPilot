"""Provide the local UE backend placeholder for Windows or script launches."""

from __future__ import annotations

from .airvln_backend import AirVlnBackend


class LocalUeBackend(AirVlnBackend):
    """Local UE backend.

    It currently shares the primitive AirSim/UAV control skeleton with the
    AirVLN-style backend. Process launch policy is owned by SimulationServer.
    """
