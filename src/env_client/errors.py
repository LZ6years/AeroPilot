"""Simulator-independent faults that require caller-owned scene teardown."""


class SceneRecoveryRequired(RuntimeError):
    """Control state cannot safely support another decision in this scene."""
