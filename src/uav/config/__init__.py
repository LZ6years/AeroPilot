"""UAV-domain action scales and safety limits.

These value objects are shared by Agent, Tools, simulator adapters and Runtime,
so they live below those layers instead of making them depend on Runtime.
"""

from src.uav.config.action import UavActionConfig, load_uav_action_config
from src.uav.config.safety import UavSafetyConfig, load_uav_safety_config

__all__ = [
    "UavActionConfig",
    "UavSafetyConfig",
    "load_uav_action_config",
    "load_uav_safety_config",
]
