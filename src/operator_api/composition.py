"""Adapt operator configuration to the existing production RuntimeFactory graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agent.config import load_model_config
from src.agent.safety import SafetySupervisor
from src.env_client import load_observation_profile
from src.env_client.simulator_client import SimulatorEnvClient
from src.runtime.configs import TaskExecutionOptions
from src.runtime.configs.map_config import load_map_config
from src.runtime.factory import RuntimeFactory
from src.runtime.logging.async_run_logger import AsyncRunLogger
from src.runtime.runners.interactive_session import InteractiveTaskSession
from src.sim_server.server import SimulationServer
from src.sim_server.preview import PreviewConfig
from src.uav.config import load_uav_action_config, load_uav_safety_config

from .config import OperatorConfig
from .schemas import CreateSession


def compose_session(config: OperatorConfig, request: CreateSession, run_dir: Path, sink: Any):
    """Called on the SDK worker; CLI classes and prompt policy are not imported."""

    maps, ports = load_map_config(config.path(config.maps_config))
    model = load_model_config(config.path(config.env_path))
    profile = load_observation_profile(None)
    recorder = AsyncRunLogger(run_dir)
    server = None
    try:
        action = load_uav_action_config(config.path("configs/uav_action_config.json"))
        safety = load_uav_safety_config(config.path("configs/uav_safety_config.json"))
        preview = PreviewConfig(width=request.preview_width, height=request.preview_height, fps=request.preview_fps)
        server = SimulationServer(
            maps, ports, run_dir=run_dir, execution_mode=request.execution,
            show_display=False, action_config=action, save_artifacts=True, preview_config=preview,
        )
        selected = next(package for package in maps if package.map_id == request.map_id)
        recorder.write_json("config/map_package.json", selected.to_dict())
        recorder.write_json("config/operator.json", request.model_dump())
        runtime = RuntimeFactory(
            config.path("configs/plugins.json"), context_config_path=config.path("configs/context.json")
        ).create_task_runtime(
            SimulatorEnvClient(server, observation_profile=profile), recorder,
            model_config=model, safety=SafetySupervisor(safety), event_sink=sink,
        )
        return InteractiveTaskSession(
            runtime, request.map_id, event_sink=sink,
            available_map_ids=[package.map_id for package in maps],
            execution_options=TaskExecutionOptions.from_observation_profile(profile),
        ), server
    except Exception:
        recorder.close()
        raise
