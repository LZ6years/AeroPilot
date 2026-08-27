"""Keep one simulator scene alive across a UAV-Flow evaluation batch.

The stable EnvClient contract deliberately exposes scene start/stop rather
than benchmark batching.  UAV-Flow therefore owns this small orchestration
layer: it starts one scene for a same-map batch, places the UAV before model
parsing begins, releases the protocol client while benchmark-only objects are
edited, and stops the owned scene exactly once at batch shutdown.

Nothing in this module receives reference paths, success labels, or metrics.
The only episode input used here is the planner-safe initial pose.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.benchmarks.core.models import PreparedBenchmarkEpisode
from evaluation.benchmarks.core.report_io import atomic_write_text
from src.env_client import SceneInstance, SimulatorEnvClient
from src.sim_server.server import SimulationServer


class UavFlowBatchSceneSession:
    """Own one SimulationServer scene for a complete UAV-Flow batch.

    A fresh TaskRuntime is still created for every case.  Only simulator
    process/protocol state is shared, so Agent memory and model context cannot
    leak between episodes.
    """

    def __init__(
        self,
        service: SimulationServer,
        client: SimulatorEnvClient,
        *,
        map_id: str,
        scene_id: str | None,
        eval_dir: str | Path,
    ) -> None:
        self.service = service
        self.client = client
        self.map_id = str(map_id)
        self.scene_id = scene_id
        self.eval_dir = Path(eval_dir)
        self.scene: SceneInstance | None = None
        self.start_count = 0
        self.restart_count = 0
        self.stop_count = 0
        self.events: list[dict[str, Any]] = []
        self._closed = False
        self._write_manifest(status="created")

    def ensure_scene(self) -> SceneInstance:
        """Start the shared scene lazily, or return the current instance."""

        if self._closed:
            raise RuntimeError("UAV-Flow batch scene session is already closed")
        if self.scene is not None:
            return self.scene
        self.scene = self.client.get_or_start_scene(self.map_id, self.scene_id)
        self.start_count += 1
        self._record(
            "scene_started",
            {
                "scene_instance_id": self.scene.scene_instance_id,
                "process_ownership": self.scene.process_ownership,
            },
        )
        return self.scene

    def prepare_episode(
        self,
        episode: PreparedBenchmarkEpisode[Any, Any],
    ) -> SceneInstance:
        """Place and validate the UAV before TaskParser can block on a model.

        TaskRuntime intentionally applies the same initial pose again inside
        its audited initialization path.  This early placement is not an Agent
        action: it removes the long-lived default/previous camera view while
        TaskParser is running and provides a simulator health check before an
        episode consumes model calls.
        """

        scene = self.ensure_scene()
        pose = episode.execution_options.initial_pose
        if pose is None:
            raise ValueError("UAV-Flow episode requires an initial pose")

        connect = self.client.execute_action(
            scene.scene_instance_id,
            episode.agent_id,
            "connect",
            {},
        )
        _require_ok(connect, "batch pre-position connect")
        placement = self.client.execute_action(
            scene.scene_instance_id,
            episode.agent_id,
            "set_pose",
            pose.to_action_args(),
        )
        _require_ok(placement, "batch episode initial pose")

        # Capturing the configured modalities forces camera synchronization and
        # validates the observable pose.  This warm-up is deliberately outside
        # the case recorder and therefore cannot enter the formal trajectory.
        observation = self.client.get_observation(
            scene.scene_instance_id,
            episode.agent_id,
            # One RGB frame is sufficient to force camera synchronization;
            # depth/segmentation warm-up would add per-case I/O without adding
            # another lifecycle guarantee.
            ["rgb", "pose"],
        )
        observed_pose = observation.data.get("pose", {})
        _validate_initial_pose(pose.to_dict(), observed_pose)
        self._record(
            "episode_prepared",
            {
                "case_id": _case_id(episode),
                "scene_instance_id": scene.scene_instance_id,
                "requested_pose": pose.to_dict(),
                "observed_pose": dict(observed_pose),
            },
        )

        # UAV-Flow's scenario controller opens a privileged UnrealCV client to
        # place people/vehicles.  Release Runtime's client first because some
        # packaged UnrealCV servers are effectively single-client.
        self.release_protocol_client()
        return scene

    def release_protocol_client(self) -> None:
        """Disconnect the SDK client without changing UE process ownership."""

        if self.scene is None:
            return
        self.service.release_scene_client(self.scene.scene_instance_id)
        self._record(
            "protocol_client_released",
            {"scene_instance_id": self.scene.scene_instance_id},
        )

    def restart_scene(self, *, reason: str) -> SceneInstance:
        """Replace an unhealthy scene while preserving batch ownership."""

        old_scene_id = self.scene.scene_instance_id if self.scene else None
        if self.scene is not None:
            self.client.stop_scene(self.scene.scene_instance_id)
            self.stop_count += 1
            self.scene = None
        self.restart_count += 1
        self._record(
            "scene_restart_requested",
            {"reason": reason, "previous_scene_instance_id": old_scene_id},
        )
        return self.ensure_scene()

    def close(self) -> None:
        """Stop or detach the shared SceneInstance exactly once."""

        if self._closed:
            return
        self._closed = True
        error: dict[str, str] | None = None
        if self.scene is not None:
            try:
                self.client.stop_scene(self.scene.scene_instance_id)
                self.stop_count += 1
            except Exception as exc:  # Preserve cleanup evidence before re-raise.
                error = {"error_type": type(exc).__name__, "error": str(exc)}
            finally:
                self.scene = None
        self._record("session_closed", {"error": error}, status="closed")
        if error is not None:
            raise RuntimeError(
                "UAV-Flow batch scene cleanup failed: " + error["error"]
            )

    def runtime_view(self, runtime: Any) -> "UavFlowEpisodeRuntime":
        """Return the executor-facing view that does not stop the batch UE."""

        return UavFlowEpisodeRuntime(runtime, self)

    def _record(
        self,
        event: str,
        payload: dict[str, Any],
        *,
        status: str = "running",
    ) -> None:
        self.events.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "payload": payload,
            }
        )
        self._write_manifest(status=status)

    def _write_manifest(self, *, status: str) -> None:
        scene = self.scene.to_dict() if self.scene is not None else None
        payload = {
            "status": status,
            "map_id": self.map_id,
            "scene_id": self.scene_id,
            "scene": scene,
            "start_count": self.start_count,
            "restart_count": self.restart_count,
            "stop_count": self.stop_count,
            "events": list(self.events),
        }
        atomic_write_text(
            self.eval_dir / "scene_session.json",
            json.dumps(payload, ensure_ascii=False, indent=2),
        )


class UavFlowEpisodeRuntime:
    """Delegate TaskRuntime while retaining batch-level scene ownership."""

    def __init__(self, runtime: Any, session: UavFlowBatchSceneSession) -> None:
        self._runtime = runtime
        self._session = session

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    def finish_reused_scene(
        self,
        map_id: str,
        state: Any,
        *,
        agent_id: str = "uav_1",
        scene_id: str | None = None,
    ) -> None:
        """Stabilize one episode, but leave process stop to the batch owner."""

        try:
            if state is not None and self._runtime.shutdown_policy != "none":
                self._runtime.safe_hover_reused_scene(
                    map_id,
                    agent_id=agent_id,
                    scene_id=scene_id,
                )
        finally:
            # Scenario cleanup follows immediately in BenchmarkEpisodeExecutor.
            # Releasing here gives its privileged connection exclusive access.
            self._session.release_protocol_client()


def _require_ok(result: Any, stage: str) -> None:
    if getattr(result, "ok", False):
        return
    raise RuntimeError(
        f"UAV-Flow simulator {stage} failed: {getattr(result, 'message', '')}"
    )


def _validate_initial_pose(
    requested: dict[str, Any],
    observed: dict[str, Any],
    *,
    position_tolerance_m: float = 0.05,
    yaw_tolerance_deg: float = 2.0,
) -> None:
    if not isinstance(observed, dict):
        raise RuntimeError("UAV-Flow simulator warm-up returned no pose")
    try:
        position_error = math.sqrt(
            sum(
                (float(observed[name]) - float(requested[name])) ** 2
                for name in ("x", "y", "z")
            )
        )
        yaw_error = abs(
            (float(observed["yaw_deg"]) - float(requested["yaw_deg"]) + 180.0)
            % 360.0
            - 180.0
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"UAV-Flow simulator warm-up returned an invalid pose: {observed!r}"
        ) from exc
    if position_error <= position_tolerance_m and yaw_error <= yaw_tolerance_deg:
        return
    raise RuntimeError(
        "UAV-Flow simulator initial pose validation failed: "
        f"position_error_m={position_error:.6f}, "
        f"position_tolerance_m={position_tolerance_m:.6f}, "
        f"yaw_error_deg={yaw_error:.6f}, "
        f"yaw_tolerance_deg={yaw_tolerance_deg:.6f}"
    )


def _case_id(episode: PreparedBenchmarkEpisode[Any, Any]) -> str | None:
    return str(episode.case_id)
