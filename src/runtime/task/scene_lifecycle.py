"""Simulator-scene lifecycle, reuse, preflight, and terminal cleanup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.env_client import EnvClient, SceneInstance
from src.env_client.errors import SceneRecoveryRequired
from src.memory.flight_memory import FlightMemory
from src.runtime.configs.task_execution_options import InitialPose, TaskExecutionOptions
from src.runtime.task.evidence import FlightEvidenceRecorder
from src.runtime.task.session import TaskSession


@dataclass(frozen=True)
class _LifecycleEvidenceSession:
    """Minimal evidence target for shutdown after TaskSession disposal."""

    scene: SceneInstance
    agent_id: str
    state: Any
    memory: FlightMemory


class SceneLifecycleService:
    """Own scene identity, reusable FlightMemory, and lifecycle actions."""

    def __init__(
        self,
        env_client: EnvClient,
        recorder: Any,
        tools: Any,
        stabilizer: Any,
        evidence: FlightEvidenceRecorder,
        journal: Any,
        memory_writer: Any,
        memory_reader: Any,
        *,
        shutdown_policy: str,
    ) -> None:
        self.env_client = env_client
        self.recorder = recorder
        self.tools = tools
        self.stabilizer = stabilizer
        self.evidence = evidence
        self.journal = journal
        self.memory_writer = memory_writer
        self.memory_reader = memory_reader
        self.shutdown_policy = shutdown_policy
        self.preflight_scene_ids: set[str] = set()
        self.unusable_scene_ids: set[str] = set()
        self.active_reused_scenes: dict[tuple[str, str | None], SceneInstance] = {}
        self.active_flight_memories: dict[tuple[str, str | None, str], FlightMemory] = {}

    def acquire(
        self, map_id: str, policy: str, scene_id: str | None
    ) -> SceneInstance:
        if policy == "reuse_running":
            key = self.scene_key(map_id, scene_id)
            known = self.active_reused_scenes.get(key)
            scene = self.env_client.get_or_start_scene(map_id, scene_id)
            self.active_reused_scenes[key] = scene
            self._record_acquired(
                scene,
                reused=known is not None
                and known.scene_instance_id == scene.scene_instance_id,
            )
            return scene
        scene = self.env_client.start_scene(map_id, scene_id)
        self._record_acquired(scene, reused=False)
        return scene

    def resolve_memory(
        self,
        map_id: str,
        scene: SceneInstance,
        agent_id: str,
        scene_policy: str,
        instruction: str,
        run_id: str,
    ) -> FlightMemory:
        key = (map_id, scene.scene_id, agent_id)
        if scene_policy == "reuse_running" and key in self.active_flight_memories:
            memory = self.active_flight_memories[key]
        else:
            memory = FlightMemory(
                run_id=run_id,
                map_id=map_id,
                scene_instance_id=scene.scene_instance_id,
                agent_id=agent_id,
                memory_reader=self.memory_reader,
            )
            if scene_policy == "reuse_running":
                self.active_flight_memories[key] = memory
        instruction_ref = self.memory_writer.write_instruction(instruction, {})
        memory.update_instruction_ref(instruction_ref, instruction)
        memory.observation_anchors.clear()
        return memory

    def prepare(self, session: TaskSession, options: TaskExecutionOptions) -> None:
        self.assert_usable(session.scene)
        self.validate_execution_options(session.scene, options)
        first_scene_use = not self.is_preflighted(session.scene)
        if first_scene_use:
            self._run_preflight(session)
        if options.initial_pose is not None:
            self._apply_initial_pose(session, options.initial_pose)
        if first_scene_use:
            self._run_preflight_hover(session)
            self.mark_preflighted(session.scene)

        stabilization = None
        if first_scene_use or options.initial_pose is not None:
            stabilization = self.stabilizer.wait_after_preflight(
                session.scene.scene_instance_id,
                session.agent_id,
                capabilities=session.scene.capabilities,
                warmup_modalities=list(session.state.observation_modalities),
            )
            self.journal.debug_event(
                (
                    "preflight_stabilization_completed"
                    if first_scene_use
                    else "episode_initialization_stabilization_completed"
                ),
                stabilization,
            )
        if first_scene_use:
            self.journal.event(
                "preflight_completed",
                {
                    "scene_instance_id": session.scene.scene_instance_id,
                    "stabilization": self.stabilization_summary(stabilization),
                },
            )
        source = "preflight" if first_scene_use else "instruction_start"
        observation = self.evidence.record_observation(
            session,
            list(session.state.observation_modalities),
            source,
            overwrite_task_initial=options.initial_pose is not None,
        )
        self.journal.debug_event(
            (
                "initial_observation_updated"
                if first_scene_use
                else "instruction_start_observation_updated"
            ),
            observation,
        )
        self.journal.event(
            "observation_updated",
            self.evidence.observation_summary(observation, source),
        )
        self.evidence.write_state(session)

    def finalize_instruction(
        self, session: TaskSession, *, scene_unusable: bool = False
    ) -> None:
        """Capture terminal evidence before optional landing and scene stop."""

        failures: list[Exception] = []
        state = session.state
        if state.connected and not scene_unusable:
            capabilities = session.scene.capabilities
            if capabilities is None or capabilities.supports("hover"):
                try:
                    result = self.tools.call(
                        session.scene.scene_instance_id,
                        session.agent_id,
                        "uav.hover",
                        {},
                    )
                    self.evidence.record_action(
                        session, "uav.hover", {}, result.to_dict()
                    )
                    if result.execution_state.get("requires_scene_reset"):
                        scene_unusable = True
                        raise SceneRecoveryRequired(result.message)
                    if not result.ok:
                        raise RuntimeError(result.message)
                except Exception as exc:
                    failures.append(exc)
            try:
                if not scene_unusable:
                    self.evidence.record_observation(
                        session,
                        list(state.observation_modalities),
                        "task_terminal",
                    )
                    session.checkpoints.publish(
                        state, session.memory, "task_terminal"
                    )
            except Exception as exc:
                failures.append(exc)
        terminal = {
            **self.evidence.state_summary(state),
            "cleanup_errors": [
                {"error_type": type(exc).__name__, "error": str(exc)}
                for exc in failures
            ],
        }
        session.transcript.append_task_terminal(terminal)
        self.journal.write_terminal(terminal)
        self.evidence.write_state(session)
        if failures:
            raise failures[0]

    def shutdown(self, session: TaskSession) -> None:
        self._shutdown(session)

    def finish_reused_scene(
        self,
        map_id: str,
        state: Any | None,
        agent_id: str,
        scene_id: str | None,
    ) -> None:
        scene = self.reused(map_id, scene_id)
        if scene is None:
            self.journal.debug_event(
                "scene_stop_skipped",
                {"map_id": map_id, "scene_id": scene_id, "reason": "not_running"},
            )
            return
        primary_error: Exception | None = None
        try:
            if state is not None:
                memory = self.active_flight_memories.get(
                    (map_id, scene.scene_id, agent_id)
                )
                if memory is not None:
                    lifecycle_session = _LifecycleEvidenceSession(
                        scene=scene,
                        agent_id=agent_id,
                        state=state,
                        memory=memory,
                    )
                    self._shutdown(lifecycle_session)
        except Exception as exc:
            primary_error = exc
            raise
        finally:
            try:
                self.stop(scene)
            except Exception as cleanup_error:
                if primary_error is None:
                    raise
                _attach_cleanup_error(primary_error, cleanup_error, "scene_stop")

    def safe_hover_reused_scene(
        self, map_id: str, agent_id: str, scene_id: str | None
    ) -> None:
        scene = self.reused(map_id, scene_id)
        if scene is None or self.is_unusable(scene):
            self.journal.debug_event(
                "continuous_error_hover_skipped", {"reason": "not_running"}
            )
            return
        if scene.capabilities and not scene.capabilities.supports("hover"):
            self.journal.debug_event(
                "continuous_error_hover_skipped",
                {
                    "reason": "backend_capability",
                    "protocol": scene.capabilities.protocol,
                    "control_mode": scene.capabilities.control_mode,
                },
            )
            return
        try:
            result = self.tools.call(
                scene.scene_instance_id, agent_id, "uav.hover", {}
            )
            self.journal.debug_event(
                "continuous_error_hover_completed", result.to_dict()
            )
        except Exception as exc:
            self.journal.debug_event(
                "continuous_error_hover_failed",
                {"error_type": type(exc).__name__, "error": str(exc)},
            )

    def reused(self, map_id: str, scene_id: str | None) -> SceneInstance | None:
        return self.active_reused_scenes.get(self.scene_key(map_id, scene_id))

    def is_preflighted(self, scene: SceneInstance) -> bool:
        return scene.scene_instance_id in self.preflight_scene_ids

    def mark_preflighted(self, scene: SceneInstance) -> None:
        self.preflight_scene_ids.add(scene.scene_instance_id)

    def is_unusable(self, scene: SceneInstance) -> bool:
        return scene.scene_instance_id in self.unusable_scene_ids

    def mark_unusable(self, scene: SceneInstance) -> None:
        self.unusable_scene_ids.add(scene.scene_instance_id)

    def assert_usable(self, scene: SceneInstance) -> None:
        if self.is_unusable(scene):
            raise SceneRecoveryRequired(
                "Scene must be recreated before another instruction"
            )

    def stop(self, scene: SceneInstance) -> None:
        self.env_client.stop_scene(scene.scene_instance_id)
        self.preflight_scene_ids.discard(scene.scene_instance_id)
        self.forget(scene)
        self.journal.debug_event("scene_stopped", scene.to_dict())

    def forget(self, scene: SceneInstance) -> None:
        self.unusable_scene_ids.discard(scene.scene_instance_id)
        for key, active in list(self.active_reused_scenes.items()):
            if active.scene_instance_id == scene.scene_instance_id:
                self.active_reused_scenes.pop(key, None)
        for key, memory in list(self.active_flight_memories.items()):
            if memory.scene_instance_id == scene.scene_instance_id:
                self.active_flight_memories.pop(key, None)

    @staticmethod
    def scene_key(map_id: str, scene_id: str | None) -> tuple[str, str | None]:
        return map_id, str(scene_id) if scene_id is not None else None

    @staticmethod
    def validate_execution_options(
        scene: SceneInstance, options: TaskExecutionOptions
    ) -> None:
        capabilities = scene.capabilities
        if capabilities is None:
            return
        supported = {
            str(value).strip().lower()
            for value in capabilities.observation_modalities
        }
        unsupported = [
            value for value in options.observation_modalities if value not in supported
        ]
        if unsupported:
            raise RuntimeError(
                "Scene does not support requested observation modalities: "
                + ", ".join(unsupported)
            )
        if options.initial_pose is not None and not capabilities.supports("set_pose"):
            raise RuntimeError(
                "Scene does not support set_pose required by initial_pose"
            )

    def _record_acquired(self, scene: SceneInstance, *, reused: bool) -> None:
        try:
            self._record_contract(scene)
            self.journal.debug_event(
                "scene_reused" if reused else "scene_started", scene.to_dict()
            )
        except Exception as original:
            if not reused:
                try:
                    self.env_client.stop_scene(scene.scene_instance_id)
                except Exception as cleanup_error:
                    _attach_cleanup_error(
                        original, cleanup_error, "scene_acquisition_rollback"
                    )
                else:
                    self.preflight_scene_ids.discard(scene.scene_instance_id)
                    self.forget(scene)
            raise

    def _record_contract(self, scene: SceneInstance) -> None:
        self.recorder.write_json("state/scene_instance.json", scene.to_dict())
        if scene.capabilities is not None:
            self.recorder.write_json(
                "state/scene_capabilities.json",
                {
                    "protocol": scene.capabilities.protocol,
                    "control_mode": scene.capabilities.control_mode,
                    "supported_actions": list(scene.capabilities.supported_actions),
                    "observation_modalities": list(
                        scene.capabilities.observation_modalities
                    ),
                    "requires_takeoff": scene.capabilities.requires_takeoff,
                    "supports_soft_reset": scene.capabilities.supports_soft_reset,
                },
            )

    def _run_preflight(self, session: TaskSession) -> None:
        scene = session.scene
        result = self.tools.call(
            scene.scene_instance_id, session.agent_id, "uav.connect", {}
        )
        self.evidence.record_action(session, "uav.connect", {}, result.to_dict())
        self.journal.debug_event("control_connected", result.to_dict())
        if not result.ok:
            raise RuntimeError(f"Preflight connect failed: {result.message}")
        capabilities = scene.capabilities
        if capabilities is None or capabilities.requires_takeoff:
            self.journal.debug_event(
                "preflight_takeoff_started",
                {"scene_instance_id": scene.scene_instance_id},
            )
            result = self.tools.call(
                scene.scene_instance_id, session.agent_id, "uav.takeoff", {}
            )
            self.evidence.record_action(
                session, "uav.takeoff", {}, result.to_dict()
            )
            self.journal.debug_event("preflight_takeoff_completed", result.to_dict())
            if not result.ok:
                raise RuntimeError(f"Preflight takeoff failed: {result.message}")
        else:
            self.journal.debug_event(
                "preflight_takeoff_skipped",
                {
                    "protocol": capabilities.protocol,
                    "control_mode": capabilities.control_mode,
                },
            )

    def _run_preflight_hover(self, session: TaskSession) -> None:
        capabilities = session.scene.capabilities
        if capabilities is None or capabilities.supports("hover"):
            result = self.tools.call(
                session.scene.scene_instance_id,
                session.agent_id,
                "uav.hover",
                {},
            )
            self.evidence.record_action(session, "uav.hover", {}, result.to_dict())
            self.journal.debug_event("preflight_hover_completed", result.to_dict())
            if not result.ok:
                raise RuntimeError(f"Preflight hover failed: {result.message}")
        else:
            self.journal.debug_event(
                "preflight_hover_skipped",
                {
                    "reason": "backend_capability",
                    "protocol": capabilities.protocol,
                    "control_mode": capabilities.control_mode,
                },
            )

    def _apply_initial_pose(
        self, session: TaskSession, initial_pose: InitialPose
    ) -> None:
        args = initial_pose.to_action_args()
        payload = {
            "scene_instance_id": session.scene.scene_instance_id,
            "agent_id": session.agent_id,
            "pose": initial_pose.to_dict(),
        }
        self.journal.debug_event("episode_initial_pose_started", payload)
        try:
            result = self.tools.call(
                session.scene.scene_instance_id,
                session.agent_id,
                "uav.set_pose",
                args,
            )
        except Exception as exc:
            self.journal.debug_event(
                "episode_initial_pose_failed",
                {**payload, "error_type": type(exc).__name__, "error": str(exc)},
            )
            raise
        if not result.ok:
            self.journal.debug_event(
                "episode_initial_pose_failed",
                {**payload, "result": result.to_dict()},
            )
            raise RuntimeError(f"Episode initial pose failed: {result.message}")
        session.memory.reset_pose_epoch(
            active_instruction=session.state.user_instruction
        )
        self.journal.debug_event(
            "episode_initial_pose_completed",
            {
                **payload,
                "result": result.to_dict(),
                "memory_reset_epoch": session.memory.reset_epoch,
            },
        )

    def _shutdown(
        self, session: TaskSession | _LifecycleEvidenceSession
    ) -> None:
        scene = session.scene
        state = session.state
        if self.is_unusable(scene):
            self.journal.debug_event(
                "shutdown_skipped", {"reason": "scene_recovery_required"}
            )
            return
        if self.shutdown_policy == "none" or not state.connected:
            self.journal.debug_event(
                "shutdown_skipped",
                {"policy": self.shutdown_policy, "connected": state.connected},
            )
            return
        self.journal.debug_event(
            "shutdown_started", {"policy": self.shutdown_policy}
        )
        capabilities = scene.capabilities
        if capabilities is None or capabilities.supports("hover"):
            result = self.tools.call(
                scene.scene_instance_id, session.agent_id, "uav.hover", {}
            )
            self.evidence.record_action(session, "uav.hover", {}, result.to_dict())
            self.journal.debug_event("shutdown_hover_completed", result.to_dict())
            if not result.ok:
                raise RuntimeError(f"Shutdown hover failed: {result.message}")
        else:
            self.journal.debug_event(
                "shutdown_hover_skipped",
                {
                    "reason": "backend_capability",
                    "protocol": capabilities.protocol,
                    "control_mode": capabilities.control_mode,
                },
            )
        if self.shutdown_policy == "land":
            if capabilities is None or capabilities.supports("land"):
                self.journal.debug_event(
                    "shutdown_land_started",
                    {"scene_instance_id": scene.scene_instance_id},
                )
                result = self.tools.call(
                    scene.scene_instance_id, session.agent_id, "uav.land", {}
                )
                self.evidence.record_action(
                    session, "uav.land", {}, result.to_dict()
                )
                self.journal.debug_event(
                    "shutdown_land_completed", result.to_dict()
                )
                if not result.ok:
                    raise RuntimeError(f"Shutdown land failed: {result.message}")
            else:
                self.journal.debug_event(
                    "shutdown_land_skipped",
                    {
                        "reason": "backend_capability",
                        "protocol": capabilities.protocol,
                        "control_mode": capabilities.control_mode,
                    },
                )
        self.evidence.write_state(session)

    @staticmethod
    def stabilization_summary(stabilization: dict | None) -> dict[str, Any]:
        data = stabilization if isinstance(stabilization, dict) else {}
        return {
            "status": data.get("status"),
            "elapsed_sec": data.get("elapsed_sec"),
            "last_velocity_norm_mps": data.get("last_velocity_norm_mps"),
            "threshold": data.get("threshold"),
            "reason": data.get("reason"),
        }


def _attach_cleanup_error(primary: Exception, cleanup: Exception, stage: str) -> None:
    errors = list(getattr(primary, "cleanup_errors", []))
    errors.append(
        {
            "stage": stage,
            "error_type": type(cleanup).__name__,
            "error": str(cleanup),
        }
    )
    primary.cleanup_errors = errors
