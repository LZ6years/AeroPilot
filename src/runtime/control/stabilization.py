"""Wait for UAV motion to settle between planned tool steps.

The stabilizer is a runtime control policy, not a model-visible action. It may
send a zero-velocity brake command and poll pose through ToolDispatcher, but
those polling observations are deliberately kept out of FlightMemory so the
model only sees the final automatic observation for a phase window.
"""

from __future__ import annotations

import time
from typing import Any

from src.agent.planning.schemas import PlannedStep
from src.env_client import SceneCapabilities
from src.runtime.configs.stabilization_config import UavStabilizationConfig
from src.tools.dispatcher import ToolDispatcher
from src.uav import pose_from_observation, velocity_norm_mps


class UavStabilizer:
    """Poll pose velocity until the UAV is stable enough for the next step."""

    def __init__(
        self,
        tools: ToolDispatcher,
        recorder: Any | None = None,
        config: UavStabilizationConfig | None = None,
    ) -> None:
        self.tools = tools
        self.recorder = recorder
        self.config = config or UavStabilizationConfig()

    def wait_after_step(
        self,
        scene_instance_id: str,
        agent_id: str,
        phase: str,
        step: PlannedStep,
        capabilities: SceneCapabilities | None = None,
    ) -> dict[str, Any]:
        """Wait after a motion step and return a serializable summary."""

        tool = step.tool or ""
        base = {
            "phase": phase,
            "tool": tool,
            "threshold": self.config.velocity_threshold_mps,
            "timeout": self.config.timeout_sec,
            "zero_velocity_brake_duration_sec": self.config.zero_velocity_brake_duration_sec,
            "hover_after_brake": self.config.hover_after_brake,
        }
        if not self.config.enabled:
            return self._skip({**base, "reason": "disabled"})
        if tool not in self.config.motion_tools:
            return self._skip({**base, "reason": "non_motion_tool"})
        if (
            capabilities is not None
            and "pose" not in capabilities.observation_modalities
        ):
            return self._skip(
                {**base, "reason": "pose_observation_unsupported"}
            )

        started = time.monotonic()
        self._event("uav_stabilization_started", base)
        brake_data = self._zero_velocity_brake(
            scene_instance_id,
            agent_id,
            base,
            capabilities,
        )
        hover_data: dict[str, Any] | None = self._optional_hover(
            scene_instance_id,
            agent_id,
            base,
            capabilities,
        )

        poll_count = 0
        last_velocity: float | None = None
        deadline = started + max(0.0, float(self.config.timeout_sec))
        while time.monotonic() <= deadline:
            poll_count += 1
            observation = self.tools.observe(scene_instance_id, agent_id, ["pose"])
            pose = pose_from_observation(observation.to_dict().get("data", {}))
            last_velocity = velocity_norm_mps(pose)
            if last_velocity is None:
                return self._skip(
                    {
                        **base,
                        "reason": "velocity_missing",
                        "elapsed_sec": round(time.monotonic() - started, 3),
                        "poll_count": poll_count,
                        "last_velocity_norm_mps": None,
                    }
                )
            if last_velocity <= self.config.velocity_threshold_mps:
                payload = {
                    **base,
                    "status": "completed",
                    "elapsed_sec": round(time.monotonic() - started, 3),
                    "poll_count": poll_count,
                    "last_velocity_norm_mps": round(last_velocity, 4),
                    "brake_result": brake_data,
                    "hover_after_brake": self.config.hover_after_brake,
                    "hover_result": hover_data,
                }
                self._event("uav_stabilization_completed", payload)
                return payload
            time.sleep(max(0.0, float(self.config.poll_interval_sec)))

        payload = {
            **base,
            "status": "timeout",
            "elapsed_sec": round(time.monotonic() - started, 3),
            "poll_count": poll_count,
            "last_velocity_norm_mps": round(last_velocity, 4) if last_velocity is not None else None,
            "brake_result": brake_data,
            "hover_after_brake": self.config.hover_after_brake,
            "hover_result": hover_data,
        }
        self._event("uav_stabilization_timeout", payload)
        return payload

    def wait_after_preflight(
        self,
        scene_instance_id: str,
        agent_id: str,
        capabilities: SceneCapabilities | None = None,
        warmup_modalities: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Wait before the initial preflight observation.

        Velocity stability can be reported before a simulator's render target
        has produced a usable frame after ``set_pose``. The optional minimum
        settle time therefore applies only here, after scene/episode
        initialization and before the first Agent-visible observation. It does
        not delay stabilization after ordinary Planner actions.
        """

        started = time.monotonic()
        step = PlannedStep.tool_step("uav.move_body", {}, "preflight stabilization")
        result = self.wait_after_step(
            scene_instance_id,
            agent_id,
            "preflight",
            step,
            capabilities=capabilities,
        )
        minimum_settle_sec = max(
            0.0, float(self.config.initialization_min_settle_sec)
        )
        normalized_modalities = tuple(
            str(value).strip().lower()
            for value in (warmup_modalities or ())
            if str(value).strip()
        )
        settle_started = time.monotonic()
        warmup_summary: dict[str, Any] | None = None
        if minimum_settle_sec > 0.0:
            self._event(
                "uav_initialization_settle_started",
                {
                    "minimum_settle_sec": minimum_settle_sec,
                    "warmup_modalities": list(normalized_modalities),
                },
            )
        if minimum_settle_sec > 0.0 and normalized_modalities:
            self._event(
                "uav_initialization_observation_warmup_started",
                {
                    "modalities": list(normalized_modalities),
                    "minimum_settle_sec": minimum_settle_sec,
                },
            )
            warmup_count = 0
            while (
                max(0.0, time.monotonic() - settle_started)
                < minimum_settle_sec
            ):
                # These observations deliberately bypass FlightMemory. Some
                # simulators advance lazy render targets only while image
                # requests are active, so a bounded polling loop is required
                # instead of one request followed by a passive sleep.
                self.tools.observe(
                    scene_instance_id,
                    agent_id,
                    list(normalized_modalities),
                )
                warmup_count += 1
                elapsed = max(0.0, time.monotonic() - settle_started)
                remaining = max(0.0, minimum_settle_sec - elapsed)
                if remaining > 0.0:
                    time.sleep(
                        min(
                            max(
                                0.001,
                                float(self.config.poll_interval_sec),
                            ),
                            remaining,
                        )
                    )
            warmup_summary = {
                "modalities": list(normalized_modalities),
                "poll_count": warmup_count,
                "elapsed_sec": round(
                    max(0.0, time.monotonic() - settle_started), 3
                ),
            }
            self._event(
                "uav_initialization_observation_warmup_completed",
                warmup_summary,
            )
        elif minimum_settle_sec > 0.0:
            time.sleep(minimum_settle_sec)
        if minimum_settle_sec > 0.0:
            payload = {
                "minimum_settle_sec": minimum_settle_sec,
                "warmup": warmup_summary,
            }
            self._event(
                "uav_initialization_settle_completed",
                {
                    **payload,
                    "settle_elapsed_sec": round(
                        max(0.0, time.monotonic() - settle_started), 3
                    ),
                },
            )
        return {
            **result,
            "initialization_min_settle_sec": minimum_settle_sec,
            "initialization_wait_sec": minimum_settle_sec,
            "initialization_observation_warmup": warmup_summary,
            "elapsed_sec": round(
                max(0.0, time.monotonic() - started), 3
            ),
        }

    def _skip(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = {**payload, "status": "skipped"}
        self._event("uav_stabilization_skipped", result)
        return result

    def _zero_velocity_brake(
        self,
        scene_instance_id: str,
        agent_id: str,
        base: dict[str, Any],
        capabilities: SceneCapabilities | None,
    ) -> dict[str, Any] | None:
        duration = float(self.config.zero_velocity_brake_duration_sec)
        if duration <= 0.0:
            return None
        if capabilities is not None and not capabilities.supports("move_body"):
            result = {
                "status": "skipped",
                "reason": "move_body_unsupported",
            }
            self._event(
                "stabilization_action_skipped",
                {**base, "action": "move_body", **result},
            )
            return result
        args = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "duration": duration}
        result = self.tools.call(scene_instance_id, agent_id, "uav.move_body", args).to_dict()
        self._event(
            "uav_stabilization_brake_requested",
            {**base, "args": args, "result": result},
        )
        return result

    def _optional_hover(
        self,
        scene_instance_id: str,
        agent_id: str,
        base: dict[str, Any],
        capabilities: SceneCapabilities | None,
    ) -> dict[str, Any] | None:
        if not self.config.hover_after_brake:
            return None
        if capabilities is not None and not capabilities.supports("hover"):
            result = {
                "status": "skipped",
                "reason": "hover_unsupported",
            }
            self._event(
                "stabilization_action_skipped",
                {**base, "action": "hover", **result},
            )
            return result
        result = self.tools.call(scene_instance_id, agent_id, "uav.hover", {}).to_dict()
        self._event("uav_stabilization_hover_requested", {**base, "result": result})
        return result

    def _event(self, name: str, payload: dict[str, Any]) -> None:
        if self.recorder is not None:
            debug_event = getattr(self.recorder, "debug_event", None)
            if callable(debug_event):
                debug_event(name, payload)
            else:
                self.recorder.event(name, payload)
