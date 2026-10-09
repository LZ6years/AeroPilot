"""One local flight session, with SDK ownership confined to its worker."""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime
from typing import Any

from src.runtime.control.runtime_control import RuntimeControl
from src.runtime.configs.map_config import load_map_config
from ..composition import compose_session
from ..config import OperatorConfig
from ..schemas import CreateSession, SubmitInstruction
from ..projections.replay import RunRepository
from .event_stream import EventStream, public_value
from .worker import SessionWorker


class OperatorSession:
    def __init__(self, manager, request: CreateSession):
        self.manager = manager
        self.request = request
        self.session_id = uuid.uuid4().hex
        self.run_dir = manager.config.path(manager.config.run_root) / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_workbench_" + self.session_id[:8])
        self.run_dir.mkdir(parents=True)
        self.run_id = manager.runs.register(self.run_dir)
        self.stream = EventStream(self.run_dir, self.session_id)
        self.worker = SessionWorker(self.session_id)
        self.lock = threading.RLock()
        self.phase = "starting"
        self.error = None
        self.runtime_session = self.server = self.preview = None
        self.preview_error = None
        self.control: RuntimeControl | None = None
        self.busy = False
        self.closing = False
        self.instruction_id = None
        self.instruction = ""
        self.latest_pose = {}
        self.summary = None
        self.clients: dict[str, float] = {}
        self.last_contact = time.monotonic()
        self.requests: dict[str, dict] = {}
        self.scene = None

    def start(self):
        self.worker.submit(self._start)

    def _start(self):
        try:
            self.runtime_session, self.server = self.manager.compose(self.manager.config, self.request, self.run_dir, self.sink)
            ready = self.runtime_session.start_scene()
            self.scene = ready["scene"]
            try:
                self.preview = self.server.open_preview(self.scene["scene_instance_id"])
            except Exception as exc:
                self.preview_error = str(exc)
            with self.lock:
                if not self.closing:
                    self.phase = "ready"
            self.stream.publish("session_ready", {"scene": self.scene, "run_id": self.run_id})
        except Exception as exc:
            with self.lock:
                self.phase, self.error = "failed", str(exc)
            self.stream.publish("session_failed", {"error": str(exc)})
            if self.runtime_session is not None:
                try:
                    self.runtime_session.shutdown()
                except Exception as cleanup:
                    self.stream.publish("scene_cleanup_failed", {"error": str(cleanup)})

    def sink(self, event: dict[str, Any]):
        kind, payload = event["kind"], event.get("payload") or {}
        with self.lock:
            self.instruction_id = payload.get("instruction_id") or self.instruction_id
            if kind == "observation_updated":
                self.latest_pose = payload.get("pose") or self.latest_pose
            instruction_id = self.instruction_id
        self.stream.publish(kind, payload, instruction_id=instruction_id)

    def snapshot(self):
        with self.lock:
            control = self.control.snapshot() if self.control else None
            return public_value({
                "session_id": self.session_id, "run_id": self.run_id, "map_id": self.request.map_id,
                "execution": self.request.execution, "phase": self.phase, "busy": self.busy,
                "control": control, "instruction_id": self.instruction_id, "instruction": self.instruction,
                "scene": self.scene, "latest_pose": self.latest_pose, "summary": self.summary,
                "error": self.error, "event_seq": self.stream.seq, "event_error": self.stream.error,
                "controller_connected": bool(self.clients),
                "preview": self.preview.stats() if self.preview else {"status": "unavailable", "error": self.preview_error},
            })

    def submit(self, request: SubmitInstruction):
        with self.lock:
            if request.request_id in self.requests:
                previous = self.requests[request.request_id]
                if previous["instruction"] != request.instruction:
                    raise ValueError("request_id was already used for a different instruction")
                return previous
            if self.phase != "ready" or self.busy or self.closing:
                raise ValueError("Session is not ready for a new instruction")
            self.busy = True
            self.control = RuntimeControl()
            self.instruction = request.instruction
            self.phase = "running"
            self.instruction_id = None
            self.last_contact = time.monotonic() if not self.clients else self.last_contact
            response = {"request_id": request.request_id, "instruction": request.instruction, "accepted": True}
            self.requests[request.request_id] = response
            self.worker.submit(self._run, request.instruction, self.control)
            return response

    def _run(self, instruction: str, control: RuntimeControl):
        try:
            summary = self.runtime_session.submit_instruction(instruction, control=control)
            with self.lock:
                self.summary = summary
                self.latest_pose = summary.get("final_pose") or self.latest_pose
            self.stream.publish("instruction_finished", summary, instruction_id=self.instruction_id)
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
            self.stream.publish("instruction_failed", {"error": str(exc)}, instruction_id=self.instruction_id)
        finally:
            control.finish()
            with self.lock:
                self.busy = False
                if not self.closing:
                    self.phase = "ready"

    def signal(self, action: str, reason="operator"):
        with self.lock:
            if not self.busy or self.control is None:
                raise ValueError("No active instruction")
            result = self.control.request(action, reason=reason)
        self.stream.publish("control_requested", result, instruction_id=self.instruction_id)
        return result

    def contact(self, client_id: str):
        with self.lock:
            self.clients[client_id] = time.monotonic()
            self.last_contact = time.monotonic()

    def disconnect(self, client_id: str):
        with self.lock:
            self.clients.pop(client_id, None)
            if not self.clients:
                self.last_contact = time.monotonic()

    def check_disconnect(self):
        with self.lock:
            now = time.monotonic()
            # Heartbeat expiry also detects a silently broken TCP connection.
            self.clients = {key: seen for key, seen in self.clients.items() if now - seen < self.manager.config.disconnect_grace_sec}
            if not self.busy or now - self.last_contact < self.manager.config.disconnect_grace_sec:
                return
            if self.control and self.control.snapshot()["desired"] == "running":
                self.signal("pause", "control_disconnect")

    def close(self):
        with self.lock:
            if self.closing or self.phase == "closed":
                return
            self.closing, self.phase = True, "closing"
            if self.busy and self.control:
                self.control.request("stop", reason="session_close")
            self.worker.submit(self._close)

    def _close(self):
        try:
            if self.runtime_session and not self.runtime_session.closed:
                self.runtime_session.shutdown()
            with self.lock:
                self.phase = "closed"
            self.stream.publish("session_closed", {})
        except Exception as exc:
            with self.lock:
                self.phase, self.error = "cleanup_failed", str(exc)
            self.stream.publish("scene_cleanup_failed", {"error": str(exc)})
        finally:
            self.stream.close()
            self.worker.retire()

    def join(self):
        self.close()
        self.worker.close()
        self.stream.close()


class SessionManager:
    def __init__(self, config: OperatorConfig, *, compose=compose_session):
        self.config, self.compose = config, compose
        self.runs = RunRepository(config.path(config.run_root))
        self.sessions: dict[str, OperatorSession] = {}
        self.lock = threading.Lock()

    def maps(self):
        maps, _ = load_map_config(self.config.path(self.config.maps_config))
        return [{"map_id": m.map_id, "scene_id": m.scene_id, "protocol": m.control_protocol,
                 "connection": m.connection_mode, "preview_supported": m.control_protocol == "airsim"} for m in maps]

    def create(self, request: CreateSession):
        if request.map_id not in {m["map_id"] for m in self.maps()}:
            raise ValueError("Unknown map_id")
        with self.lock:
            if any(s.phase not in {"closed", "failed"} for s in self.sessions.values()):
                raise ValueError("Close the current scene before opening another")
            session = OperatorSession(self, request)
            self.sessions[session.session_id] = session
            session.start()
            return session.snapshot()

    def get(self, identity):
        with self.lock:
            if identity not in self.sessions:
                raise KeyError("Unknown session ID")
            return self.sessions[identity]

    def check_disconnects(self):
        for session in list(self.sessions.values()):
            session.check_disconnect()

    def close(self):
        for session in list(self.sessions.values()):
            session.join()
