"""Loopback operator HTTP API, reconnectable events, and independent MJPEG."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import OperatorConfig
from .schemas import CreateSession, SubmitInstruction, ControlRequest
from .services.sessions import SessionManager
from .services.event_stream import public_value
from .projections.replay import read_json


def create_app(config: OperatorConfig | None = None, *, manager: SessionManager | None = None):
    config = config or OperatorConfig()
    manager = manager or SessionManager(config)

    async def monitor():
        while True:
            manager.check_disconnects()
            await asyncio.sleep(0.1)

    @asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(monitor())
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            await asyncio.to_thread(manager.close)

    app = FastAPI(title="FlyAgent Flight Workbench", version="0.1.0", lifespan=lifespan)
    app.state.manager = manager

    @app.exception_handler(KeyError)
    async def missing(_, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ValueError)
    async def conflict(_, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": "0.1.0", "default_map_id": config.default_map_id,
                "disconnect_grace_sec": config.disconnect_grace_sec}

    @app.get("/api/maps")
    def maps():
        return manager.maps()

    @app.get("/api/sessions")
    def sessions():
        return [s.snapshot() for s in list(manager.sessions.values())]

    @app.post("/api/sessions", status_code=202)
    def create(request: CreateSession):
        return manager.create(request)

    @app.get("/api/sessions/{identity}")
    def session(identity: str):
        return manager.get(identity).snapshot()

    @app.delete("/api/sessions/{identity}", status_code=202)
    def close(identity: str):
        target = manager.get(identity)
        target.close()
        return target.snapshot()

    @app.post("/api/sessions/{identity}/instructions", status_code=202)
    def instruction(identity: str, request: SubmitInstruction):
        return manager.get(identity).submit(request)

    @app.post("/api/sessions/{identity}/control", status_code=202)
    def control(identity: str, request: ControlRequest):
        return manager.get(identity).signal(request.action)

    @app.get("/api/sessions/{identity}/events")
    def events(identity: str, after: int = 0):
        return manager.get(identity).stream.since(after)

    @app.websocket("/api/sessions/{identity}/ws")
    async def websocket(socket: WebSocket, identity: str, after: int = 0):
        try:
            target = manager.get(identity)
        except KeyError:
            await socket.close(code=1008)
            return
        await socket.accept()
        client_id = uuid.uuid4().hex
        target.contact(client_id)

        async def receive():
            while True:
                message = await socket.receive_json()
                if message.get("type") == "ping":
                    target.contact(client_id)

        async def send():
            cursor = max(0, after)
            while True:
                batch = target.stream.since(cursor)
                for event in batch:
                    await socket.send_json(event)
                    cursor = event["seq"]
                await asyncio.sleep(0.1)

        tasks = [asyncio.create_task(receive()), asyncio.create_task(send())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            target.disconnect(client_id)

    @app.get("/api/sessions/{identity}/preview.mjpeg")
    async def preview(identity: str):
        target = manager.get(identity)
        if target.preview is None:
            raise HTTPException(503, target.preview_error or "Preview is not ready")

        async def frames():
            last = None
            while target.phase not in {"closed", "failed", "cleanup_failed"}:
                frame = target.preview.latest()
                if frame and frame.frame_id != last and target.preview.stats()["status"] == "streaming":
                    last = frame.frame_id
                    header = (f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame.jpeg)}\r\n"
                              f"X-Frame-Id: {frame.frame_id}\r\nX-Captured-At: {frame.captured_at}\r\n"
                              f"X-Capture-Ms: {frame.capture_ms:.2f}\r\n\r\n").encode()
                    yield header + frame.jpeg + b"\r\n"
                await asyncio.sleep(0.01)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.get("/api/runs")
    def runs():
        return manager.runs.list()

    @app.get("/api/runs/{identity}/timeline")
    def timeline(identity: str):
        return manager.runs.timeline(identity)

    @app.get("/api/runs/{identity}/decisions/{decision_id}")
    def decision(identity: str, decision_id: str, instruction_id: str | None = None):
        return manager.runs.decision(identity, decision_id, instruction_id)

    @app.get("/api/runs/{identity}/decisions/{decision_id}/audit")
    def audit(identity: str, decision_id: str, instruction_id: str | None = None):
        selected = manager.runs.decision(identity, decision_id, instruction_id)
        reference = selected.get("debug_ref")
        return public_value(read_json(manager.runs.child(manager.runs.get(identity), reference))) if reference else {}

    @app.get("/api/runs/{identity}/assets/{asset_id}")
    def asset(identity: str, asset_id: str):
        return FileResponse(manager.runs.asset(identity, asset_id), headers={"Cache-Control": "private, max-age=3600"})

    dist = config.root / "frontend/dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="workbench")
    return app
