"""Read-only adapters for saved transcripts and legacy action journals.

Only explicitly selected model-visible fields cross this boundary. Benchmark
results, reference trajectories, absolute poses and arbitrary Context fields
are never forwarded. This module does not import Agent or Runtime contracts.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any

from PIL import Image


# Recursive allowlist: unknown plugin payloads remain missing evidence rather
# than accidentally admitting evaluator/debug truth in a model prompt.
_FIELDS = frozenset({
    "vx", "vy", "vz", "duration", "yaw_deg", "yaw_rate", "speed", "distance",
    "observation_id", "query", "include_distance", "candidates", "candidate_id",
    "label", "confidence", "bbox", "bbox_xyxy", "distance_m", "distance_status",
    "distance_geometry", "depth_status", "status", "reason", "message",
    "forward_m", "right_m", "up_m", "planar_m", "yaw_delta_deg",
    "instruction_start_relative_uav_motion", "last_window_uav_motion", "field_semantics",
    "steps", "id", "description", "explanation", "outcome", "evidence_refs",
    "intervention", "original", "applied", "requested_args", "executed_args",
})


def _visible(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _visible(v) for k, v in value.items() if k in _FIELDS}
    if isinstance(value, list):
        return [_visible(v) for v in value]
    return value if value is None or isinstance(value, (str, int, float, bool)) else None


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Evidence path escapes the source run")
    return resolved


@dataclass
class EvidenceBundle:
    """Stable evidence IDs, original RGB bytes, and auditable source digests.

    Images are addressed by observation ID. They are not thumbnails or rendered
    HTML screenshots. Missing/invalid RGB is recorded in the public index.
    """
    source: Path
    instruction: str
    timeline: list[dict[str, Any]]
    observations: dict[str, dict[str, Any]]
    sources: dict[str, str]
    limitations: list[str] = field(default_factory=list)
    images: dict[str, tuple[str, bytes]] = field(default_factory=dict, repr=False)

    @property
    def evidence_ids(self) -> set[str]:
        return {row["id"] for row in self.timeline} | set(self.observations)

    def index(self) -> dict[str, Any]:
        return {"instruction": self.instruction, "timeline": self.timeline,
                "observations": self.observations, "source_sha256": self.sources,
                "limitations": self.limitations}

    def image_part(self, observation_id: str) -> dict[str, Any]:
        mime, data = self.images[observation_id]
        return {"type": "image_url", "image_url": {
            "url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"}}


def read_run(source: Path, *, instruction_id: str | None = None,
             instruction: str | None = None) -> EvidenceBundle:
    """Load one task, rejecting ambiguous multi-task logs rather than mixing them.

    Native runs use memory/agent_transcript.jsonl; older runs may have only
    memory/actions.jsonl (or these journals directly in the selected directory).
    Legacy instructions must be supplied explicitly, never inferred from truth.
    """
    root = source.resolve(strict=True)
    memory = root / "memory" if (root / "memory").is_dir() else root
    sources: dict[str, str] = {}

    def rows(name: str) -> list[dict[str, Any]]:
        path = _inside(root, memory / name)
        if not path.exists():
            return []
        raw = path.read_bytes()
        sources[path.relative_to(root).as_posix()] = hashlib.sha256(raw).hexdigest()
        result = []
        for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object in {name}:{number}")
                result.append({**value, "_source": f"{path.relative_to(root).as_posix()}#{number}"})
        return result

    transcript = rows("agent_transcript.jsonl")
    timeline: list[dict[str, Any]] = []
    observation_ids: set[str] = set()
    limitations = ["Self-reported completion is a claim, not verified flight success.",
                  "Unknown payload fields are omitted; no benchmark truth is loaded."]
    if transcript:
        tasks = {r.get("instruction_id") for r in transcript if r.get("kind") == "user_instruction"}
        if instruction_id is None:
            if len(tasks) != 1:
                raise ValueError("Select instruction_id for a multi-task transcript")
            instruction_id = next(iter(tasks))
        if instruction_id not in tasks:
            raise ValueError("Unknown instruction_id")
        selected = [r for r in transcript if r.get("instruction_id") == instruction_id]
        saved = next(r["payload"]["instruction"] for r in selected if r["kind"] == "user_instruction")
        if instruction is not None and instruction != saved:
            raise ValueError("Supplied instruction differs from the original")
        instruction = saved
        for row in selected:
            payload, kind = row.get("payload", {}), row.get("kind")
            item = {"id": row["id"], "kind": kind, "timestamp": row.get("timestamp"),
                    "source_ref": row["_source"]}
            if kind == "state_checkpoint":
                obs = payload.get("observation_id")
                if not obs:
                    continue
                observation_ids.add(obs)
                item["observation_id"] = obs
                fragments = payload.get("context", {}).get("fragments", [])
                item["measured_self_motion"] = [_visible(f.get("content", {})) for f in fragments
                                               if f.get("name") == "action"]
            elif kind == "assistant_turn":
                item["unverified_intent"] = payload.get("public_intent")
                item["requested_calls"] = [{"call_id": c.get("tool_call_id"),
                    "tool": c.get("tool_name"), "args": _visible(c.get("args", {}))}
                    for c in payload.get("tool_calls", [])]
            elif kind == "tool_result":
                item.update(call_id=payload.get("tool_call_id"), tool=payload.get("tool_name"),
                            status=payload.get("status"), message=payload.get("message"),
                            requested_args=_visible(payload.get("requested_args", {})),
                            executed_args=_visible(payload.get("executed_args", {})),
                            result=_visible(payload.get("data", {})),
                            safety_interventions=_visible(payload.get("safety_interventions", [])))
            else:
                continue
            timeline.append(item)
    else:
        if not instruction:
            raise ValueError("Legacy action logs require the original instruction")
        actions = rows("actions.jsonl")
        if not actions:
            raise ValueError("No supported transcript or action journal found")
        if instruction_id is not None:
            raise ValueError("Legacy actions cannot isolate instruction_id")
        limitations.append("Legacy actions have no task partition or checkpoint pairing; caller must supply a single-task run.")
        for row in actions:
            result = row.get("result", {})
            timeline.append({"id": row["id"], "kind": "action_result", "source_ref": row["_source"],
                             "timestamp": row.get("timestamp"), "tool": row.get("tool"),
                             "requested_args": _visible(row.get("args", {})),
                             "status": result.get("ok"), "message": result.get("message"),
                             "result": _visible(result.get("data", {}))})

    observations = rows("observations.jsonl")
    frames = rows("frames.jsonl")
    bundle = EvidenceBundle(root, instruction, timeline, {}, sources, limitations)
    if not transcript:
        observation_ids = {r["id"] for r in observations}
    observation_rows = {r["id"]: r for r in observations}
    for obs in sorted(observation_ids):
        recorded = observation_rows.get(obs, {})
        info = {"timestamp": recorded.get("timestamp"), "source_ref": recorded.get("_source"),
                "image_status": "missing", "reason": "No original RGB frame"}
        bundle.observations[obs] = info
        rgb = [f for f in frames if f.get("observation_id") == obs and f.get("modality") == "rgb"]
        if len(rgb) != 1:
            info["reason"] = "Missing or ambiguous RGB frame association"
            continue
        frame = rgb[0]
        info.update(frame_id=frame["id"], frame_source_ref=frame["_source"])
        try:
            metadata = frame.get("metadata", {})
            if metadata.get("base64"):
                data = base64.b64decode(metadata["base64"], validate=True)
            elif frame.get("path"):
                path = _inside(root, root / frame["path"])
                data = path.read_bytes()
                sources[path.relative_to(root).as_posix()] = hashlib.sha256(data).hexdigest()
            else:
                raise ValueError("No embedded or local original RGB")
            with Image.open(BytesIO(data)) as img:
                mime = Image.MIME.get(img.format)
                if mime not in ("image/png", "image/jpeg", "image/webp"):
                    raise ValueError("Unsupported original image encoding")
                size = list(img.size)
                img.verify()
            bundle.images[obs] = (mime, data)
            info.update(image_status="available", reason=None, dimensions=size,
                        sha256=hashlib.sha256(data).hexdigest())
        except (OSError, ValueError) as exc:
            info.update(image_status="invalid", reason=type(exc).__name__)
    ids = [r["id"] for r in timeline] + list(bundle.observations)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate evidence IDs")
    return bundle
