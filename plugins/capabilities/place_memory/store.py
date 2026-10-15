"""Task-local observed-place memory; no flight control or benchmark truth.

Snapshots are atomically replaced; rendered versions and original photos are
immutable. A task owns its store through (run directory, instruction id).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

DEFAULTS = {"merge_horizontal_m": 10.0, "merge_vertical_m": 5.0, "max_photos": 2}


def annotation_status(node):
    """Describe saved coverage without deciding whether new views are useful.

    Legacy notes lack a save boundary; their coverage remains unknown instead
    of inventing a last-save observation or claiming fresh evidence was saved.
    """
    annotated = bool(node["summary"] or node["landmarks"])
    saved_count = node.get("annotation_visit_count")
    added = max(0, len(node["visits"]) - saved_count) if annotated and saved_count is not None else None
    return {"is_annotated": annotated,
            "last_saved_observation_id": node.get("last_saved_observation_id"),
            "new_observations_since_save": added}


def place_summary(graph, node):
    """Compact place state shared by checkpoints, tool results and map labels."""
    photos = list(dict.fromkeys(p["observation_id"] for p in node["photos"]))
    return {"id": node["id"], "summary": node["summary"], "landmarks": node["landmarks"],
            "latest_observation_id": node["visits"][-1],
            "photo_observation_ids": photos, **annotation_status(node)}


def read_records(path):
    """Read complete JSONL records; a crash may leave only the last line torn."""
    if not Path(path).exists():
        return []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return records


def atomic_json(path, data):
    """Keep the previous complete snapshot if writing/replacement fails."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.02 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


class InvalidPlacePhotos(ValueError):
    """Invalid selection with task-local options for a corrective tool call."""

    def __init__(self, place_id, selected, available):
        super().__init__("photos must be up to two distinct original RGB observations at this place")
        self.data = {"place_id": place_id,
                     "invalid_photo_observation_ids": [oid for oid in selected if oid not in available],
                     "available_photo_observation_ids": available[-10:],
                     "available_photo_total": len(available),
                     "repair": "Use inspect_place with this place_id for all available photo observation IDs; select at most two distinct IDs."}


def available_photo_ids(graph, node):
    """Return observation choices in visit order, independent of saved selection."""
    return [oid for oid in node["visits"] if graph["observations"].get(oid, {}).get("photos")]


def historical_photo(graph, photo):
    """Recover legacy metadata only from the exact historical observation."""
    oid = photo["observation_id"]
    observation = graph["observations"].get(oid, {})
    measured = observation if "position" in observation else next(
        (p for p in graph.get("trajectory", []) if p.get("observation_id") == oid), {})
    return {**photo, "position_right_forward_up_m": measured.get("position"),
            "uav_yaw_relative_initial_rad": measured.get("yaw_rad")}


class PlaceStore:
    """Own one task's annotations and deterministic geometry reconstruction."""

    def __init__(self, run_dir, instruction_id, settings=None):
        if not instruction_id:
            raise ValueError("place memory requires an instruction id")
        self.run_dir = Path(run_dir).resolve()
        self.instruction_id = str(instruction_id)
        self.settings = {**DEFAULTS, **(settings or {})}
        if set(self.settings) != set(DEFAULTS):
            raise ValueError("unknown place-memory setting")
        if any(not math.isfinite(float(self.settings[k])) or float(self.settings[k]) <= 0
               for k in ("merge_horizontal_m", "merge_vertical_m")) or self.settings["max_photos"] != 2:
            raise ValueError("invalid place-memory thresholds/photo budget")
        scope = hashlib.sha256(self.instruction_id.encode()).hexdigest()[:16]
        self.directory = self.run_dir / "contexts/place_memory" / scope
        self.path = self.directory / "graph.json"

    def load(self):
        if not self.path.exists():
            raise ValueError("place map has not been built for this task")
        graph = json.loads(self.path.read_text(encoding="utf-8"))
        if graph["instruction_id"] != self.instruction_id:
            raise ValueError("place memory task mismatch")
        return graph

    def photo_path(self, ref):
        path = (self.run_dir / ref).resolve()
        if self.run_dir not in path.parents or not path.is_file():
            raise ValueError("photo is missing or outside this task's run")
        return path

    def list_saved_places(self, offset, limit):
        """Read the current saved-only directory without producing artifacts."""
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("offset must be nonnegative and limit must be 1..20")
        nodes = sorted((n for n in self.load()["nodes"] if annotation_status(n)["is_annotated"]),
                       key=lambda n: int(n["id"][1:]))
        end = offset + limit
        return {"places": [{"place_id": n["id"], "summary": n["summary"] or "; ".join(n["landmarks"])}
                           for n in nodes[offset:end]], "total": len(nodes),
                "next_offset": end if end < len(nodes) else None,
                "evidence_origin": "historical_memory"}

    def update(self, start_observation_id, latest_observation_id):
        """Rebuild from complete run evidence, limited to the active anchor/end."""
        records = read_records(self.run_dir / "memory/observations.jsonl")
        ids = [r["id"] for r in records]
        if start_observation_id not in ids or latest_observation_id not in ids:
            raise ValueError("task observation anchors are missing")
        first, last = ids.index(start_observation_id), ids.index(latest_observation_id)
        if last < first:
            raise ValueError("observation boundary precedes task start")
        records = records[first:last + 1]
        poses = {r["id"]: r for r in read_records(self.run_dir / "memory/poses.jsonl")}
        frames = {r["id"]: r for r in read_records(self.run_dir / "memory/frames.jsonl")}
        actions = read_records(self.run_dir / "memory/actions.jsonl")
        old = self.load() if self.path.exists() else {"nodes": [], "committed_calls": {}}
        annotations = {n["id"]: n for n in old["nodes"]}
        graph = {"instruction_id": self.instruction_id, "settings": self.settings,
                 "nodes": [], "edges": [], "trajectory": [], "observations": {},
                 "committed_calls": old.get("committed_calls", {})}
        previous = None
        for observation in records:
            raw = poses.get(observation.get("pose_ref", {}).get("id"))
            if not raw:
                raise ValueError("observation has no measured pose")
            if raw.get("coordinate_frame", "canonical_unspecified") not in {
                    "canonical_unspecified", "unrealcv_world_m", "uav_flow_initial_local_canonical_m"}:
                raise ValueError("place map requires canonical positive-Z-up poses")
            pose = raw["pose"]
            position = pose.get("position", pose)
            point = [float(position[k]) for k in ("x", "y", "z")]
            yaw = float(pose.get("euler_rad", {}).get("yaw", math.radians(pose.get("yaw_deg", 0))))
            if not all(math.isfinite(v) for v in [*point, yaw]):
                raise ValueError("nonfinite observed pose")
            if "origin" not in graph:
                graph.update(origin=point, initial_yaw_rad=yaw)
            origin = graph["origin"]
            dx, dy = point[0] - origin[0], point[1] - origin[1]
            a = graph["initial_yaw_rad"]
            # Forward is map-up; right is map-right. Height stays relative to start.
            local = [-dx * math.sin(a) + dy * math.cos(a),
                     dx * math.cos(a) + dy * math.sin(a), point[2] - origin[2]]
            graph["trajectory"].append({"observation_id": observation["id"], "position": local,
                                        "yaw_rad": yaw - a, "timestamp": observation["timestamp"]})
            nearby = [n for n in graph["nodes"]
                      if math.dist(n["position"][:2], local[:2]) <= self.settings["merge_horizontal_m"]
                      and abs(n["position"][2] - local[2]) <= self.settings["merge_vertical_m"]]
            node = min(nearby, key=lambda n: math.dist(n["position"], local)) if nearby else None
            if node is None:
                identity = f"P{len(graph['nodes']) + 1:03d}"
                saved = annotations.get(identity, {})
                node = {"id": identity, "position": local, "visits": [], "photos": [], "actions": [],
                        "summary": saved.get("summary", ""), "landmarks": saved.get("landmarks", []),
                        "evidence_refs": saved.get("evidence_refs", []),
                        "selected_photos": saved.get("selected_photos", []),
                        "last_saved_observation_id": saved.get("last_saved_observation_id"),
                        "annotation_visit_count": saved.get("annotation_visit_count")}
                graph["nodes"].append(node)
            node["visits"].append(observation["id"])
            node["actions"].extend({"id": r["id"], "tool": r["tool"], "args": r.get("args", {}),
                                     "ok": r.get("result", {}).get("ok"), "timestamp": r["timestamp"]}
                                    for r in actions if r["timestamp"] <= observation["timestamp"]
                                    and r["timestamp"] > (previous["timestamp"] if previous else observation["timestamp"])
                                    and r["tool"].startswith("uav."))
            photos = []
            for ref in observation.get("frame_refs", []):
                frame = frames.get(ref["id"], {})
                if frame.get("modality") not in {"rgb", "scene", "color"}:
                    continue
                value = frame.get("path") or frame.get("metadata", {}).get("artifact_path")
                if value:
                    path = Path(value)
                    path = path if path.is_absolute() else self.run_dir / path
                    if path.is_file() and self.run_dir in path.resolve().parents:
                        photos.append({"ref": path.resolve().relative_to(self.run_dir).as_posix(),
                                       "observation_id": observation["id"], "frame_id": frame["id"],
                                       "timestamp": observation["timestamp"]})
            graph["observations"][observation["id"]] = {
                "place_id": node["id"], "photos": photos, "position": local,
                "yaw_rad": yaw - a, "timestamp": observation["timestamp"]}
            node["photos"].extend(photos[:max(0, 2 - len(node["photos"]))])
            if previous and previous["place_id"] != node["id"]:
                graph["edges"].append({"from": previous["place_id"], "to": node["id"],
                                       "observation_id": observation["id"]})
            previous = {"place_id": node["id"], "timestamp": observation["timestamp"]}
            graph["current_place_id"] = node["id"]
        for node in graph["nodes"]:
            if node["selected_photos"]:
                node["photos"] = [p for oid in node["selected_photos"]
                                  for p in graph["observations"].get(oid, {}).get("photos", [])[:1]][:2]
        # Retain measured intermediate action poses too, not just the sparse
        # observation sites. Never extend beyond this checkpoint's boundary.
        trajectory = []
        a = graph["initial_yaw_rad"]
        for raw in poses.values():
            if not records[0]["timestamp"] <= raw["timestamp"] <= records[-1]["timestamp"]:
                continue
            if raw.get("coordinate_frame", "canonical_unspecified") not in {
                    "canonical_unspecified", "unrealcv_world_m", "uav_flow_initial_local_canonical_m"}:
                raise ValueError("trajectory contains a noncanonical pose")
            pose = raw["pose"]; p = pose.get("position", pose)
            dx, dy = float(p["x"]) - origin[0], float(p["y"]) - origin[1]
            yaw = float(pose.get("euler_rad", {}).get("yaw", math.radians(pose.get("yaw_deg", 0))))
            local = [-dx * math.sin(a) + dy * math.cos(a), dx * math.cos(a) + dy * math.sin(a), float(p["z"]) - origin[2]]
            if not all(math.isfinite(v) for v in [*local, yaw]):
                raise ValueError("nonfinite trajectory")
            trajectory.append({"observation_id": raw.get("observation_id"), "position": local,
                               "yaw_rad": yaw - a, "timestamp": raw["timestamp"]})
        graph["trajectory"] = sorted(trajectory, key=lambda p: p["timestamp"])
        atomic_json(self.path, graph)
        return graph

    def annotate(self, args, context):
        graph = self.load()
        call_id = context["tool_call_id"]
        digest = hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()
        if call_id in graph["committed_calls"]:
            if graph["committed_calls"][call_id] != digest:
                raise ValueError("reused call id has different annotation")
            return graph
        node = next((n for n in graph["nodes"] if n["id"] == args["place_id"]), None)
        if node is None:
            raise ValueError("unknown place id in this task")
        refs = args["evidence_refs"]
        if not refs or not set(refs).issubset(context.get("evidence_refs", [])):
            raise ValueError("annotation requires prior environmental evidence from this task")
        photos = args.get("photo_observation_ids", [])
        available = available_photo_ids(graph, node)
        if len(photos) > 2 or len(set(photos)) != len(photos) or not set(photos).issubset(available):
            raise InvalidPlacePhotos(node["id"], photos, available)
        # This boundary records the visits present when the note was saved,
        # not a claim that every photo or landmark has been understood.
        node.update(summary=args["summary"], landmarks=args.get("landmarks", []),
                    evidence_refs=refs, last_saved_observation_id=node["visits"][-1],
                    annotation_visit_count=len(node["visits"]))
        if photos:
            node["selected_photos"] = photos
            node["photos"] = [graph["observations"][oid]["photos"][0] for oid in photos]
        graph["committed_calls"][call_id] = digest
        atomic_json(self.path, graph)
        return graph
