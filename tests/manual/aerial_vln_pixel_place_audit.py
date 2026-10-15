"""Sandbox admission of pixel/place capabilities with saved simulator evidence.

Original records and photos are read-only. Copies keep real acquisition metadata;
the invalid-depth fixture is explicitly derived and never called a real capture.
No navigation, detector, score or benchmark truth enters this sandbox.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from plugins.capabilities.pixel_depth.measure_pixel_depth.scripts.handler import measure_pixel_depth
from plugins.capabilities.place_memory.store import PlaceStore
from plugins.capabilities.place_memory.render import render_map
from plugins.capabilities.place_memory.save_place.scripts.handler import handle as save_place
from plugins.capabilities.place_memory.inspect_place.scripts.handler import handle as inspect_place
from plugins.capabilities.place_memory.list_places.scripts.handler import handle as list_places
from src.plugins.contracts.skill import SkillCapabilityFailure
from src.agent.context.request_assembler import _media_label


def admit_saved_evidence(root, previous_root, write, digest):
    """Exercise exact pixels, rejected coordinates and immutable place versions."""
    previous = json.loads((previous_root / "progress.json").read_text(encoding="utf-8"))
    original = Path(previous["cases"][0]["attempts"][-1]["result"]["run_dir"])
    sandbox = root / "capability_admission"
    sandbox.mkdir(parents=True, exist_ok=False)
    protected = {}
    records = {}
    for name in ("observations", "poses", "frames", "actions"):
        path = original / f"memory/{name}.jsonl"
        protected[str(path)] = digest(path)
        records[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Keep the first two genuine checkpoints; photo files must belong to this
    # sandbox so place memory cannot cite another task's original paths.
    observations = records["observations"][:2]
    ids = {r["id"] for r in observations}
    selected_frames = []
    for raw in records["frames"]:
        if raw.get("observation_id") not in ids:
            continue
        frame = copy.deepcopy(raw)
        metadata = frame.get("metadata", {})
        path = Path(frame.get("path") or metadata["artifact_path"])
        protected[str(path)] = digest(path)
        target = sandbox / "captures" / (frame["id"] + path.suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        metadata.pop("base64", None)
        metadata["artifact_path"] = str(target)
        frame["path"] = str(target)
        selected_frames.append(frame)
    for name, rows in {**records, "observations": observations, "frames": selected_frames}.items():
        path = sandbox / f"memory/{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    first_id = observations[0]["id"]
    frames = {f["modality"]: {"path": f["path"], "id": f["id"], "metadata": f["metadata"]}
              for f in selected_frames if f["observation_id"] == first_id}
    observation = {"observation_id": first_id, "frames": frames}
    depth = np.load(frames["depth"]["path"])
    valid = np.argwhere(np.isfinite(depth) & (depth > 0))
    if not len(valid):
        raise ValueError("Saved real depth has no valid point for admission")
    y, x = map(int, valid[len(valid) // 2])
    result = measure_pixel_depth(observation, first_id, x, y, run_dir=str(sandbox), tool_call_id="valid")
    checks = {"valid_exact_pixel": result.data["distance_status"] == "ok"
              and result.data["distance_m"] == round(float(depth[y, x]), 1),
        "ray_range_geometry": result.data["depth_geometry"] == "ray_range",
        "marker_dimensions": Image.open(result.artifact_refs[0]).size == Image.open(frames["rgb"]["path"]).size}
    # Invalid fixture only changes the selected depth element; valid neighbors
    # must not cause retargeting. Preserve the original saved depth file.
    derived = depth.copy(); derived[y, x] = np.nan
    invalid_path = sandbox / "derived_invalid_depth.npy"; np.save(invalid_path, derived)
    invalid_observation = copy.deepcopy(observation)
    invalid_observation["frames"]["depth"]["path"] = str(invalid_path)
    invalid_observation["frames"]["depth"]["metadata"]["artifact_path"] = str(invalid_path)
    invalid_result = measure_pixel_depth(invalid_observation, first_id, x, y,
        run_dir=str(sandbox), tool_call_id="derived_invalid")
    checks["invalid_center_not_retargeted"] = invalid_result.data["distance_m"] is None
    rejected = []
    for obs_id, px, py in [("wrong_observation", x, y), (first_id, -1, y), (first_id, depth.shape[1], y), (first_id, True, y)]:
        try:
            measure_pixel_depth(observation, obs_id, px, py)
        except ValueError as exc:
            rejected.append({"observation_id": obs_id, "pixel_x": px, "pixel_y": py, "reason": str(exc)})
    checks["observation_and_pixel_rejections"] = len(rejected) == 4
    store = PlaceStore(sandbox, "saved_evidence_admission")
    initial = store.update(first_id, first_id)
    first_map, _ = render_map(store, initial)
    first_path = sandbox / first_map["media_ref"]["ref"]
    first_hash = digest(first_path)
    graph = store.update(first_id, observations[-1]["id"])
    node = graph["nodes"][0]
    context = {"run_dir": str(sandbox), "instruction_id": store.instruction_id,
        "tool_call_id": "save_admission", "evidence_refs": [first_id]}
    args = {"place_id": node["id"], "summary": "Sandbox note from a saved observation; model judgment.",
        "landmarks": [], "evidence_refs": [first_id], "photo_observation_ids": [first_id]}
    saved = save_place({"runtime_context": context, "args": args})
    listing = {"runtime_context": context, "args": {"offset": 0, "limit": 10}}
    listed = list_places(listing)
    checks["saved_only_directory"] = listed.data["places"] == [{"place_id": node["id"], "summary": args["summary"]}]
    before = store.path.read_bytes()
    try:
        save_place({"runtime_context": {**context, "tool_call_id": "invalid_photo"},
                    "args": {**args, "photo_observation_ids": ["unknown_observation"]}})
    except SkillCapabilityFailure as exc:
        checks["photo_repair_no_write"] = (store.path.read_bytes() == before and
            first_id in exc.data["available_photo_observation_ids"])
    else:
        checks["photo_repair_no_write"] = False
    revised = {**args, "summary": "Revised sandbox uncertainty.", "photo_observation_ids": []}
    save_place({"runtime_context": {**context, "tool_call_id": "revision"}, "args": revised})
    checks["same_arguments_read_fresh_state"] = list_places(listing).data["places"][0]["summary"] == revised["summary"]
    inspected = inspect_place({"runtime_context": context, "args": {"place_id": node["id"]}})
    photo = inspected.data["photos"][0]
    photo_path = store.photo_path(photo["ref"])
    checks.update(map_size=Image.open(first_path).size == (1600, 1100),
        map_version_immutable=digest(first_path) == first_hash,
        saved_map_is_new=saved.model_inputs[0]["media_ref"]["ref"] != first_map["media_ref"]["ref"],
        original_photo_identity=digest(photo_path) == digest(frames["rgb"]["path"]),
        photo_observation_identity=photo["observation_id"] == first_id,
        historical_photo_role="HISTORY" in _media_label(inspected.model_inputs[0]["media_ref"], "HISTORY"),
        map_temporal_role=_media_label(saved.model_inputs[0]["media_ref"], "HISTORY").startswith("CONTEXT IMAGE "),
        original_evidence_unchanged=all(digest(Path(path)) == value for path, value in protected.items()))
    data = {"passed": all(checks.values()), "checks": checks, "protected_sources": protected,
        "notice": "Isolated saved-evidence capability calls; invalid-depth fixture is derived, not a new sensor capture.",
        "valid_pixel": {"args": {"observation_id": first_id, "pixel_x": x, "pixel_y": y},
            "data": result.data, "marker": result.artifact_refs},
        "derived_invalid_pixel": invalid_result.data, "rejections": rejected,
        "save_place": {"args": args, "data": saved.data, "model_inputs": saved.model_inputs},
        "inspect_place": {"data": inspected.data, "model_inputs": inspected.model_inputs}}
    write(root / "pixel_place_admission.json", data)
    return data


def audit_capability_records(run, events):
    """Verify successful pixel/photo receipts against original sensor files.

    Rejected model calls remain outcomes, not integrity failures. This reader
    checks coordinates, identity and bytes, never landmark or route semantics.
    """
    import hashlib
    receipts = [e["payload"] for e in events if e.get("kind") == "tool_result"
                and e["payload"].get("tool_name", "").startswith(("skill.pixel_depth.", "skill.place_memory."))]
    if not receipts:
        return {"available": True, "failures": [], "pixel_checks": [], "photo_checks": []}
    frame_path = run / "memory/frames.jsonl"
    frames = {f["id"]: f for f in map(json.loads, frame_path.read_text(encoding="utf-8").splitlines())}
    observations_path, poses_path = run / "memory/observations.jsonl", run / "memory/poses.jsonl"
    observations = {o["id"]: o for o in map(json.loads, observations_path.read_text(encoding="utf-8").splitlines())} if observations_path.exists() else {}
    poses = {p["id"]: p for p in map(json.loads, poses_path.read_text(encoding="utf-8").splitlines())} if poses_path.exists() else {}
    anchor_id = next((e.get("payload", {}).get("observation_id") for e in events if e.get("kind") == "state_checkpoint"), None)

    def measured(oid):
        pose = poses.get(observations.get(oid, {}).get("pose_ref", {}).get("id"), {}).get("pose")
        if pose is None:
            return None
        position = pose.get("position", pose)
        return ([float(position[k]) for k in ("x", "y", "z")],
                float(pose.get("euler_rad", {}).get("yaw", math.radians(pose.get("yaw_deg", 0)))))

    anchor = measured(anchor_id)
    def path_of(frame):
        path = Path(frame.get("path") or frame.get("metadata", {}).get("artifact_path"))
        return path if path.is_absolute() else run / path
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    calls = {}; current = None; failures = []; pixels = []; photos = []; unavailable = []
    receipt_path = run.parents[1] / "receipt.json"
    cleanup_receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {}
    depth_pruned = cleanup_receipt.get("cleanup_verified") is True and cleanup_receipt.get("temporary_depth_bytes", 0) > 0
    for event in events:
        p = event.get("payload", {})
        if event.get("kind") == "state_checkpoint":
            current = p.get("observation_id")
        elif event.get("kind") == "assistant_turn":
            calls.update({c["tool_call_id"]: c for c in p.get("tool_calls", [])})
        elif event.get("kind") == "tool_result" and p.get("status") == "ok":
            args = calls.get(p.get("tool_call_id"), {}).get("args", {})
            data = p.get("data") or {}
            if p.get("tool_name") == "skill.pixel_depth.measure_pixel_depth":
                oid = args.get("observation_id"); x = args.get("pixel_x"); y = args.get("pixel_y")
                relevant = [f for f in frames.values() if f.get("observation_id") == oid]
                rgb = next((f for f in relevant if f.get("modality") == "rgb"), None)
                depth = next((f for f in relevant if f.get("modality") == "depth"), None)
                problems = []
                depth_verified = None
                if oid != current or oid != data.get("observation_id"):
                    problems.append("pixel_observation_mismatch")
                if rgb is None:
                    problems.append("pixel_rgb_missing")
                else:
                    with Image.open(path_of(rgb)) as im:
                        size = im.size
                    if (type(x) is not int or type(y) is not int or not (0 <= x < size[0] and 0 <= y < size[1])
                            or [x, y] != [data.get("pixel_x"), data.get("pixel_y")]):
                        problems.append("pixel_coordinate_mismatch")
                    elif data.get("distance_status") == "ok" and depth:
                        if path_of(depth).is_file():
                            value = float(np.load(path_of(depth), mmap_mode="r")[y, x])
                            depth_verified = np.isfinite(value) and value > 0 and round(value, 1) == data.get("distance_m")
                            if not depth_verified:
                                problems.append("pixel_distance_mismatch")
                        elif depth_pruned:
                            unavailable.append({"tool_call_id": p.get("tool_call_id"), "reason": "raw_depth_pruned_after_report"})
                        else:
                            problems.append("pixel_depth_missing")
                for model_input in p.get("model_inputs") or []:
                    ref = model_input["media_ref"]
                    marker = run / ref["ref"]
                    if not marker.is_file() or ref.get("observation_id") != oid:
                        problems.append("pixel_marker_mismatch")
                pixels.append({"tool_call_id": p.get("tool_call_id"), "observation_id": oid,
                               "raw_depth_verified": bool(depth_verified) if depth_verified is not None else None,
                               "failures": problems})
                failures.extend(problems)
            elif p.get("tool_name") == "skill.place_memory.inspect_place":
                for photo in data.get("photos", []):
                    frame = frames.get(photo.get("frame_id"))
                    original = path_of(frame) if frame else None
                    copied = run / photo["ref"]
                    consistent = (frame is not None and frame.get("observation_id") == photo["observation_id"]
                        and frame.get("timestamp") == photo.get("timestamp")
                        and original.is_file() and copied.is_file() and sha(original) == sha(copied))
                    photos.append({"tool_call_id": p.get("tool_call_id"), "frame_id": photo.get("frame_id"),
                        "observation_id": photo.get("observation_id"), "consistent": bool(consistent)})
                    if not consistent:
                        failures.append("place_photo_reference_mismatch")
                    historical = measured(photo["observation_id"])
                    # Legacy receipts omit spatial fields. Do not retroactively
                    # label those historical requests as newly enriched evidence.
                    if "position_right_forward_up_m" in photo and anchor and historical:
                        origin, initial_yaw = anchor
                        point, yaw = historical
                        dx, dy = point[0] - origin[0], point[1] - origin[1]
                        expected = [-dx * math.sin(initial_yaw) + dy * math.cos(initial_yaw),
                                    dx * math.cos(initial_yaw) + dy * math.sin(initial_yaw), point[2] - origin[2]]
                        actual = photo.get("position_right_forward_up_m")
                        heading = photo.get("uav_yaw_relative_initial_rad")
                        matches = (isinstance(actual, list) and len(actual) == 3 and heading is not None
                                   and all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(actual, expected))
                                   and math.isclose(heading, yaw - initial_yaw, abs_tol=1e-6))
                        photos[-1]["historical_pose_consistent"] = matches
                        if not matches:
                            failures.append("place_photo_historical_pose_mismatch")
    return {"available": True, "failures": failures, "pixel_checks": pixels, "photo_checks": photos,
            "unavailable_checks": unavailable}


def audit_place_timeline(events):
    """Replay tool receipts in order; never consult the final mutable graph."""
    calls, saved, rows, failures = {}, {}, [], []
    previous_reads = {}
    revision = 0
    for event in events:
        payload = event.get("payload", {})
        if event.get("kind") == "assistant_turn":
            calls.update({c["tool_call_id"]: c for c in payload.get("tool_calls", [])})
        if event.get("kind") != "tool_result":
            continue
        name = payload.get("tool_name", "")
        if not name.startswith("skill.place_memory."):
            continue
        call = calls.get(payload.get("tool_call_id"))
        if call is None:
            failures.append("place_result_without_model_call")
            continue
        args, data = call.get("args", {}), payload.get("data") or {}
        row = {"tool_call_id": payload["tool_call_id"], "tool": name,
               "status": payload.get("status"), "args": args, "data": data,
               "saved_ids_before": sorted(saved), "failures": []}
        if payload.get("status") == "ok":
            pid = args.get("place_id")
            if name.endswith(".save_place"):
                saved[pid] = {"summary": args["summary"], "landmarks": args["landmarks"], "evidence_refs": args["evidence_refs"]}
                revision += 1
            elif name.endswith(".list_places"):
                ordered = sorted(saved, key=lambda value: int(value[1:]))
                offset, limit = args["offset"], args["limit"]
                expected = [{"place_id": key, "summary": saved[key]["summary"] or "; ".join(saved[key]["landmarks"])}
                            for key in ordered[offset:offset + limit]]
                if data.get("places") != expected or data.get("total") != len(saved):
                    row["failures"].append("directory_differs_from_prior_saves")
                if payload.get("model_inputs"):
                    row["failures"].append("directory_added_images")
            elif name.endswith(".inspect_place") and pid in saved:
                if any(data.get(key) != value for key, value in saved[pid].items()):
                    row["failures"].append("inspection_differs_from_prior_save")
            if name.endswith((".list_places", ".inspect_place")):
                key = (name, json.dumps(args, sort_keys=True))
                row["repeated_after_save"] = key in previous_reads and previous_reads[key] < revision
                previous_reads[key] = revision
        failures.extend(row["failures"])
        rows.append(row)
    return {"calls": rows, "failures": failures, "saved_places": len(saved),
            "notice": "Receipt consistency only. Revisit recognition, correction quality and avoided search require human review."}


def write_place_report(root, pairs, write):
    """Write per-case autonomous tool coverage and time-local evidence review."""
    import html
    cases = []
    for pair in pairs:
        result = pair.get("current") or {}
        run = result.get("run_dir")
        path = Path(run) / "memory/agent_transcript.jsonl" if run else None
        if path is None or not path.is_file():
            cases.append({"index": pair["index"], "available": False})
            continue
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        cases.append({"index": pair["index"], "available": True, "source": str(path), **audit_place_timeline(events)})
    coverage = {name: sum(any(c["tool"].endswith("." + name) for c in row.get("calls", [])) for row in cases)
                for name in ("save_place", "list_places", "inspect_place")}
    data = {"cases": cases, "coverage": coverage, "selected_cases": len(pairs),
            "failures": [item for row in cases for item in row.get("failures", [])],
            "notice": "Autonomous navigation calls only; directed admission excluded. Low use means insufficient coverage, not proven benefit. Current v3/optional composite differs from historical protocol and prompts; no isolated causal attribution."}
    write(root / "place_memory_audit.json", data)
    (root / "place_memory_audit.html").write_text('<!doctype html><meta charset="utf-8"><title>Place memory audit</title><h1>Place memory audit</h1><pre style="white-space:pre-wrap">' + html.escape(json.dumps(data, ensure_ascii=False, indent=2)) + '</pre>', encoding="utf-8")
    return data
