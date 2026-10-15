"""Read-only action/observation and knowledge diagnostics for teaching changes.

Distances below describe commands, not obstacle clearance or target range. Actual
motion belongs to a complete observed window, including historical multi-action
windows. Axis counts do not automatically classify a move as viewpoint recovery.
"""
from collections import Counter
import math
import statistics


def number(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def command_motion(args):
    """Keep unavailable command measurements unknown; never invent coordinates."""
    values = {key: number(args.get(key)) for key in ("vx", "vy", "vz", "duration")}
    if any(value is None for value in values.values()) or values["duration"] <= 0:
        return None
    return {"planar_m": math.hypot(values["vx"], values["vy"]) * values["duration"],
            "up_m": values["vz"] * values["duration"]}


def distribution(values):
    known = [value for value in values if value is not None]
    return {"count": len(known), "mean": statistics.mean(known) if known else None,
            "peak": max(known) if known else None}


def analyze(events):
    calls = {}; actions = []; windows = []; pending = []; last_observation = None
    knowledge = Counter(); knowledge_receipts = []
    for event in events:
        payload = event.get("payload", {})
        if event.get("kind") == "assistant_turn":
            for call in payload.get("tool_calls", []):
                calls[call.get("tool_call_id")] = call
        elif event.get("kind") == "tool_result":
            call = calls.get(payload.get("tool_call_id"))
            if call is None:
                continue
            name = call.get("tool_name"); args = call.get("args") or {}
            if name == "knowledge.load":
                data = payload.get("data") or {}
                label = args.get("name", "unknown")
                knowledge[label + ":" + payload.get("status", "unknown")] += 1
                knowledge_receipts.append({"tool_call_id": call["tool_call_id"], "name": label,
                    "status": payload.get("status"), "reference_paths": args.get("reference_paths"),
                    "body_included": data.get("body_included"), "version": data.get("version")})
            if name not in {"uav.move_body", "uav.turn_relative"}:
                continue
            action = {"tool_call_id": call["tool_call_id"], "tool": name,
                "status": payload.get("status"),
                "requested_args": args, "executed_args": payload.get("executed_args"),
                "diagnostic": payload.get("diagnostic")}
            if name == "uav.move_body":
                action["requested_motion"] = command_motion(args)
                action["executed_command_motion"] = command_motion(payload.get("executed_args") or {})
            actions.append(action)
            if payload.get("status") not in {"rejected", "skipped"}:
                pending.append(action)
        elif event.get("kind") == "state_checkpoint":
            current = payload.get("observation_id")
            if payload.get("reason") == "physical_window_completed":
                motion = next((f.get("content", {}).get("last_window_uav_motion")
                    for f in payload.get("fragments", []) if f.get("name") == "motion"), None)
                windows.append({"observation_before_id": last_observation, "observation_after_id": current,
                    "actions": pending, "actual_motion_for_whole_window": motion})
                pending = []
            last_observation = current
    if pending:
        windows.append({"observation_before_id": last_observation, "observation_after_id": None,
            "actions": pending, "actual_motion_for_whole_window": None})
    attempted = [a for a in actions if a["status"] not in {"rejected", "skipped"}]
    moves = [a for a in attempted if a["tool"] == "uav.move_body"]
    lengths = [(a.get("requested_motion") or {}).get("planar_m") for a in moves]
    axes = Counter()
    for a in moves:
        for label, key, predicate in (("vertical", "vz", lambda v: v != 0),
                                      ("lateral", "vy", lambda v: v != 0),
                                      ("backward_body", "vx", lambda v: v < 0)):
            value = number(a["requested_args"].get(key))
            if value is not None and predicate(value):
                axes[label] += 1
    return {"knowledge_by_name": dict(knowledge), "knowledge_receipts": knowledge_receipts,
        "movement_choices": {"requested_planar": distribution(lengths),
            "requested_planar_over_20m": sum(v is not None and v > 20 for v in lengths),
            "requested_vertical_abs": distribution([abs(m["up_m"]) for a in moves
                if (m := a.get("requested_motion")) is not None]),
            "requested_yaw_abs": distribution([abs(v) for a in attempted if a["tool"] == "uav.turn_relative"
                and (v := number(a["requested_args"].get("yaw_deg"))) is not None]),
            "axis_choices": dict(axes),
            "notice": "20 m is a reporting threshold, not a safety limit. Command length is not actual displacement; axis counts do not prove recovery purpose."},
        "action_observation_timeline": windows,
        "max_calls_in_observed_window": max((len(w["actions"]) for w in windows if w["observation_after_id"]), default=None)}
