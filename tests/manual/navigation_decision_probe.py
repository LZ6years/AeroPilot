"""Frozen navigation decision probes; no physical or perception tools execute.

prepare freezes historical prefixes and counterfactual text observations. run
requests up to three decisions per sample, supplying only local knowledge or
plan-storage acknowledgements. Other calls end the sample for human review.
Synthetic observations test conditional reasoning, not visual accuracy. Historical
prefixes retain their original calls and results, including retired parameters.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent.planning.model_action_planner import ModelActionPlanner
from src.agent.planning.tool_contract_validator import validate_native_turn
from src.agent.transcript import AgentTranscript, AgentToolCall, AssistantTurn, ToolResult
from src.memory.context.contracts import (
    INSTRUCTION_ENVELOPE_SCHEMA,
    STATE_CHECKPOINT_SCHEMA,
    TRANSCRIPT_SCHEMA,
    ContextFragment,
    InstructionEnvelope,
    StateCheckpoint,
    checkpoint_fingerprint,
)
from src.memory.context.media_refs import image_media_ref
from src.memory.stores import MemoryEventWriter
from src.llm.providers.openai_compatible import public_response_snapshot
from src.plugins import PluginRegistry
from src.plugins.skills.knowledge_service import KnowledgeService
from src.agent.config import load_model_config
from src.runtime.context.checkpoints import UnsupportedRunSchema
from src.tools.registry import default_registry
from tests.manual.flight_experience_batch_probe import hashes
from tests.manual.tool_contract_behavior_probe import build_transcript


SYNTHETIC = (
    ("inspect_closer", "Identify the marked equipment; stop when its identity is clear.",
     "An uncertain equipment candidate is distant. Its distinguishing label is too small. The approach is unobstructed and there is ample separation."),
    ("inspect_side", "Identify the marked equipment; stop when its identity is clear.",
     "An uncertain equipment candidate is close. Moving forward would enter its occupied space; a clear lateral path can expose its obscured distinguishing label."),
    ("candidate_rejected", "Find the service entrance.",
     "A closer view of the prior candidate shows a permanently sealed panel, contradicting the entrance interpretation. Another unexplored corridor is visible and unobstructed."),
    ("temporary_occlusion", "Pass the identified marker on its left and continue along the corridor.",
     "The identified marker moved from ahead to behind after measured forward translation on its left. It is now outside view; the same corridor continues clearly ahead."),
    ("occluded_identity", "Inspect the identified equipment's label.",
     "Equipment identity was established in earlier clear views. A support column temporarily hides its label after a lateral move; no contrary identity evidence appeared. A small safe lateral movement can restore the view. The label has not yet been read."),
    ("explicit_altitude", "Ascend a total of 12 metres, then stop.",
     "Measured ascent is 5 metres. Vertical clearance is available. The ground reference has left the forward view; this does not change the requested ascent."),
    ("motion_done", "Turn right by 12 degrees, then stop.",
     "The completed action has measured yaw change +12 degrees; vehicle stable. No landmark requirement was given."),
    ("node_not_reached", "Turn left at the corridor junction, then continue.",
     "A junction is visible ahead. Continuous side walls still border the current position; the left opening has not been reached. Forward approach is clear."),
    ("known_corridor", "Continue along the clear corridor to the next marker.",
     "The corridor is straight and unobstructed. The next marker is visibly far ahead. Prior forward actions produced consistent translation and visual change; no intervening decision point is apparent."),
    ("facing_done", "Face the identified sign, then stop.",
     "The identified sign is clearly visible in the forward region and its identity is unambiguous. No exact centering or proximity was requested."),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(output: Path, source: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    samples, evidence = [], {}
    for number, obs, name in ((1, "obs_0002", "height_observation"),
                              (2, "obs_0003", "motion_route_relation"),
                              (3, "obs_0008", "route_continuation"),
                              (6, "obs_0001", "direction_only")):
        case = next((source / "cases").glob(f"{number:02d}_*"))
        audit_path = next(case.rglob("audit.json"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        prefix = []
        for event in audit["agent_transcript"]:
            prefix.append(event)
            if event["kind"] == "state_checkpoint" and event["payload"]["observation_id"] == obs:
                break
        else:
            raise ValueError(f"Missing frozen checkpoint: {name}/{obs}")
        selected = {e["payload"]["observation_id"] for e in prefix if e["kind"] == "state_checkpoint"}
        frames = {f["observation_id"]: f["display_path"] for f in audit["frame_assets"]
                  if f["modality"] == "rgb" and f["observation_id"] in selected}
        if set(frames) != selected:
            raise ValueError("Missing original RGB in historical prefix")
        evidence[str(audit_path.resolve())] = digest(audit_path)
        for path in frames.values():
            evidence[path] = digest(Path(path))
        samples.append({"name": name, "kind": "historical_prefix", "events": prefix,
                        "frames": frames, "source_audit": str(audit_path.resolve())})
    samples += [{"name": name, "kind": "synthetic_counterfactual", "instruction": instruction,
                 "scene": "Synthetic observation for conditional reasoning only: " + scene,
                 "image": None} for name, instruction, scene in SYNTHETIC]
    write(output / "manifest.json", {"samples": samples, "evidence_sha256": evidence,
          "source_hashes": hashes(), "model": load_model_config().redacted_summary(),
          "tools_executed": False, "max_rounds": 3})


def historical_transcript(sample: dict, output: Path) -> AgentTranscript:
    """Rehydrate a V3 prefix, never audit truth, future events, or V2 state."""
    transcript = AgentTranscript(MemoryEventWriter(output))
    envelope: InstructionEnvelope | None = None
    for event in sample["events"]:
        if event.get("schema_version") != TRANSCRIPT_SCHEMA:
            raise UnsupportedRunSchema("historical probe accepts agent_transcript_v3 only")
        kind, payload = event["kind"], copy.deepcopy(event["payload"])
        if kind == "instruction_envelope":
            if payload.get("schema_version") != INSTRUCTION_ENVELOPE_SCHEMA:
                raise UnsupportedRunSchema("missing instruction_envelope_v3")
            envelope = InstructionEnvelope(
                instruction_id="offline_prefix",
                protocol_context=str(payload.get("protocol_context") or ""),
                identity=dict(payload.get("identity") or {}),
                user_preferences=dict(payload.get("user_preferences") or {}),
                knowledge_catalog=tuple(payload.get("knowledge_catalog") or ()),
                static_module_guidance=tuple(payload.get("static_module_guidance") or ()),
            )
        elif kind == "user_instruction":
            if envelope is None:
                raise UnsupportedRunSchema("instruction envelope must precede user instruction")
            transcript.start_instruction(payload["instruction"], "offline_prefix", envelope)
        elif kind == "assistant_turn":
            transcript.append_assistant(AssistantTurn(payload["turn_id"], payload["public_intent"],
                [AgentToolCall(**call) for call in payload["tool_calls"]],
                payload.get("public_intent_source", "provider_content")))
        elif kind == "tool_result":
            transcript.append_tool_result(ToolResult(**payload))
        elif kind == "state_checkpoint":
            if payload.get("schema_version") != STATE_CHECKPOINT_SCHEMA:
                raise UnsupportedRunSchema("historical probe accepts state_checkpoint_v3 only")
            obs = str(payload["observation_id"])
            fragments: list[ContextFragment] = []
            for raw_fragment in payload.get("fragments", []):
                inputs = tuple(copy.deepcopy(raw_fragment.get("media_refs") or ()))
                if raw_fragment.get("name") == "image":
                    # Attach only the exact RGB associated with this recorded
                    # observation. Encoding remains transient in projection.
                    ref = image_media_ref(
                        sample["frames"][obs],
                        observation_id=obs,
                        frame_id=f"recorded_{obs}",
                        role="current_observation",
                        usage="Recorded image for this historical observation.",
                    )
                    inputs = (ref.to_model_input(),)
                fragments.append(ContextFragment(
                    str(raw_fragment.get("name") or "unknown"),
                    dict(raw_fragment.get("content") or {}),
                    inputs,
                ))
            fragment_tuple = tuple(fragments)
            transcript.append_state(StateCheckpoint(
                checkpoint_id=str(payload.get("checkpoint_id") or "checkpoint_0001"),
                reason=payload.get("reason", "physical_window_completed"),
                observation_id=obs,
                fragments=fragment_tuple,
                fingerprint=checkpoint_fingerprint(obs, fragment_tuple),
            ))
        elif kind == "protocol_feedback":
            transcript.append_protocol_feedback(payload["message"])
    return transcript


def run(output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    config = load_model_config()
    if config.redacted_summary() != manifest["model"] or hashes() != manifest["source_hashes"]:
        raise ValueError("Frozen model or source changed; prepare a new experiment")
    for name, expected in manifest["evidence_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError("Frozen evidence changed")
    plugins = PluginRegistry.from_config()
    registry = default_registry(plugins)
    rows = []
    for sample in manifest["samples"]:
        directory = output / sample["name"]
        if (directory / "result.json").exists():
            rows.append(json.loads((directory / "result.json").read_text(encoding="utf-8")))
            continue
        transcript = (historical_transcript(sample, directory) if sample["kind"] == "historical_prefix"
                      else build_transcript(sample, directory, plugins)[0])
        knowledge = KnowledgeService(plugins.skill_catalog)
        planner = ModelActionPlanner(config)
        task_plan = None
        row = {"name": sample["name"], "kind": sample["kind"], "rounds": [], "semantic_review": "pending"}
        for index in range(manifest["max_rounds"]):
            started = time.monotonic()
            result = {}
            try:
                turn = planner.plan_turn(
                    transcript,
                    registry,
                    current_task_state={
                        "task_plan": copy.deepcopy(task_plan),
                        "loaded_knowledge": knowledge.loaded_index(),
                    },
                )
                validate_native_turn(turn, registry, allowed_names={s.name for s in registry.model_specs()},
                    latest_observation={"observation_id": transcript.latest_observation_id()},
                    prior_call_ids={c.tool_call_id for e in transcript.items if e.kind == "assistant_turn" for c in e.payload.tool_calls})
                result.update(status="valid", turn=turn.to_dict())
            except Exception as exc:
                result.update(status="failed", error_type=type(exc).__name__,
                              error=str(exc), diagnostic=getattr(exc, "diagnostic", None))
                turn = None
            media_refs = []
            for media in planner.last_exchange.get("media", []):
                entry = {k: v for k, v in media.items() if k != "data"}
                if isinstance(media.get("data"), bytes):
                    path = directory / "media" / (media["sha256"] + media["suffix"])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(media["data"])
                    entry["path"] = str(path.resolve())
                media_refs.append(entry)
            result.update(elapsed_sec=time.monotonic()-started,
                          usage=planner.last_response.get("usage"),
                          request=planner.last_request, request_media=media_refs,
                          response=public_response_snapshot(planner.last_response))
            write(directory / f"round_{index+1}.json", result)
            row["rounds"].append({k: v for k, v in result.items() if k not in ("request", "response")})
            if turn is None or any(c.tool_name not in {"knowledge.load", "runtime.update_plan"} for c in turn.tool_calls):
                break
            transcript.append_assistant(turn)
            for call in turn.tool_calls:
                data = (knowledge.load(**call.args) if call.tool_name == "knowledge.load" else {"plan": call.args})
                transcript.append_tool_result(ToolResult(call.tool_call_id, call.tool_name, "ok",
                    "Offline local guidance or plan storage; no flight executed.", data=data,
                    requested_args=call.args, executed_args=call.args))
            if turn.tool_calls[0].tool_name == "runtime.update_plan":
                task_plan = copy.deepcopy(turn.tool_calls[0].args)
        write(directory / "result.json", row)
        rows.append(row)
        write(output / "results.json", rows)
        print(json.dumps({"name": row["name"], "rounds": len(row["rounds"]),
                          "status": row["rounds"][-1]["status"]}), flush=True)
    body = "".join(f'<h2>{html.escape(r["name"])}</h2><pre>{html.escape(json.dumps(r,ensure_ascii=False,indent=2))}</pre>' for r in rows)
    (output / "index.html").write_text('<meta charset="utf-8"><title>Navigation decision probes</title><h1>Offline decisions; semantic review required</h1>'+body, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=ROOT / "tests/manual_outputs/flight_exp_5x2_20260927")
    args = parser.parse_args()
    prepare(args.output, args.source) if args.command == "prepare" else run(args.output)
