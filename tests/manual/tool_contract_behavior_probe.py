"""Paired synthetic selection audit; requests model turns but NEVER executes tools.

Run --phase before, then --phase after --baseline <before/report.json>.
Each phase makes at most ten requests, with no transport retries. Corpus,
provider settings fingerprint and contracts are checked before the after phase.
Fresh output directories preserve snapshots; no credentials or hidden reasoning
are exported. This is a small selection diagnostic, not a navigation benchmark.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.agent.planning.model_action_planner import ModelActionPlanner
from src.agent.planning.tool_contract_validator import validate_native_turn
from src.agent.transcript import AgentToolCall, AgentTranscript, AssistantTurn, ToolResult
from src.memory.context.contracts import (
    ContextFragment,
    InstructionEnvelope,
    StateCheckpoint,
    checkpoint_fingerprint,
)
from src.memory.context.documents import ContextDocuments
from src.memory.context.media_refs import image_media_ref
from src.memory.context.modules.image import CURRENT_IMAGE_USAGE
from src.memory.stores import MemoryEventWriter
from src.plugins import PluginRegistry
from src.agent.config import load_model_config
from src.uav.config import load_uav_safety_config
from src.tools.registry import default_registry


CASES = (
    ("simple_turn_right", "Turn right by 15 degrees.", "uav.turn_relative"),
    ("simple_turn_left", "Turn left by 10 degrees.", "uav.turn_relative"),
    ("simple_translate", "Move forward at 0.5 m/s for 1 second.", "uav.move_body"),
    ("complex_ordered", "Find the red building, fly around its left side, then find its rear entrance and stop facing it.", "runtime.update_plan"),
    ("complex_search", "Search for the white van, approach it while avoiding obstacles, inspect both sides, and finish above its rear bumper.", "runtime.update_plan"),
    ("complex_route", "Follow the road to the bridge, cross it, locate the second junction, then turn left and stop beside the blue gate.", "runtime.update_plan"),
    ("explicit_knowledge", "Read the available orbit_and_surround guidance before any flight. Reading that guidance is my current request.", "knowledge.load"),
    ("explicit_distance", "Measure the current distance to the white truck visible on the right. Do not move the vehicle.", "skill.visual_grounding.measure_visible_depth"),
    ("supported_completion", "Turn right by 15 degrees, then finish.", "runtime.report_completed"),
    ("unsafe_blocked", "Move forward one metre if it can be done safely.", "runtime.report_blocked"),
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def without_descriptions(value):
    """Strip schema annotations, not an argument named description or extensions."""
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    schema = copy.deepcopy(value)
    schema.pop("description", None)
    for key in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
        if isinstance(schema.get(key), dict):
            schema[key] = {name: without_descriptions(item) for name, item in schema[key].items()}
    for key in ("items", "additionalItems", "additionalProperties", "contains", "propertyNames",
                "not", "if", "then", "else", "unevaluatedProperties", "unevaluatedItems"):
        if key in schema:
            item = schema[key]
            schema[key] = ([without_descriptions(child) for child in item] if isinstance(item, list)
                           else without_descriptions(item))
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        if isinstance(schema.get(key), list):
            schema[key] = [without_descriptions(item) for item in schema[key]]
    return schema


def tool_snapshot(registry):
    return [{"name": name, "wire_name": registry.get(name).wire_name,
             "kind": registry.get(name).kind, "description": registry.get(name).description,
             "runtime": registry.get(name).runtime.to_dict(), "schema": registry.get(name).input_schema,
             "input_examples": list(registry.get(name).input_examples),
             "provider": registry.get(name).to_provider_tool()} for name in registry.names()]


def structural_contracts(snapshot):
    return [{key: without_descriptions(tool[key]) if key == "schema" else copy.deepcopy(tool[key])
             for key in ("name", "wire_name", "kind", "runtime", "schema")}
            for tool in snapshot]


def source_snapshot():
    """Capture public prompt sources only, never .env or private plugin settings."""
    paths = [Path("memory/FLYAGENT.md"), Path("memory/USER.md"), Path("AGENT.md"),
             *Path("src/agent/planning/prompts").glob("*.txt"),
             *Path("plugins/knowledge/uav_flight").rglob("*.md")]
    return {path.as_posix(): path.read_text(encoding="utf-8") for path in sorted(paths)}


def corpus():
    """Fixed synthetic cases; the recorded fixture is an illustration, not a live view."""
    fixture = Path("tests/fixtures/visual_description_images/frame_000001.png")
    return [{"name": name, "instruction": instruction, "expected": expected,
             "image": str(fixture) if name == "explicit_distance" else None,
             "scene": ("Synthetic safety observation: immediate flight area is obstructed in every direction; "
                       "sensors cannot establish any safe movement or evidence-producing recovery."
                       if name == "unsafe_blocked" else
                       "Synthetic offline observation: vehicle airborne and stable, immediate flight area clear. "
                       "No target position is established unless shown in the attached recorded illustration.")}
            for name, instruction, expected in CASES]


def build_transcript(case, output, plugins):
    transcript = AgentTranscript(MemoryEventWriter(output))
    documents = ContextDocuments()
    catalog = tuple(
        {"name": skill.name, "description": skill.description}
        for skill in plugins.skill_catalog.skills if skill.capability is None
    )
    transcript.start_instruction(
        case["instruction"],
        "synthetic_instruction",
        InstructionEnvelope(
            instruction_id="synthetic_instruction",
            protocol_context="",
            identity=documents.identity,
            user_preferences=documents.user_preferences,
            knowledge_catalog=catalog,
            static_module_guidance=(),
        ),
    )
    observation = {"observation_id": "synthetic_obs_1", "description": case["scene"]}
    if case["name"] == "supported_completion":
        _append_synthetic_checkpoint(
            transcript, "synthetic_obs_0", "Synthetic pre-action state.", index=1
        )
        transcript.append_assistant(AssistantTurn("synthetic_prior_turn", "Turn right by 15 degrees.",
                                   [AgentToolCall("synthetic_prior_call", "uav.turn_relative", {"yaw_deg": 15})]))
        transcript.append_tool_result(ToolResult("synthetic_prior_call", "uav.turn_relative", "ok",
            "Synthetic recorded result: requested 15-degree right turn executed; measured yaw change +15 degrees.",
            data={"observation_id": "synthetic_obs_1", "yaw_delta_deg": 15}))
        observation["description"] = "Synthetic post-action observation: measured right turn is 15 degrees; stable."
    fragments = [
        ContextFragment(
            "evaluation",
            {
                "observation": observation,
                "evaluation_note": (
                    "Synthetic offline evaluation; attached media, if any, is a "
                    "recorded illustration, not a live observation."
                ),
            },
        )
    ]
    if case["image"]:
        ref = image_media_ref(
            case["image"],
            observation_id="synthetic_obs_1",
            frame_id="synthetic_frame_1",
            role="current_observation",
            usage=CURRENT_IMAGE_USAGE,
            scene_width=224,
            scene_height=224,
        )
        fragments.append(ContextFragment("image", {}, (ref.to_model_input(),)))
    index = 2 if case["name"] == "supported_completion" else 1
    fragments_tuple = tuple(fragments)
    transcript.append_state(
        StateCheckpoint(
            checkpoint_id=f"checkpoint_{index:04d}",
            reason="instruction_start" if index == 1 else "physical_window_completed",
            observation_id=observation["observation_id"],
            task_plan=None,
            loaded_knowledge=(),
            fragments=fragments_tuple,
            fingerprint=checkpoint_fingerprint(
                observation["observation_id"], None, (), fragments_tuple
            ),
        )
    )
    return transcript, observation


def _append_synthetic_checkpoint(transcript, observation_id, note, *, index):
    fragment = ContextFragment("evaluation", {"evaluation_note": note})
    transcript.append_state(
        StateCheckpoint(
            checkpoint_id=f"checkpoint_{index:04d}",
            reason="instruction_start",
            observation_id=observation_id,
            task_plan=None,
            loaded_knowledge=(),
            fragments=(fragment,),
            fingerprint=checkpoint_fingerprint(observation_id, None, (), (fragment,)),
        )
    )


def selection_metrics(case, turn):
    """Report case-oracle discrepancies, not a universal judgment of necessity."""
    names = [call.tool_name for call in turn.tool_calls]
    expected = case["expected"]
    arguments_match = True
    if case["name"] == "explicit_distance":
        arguments_match = any(c.tool_name == expected and "include_distance" not in c.args
                              and c.args.get("observation_id") == "synthetic_obs_1" for c in turn.tool_calls)
    elif case["name"] == "explicit_knowledge":
        arguments_match = any(c.tool_name == expected and c.args.get("name") == "orbit_and_surround"
                              for c in turn.tool_calls)
    return {"selected_tools": names, "unnecessary_calls": [name for name in names if name != expected],
            "omitted_expected_call": expected not in names, "expected_arguments_match": arguments_match,
            "matched": bool(names) and all(name == expected for name in names) and arguments_match}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = replace(load_model_config(args.env_path), timeout_sec=45,
                     provider_timeout_retries=0, provider_connection_retries=0)
    if not config.api_key:
        raise SystemExit("No model credentials configured; behavior probe not run.")
    if (args.phase == "after") != (args.baseline is not None):
        parser.error("--baseline is required only for --phase after")
    output = args.output_dir or Path(tempfile.mkdtemp(prefix="flyagent-tool-contract-"))
    if args.output_dir:
        output.mkdir(parents=True, exist_ok=False)
    plugins = PluginRegistry.from_config()
    registry = default_registry(plugins, load_uav_safety_config())
    cases = corpus()
    tools = tool_snapshot(registry)
    settings_hash = digest({"model": config.model_name, "provider": config.provider, "endpoint": config.base_url,
                            "prompt_path": config.action_planner_prompt_path, "temperature": 0,
                            "timeout": config.timeout_sec})
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
    if baseline:
        if baseline["corpus_hash"] != digest(cases) or baseline["settings_hash"] != settings_hash:
            raise SystemExit("Paired corpus/provider settings differ; no requests sent.")
        if structural_contracts(baseline["tools"]) != structural_contracts(tools):
            raise SystemExit("Structural tool contracts changed; no requests sent.")
    allowed = {spec.name for spec in registry.model_specs()}
    rows = []
    # Freeze every prompt/tool in one phase before making requests; edits between
    # phases cannot silently mix versions within a baseline run.
    planners = [ModelActionPlanner(config) for _ in cases]
    inputs = [build_transcript(case, output / case["name"], plugins) for case in cases]
    sources = source_snapshot()
    for case, planner, (transcript, observation) in zip(cases, planners, inputs):
        row = {"case": case["name"], "expected": case["expected"]}
        try:
            turn = planner.plan_turn(transcript, registry)
            row["turn"] = turn.to_dict()
            row.update(selection_metrics(case, turn))
            validate_native_turn(turn, registry, allowed_names=allowed, latest_observation=observation,
                                 prior_call_ids={"synthetic_prior_call"} if case["name"] == "supported_completion" else set())
            for call in turn.tool_calls:
                if call.tool_name in {"runtime.report_completed", "runtime.report_blocked"}:
                    refs = set(call.args["evidence_refs"])
                    if not refs.issubset(transcript.prior_evidence_reference_ids()):
                        raise ValueError("Terminal references are not prior environmental evidence")
            row["valid"] = True
        except Exception as exc:
            # Never print provider exceptions: they can contain URLs or credentials.
            row.update(valid=False, matched=False, error_type=type(exc).__name__)
        rows.append(row)
        (output / case["name"] / "request.json").write_text(
            json.dumps(planner.last_request, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in row.items() if key != "turn"}), flush=True)
        if row.get("error_type") and "Provider" in row["error_type"]:
            break
    report = {"phase": args.phase, "model": config.model_name, "settings_hash": settings_hash,
              "corpus_hash": digest(cases), "corpus": cases, "tools": tools, "sources": sources,
              "synthetic": True, "tools_executed": False, "cases": rows,
              "matched": sum(row["matched"] for row in rows), "total": len(rows)}
    if baseline:
        report["paired_baseline"] = str(args.baseline.resolve())
        report["comparison"] = [{"case": before["case"], "before": before.get("selected_tools"),
                                  "after": after.get("selected_tools"), "before_matched": before["matched"],
                                  "after_matched": after["matched"]}
                                 for before, after in zip(baseline["cases"], rows)]
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {output / 'report.json'}")


if __name__ == "__main__":
    main()
