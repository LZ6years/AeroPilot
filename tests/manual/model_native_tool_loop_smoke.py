"""Exercise FlyAgent's real multimodal native tool loop without a simulator.

The probe uses the production ``ModelActionPlanner``, canonical ``ToolRegistry``,
and ``AgentTranscript`` provider projection. It normally performs three paid
model calls: unavailable measurement, valid measurement, then one
a direct UAV turn after synthetic ToolResults. A genuinely
invalid provider response may add a bounded repair round. No UAV command is
executed.

PowerShell example:

    $env:PYTHONIOENCODING='utf-8'
    python tests/manual/model_native_tool_loop_smoke.py --env-path .env

Outputs are retained under ``runs/manual_native_tool_loop_provider_probe`` by
default so ``memory/agent_transcript.jsonl`` and ``probe_summary.json`` can be
inspected after the call.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.planning.model_action_planner import (
    AgentProtocolError,
    ModelActionPlanner,
)
from src.agent.transcript import AgentTranscript, ToolResult
from src.memory.context.contracts import (
    ContextFragment,
    InstructionEnvelope,
    StateCheckpoint,
    checkpoint_fingerprint,
)
from src.memory.context.media_refs import image_media_ref
from src.memory.stores import MemoryEventWriter
from src.plugins import PluginRegistry
from src.agent.config import load_model_config
from src.tools.registry import default_registry


PROBE_PROMPT = Path(__file__).with_name(
    "native_tool_loop_provider_probe_system.txt"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "manual_native_tool_loop_provider_probe"
ALLOWED_TOOLS = {
    "skill.visual_grounding.measure_visible_depth",
    "uav.turn_relative",
}


def main() -> None:
    args = _parser().parse_args()
    output_dir = _output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    config = load_model_config(args.env_path)
    if not config.api_key:
        raise SystemExit(
            "Missing FLYAGENT_MODEL_API_KEY in environment or .env; "
            "the key will never be printed."
        )

    registry = _registry()
    transcript = AgentTranscript(MemoryEventWriter(output_dir))
    image_paths = _write_probe_images(output_dir / "artifacts" / "rgb_depth")
    detection_evidence_path = _write_probe_confirmation(
        image_paths["obs_probe_0001"],
        output_dir / "artifacts" / "tool_evidence" / "detection_candidates.png",
        color=(255, 80, 40),
    )
    distance_evidence_path = _write_probe_confirmation(
        image_paths["obs_probe_0001"],
        output_dir / "artifacts" / "tool_evidence" / "distance_candidates.png",
        color=(30, 180, 255),
    )
    instruction = (
        "parallel_query_probe: request independent person localization and building measurement, then one small right turn."
        if args.parallel_queries
        else "Locate the visible person, measure it only when needed, then request one small right turn."
    )
    transcript.start_instruction(
        instruction,
        "provider_protocol_probe",
        InstructionEnvelope(
            instruction_id="provider_protocol_probe",
            protocol_context="",
            identity={},
            user_preferences={},
            knowledge_catalog=(),
            static_module_guidance=(),
        ),
    )
    for index, observation_id in enumerate(
        ("obs_probe_start", "obs_probe_before", "obs_probe_0001"), start=1
    ):
        transcript.append_state(
            _probe_checkpoint(image_paths[observation_id], observation_id, index)
        )

    planner = ModelActionPlanner(config, prompt_path=str(PROBE_PROMPT))
    if args.parallel_queries:
        _parallel_query_probe(planner, transcript, registry, output_dir, detection_evidence_path)
        return
    first_turn, first_request, first_attempts = _plan_with_protocol_repair(
        planner,
        transcript,
        registry,
        stage="unavailable_measurement",
    )
    _assert_grounding_turn(first_turn)
    transcript.append_assistant(first_turn)
    detection_call = first_turn.tool_calls[0]
    transcript.append_tool_result(
        ToolResult(
            tool_call_id=detection_call.tool_call_id,
            tool_name=detection_call.tool_name,
            status="ok",
            message="Synthetic candidate has unavailable metric depth.",
            data={
                "observation_id": "obs_probe_0001",
                "query": "the visible person",
                "include_distance": True,
                "candidates": [
                    {
                        "candidate_id": "candidate_01",
                        "label": "person",
                        "confidence": 0.91,
                        "box_normalized": [0.4, 0.1, 0.6, 0.8],
                        "distance_m": None,
                        "distance_status": "metric_depth_unavailable",
                    }
                ],
                "audit_details": {"grounding_backend": "probe_stub"},
            },
            artifact_refs=[str(detection_evidence_path)],
            requested_args=dict(detection_call.args),
            executed_args=dict(detection_call.args),
            model_inputs=[
                image_media_ref(
                    detection_evidence_path,
                    observation_id="obs_probe_0001",
                    frame_id="detection_candidates",
                    role="visual_grounding_candidates",
                    usage="Confirm candidate_01 against the instruction; confidence alone is not proof.",
                ).to_model_input()
            ],
        )
    )

    second_turn, second_request, second_attempts = _plan_with_protocol_repair(
        planner,
        transcript,
        registry,
        stage="distance_enabled_grounding",
    )
    _assert_grounding_turn(second_turn)
    transcript.append_assistant(second_turn)
    distance_call = second_turn.tool_calls[0]
    transcript.append_tool_result(
        ToolResult(
            tool_call_id=distance_call.tool_call_id,
            tool_name=distance_call.tool_name,
            status="ok",
            message="Candidate distance is robust metric depth to the visible surface.",
            data={
                "observation_id": "obs_probe_0001",
                "query": "the visible person",
                "include_distance": True,
                "candidates": [
                    {
                        "candidate_id": "candidate_01",
                        "label": "person",
                        "confidence": 0.91,
                        "box_normalized": [0.4, 0.1, 0.6, 0.8],
                        "distance_m": 2.9,
                        "distance_status": "ok",
                    }
                ],
                "audit_details": {"sampling_source": "eroded_mask", "mad_m": 0.06},
            },
            artifact_refs=[str(distance_evidence_path)],
            requested_args=dict(distance_call.args),
            executed_args=dict(distance_call.args),
            model_inputs=[
                image_media_ref(
                    distance_evidence_path,
                    observation_id="obs_probe_0001",
                    frame_id="distance_candidates",
                    role="visual_grounding_candidates",
                    usage="Candidate evidence associated with the robust distance result.",
                ).to_model_input()
            ],
        )
    )

    third_turn, third_request, third_attempts = _plan_with_protocol_repair(
        planner,
        transcript,
        registry,
        stage="direct_physical",
    )
    _assert_physical_turn(third_turn, registry)
    transcript.append_assistant(third_turn)
    for call in third_turn.tool_calls:
        transcript.append_tool_result(
            ToolResult(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                status="ok",
                message=(
                    "provider protocol accepted; physical execution intentionally skipped"
                    if registry.get(call.tool_name).runtime.effect == "physical"
                    else "provider protocol tool accepted"
                ),
                data={"probe_only": True, "executed": False},
                requested_args=dict(call.args),
                executed_args=dict(call.args),
            )
        )

    _assert_provider_requests(
        [first_request, second_request, third_request],
        [first_turn, second_turn],
    )
    _assert_closed_transcript(transcript.to_dict())
    summary = {
        "status": "ok",
        "probe_only": True,
        "physical_execution": False,
        "model_config": config.redacted_summary(),
        "chat_completions_url": config.chat_completions_url(),
        "first_turn": first_turn.to_dict(),
        "second_turn": second_turn.to_dict(),
        "third_turn": third_turn.to_dict(),
        "protocol_attempts": {
            "first_turn": first_attempts,
            "second_turn": second_attempts,
            "third_turn": third_attempts,
        },
        "first_request": first_request,
        "second_request": second_request,
        "third_request": third_request,
        "transcript_path": str(
            output_dir / "memory" / "agent_transcript.jsonl"
        ),
    }
    summary_path = output_dir / "probe_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(_terminal_summary(summary, output_dir), indent=2, ensure_ascii=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Make three real model calls to validate FlyAgent multimodal native "
            "ToolCall/ToolResult replay without starting a simulator."
        )
    )
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--parallel-queries", action="store_true", help="Probe two independent same-turn queries and replay both results before motion (two paid model requests).")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="New output directory; omitted uses a timestamped directory under runs/.",
    )
    return parser


def _parallel_query_probe(planner, transcript, registry, output_dir, evidence_path):
    """Protocol-only double-call probe; synthetic candidates do not test detection quality."""
    first, first_request, first_attempts = _plan_with_protocol_repair(planner, transcript, registry, stage="parallel_queries")
    calls = first.tool_calls
    if len(calls) != 2 or any(call.tool_name != "skill.visual_grounding.measure_visible_depth" for call in calls):
        raise RuntimeError("Parallel probe did not produce two independent grounding calls; branch not observed.")
    if any("include_distance" in call.args for call in calls):
        raise RuntimeError("Parallel probe accepts measurement-only inputs.")
    if any(call.args.get("observation_id") != "obs_probe_0001" for call in calls):
        raise RuntimeError("Parallel probe referenced stale observation.")
    transcript.append_assistant(first)
    for i, call in enumerate(calls):
        distance = i == 1
        transcript.append_tool_result(ToolResult(
            tool_call_id=call.tool_call_id, tool_name=call.tool_name, status="ok",
            message="Synthetic protocol evidence only; no detector or UAV executed.",
            data={"observation_id": "obs_probe_0001", "query": call.args["query"], "candidates": [{"candidate_id": "candidate_01", "distance_m": 2.9 if distance else None, "distance_status": "ok" if distance else "metric_depth_unavailable"}]},
            requested_args=dict(call.args), executed_args=dict(call.args),
            model_inputs=[image_media_ref(
                evidence_path,
                observation_id="obs_probe_0001",
                frame_id="parallel_candidates",
                role="visual_grounding_candidates",
                usage="Synthetic probe evidence, not detector accuracy validation.",
            ).to_model_input()],
        ))
    second, second_request, second_attempts = _plan_with_protocol_repair(planner, transcript, registry, stage="physical_after_parallel_queries")
    _assert_physical_turn(second, registry)
    replay = [message.get("tool_call_id") for message in second_request["messages"] if message.get("role") == "tool"]
    if replay != [call.tool_call_id for call in calls]:
        raise RuntimeError("Provider request did not replay both results exactly once in order.")
    if any("response_format" in request for request in (first_request, second_request)):
        raise RuntimeError("Unexpected response_format in native probe.")
    transcript.append_assistant(second)
    for call in second.tool_calls:
        transcript.append_tool_result(ToolResult(tool_call_id=call.tool_call_id, tool_name=call.tool_name, status="ok", message="Protocol accepted; execution intentionally skipped.", data={"probe_only": True, "executed": False}))
    _assert_closed_transcript(transcript.to_dict())
    summary = {"status": "ok", "probe_only": True, "parallel_queries": True, "physical_execution": False, "attempts": [first_attempts, second_attempts], "requests": [first_request, second_request], "turns": [first.to_dict(), second.to_dict()]}
    (output_dir / "probe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "parallel_queries": True, "output_dir": str(output_dir)}))


def _output_dir(value: str | None) -> Path:
    if value:
        return Path(value).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return DEFAULT_OUTPUT_ROOT / stamp


def _registry():
    plugins = PluginRegistry.discover(
        {
            "plugins": {
                "roots": ["plugins/capabilities"],
                "enabled": ["capabilities.visual_grounding"],
            }
        }
    )
    registry = default_registry(plugins)
    missing = ALLOWED_TOOLS - set(registry.names())
    if missing:
        raise RuntimeError(f"Probe registry is missing tools: {sorted(missing)}")
    return registry


def _probe_checkpoint(
    image_path: Path, observation_id: str, index: int
) -> StateCheckpoint:
    frame_id = f"frame_{observation_id}"
    media = image_media_ref(
        image_path,
        observation_id=observation_id,
        frame_id=frame_id,
        role="current_observation",
        usage="Original RGB preserves scene colors. Use grounding only when the next decision requires it.",
        scene_width=240,
        scene_height=180,
    )
    fragments = (
        ContextFragment("image", {}, (media.to_model_input(),)),
    )
    return StateCheckpoint(
        checkpoint_id=f"checkpoint_{index:04d}",
        reason=("instruction_start" if index == 1 else "physical_window_completed"),
        observation_id=observation_id,
        task_plan=None,
        loaded_knowledge=(),
        fragments=fragments,
        fingerprint=checkpoint_fingerprint(
            observation_id, None, (), fragments
        ),
    )


def _write_probe_images(directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for index, observation_id in enumerate(
        ("obs_probe_start", "obs_probe_before", "obs_probe_0001"),
        start=1,
    ):
        path = directory / f"provider_probe_{index}.png"
        image = Image.new("RGB", (240, 180), (255, 255 - index * 12, 255 - index * 7))
        draw = ImageDraw.Draw(image)
        draw.ellipse((102, 22, 138, 58), fill=(42, 72, 110))
        draw.rectangle((96, 58, 144, 128), fill=(49, 105, 158))
        draw.line((106, 128, 88, 166), fill=(42, 72, 110), width=10)
        draw.line((134, 128, 152, 166), fill=(42, 72, 110), width=10)
        image.save(path)
        paths[observation_id] = path
    return paths


def _write_probe_confirmation(
    source_path: Path, output_path: Path, *, color: tuple[int, int, int]
) -> Path:
    """Create synthetic candidate evidence for protocol validation."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((96, 22, 144, 166), outline=color, width=3)
    draw.text((98, 24), "candidate_01", fill=color)
    image.save(output_path)
    return output_path


def _plan_with_protocol_repair(
    planner: ModelActionPlanner,
    transcript: AgentTranscript,
    registry,
    *,
    stage: str,
):
    """Mirror Runtime's transcript-native repair loop for provider responses."""

    attempts: list[dict[str, Any]] = []
    max_rounds = planner.model_config.max_tool_rounds
    for attempt in range(1, max_rounds + 1):
        try:
            turn = planner.plan_turn(
                transcript,
                registry,
                allowed_tool_names=ALLOWED_TOOLS,
            )
        except AgentProtocolError as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "protocol_rejected",
                    "transport": copy.deepcopy(
                        planner.last_exchange.get("transport", {})
                    ),
                    "error": str(exc),
                    "request": copy.deepcopy(planner.last_request),
                    "response": copy.deepcopy(exc.rejected_turn),
                }
            )
            if exc.rejected_turn:
                transcript.append_provider_rejection(exc.rejected_turn)
            transcript.append_protocol_feedback(
                f"Protocol error: {exc}. Return a corrected native tool-call turn."
            )
            continue
        attempts.append(
            {
                "attempt": attempt,
                "status": "ok",
                "transport": copy.deepcopy(
                    planner.last_exchange.get("transport", {})
                ),
                "request": copy.deepcopy(planner.last_request),
            }
        )
        return turn, copy.deepcopy(planner.last_request), attempts
    raise RuntimeError(
        f"Provider did not produce a valid {stage} turn within {max_rounds} rounds"
    )


def _assert_grounding_turn(turn) -> None:
    names = [call.tool_name for call in turn.tool_calls]
    if names != ["skill.visual_grounding.measure_visible_depth"]:
        raise RuntimeError(
            "Grounding stage must contain only the read-only visual-grounding tool; "
            f"received={names}"
        )
    if turn.tool_calls[0].args.get("observation_id") != "obs_probe_0001":
        raise RuntimeError(
            "Visual-grounding call did not preserve the image observation_id: "
            f"{turn.tool_calls[0].args}"
        )
    if "include_distance" in turn.tool_calls[0].args:
        raise RuntimeError(
            "Measurement-only call contains retired include_distance: "
            f"{turn.tool_calls[0].args}"
        )


def _assert_physical_turn(turn, registry) -> None:
    from src.agent.planning.tool_contract_validator import validate_native_turn
    effect = validate_native_turn(turn, registry, allowed_names=ALLOWED_TOOLS,
                                  latest_observation={"observation_id": "obs_probe_0001"})
    if effect != "physical" or len(turn.tool_calls) != 1:
        raise RuntimeError("Final probe turn must request one physical call directly")


def _assert_provider_requests(
    requests: list[dict[str, Any]],
    grounding_turns: list,
) -> None:
    first_request, second_request, third_request = requests
    if any("response_format" in request for request in requests):
        raise RuntimeError("Native tool-loop request unexpectedly used response_format")
    if _count_provider_media(first_request) != 3:
        raise RuntimeError(
            "First provider request did not include all three chronological images"
        )
    serialized = json.dumps(first_request, ensure_ascii=False)
    if serialized.count("HISTORY observation_id=") != 2 or serialized.count(
        "CURRENT observation_id=obs_probe_0001"
    ) != 1:
        raise RuntimeError("Provider image labels do not identify two HISTORY images and one CURRENT image")
    if serialized.count("display_size=240x180") != 3 or serialized.count(
        "Original RGB preserves scene colors"
    ) != 3 or "bottom color bar" in serialized:
        raise RuntimeError(
            "Provider image labels do not match the original RGB probe images"
        )
    if _count_provider_media(second_request) < 4:
        raise RuntimeError(
            "Second provider request omitted chronological or detection evidence images"
        )
    if _count_provider_media(third_request) < 5:
        raise RuntimeError("Third provider request omitted grounding evidence images")
    for request in (second_request, third_request):
        if "TOOL EVIDENCE tool_call_id=" not in json.dumps(request, ensure_ascii=False):
            raise RuntimeError("Provider request omitted grounding candidate evidence")
    tool_messages = [
        item
        for item in third_request.get("messages", [])
        if isinstance(item, dict) and item.get("role") == "tool"
    ]
    replayed_result_ids = {str(item.get("tool_call_id")) for item in tool_messages}
    expected_result_ids = {
        turn.tool_calls[0].tool_call_id for turn in grounding_turns
    }
    if not expected_result_ids.issubset(replayed_result_ids):
        raise RuntimeError(
            "Third provider request did not replay all grounding ToolResults"
        )
    assistant_messages = [
        item
        for item in third_request.get("messages", [])
        if isinstance(item, dict)
        and item.get("role") == "assistant"
        and any(
            call.get("id") in expected_result_ids
            for call in item.get("tool_calls", [])
            if isinstance(call, dict)
        )
    ]
    if len(assistant_messages) != len(grounding_turns):
        raise RuntimeError("Third provider request did not replay both grounding AssistantTurns")
    for replayed, turn in zip(assistant_messages, grounding_turns):
        expected_content = (
            turn.public_intent
            if turn.public_intent_source == "provider_content"
            else None
        )
        if replayed.get("content") != expected_content:
            raise RuntimeError(
                "Provider history fabricated or dropped AssistantTurn content: "
                f"source={turn.public_intent_source!r}"
            )


def _count_provider_media(value: Any) -> int:
    if isinstance(value, str):
        return int(value.startswith("[provider_media sha256="))
    if isinstance(value, dict):
        return sum(_count_provider_media(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_provider_media(item) for item in value)
    return 0


def _assert_closed_transcript(value: dict[str, Any]) -> None:
    calls: set[str] = set()
    results: set[str] = set()
    for item in value.get("items", []):
        payload = item.get("payload", {})
        if item.get("kind") == "assistant_turn":
            calls.update(
                str(call.get("tool_call_id"))
                for call in payload.get("tool_calls", [])
            )
        elif item.get("kind") == "tool_result":
            results.add(str(payload.get("tool_call_id")))
    if calls != results:
        raise RuntimeError(
            "Probe transcript is not closed: "
            f"missing_results={sorted(calls - results)}, "
            f"orphan_results={sorted(results - calls)}"
        )


def _terminal_summary(summary: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "status": summary["status"],
        "probe_only": True,
        "physical_execution": False,
        "model_config": summary["model_config"],
        "first_turn_tools": [
            call["tool_name"] for call in summary["first_turn"]["tool_calls"]
        ],
        "second_turn_tools": [
            call["tool_name"] for call in summary["second_turn"]["tool_calls"]
        ],
        "third_turn_tools": [
            call["tool_name"] for call in summary["third_turn"]["tool_calls"]
        ],
        "first_turn_intent_source": summary["first_turn"]["public_intent_source"],
        "second_turn_intent_source": summary["second_turn"]["public_intent_source"],
        "third_turn_intent_source": summary["third_turn"]["public_intent_source"],
        "response_format_present": False,
        "tool_result_replayed": True,
        "transport_traces": {
            name: [attempt.get("transport", {}) for attempt in attempts]
            for name, attempts in summary["protocol_attempts"].items()
        },
        "output_dir": str(output_dir),
        "transcript_path": summary["transcript_path"],
        "summary_path": str(output_dir / "probe_summary.json"),
    }


if __name__ == "__main__":
    main()
