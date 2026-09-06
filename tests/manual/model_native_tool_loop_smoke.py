"""Exercise FlyAgent's real multimodal native tool loop without a simulator.

The probe uses the production ``ModelActionPlanner``, canonical ``ToolRegistry``,
and ``AgentTranscript`` provider projection. It performs two paid model calls:
one read-only person-detector request, followed by one phase commit plus a UAV
turn request after a synthetic detector ToolResult. No UAV command is executed.

PowerShell example:

    $env:PYTHONIOENCODING='utf-8'
    python tests/manual/model_native_tool_loop_smoke.py --env-path .env

Outputs are retained under ``runs/manual_native_tool_loop_provider_probe`` by
default so ``memory/agent_transcript.jsonl`` and ``probe_summary.json`` can be
inspected after the call.
"""

from __future__ import annotations

import argparse
import base64
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
from src.memory.stores import MemoryEventWriter
from src.plugins import PluginRegistry
from src.runtime.configs.model_config import load_model_config
from src.tools.registry import default_registry


PROBE_PROMPT = Path(__file__).with_name(
    "native_tool_loop_provider_probe_system.txt"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "manual_native_tool_loop_provider_probe"
ALLOWED_TOOLS = {
    "skill.person_detector.detect",
    "runtime.commit_phase_turn",
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
    image_path = output_dir / "artifacts" / "rgb" / "provider_probe.png"
    _write_probe_image(image_path)
    transcript.start_instruction(
        "Confirm the visible person through the detector, then request one small right turn.",
        "provider_protocol_probe",
    )
    transcript.append_state(_probe_context(image_path), "obs_probe_0001")

    planner = ModelActionPlanner(config, prompt_path=str(PROBE_PROMPT))
    first_turn, first_request, first_attempts = _plan_with_protocol_repair(
        planner,
        transcript,
        registry,
        stage="read_only_detector",
    )
    _assert_first_turn(first_turn)
    transcript.append_assistant(first_turn)
    detector_call = first_turn.tool_calls[0]
    transcript.append_tool_result(
        ToolResult(
            tool_call_id=detector_call.tool_call_id,
            tool_name=detector_call.tool_name,
            status="ok",
            message="probe detector confirmed one visible person",
            data={
                "mock": True,
                "observation_id": "obs_probe_0001",
                "frame_id": "frame_probe_rgb_0001",
                "detections": [
                    {
                        "class": "person",
                        "confidence": 0.97,
                        "bbox": [0.39, 0.18, 0.61, 0.86],
                    }
                ],
            },
            artifact_refs=["frame_probe_rgb_0001"],
            requested_args=dict(detector_call.args),
            executed_args=dict(detector_call.args),
        )
    )

    second_turn, second_request, second_attempts = _plan_with_protocol_repair(
        planner,
        transcript,
        registry,
        stage="commit_and_physical",
    )
    _assert_second_turn(second_turn, registry)
    transcript.append_assistant(second_turn)
    for call in second_turn.tool_calls:
        transcript.append_tool_result(
            ToolResult(
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                status="ok",
                message=(
                    "provider protocol accepted; physical execution intentionally skipped"
                    if registry.get(call.tool_name).runtime.effect == "physical"
                    else "provider protocol commit accepted"
                ),
                data={"probe_only": True, "executed": False},
                requested_args=dict(call.args),
                executed_args=dict(call.args),
            )
        )

    _assert_provider_requests(
        first_request,
        second_request,
        detector_call.tool_call_id,
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
        "protocol_attempts": {
            "first_turn": first_attempts,
            "second_turn": second_attempts,
        },
        "first_request": first_request,
        "second_request": second_request,
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
            "Make two real model calls to validate FlyAgent multimodal native "
            "ToolCall/ToolResult replay without starting a simulator."
        )
    )
    parser.add_argument("--env-path", default=".env")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="New output directory; omitted uses a timestamped directory under runs/.",
    )
    return parser


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
                "enabled": ["capabilities.perception"],
            }
        }
    )
    registry = default_registry(plugins)
    missing = ALLOWED_TOOLS - set(registry.names())
    if missing:
        raise RuntimeError(f"Probe registry is missing tools: {sorted(missing)}")
    return registry


def _probe_context(image_path: Path) -> dict[str, Any]:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "active_instruction": (
            "Confirm the visible person through the detector, then request "
            "one small right turn."
        ),
        "current_phase": {
            "name": "provider_protocol_probe",
            "goal": "confirm the person and request a small right turn",
            "done_when": [
                "the UAV has turned right after the detector confirmed the person"
            ],
            "continue_when": ["the requested right turn has not executed"],
        },
        "phase_strategy": None,
        "phase_progress_evidence": {
            "summary": "No physical action has executed in this protocol probe."
        },
        "fragments": [
            {
                "name": "image",
                "content": {"observation_id": "obs_probe_0001"},
                "model_inputs": [
                    {
                        "type": "image",
                        "input_id": "frame_probe_rgb_0001",
                        "frame_id": "frame_probe_rgb_0001",
                        "observation_id": "obs_probe_0001",
                        "role": "current_observation",
                        "usage": "Current RGB image for the provider protocol probe.",
                        "mime_type": "image/png",
                        "encoding": "data_url",
                        "data": f"data:image/png;base64,{encoded}",
                        "artifact_ref": str(image_path),
                    }
                ],
            }
        ],
    }


def _write_probe_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (240, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((102, 22, 138, 58), fill=(42, 72, 110))
    draw.rectangle((96, 58, 144, 128), fill=(49, 105, 158))
    draw.line((106, 128, 88, 166), fill=(42, 72, 110), width=10)
    draw.line((134, 128, 152, 166), fill=(42, 72, 110), width=10)
    draw.text((12, 12), "provider tool-loop probe", fill=(24, 35, 52))
    image.save(path)


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
                    "error": str(exc),
                    "request": copy.deepcopy(planner.last_request),
                    "response": _provider_response_summary(
                        planner.last_response
                    ),
                }
            )
            transcript.append_protocol_feedback(
                f"Protocol error: {exc}. Return a corrected native tool-call turn."
            )
            continue
        attempts.append(
            {
                "attempt": attempt,
                "status": "ok",
                "request": copy.deepcopy(planner.last_request),
            }
        )
        return turn, copy.deepcopy(planner.last_request), attempts
    raise RuntimeError(
        f"Provider did not produce a valid {stage} turn within {max_rounds} rounds"
    )


def _provider_response_summary(response: dict[str, Any]) -> dict[str, Any]:
    """Keep protocol evidence without persisting provider-private reasoning."""

    choices = response.get("choices")
    choices = choices if isinstance(choices, list) else []
    message = choices[0].get("message", {}) if choices else {}
    message = message if isinstance(message, dict) else {}
    raw_calls = message.get("tool_calls")
    raw_calls = raw_calls if isinstance(raw_calls, list) else []
    return {
        "response_keys": sorted(response),
        "finish_reason": (
            choices[0].get("finish_reason")
            if choices and isinstance(choices[0], dict)
            else None
        ),
        "assistant_content_present": bool(message.get("content")),
        "tool_names": [
            call.get("function", {}).get("name")
            for call in raw_calls
            if isinstance(call, dict)
            and isinstance(call.get("function"), dict)
        ],
    }


def _assert_first_turn(turn) -> None:
    names = [call.tool_name for call in turn.tool_calls]
    if names != ["skill.person_detector.detect"]:
        raise RuntimeError(
            "First provider turn must contain only the read-only detector; "
            f"received={names}"
        )
    if turn.tool_calls[0].args.get("observation_id") != "obs_probe_0001":
        raise RuntimeError(
            "Detector did not preserve the image observation_id: "
            f"{turn.tool_calls[0].args}"
        )


def _assert_second_turn(turn, registry) -> None:
    names = [call.tool_name for call in turn.tool_calls]
    if not names or names[0] != "runtime.commit_phase_turn":
        raise RuntimeError(
            "Second provider turn must begin with runtime.commit_phase_turn; "
            f"received={names}"
        )
    if len(names) < 2 or any(
        registry.get(name).runtime.effect != "physical" for name in names[1:]
    ):
        raise RuntimeError(
            "Second provider turn must include a physical call after commit; "
            f"received={names}"
        )


def _assert_provider_requests(
    first_request: dict[str, Any],
    second_request: dict[str, Any],
    detector_call_id: str,
) -> None:
    if "response_format" in first_request or "response_format" in second_request:
        raise RuntimeError("Native tool-loop request unexpectedly used response_format")
    if not _contains_value(first_request, "[image_data_url]"):
        raise RuntimeError("First provider request did not include a redacted image input")
    tool_messages = [
        item
        for item in second_request.get("messages", [])
        if isinstance(item, dict) and item.get("role") == "tool"
    ]
    if not any(
        item.get("tool_call_id") == detector_call_id for item in tool_messages
    ):
        raise RuntimeError(
            "Second provider request did not replay the first ToolResult"
        )


def _contains_value(value: Any, expected: Any) -> bool:
    if value == expected:
        return True
    if isinstance(value, dict):
        return any(_contains_value(item, expected) for item in value.values())
    if isinstance(value, list):
        return any(_contains_value(item, expected) for item in value)
    return False


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
        "response_format_present": False,
        "tool_result_replayed": True,
        "output_dir": str(output_dir),
        "transcript_path": summary["transcript_path"],
        "summary_path": str(output_dir / "probe_summary.json"),
    }


if __name__ == "__main__":
    main()
