r"""No-flight Responses admission for instruction injection and FlyAgent tools.

The probe executes six model requests through FlyAgent's shared ``ModelClient``.
It never executes returned tools and never writes credentials to its report.

PowerShell, after placing a rotated credential in the process environment:

    & 'F:\conda_env\flyagent\python.exe' tests/manual/provider_instruction_probe.py `
      --expected-base-url https://api.ikuncode.cc/v1 `
      --expected-model gpt-6.1-sol `
      --output runs/diagnostics/ikuncode_instruction_probe
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import sys
import time
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.config import ModelConfig, load_model_config
from src.agent.context.compression import (
    CompressionRequest,
    CompressionWindow,
    EvidenceCatalogEntry,
    ModelHistoryCompactor,
)
from src.agent.context.compression.validator import CompressionValidator
from src.llm import ModelClient, ModelRequest, ModelResponse, create_model_client
from src.llm.audit import redact_provider_value


CODE_AGENT_TERMS = (
    "repository", "codebase", "terminal", "shell", "patch", "pull request",
    "workspace", "coding agent", "request_user_input", "mcp", "web search",
)
UNKNOWN_TOOL_TERMS = (
    "shell", "exec", "terminal", "web", "mcp", "patch", "request_user_input",
)


def _tool(name: str, description: str = "Diagnostic no-op function.") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    }


def _call(
    client: ModelClient,
    name: str,
    request: ModelRequest,
) -> tuple[ModelResponse, dict[str, Any]]:
    started = time.monotonic()
    response = client.request(request, error_prefix=f"{name} admission failed")
    elapsed = time.monotonic() - started
    calls = response.tool_calls
    public_instructions = response.data.get("instructions")
    instructions_text = (
        public_instructions.lower() if isinstance(public_instructions, str) else ""
    )
    return response, {
        "name": name,
        "elapsed_seconds": elapsed,
        "request_id": response.transport_trace.get("request_id"),
        "usage": response.usage,
        "token_attribution": _token_attribution(response.usage),
        "response_instructions": redact_provider_value(
            public_instructions
        ),
        "response_instructions_visibility": (
            "visible" if public_instructions is not None else "not_returned"
        ),
        "response_instructions_codex_marker": "you are codex" in instructions_text,
        "text": response.text,
        "tool_calls": calls,
        "transport": response.transport_trace,
    }


def run_probe(
    config: ModelConfig,
    output_dir: str | Path,
    *,
    client: ModelClient | None = None,
    expected_base_url: str | None = None,
    expected_model: str | None = None,
) -> dict[str, Any]:
    if not config.api_key and client is None:
        raise ValueError("FLYAGENT_MODEL_API_KEY is required and will not be logged")
    if expected_base_url and config.base_url.rstrip("/") != expected_base_url.rstrip("/"):
        raise ValueError("configured base URL differs from the explicit admission target")
    if expected_model and config.model_name != expected_model:
        raise ValueError("configured model differs from the explicit admission target")
    responses_config = replace(config, api_protocol="responses", stream=False)
    model_client = client or create_model_client(responses_config.to_provider_settings())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []

    base_response, base = _call(
        model_client,
        "instructions_visibility",
        ModelRequest(
            messages=[
                {"role": "system", "content": "Reply with exactly ADMISSION_OK."},
                {"role": "user", "content": "Run the no-flight provider admission."},
            ],
            parameters={"reasoning_effort": "low"},
        ),
    )
    base["accepted"] = isinstance(base_response.text, str)
    records.append(base)

    single_response, single = _call(
        model_client,
        "single_tool_call",
        ModelRequest(
            messages=[
                {"role": "system", "content": "Call ping exactly once. Do not answer in text."},
                {"role": "user", "content": "Ping value one."},
            ],
            tools=[_tool("ping")],
            parameters={"tool_choice": "auto", "reasoning_effort": "low"},
        ),
    )
    single["accepted"] = _exact_tool_names(single_response, ["ping"])
    records.append(single)

    expected_parallel = ["ping_one", "ping_two", "ping_three"]
    parallel_response, parallel = _call(
        model_client,
        "parallel_tool_calls",
        ModelRequest(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "In one assistant turn call ping_one, ping_two, and ping_three once each. "
                        "Do not wait for results and do not answer in text."
                    ),
                },
                {"role": "user", "content": "Run all three independent no-op checks."},
            ],
            tools=[_tool(name) for name in expected_parallel],
            parameters={
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "reasoning_effort": "low",
            },
        ),
    )
    parallel["accepted"] = _exact_tool_names(parallel_response, expected_parallel)
    records.append(parallel)

    authority_tools = [
        _tool("uav_hover", "Hold the UAV at its current position."),
        _tool("runtime_update_plan", "Record a concise UAV task plan."),
        _tool("runtime_stop", "Stop the current UAV task."),
    ]
    authority_response, authority = _call(
        model_client,
        "flyagent_authority_canary",
        ModelRequest(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the FlyAgent UAV planner. Only use the provided UAV/runtime "
                        "functions. For this no-flight canary call uav_hover once. Do not discuss "
                        "software, repositories, terminals, patches, or tools not provided here."
                    ),
                },
                {"role": "user", "content": "Hold position for the authority canary."},
            ],
            tools=authority_tools,
            parameters={"tool_choice": "auto", "reasoning_effort": "low"},
        ),
    )
    supplied_authority = {item["function"]["name"] for item in authority_tools}
    unknown = sorted({
        call.get("function", {}).get("name")
        for call in authority_response.tool_calls
        if call.get("function", {}).get("name") not in supplied_authority
    })
    visible_authority = json.dumps(
        {"text": authority_response.text, "tool_calls": authority_response.tool_calls},
        ensure_ascii=False,
    ).lower()
    term_hits = sorted(term for term in CODE_AGENT_TERMS if term in visible_authority)
    authority.update(
        unknown_tools=unknown,
        code_agent_term_hits=term_hits,
        accepted=(not unknown and not term_hits and _exact_tool_names(authority_response, ["uav_hover"])),
    )
    records.append(authority)

    compression_request = CompressionRequest(
        source_digest="instruction-probe",
        instruction="Inspect the tower, remember the result, and continue safely.",
        windows=(CompressionWindow(
            "window_0001",
            0,
            2,
            (
                {
                    "kind": "assistant_turn",
                    "public_intent": "Inspect the tower.",
                    "tool_calls": [{"id": "call_old", "tool_name": "inspect"}],
                },
                {
                    "kind": "tool_result",
                    "tool_call_id": "call_old",
                    "status": "ok",
                    "result_ref": "observation_old",
                },
            ),
        ),),
        evidence_catalog=(
            EvidenceCatalogEntry("E0001", "call_old", "tool_call"),
            EvidenceCatalogEntry("E0002", "observation_old", "observation"),
        ),
        current_state_hint={"connected": True, "airborne": True},
        deterministic_facts={"open_work": ["continue toward the tower"]},
    )
    compactor = ModelHistoryCompactor(
        model_client,
        summary_max_tokens=4096,
        thinking_parameters={"reasoning_effort": "low"},
    )
    compaction_started = time.monotonic()
    compaction_result = compactor.compact(compression_request)
    compaction_validation = CompressionValidator(4096).validate(
        compaction_result.draft,
        compression_request,
    )
    compactor_record = {
        "name": "history_compactor",
        "elapsed_seconds": time.monotonic() - compaction_started,
        "usage": compaction_result.usage,
        "token_attribution": _token_attribution(compaction_result.usage),
        "transport": compaction_result.transport_trace,
        "finish_reason": compaction_result.finish_reason,
        "draft": compaction_result.draft.to_dict(),
        "validation_errors": list(compaction_validation.errors),
        "accepted": compaction_validation.accepted,
    }
    records.append(compactor_record)

    image_response, image_record = _call(
        model_client,
        "single_rgb_image",
        ModelRequest(
            messages=[
                {
                    "role": "system",
                    "content": "Inspect the supplied RGB image, then call image_ping exactly once.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Confirm that a red square is visible."},
                        {"type": "image_url", "image_url": {"url": _probe_image_url()}},
                    ],
                },
            ],
            tools=[_tool("image_ping", "Record that the image path was accepted.")],
            parameters={"tool_choice": "auto", "reasoning_effort": "low"},
        ),
    )
    image_record["accepted"] = _exact_tool_names(image_response, ["image_ping"])
    records.append(image_record)

    all_unknown_tools = sorted({
        name
        for record in records
        for name in record.get("unknown_tools", [])
        if isinstance(name, str)
    })
    code_agent_hits = sorted({
        term
        for record in records
        for term in record.get("code_agent_term_hits", [])
    })
    instruction_tokens = max(
        [value for record in records for value in _instruction_token_values(record)] or [0]
    )
    instructions_visible = any(
        record.get("response_instructions_visibility") == "visible"
        for record in records
    )
    codex_instruction_visible = any(
        record.get("response_instructions_codex_marker") is True
        for record in records
    )
    blockers = []
    if all_unknown_tools:
        blockers.append("unknown_tools")
    if code_agent_hits:
        blockers.append("code_agent_behavior")
    if any(not record.get("accepted") for record in records):
        blockers.append("protocol_admission_failed")
    warnings = []
    if instruction_tokens >= 4000:
        warnings.append("upstream_instructions_token_overhead")
    if codex_instruction_visible:
        warnings.append("upstream_codex_instruction_visible")
    report = {
        "schema_version": "provider_instruction_probe_v1",
        "status": "passed" if not blockers else "failed",
        "config": responses_config.redacted_summary(),
        "credentials_persisted": False,
        "instructions_content_visible": instructions_visible,
        "codex_instruction_visible": codex_instruction_visible,
        "max_reported_instruction_tokens": instruction_tokens,
        "warnings": warnings,
        "blockers": sorted(set(blockers)),
        "unknown_tools": all_unknown_tools,
        "code_agent_term_hits": code_agent_hits,
        "records": records,
    }
    report_path = output / "provider_instruction_probe.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _exact_tool_names(response: ModelResponse, expected: list[str]) -> bool:
    calls = response.tool_calls
    names = [
        call.get("function", {}).get("name")
        for call in calls
        if isinstance(call, dict) and isinstance(call.get("function"), dict)
    ]
    identifiers = [call.get("id") for call in calls if isinstance(call, dict)]
    arguments_valid = True
    for call in calls:
        try:
            arguments = json.loads(call["function"]["arguments"])
        except (KeyError, TypeError, json.JSONDecodeError):
            arguments_valid = False
            break
        if not isinstance(arguments, dict):
            arguments_valid = False
            break
    return (
        sorted(names) == sorted(expected)
        and len(names) == len(expected)
        and all(isinstance(value, str) and value for value in identifiers)
        and len(set(identifiers)) == len(identifiers)
        and arguments_valid
    )


def _token_attribution(usage: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(usage, dict):
        return {}
    result: dict[str, Any] = {}

    def visit(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                lowered = path.lower()
                if any(term in lowered for term in ("instruction", "input", "tool")):
                    result[path] = child
                visit(child, path)

    visit(usage)
    return result


def _instruction_token_values(record: dict[str, Any]) -> list[int]:
    values: list[int] = []
    for key, value in (record.get("token_attribution") or {}).items():
        if "instruction" in key.lower() and isinstance(value, int) and not isinstance(value, bool):
            values.append(value)
    return values


def _probe_image_url() -> str:
    image = Image.new("RGB", (96, 72), "white")
    ImageDraw.Draw(image).rectangle((20, 14, 68, 58), fill=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-base-url", default="")
    parser.add_argument("--expected-model", default="")
    args = parser.parse_args()
    report = run_probe(
        load_model_config(args.env_path),
        args.output,
        expected_base_url=args.expected_base_url or None,
        expected_model=args.expected_model or None,
    )
    print(json.dumps({
        "status": report["status"],
        "warnings": report["warnings"],
        "blockers": report["blockers"],
        "output": str(Path(args.output) / "provider_instruction_probe.json"),
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
