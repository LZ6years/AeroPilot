"""Offline contract tests for the no-flight provider instruction probe."""
from __future__ import annotations

import json
from pathlib import Path

from src.agent.config import ModelConfig
from src.llm import ModelResponse
from tests.manual.provider_instruction_probe import run_probe


def _text_response(text, *, instructions=None, usage=None):
    data = {
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": usage or {},
    }
    if instructions is not None:
        data["instructions"] = instructions
    return ModelResponse(
        data,
        transport_trace={
            "backend": "openai_python_sdk",
            "request_id": "req_test",
            "final_status": "success",
        },
    )


def _tool_response(*names):
    return ModelResponse({
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {"name": name, "arguments": '{"value":"ok"}'},
                    }
                    for index, name in enumerate(names, start=1)
                ],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"input_tokens_details": {"instructions_tokens": 4216}},
    }, transport_trace={"backend": "openai_python_sdk", "final_status": "success"})


class _ProbeClient:
    def __init__(self):
        self.requests = []
        self.responses = [
            _text_response(
                "ADMISSION_OK",
                instructions="public provider instruction",
                usage={"input_tokens_details": {"instructions_tokens": 4216}},
            ),
            _tool_response("ping"),
            _tool_response("ping_one", "ping_two", "ping_three"),
            _tool_response("uav_hover"),
            _text_response(json.dumps({
                "mission_progress": [{"summary": "Tower inspected.", "evidence_refs": ["E0001"]}],
                "decisions_and_outcomes": [],
                "failures_and_recovery": [],
            })),
            _tool_response("image_ping"),
        ]

    def request(self, request, *, error_prefix="Model request failed", transport_trace=None):
        self.requests.append(request)
        return self.responses[len(self.requests) - 1]


def test_probe_covers_all_protocols_without_persisting_key_or_image(tmp_path):
    client = _ProbeClient()
    output = tmp_path / "probe"
    report = run_probe(
        ModelConfig(
            model_name="gpt-6.1-sol",
            api_key="SECRET-PROBE-KEY",
            base_url="https://api.ikuncode.cc/v1",
            api_protocol="responses",
        ),
        output,
        client=client,
        expected_base_url="https://api.ikuncode.cc/v1",
        expected_model="gpt-6.1-sol",
    )

    assert report["status"] == "passed"
    assert report["warnings"] == ["upstream_instructions_token_overhead"]
    assert report["max_reported_instruction_tokens"] == 4216
    assert report["instructions_content_visible"] is True
    assert report["codex_instruction_visible"] is False
    assert [record["name"] for record in report["records"]] == [
        "instructions_visibility",
        "single_tool_call",
        "parallel_tool_calls",
        "flyagent_authority_canary",
        "history_compactor",
        "single_rgb_image",
    ]
    compactor_request = client.requests[4]
    assert "data:image" not in json.dumps(compactor_request.messages)
    schema = compactor_request.parameters["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["mission_progress"]["items"]["properties"]["evidence_refs"]["items"]["enum"] == ["E0001", "E0002"]
    image_request = client.requests[5]
    assert "data:image/png;base64," in json.dumps(image_request.messages)
    persisted = (output / "provider_instruction_probe.json").read_text(encoding="utf-8")
    assert "SECRET-PROBE-KEY" not in persisted
    assert "data:image/png;base64," not in persisted


def test_probe_rejects_wrong_target_before_creating_output(tmp_path):
    output = tmp_path / "probe"
    try:
        run_probe(
            ModelConfig(model_name="other", api_key="key", base_url="https://wrong.test/v1"),
            output,
            client=_ProbeClient(),
            expected_base_url="https://api.ikuncode.cc/v1",
            expected_model="gpt-6.1-sol",
        )
    except ValueError as exc:
        assert "base URL" in str(exc)
    else:
        raise AssertionError("target mismatch was accepted")
    assert not output.exists()
