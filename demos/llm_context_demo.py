"""Two text-only rounds showing explicit history ownership and stateless LLM IO.

Run ``python -m demos.llm_context_demo --output-dir <new-directory>``.
The default is fully offline. ``--real`` opts into a configured provider; this
demo never initializes Runtime, a simulator, or a flight plan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.agent.context.projection import HistoryProjector
from src.agent.transcript import AgentTranscript, AssistantTurn
from src.llm import MessageBuilder, ModelRequest, ModelResponse, ProviderRequestError, ProviderSettings, create_model_client
from src.llm.audit import snapshot_request
from src.llm.providers.openai_compatible import encode_request, public_response_snapshot
from src.memory.stores import MemoryEventWriter
from src.memory.context.contracts import (
    ContextFragment,
    InstructionEnvelope,
    StateCheckpoint,
    checkpoint_fingerprint,
)
from src.agent.config import load_model_config
from src.tools.registry import ToolRegistry


class MockClient:
    def request(self, request: ModelRequest, *, error_prefix: str = "Model request failed",
                transport_trace: dict | None = None) -> ModelResponse:
        trace = {"attempt_count": 1, "retry_count": 0, "final_status": "success"}
        if transport_trace is not None:
            transport_trace.update(trace)
        return ModelResponse({"choices": [{"message": {
            "role": "assistant", "content": "已根据本轮状态和历史生成回答。"}}]}, trace)


def build_request(transcript: AgentTranscript) -> tuple[ModelRequest, dict[str, Any]]:
    # History owns facts. The builder only owns this request's independent copy.
    projection = HistoryProjector().project(transcript, ToolRegistry())
    builder = MessageBuilder().add_system("根据提供的状态简短回答，不执行工具。", key="instruction")
    builder.extend(projection.messages, key="history")
    builder.add_user("临时消息", key="temporary")
    builder.get("temporary")
    builder.replace("temporary", [{"role": "user", "content": "临时消息已替换"}])
    builder.remove("temporary")
    return ModelRequest(builder.build(), parameters={"temperature": 0}), {
        "history_projection": projection.audit,
        "builder_example": {"removed_before_sending": True},
    }


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_demo(output: Path, *, client: Any, settings: ProviderSettings, mode: str) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    transcript = AgentTranscript(MemoryEventWriter(output))
    envelope = InstructionEnvelope(
        instruction_id="demo_instruction",
        protocol_context="",
        identity={},
        user_preferences={},
        knowledge_catalog=(),
        static_module_guidance=(),
    )
    transcript.start_instruction(
        "解释两轮请求如何使用当前状态和历史。", "demo_instruction", envelope
    )
    for index in (1, 2):
        fragment = ContextFragment("demo", {"round": index})
        fingerprint = checkpoint_fingerprint(f"obs_{index}", (fragment,))
        transcript.append_state(
            StateCheckpoint(
                checkpoint_id=f"checkpoint_{index:06d}",
                reason=("instruction_start" if index == 1 else "physical_window_completed"),
                observation_id=f"obs_{index}",
                fragments=(fragment,),
                fingerprint=fingerprint,
            )
        )
        request, audit = build_request(transcript)
        safe_request, _ = snapshot_request(encode_request(settings, request))
        prefix = output / f"round_{index:02d}"
        _write(prefix.with_name(prefix.name + "_request.json"), {
            "mock": mode == "mock", "request": safe_request, "audit": audit,
        })
        transport: dict[str, Any] = {}
        try:
            response = client.request(request, transport_trace=transport)
        except ProviderRequestError as exc:
            # Never persist arbitrary exception text: servers may echo secrets.
            _write(prefix.with_name(prefix.name + "_failure.json"), {
                "error_type": type(exc).__name__, "category": exc.category,
                "retry_exhausted": exc.retry_exhausted, "transport": transport,
            })
            raise
        _write(prefix.with_name(prefix.name + "_response.json"), {
            "response": public_response_snapshot(response.data), "transport": transport,
        })
        transcript.append_assistant(AssistantTurn(f"turn_{index}", response.text or ""))
    _write(output / "final_history.json", transcript.to_dict())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--env-path", default=".env")
    args = parser.parse_args(argv)
    if args.real:
        config = load_model_config(args.env_path)
        if not config.api_key:
            parser.error("real mode requires configured provider credentials")
        settings = config.to_provider_settings()
        client = create_model_client(settings)
    else:
        settings, client = ProviderSettings(model_name="offline-demo"), MockClient()
    try:
        run_demo(args.output_dir, client=client, settings=settings,
                 mode="real" if args.real else "mock")
    except FileExistsError:
        print("Output directory already exists; existing records were not changed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
