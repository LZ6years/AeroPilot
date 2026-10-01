"""OpenAI-compatible native tool-call planner for FlyAgent.

This provider boundary has no JSON-plan fallback. It consumes the canonical
AgentTranscript and returns one AssistantTurn whose tool calls are resolved to
FlyAgent's canonical ToolSpec names.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.llm import DefaultModelClient, MessageBuilder, ModelClient, ModelRequest
from src.llm.audit import snapshot_request, redact_provider_value
from src.llm.providers.openai_compatible import encode_request, public_response_snapshot
from src.agent.context.compression import (
    HistoryCompactionCoordinator,
    HistoryProjectionState,
)
from src.agent.context.projection import HistoryProjector
from src.agent.context.window_manager import ContextWindowManager
from src.agent.config import (
    ModelConfig,
    PlannerConfig,
    ProviderConfig,
    ToolLoopConfig,
    load_model_config,
)
from src.agent.transcript import (
    AgentToolCall,
    AgentTranscript,
    AssistantTurn,
    PublicIntentSource,
)

if TYPE_CHECKING:
    from src.tools.registry import ToolRegistry


class ModelActionPlannerConfigError(RuntimeError):
    """Raised when the native tool-call planner has unusable configuration."""


class AgentProtocolError(ValueError):
    """Raised when a provider response cannot be represented canonically.

    ``rejected_turn`` contains only the safe response fields needed to audit
    what the provider attempted. It intentionally excludes private reasoning
    fields and arbitrary provider metadata.
    """

    def __init__(
        self,
        message: str,
        *,
        rejected_turn: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.rejected_turn = dict(rejected_turn or {})


class ModelActionPlanner:
    planner_mode = "model"

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        prompt_path: str | None = None,
        *,
        max_repair_attempts: int | None = None,
        repair_planner: Any | None = None,
        history_projector: HistoryProjector | None = None,
        context_window_manager: ContextWindowManager | None = None,
        client: ModelClient | None = None,
        provider_config: ProviderConfig | None = None,
        planner_config: PlannerConfig | None = None,
        tool_loop_config: ToolLoopConfig | None = None,
    ) -> None:
        if max_repair_attempts is not None or repair_planner is not None:
            raise ModelActionPlannerConfigError(
                "The native tool loop repairs errors in transcript; separate repair planners are unsupported."
            )
        if model_config is not None and any(
            item is not None
            for item in (provider_config, planner_config, tool_loop_config)
        ):
            raise ModelActionPlannerConfigError(
                "pass either ModelConfig or the independent planner configs"
            )
        aggregate = model_config
        if aggregate is None and all(
            item is None
            for item in (provider_config, planner_config, tool_loop_config)
        ):
            aggregate = load_model_config()
        if aggregate is not None:
            provider_config = aggregate.provider_config()
            planner_config = aggregate.planner_config()
            tool_loop_config = aggregate.tool_loop_config()
        self.provider_config = provider_config or ProviderConfig()
        self.planner_config = planner_config or PlannerConfig()
        self.tool_loop_config = tool_loop_config or ToolLoopConfig()
        # Compatibility for standalone diagnostics; production composition
        # passes the three narrow contracts above.
        self.model_config = aggregate or ModelConfig(
            provider=self.provider_config.provider,
            model_name=self.provider_config.model_name,
            api_key=self.provider_config.api_key,
            base_url=self.provider_config.base_url,
            thinking_mode=self.provider_config.thinking_mode,
            timeout_sec=self.provider_config.timeout_sec,
            provider_timeout_retries=self.provider_config.timeout_retries,
            provider_connection_retries=self.provider_config.connection_retries,
            max_tool_rounds=self.tool_loop_config.max_rounds,
            max_consecutive_protocol_rejections=(
                self.tool_loop_config.max_consecutive_protocol_rejections
            ),
            action_planner_prompt_path=self.planner_config.prompt_path,
            task_progress_reminder_interval=(
                self.planner_config.task_progress_reminder_interval
            ),
        )
        self.client = client if client is not None else DefaultModelClient(
            self.provider_config.to_provider_settings()
        )
        # Standalone callers default to full history. Runtime injects the
        # configured projector bound to its actual Context Module instances.
        self.history_projector = history_projector or HistoryProjector()
        self.context_window_manager = context_window_manager
        configured_prompt = prompt_path or self.planner_config.prompt_path or None
        self.system_prompt = self._load_system_prompt(configured_prompt)
        prompt_dir = Path(__file__).resolve().parent / "prompts"
        self.tool_user_prompt = (prompt_dir / "tool_user_system.txt").read_text(encoding="utf-8").strip()
        self.task_progress_reminder = (prompt_dir / "task_progress_reminder.txt").read_text(encoding="utf-8").strip()
        self.last_request: dict[str, Any] = {}
        self.last_response: dict[str, Any] = {}
        # In-memory only until Runtime writes a decision artifact. Image bytes
        # are separated from the JSON request so the durable record can point
        # at content-addressed assets without embedding base64.
        self.last_exchange: dict[str, Any] = {}

    def plan_turn(
        self,
        transcript: AgentTranscript,
        registry: "ToolRegistry",
        *,
        allowed_tool_names: set[str] | None = None,
        decision_index: int | None = None,
        history_projection_state: HistoryProjectionState | None = None,
        history_compaction_coordinator: HistoryCompactionCoordinator | None = None,
    ) -> AssistantTurn:
        """Request one turn; Runtime supplies a one-based, instruction-local index.

        Standalone callers may omit the index to disable periodic reminders.
        Transport retries reuse this request and never advance the decision clock.
        """

        if decision_index is not None and (type(decision_index) is not int or decision_index < 1):
            raise ValueError("decision_index must be a positive integer or None")

        if not self.provider_config.api_key:
            raise ModelActionPlannerConfigError(
                "Model action planner requires FLYAGENT_MODEL_API_KEY"
            )
        if self.provider_config.provider != "openai_compatible":
            raise ModelActionPlannerConfigError(
                f"Unsupported model provider: {self.provider_config.provider}"
            )
        specs = registry.model_specs(allowed_tool_names)
        if not specs:
            raise ModelActionPlannerConfigError(
                "at least one model tool must be exposed in a mission turn"
            )
        envelope = transcript.instruction_envelope()
        prefix = MessageBuilder()
        prefix.extend(
            self.history_projector.assembler.memory_system_messages(envelope),
            key="memory_documents",
        )
        prefix.add_system(self.system_prompt, key="planner_instruction")
        prefix.add_system(self.tool_user_prompt, key="tool_user_instruction")
        interval = self.planner_config.task_progress_reminder_interval
        reminder: dict[str, Any] | None = None
        # A transient request instruction, never a durable Transcript event.
        # Append after the complete history so tool call/result pairs stay intact.
        if (interval and decision_index is not None and decision_index > 1
                and (decision_index - 1) % interval == 0):
            reminder = {"role": "system", "content": self.task_progress_reminder}
        parameters = {"tool_choice": "auto", "temperature": 0}
        if self.provider_config.thinking_mode:
            parameters["thinking"] = {"type": self.provider_config.thinking_mode}
        provider_tools = [spec.to_provider_tool() for spec in specs]
        fixed_messages = prefix.build()
        if reminder is not None:
            fixed_messages = [*fixed_messages, reminder]
        projection = (
            self.context_window_manager.project(
                transcript,
                registry,
                state=history_projection_state,
                coordinator=history_compaction_coordinator,
                fixed_messages=fixed_messages,
                tools=provider_tools,
                parameters=parameters,
            )
            if self.context_window_manager is not None
            else self.history_projector.project(transcript, registry)
        )
        builder = MessageBuilder()
        builder.extend(prefix.build(), key="fixed_prefix")
        builder.extend(projection.messages, key="history")
        if reminder is not None:
            builder.extend([reminder], key="task_progress_reminder")
        request = ModelRequest(
            messages=builder.build(),
            tools=provider_tools,
            parameters=parameters,
        )
        # Preserve the existing subclass hook and failed-request audit, while
        # delegating wire-envelope construction to the provider adapter.
        payload = encode_request(self.provider_config.to_provider_settings(), request)
        request_snapshot, media = snapshot_request(payload)
        annotate_request_media(payload, media)
        self.last_request = request_snapshot
        self.last_response = {}
        self.last_exchange = {
            "request": copy.deepcopy(request_snapshot),
            "media": media,
            "response": None,
            "history_projection": projection.audit,
        }
        data = self._request(payload)
        if self.context_window_manager is not None:
            usage = data.get("usage") if isinstance(data, dict) else None
            self.context_window_manager.observe_usage(
                history_projection_state,
                usage if isinstance(usage, dict) else None,
            )
            if isinstance(usage, dict):
                actual = usage.get("prompt_tokens", usage.get("input_tokens"))
                if type(actual) is int and actual > 0:
                    projection.audit["actual_input_tokens"] = actual
        self.last_response = dict(data)
        self.last_exchange["response"] = public_response_snapshot(data)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            error = "provider response missing choices[0].message"
            raise AgentProtocolError(
                error,
                rejected_turn=_provider_rejection_snapshot(data, error),
            ) from exc
        if not isinstance(message, dict):
            error = "provider choices[0].message must be an object"
            raise AgentProtocolError(
                error,
                rejected_turn=_provider_rejection_snapshot(data, error),
            )
        try:
            return _parse_assistant_turn(message, registry)
        except AgentProtocolError as exc:
            raise AgentProtocolError(
                str(exc),
                rejected_turn=_provider_rejection_snapshot(data, str(exc)),
            ) from exc

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        # Transport attempts are debug evidence, not transcript turns. The
        # shared helper updates this object in place even when it ultimately
        # raises, so Runtime can persist an exhausted retry sequence.
        transport_trace: dict[str, Any] = {}
        self.last_exchange["transport"] = transport_trace
        request = ModelRequest(
            messages=payload["messages"],
            tools=payload.get("tools"),
            parameters={key: value for key, value in payload.items()
                        if key not in {"model", "messages", "tools"}},
        )
        return self.client.request(
            request, error_prefix="Native tool-call request failed",
            transport_trace=transport_trace,
        ).data

    @staticmethod
    def _load_system_prompt(prompt_path: str | None) -> str:
        path = (
            Path(prompt_path)
            if prompt_path
            else Path(__file__).resolve().parent / "prompts" / "agent_tool_loop_system.txt"
        )
        return path.read_text(encoding="utf-8").strip()


def _parse_assistant_turn(message: dict[str, Any], registry: "ToolRegistry") -> AssistantTurn:
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise AgentProtocolError("assistant tool_calls must be a list")
    calls: list[AgentToolCall] = []
    seen: set[str] = set()
    for raw in raw_calls:
        if not isinstance(raw, dict):
            raise AgentProtocolError("assistant tool call must be an object")
        if raw.get("type") != "function":
            raise AgentProtocolError("assistant tool call type must be function")
        call_id = str(raw.get("id", "")).strip()
        function = raw.get("function")
        if not call_id or call_id in seen or not isinstance(function, dict):
            raise AgentProtocolError("assistant tool calls require unique ids and function objects")
        seen.add(call_id)
        wire_name = str(function.get("name", "")).strip()
        if not wire_name:
            raise AgentProtocolError(
                f"provider tool call {call_id} has an empty function name"
            )
        try:
            spec = registry.from_wire_name(wire_name)
        except KeyError as exc:
            raise AgentProtocolError(f"unknown provider tool: {wire_name}") from exc
        try:
            args = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            raw = function.get("arguments") or "{}"
            nearby = raw[max(0, exc.pos - 32):exc.pos + 48]
            raise AgentProtocolError(
                f"tool {wire_name} arguments are not valid JSON: {exc.msg}; "
                f"line {exc.lineno}, column {exc.colno}, offset {exc.pos}; "
                f"near {json.dumps(nearby, ensure_ascii=True)}. "
                "Resubmit a valid JSON object preserving the intended action. "
                "JSON numbers cannot have leading zeros (for example 00.3 is invalid); "
                "use valid decimal syntax such as 0.3 or -0.3 only if that is your intended value. "
                "No call in this turn executed."
            ) from exc
        except TypeError as exc:
            raise AgentProtocolError(
                f"tool {wire_name} arguments are not valid JSON"
            ) from exc
        if not isinstance(args, dict):
            raise AgentProtocolError(f"tool {wire_name} arguments must be an object")
        calls.append(AgentToolCall(call_id, spec.name, args))
    content = message.get("content")
    public_intent = content.strip() if isinstance(content, str) else ""
    if not calls:
        raise AgentProtocolError(
            "assistant turn must call a tool; use runtime.report_completed or runtime.report_blocked to finish the task"
        )
    public_intent_source: PublicIntentSource = "provider_content"
    if not public_intent:
        public_intent = _fallback_public_intent(calls)
        public_intent_source = "tool_arguments_fallback"
    turn_id = str(message.get("id") or f"assistant_{calls[0].tool_call_id}")
    return AssistantTurn(
        turn_id=turn_id,
        public_intent=public_intent,
        tool_calls=calls,
        public_intent_source=public_intent_source,
    )


def _fallback_public_intent(calls: list[AgentToolCall]) -> str:
    """Describe model-authored Tool arguments without inventing reasoning."""

    if len(calls) == 1:
        observation_id = calls[0].args.get("observation_id")
        if isinstance(observation_id, str) and observation_id.strip():
            return f"Request {calls[0].tool_name} for observation {observation_id.strip()}."
    names = ", ".join(call.tool_name for call in calls)
    return f"Request tool call(s): {names}."


def _provider_rejection_snapshot(
    response: dict[str, Any], error: str
) -> dict[str, Any]:
    """Extract auditable wire evidence without persisting hidden reasoning."""

    choices = response.get("choices")
    choices = choices if isinstance(choices, list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message")
    message = message if isinstance(message, dict) else {}
    content = message.get("content")
    raw_calls = message.get("tool_calls")
    raw_calls = raw_calls if isinstance(raw_calls, list) else []
    tool_calls = []
    for raw in raw_calls:
        raw = raw if isinstance(raw, dict) else {}
        function = raw.get("function")
        function = function if isinstance(function, dict) else {}
        arguments = function.get("arguments")
        parsed_arguments: Any = arguments
        arguments_status = "missing"
        if arguments is not None:
            try:
                parsed_arguments = (
                    json.loads(arguments)
                    if isinstance(arguments, str)
                    else arguments
                )
                arguments_status = (
                    "parsed" if isinstance(parsed_arguments, dict) else "invalid"
                )
            except (json.JSONDecodeError, TypeError):
                arguments_status = "invalid"
        tool_calls.append(
            {
                "tool_call_id": str(raw.get("id") or ""),
                "type": raw.get("type"),
                "wire_name": str(function.get("name") or ""),
                "arguments": redact_provider_value(parsed_arguments),
                "arguments_status": arguments_status,
            }
        )
    return {
        "error": error,
        "finish_reason": choice.get("finish_reason"),
        "content_present": bool(content.strip()) if isinstance(content, str) else False,
        "content": redact_provider_value(content) if isinstance(content, str) else None,
        "tool_calls": tool_calls,
    }


def annotate_request_media(
    payload: dict[str, Any], media: list[dict[str, Any]],
) -> None:
    """Add agent-owned temporal/frame labels to generic communication snapshots."""
    for item in media:
        item.update(_request_media_label(payload, item.get("request_path")))


def _request_media_label(
    payload: dict[str, Any], request_path: Any
) -> dict[str, Any]:
    """Copy the adjacent model-visible image label into the audit snapshot."""

    if not isinstance(request_path, list) or len(request_path) < 6:
        return {}
    try:
        messages_index = request_path.index("messages")
        message_index = int(request_path[messages_index + 1])
        content_index = int(request_path[messages_index + 3])
        content = payload["messages"][message_index]["content"]
        label = content[content_index - 1]
    except (ValueError, TypeError, KeyError, IndexError):
        return {}
    if not isinstance(label, dict) or label.get("type") != "text":
        return {}
    text = label.get("text")
    if not isinstance(text, str):
        return {}
    lines = text.splitlines()
    if not lines or not lines[0].startswith(
        ("HISTORY ", "CURRENT ", "TOOL EVIDENCE ")
    ):
        return {}
    first = lines[0].split()
    role = "TOOL_EVIDENCE" if first[:2] == ["TOOL", "EVIDENCE"] else first[0]
    result: dict[str, Any] = {"timeline_role": role, "label": text}
    for field in first[1:]:
        key, separator, value = field.partition("=")
        if separator and key in {"observation_id", "frame_id", "tool_call_id"}:
            result[key] = value
    for line in lines[1:]:
        for key in ("scene_size", "display_size", "scene_bounds_in_display_pixels"):
            prefix = f"{key}="
            if line.startswith(prefix):
                result[key] = line.removeprefix(prefix)
    return result
