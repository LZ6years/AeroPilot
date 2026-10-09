# Model communication boundary

This package is stateless. It knows neither task phases nor skills, benchmarks,
observation IDs, context modules, nor transcript selection rules. Callers own
message order, prompts, tools, and interpretation of the result.

## Ownership

| Component | Responsibility |
| --- | --- |
| Caller | Build the complete request and interpret the normalized response |
| `ModelRequest` / `ModelResponse` | Stable provider-independent request seam |
| `factory.py` | Construct the supported client |
| `DefaultModelClient` | Call `openai.OpenAI`, capture safe audit data, normalize output |
| `providers/openai_compatible.py` | Chat Completions fields and buffered SSE normalization |
| `providers/openai_responses.py` | Responses input translation and Chat-style output normalization |
| `audit.py` | Separate image bytes and redact diagnostic snapshots |

The only production provider name is `openai_compatible`. It supports OpenAI
and compatible third-party gateways through the official OpenAI Python SDK.
Both Chat Completions and Responses use the same client contract. Context V2
continues to use Chat-style `system/user/assistant/tool` messages internally;
the Responses adapter translates only at the provider boundary.

`DefaultModelClient` reuses one synchronous SDK client. It does not set a
top-level Responses `instructions` field, use `previous_response_id`, retain
conversation state, or execute tools. Responses requests always set
`store=false`. Unknown compatible-provider generation fields are sent through
the SDK's `extra_body` mechanism.

The two historical retry settings remain accepted for configuration
compatibility. The SDK receives their maximum as `max_retries`; its internal
wire-attempt count is not exposed or fabricated in audits. Transport audits
record the SDK version, configured retry limit, final category, status code,
request ID when available, and elapsed time.

## Calling from an application

```python
from src.llm import MessageBuilder, ModelRequest, create_model_client

client = create_model_client(model_config.to_provider_settings())
builder = MessageBuilder()
builder.add_system(system_text, key="instructions")
builder.extend(selected_history_messages, key="history")
builder.add_user(user_content, key="current")

response = client.request(
    ModelRequest(messages=builder.build(), parameters={"temperature": 0}),
)
print(response.text)
```

Each request includes the complete caller-owned history. `AgentTranscript`
remains the only conversation source of truth. Images are separated from JSON
audit snapshots, and callers persist only the safe public snapshots.

Chat streaming is consumed through the SDK raw-response interface so existing
ToolCall-delta merging and ordinary-JSON fallback remain available. A stream is
never restarted after consumption begins; if a provider disconnects after a
complete terminal assistant turn, that buffered turn may still be accepted.

## Verification

```powershell
& 'F:\conda_env\flyagent\python.exe' -m pytest -q tests/test_llm.py tests/test_llm_callers.py tests/test_provider_http.py
& 'F:\conda_env\flyagent\python.exe' -m pytest -q
```
