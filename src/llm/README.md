# Model communication boundary

This package is stateless. It knows neither task phases nor skills, benchmarks,
observation IDs, context modules or transcript selection rules. There are no
request profiles, recipes, prompt paths or implicit system messages.

## Ownership

| Component | Responsibility |
| --- | --- |
| Caller | Load prompts, choose messages/order/tools/options, interpret responses |
| MessageBuilder | Optional ordered, independently copied message blocks |
| ModelRequest / ModelResponse | Complete request and response envelope |
| DefaultModelClient | Encode, snapshot, send and decode one request |
| providers/openai_compatible.py | Chat Completions wire fields, URL and authorization |
| transport.py | Byte-oriented HTTP, timeout and bounded retry policy |
| audit.py | In-memory image separation and generic redaction |

Only the caller persists diagnostics. Image bytes in response.media are
in-memory data, not JSON-ready audit records; persist assets and replace bytes
with references before writing a report. Use response_snapshot for public
response audit, not the raw data envelope, which may include private provider
fields. Agent-owned temporal labels are attached outside this package.

The current adapter supports OpenAI-compatible Chat Completions only. Message
and tool dictionaries deliberately retain that existing representation;
another provider must implement an explicit adapter rather than silently
dropping unsupported options. This is not a universal multimodal schema.

## Calling from an application

```python
from src.llm import DefaultModelClient, MessageBuilder, ModelRequest

# The application owns configuration loading and content selection.
client = DefaultModelClient(model_config.to_provider_settings())
builder = MessageBuilder()
builder.add_system(system_text, key="instructions")
builder.extend(selected_history_messages, key="history")
builder.add_user(user_content, key="current")

response = client.request(
    ModelRequest(messages=builder.build(), parameters={"temperature": 0}),
)
print(response.text)
# Interpret response.tool_calls in the application; the client never executes them.
```

Builder keys are arbitrary local handles and are never sent to the provider.
get(key), replace(key, messages), and remove(key) fail for unknown keys;
duplicate keys fail. Replacement retains the block position. Empty blocks are
allowed. Reads and builds return independent nested copies. No content-based
deduplication, role reordering or tool-result repair occurs.

ModelRequest may also be built directly. Callers choose JSON response mode,
tool choice and generation limits explicitly; the client supplies no
business-dependent defaults. Each HTTP call receives its complete messages;
the client does not accumulate system prompts or conversation history.

The optional error_prefix defaults to "Model request failed". Framework callers
can still supply a specific diagnostic label.

ModelResponse has read-only conveniences for the first Chat Completions choice:
text returns string content (including an empty string) or None; it never uses
hidden reasoning or synthesizes intent. tool_calls returns a deep copy of the
wire list (arguments remain JSON strings), or [] when absent. usage returns a
deep copy of the provider's usage dictionary, or None, never fabricated zeros.
Malformed or structured content remains in data for caller-owned validation.
The dictionaries in data remain the existing advanced raw access interface;
these convenience properties do not make the entire response deeply immutable.

Ordinary users need not import providers/, transport.py or audit.py.
encode_request constructs the provider's wire envelope, not the conversation;
public_response_snapshot selects public fields for durable diagnostic records,
not for interpreting an answer. These are implementation/audit utilities used
by the client and specialized reporting callers. Basic calls use only
client.request and the response properties.

## Injection, failure and compatibility

Parser, Updater, Planner and Skill Selector accept an optional client implementing
ModelClient.request. A fake can return ModelResponse without opening a socket.
Existing constructors still work and construct the default client.

ModelConfig remains upstream and exports only ProviderSettings. Its URL helper
delegates to the provider adapter. Planner's existing private _request(payload)
hook remains for deterministic subclasses; canonical ToolCall normalization,
fallback public intent and rejection snapshots remain owned by Planner.

HTTP retries operate on the same serialized bytes and authorization headers.
They do not append transcript turns or execute tools. JSON decoding and business
contract failures are not retried by transport. A caller-supplied transport_trace
is populated even when retries exhaust, preserving failure diagnostics.

The old Agent HTTP module is removed, with no duplicate implementation or
import shim. Evaluation-only judges remain outside this migration.

## Verification

From the repository root:

```powershell
& 'F:\conda_env\flyagent\python.exe' -m pytest -q tests/test_llm.py tests/test_llm_callers.py tests/test_provider_http.py
& 'F:\conda_env\flyagent\python.exe' -m pytest -q
```

The first command tests tooling, dependency direction, wire equivalence and
transport failures without a real provider. V2 context assembly and lazy media
hydration can be inspected with `tests/manual/context_assembly_probe.py`; it
never calls a model and does not read V1 runs.
