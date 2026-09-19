"""Contracts for pure image selection, immutable history and provider projection."""

import base64
import copy
import json
from io import BytesIO

import pytest
from PIL import Image

from src.agent.history_projection import HistoryProjector
from src.agent.transcript import AgentTranscript, AgentToolCall, AssistantTurn, ToolResult
from src.memory.context.modules.image import DefaultImageContextModule
from src.memory.context.modules.base import ContextModule
from src.memory.context.registry import ContextModuleRegistry, ContextModuleConfigError
from src.memory.context.schemas import HistoryImageCandidate
from src.memory.stores import MemoryEventWriter
from src.tools.registry import default_registry


POLICY = {"mode": "recency_log", "max_observation_images": 20, "keep_recent_observations": 3}


def projector():
    return HistoryProjector({"image": (DefaultImageContextModule(), POLICY)})


def add_state(transcript, number, *, missing=False):
    # Identical bytes for ALL observations deliberately exercise no-motion frames.
    output = BytesIO()
    Image.new("RGB", (4, 4), "blue").save(output, format="PNG")
    media = {
        "type": "image", "role": "current_observation", "encoding": "data_url",
        "data": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode(),
        "observation_id": f"obs_{number}", "frame_id": f"frame_{number}",
    }
    transcript.append_state({
        "current_phase": {"name": f"phase_{number // 10}"},
        "phase_strategy": {"active_stage": number},
        "fragments": [{"name": "image", "model_inputs": [] if missing else [media]}],
    }, f"obs_{number}")


def history(tmp_path, total):
    transcript = AgentTranscript(MemoryEventWriter(tmp_path))
    transcript.start_instruction("Inspect the route", "inst_1")
    for number in range(1, total + 1):
        add_state(transcript, number)
    return transcript


@pytest.mark.parametrize("total", [0, 1, 3, 20, 21, 100, 1000])
def test_sampling_count_recent_first_order_and_isolation(tmp_path, total):
    transcript = history(tmp_path, total)
    before = copy.deepcopy(transcript.to_dict())
    result = projector().project(transcript, default_registry())
    chosen = result.audit["policies"]["image"]["selected_observation_ids"]
    assert len(chosen) == min(total, 20)
    assert len(set(chosen)) == len(chosen)
    assert chosen == sorted(chosen, key=lambda value: int(value.split("_")[1]))
    if total:
        assert chosen[0] == "obs_1"
        assert chosen[-min(total, 3):] == [f"obs_{i}" for i in range(max(1, total-2), total+1)]
    assert result.audit["observation_image_count"] == min(total, 20)
    assert result == projector().project(transcript, default_registry())
    assert transcript.to_dict() == before
    # Every state and its strategy remain at their original chronological place.
    assert len(result.messages) == total + 1
    for number, message in enumerate(result.messages[1:], 1):
        context = json.loads(message["content"][-1]["text"])
        assert context["phase_strategy"]["active_stage"] == number


def test_known_log_sample():
    candidates = tuple(HistoryImageCandidate(f"obs_{i}", i-1) for i in range(1, 101))
    selected = DefaultImageContextModule().select_history_images(candidates, POLICY)
    assert selected == tuple(f"obs_{i}" for i in [1,19,34,46,56,64,71,77,81,85,88,90,92,94,95,96,97,98,99,100])


@pytest.mark.parametrize("changes", [
    {"mode": "unknown"}, {"max_observation_images": True},
    {"max_observation_images": 20.0}, {"keep_recent_observations": 0},
    {"max_observation_images": 4}, {"unexpected": 1},
])
def test_invalid_policy_fails(changes):
    with pytest.raises(ValueError):
        HistoryProjector({"image": (DefaultImageContextModule(), {**POLICY, **changes})})


def test_all_mode_and_snapshot_copy(tmp_path):
    transcript = history(tmp_path, 23)
    baseline = transcript.provider_messages(default_registry())
    all_mode = HistoryProjector({"image": (DefaultImageContextModule(), {"mode": "all"})})
    assert all_mode.project(transcript, default_registry()).messages == baseline
    snapshot = transcript.snapshot()
    snapshot[1].payload["context"]["phase_strategy"]["active_stage"] = 999
    assert transcript.snapshot()[1].payload["context"]["phase_strategy"]["active_stage"] == 1


def test_repeated_checkpoint_missing_current_and_tool_result(tmp_path):
    transcript = history(tmp_path, 25)
    call = AgentToolCall("call_1", "uav.hover", {})
    transcript.append_assistant(AssistantTurn("a1", "Hold.", [call]))
    evidence = tmp_path / "evidence.png"
    Image.new("RGB", (2, 2)).save(evidence)
    transcript.append_tool_result(ToolResult("call_1", "uav.hover", "ok", model_inputs=[{
        "type": "image", "path": str(evidence), "observation_id": "obs_25",
    }]))
    transcript.append_protocol_feedback("Review the evidence.")
    add_state(transcript, 25, missing=True)
    result = projector().project(transcript, default_registry())
    assert result.audit["observation_count"] == 25
    assert result.audit["observation_image_count"] == 19
    assert result.audit["tool_image_count"] == 1
    assert result.audit["missing_images"] == [{"observation_id": "obs_25", "reason": "no_provider_ready_image"}]
    text = json.dumps(result.messages)
    assert "CURRENT observation_id=" not in text
    assert "HISTORY observation_id=obs_24" in text
    assert "Observation image unavailable" in text
    full = HistoryProjector().project(transcript, default_registry()).messages
    assert [m for m in full if m["role"] in {"assistant", "tool"}] == [m for m in result.messages if m["role"] in {"assistant", "tool"}]
    assert result.messages[-2]["content"] == "Review the evidence."


class ReplacementImage(DefaultImageContextModule):
    def validate_history_policy(self, config):
        if config != {"mode": "last"}:
            raise ValueError("unsupported")

    def select_history_images(self, candidates, config):
        return tuple(item.observation_id for item in candidates[-1:])


def test_registry_routes_actual_replacement_and_copies_settings(tmp_path):
    replacement = ReplacementImage()
    config = {"image": {"mode": "last"}}
    registry = ContextModuleRegistry({"image": replacement}, history_policies=config)
    config["image"]["mode"] = "changed"
    assert registry.history_bindings()["image"][0] is replacement
    result = HistoryProjector(registry.history_bindings()).project(history(tmp_path, 4), default_registry())
    assert result.audit["observation_image_count"] == 1
    assert result.audit["policies"]["image"]["selected_observation_ids"] == ["obs_4"]
    with pytest.raises(ContextModuleConfigError):
        ContextModuleRegistry({}, history_policies={"image": POLICY})


@pytest.mark.parametrize("output", [("obs_1", "obs_1"), ("unknown",), (1,)])
def test_bad_plugin_selection_rejected(tmp_path, output):
    module = ReplacementImage()
    module.select_history_images = lambda candidates, config: output
    with pytest.raises(ValueError, match="Invalid history selection"):
        HistoryProjector({"image": (module, {"mode": "last"})}).project(history(tmp_path, 4), default_registry())


def test_optional_base_interface_does_not_enable_unknown_modes():
    class ExistingModule(ContextModule):
        name = "existing"
        def build(self, *args): pass
        def describe_output(self): return {}
    module = ExistingModule()
    assert module.select_history_images((HistoryImageCandidate("obs_1", 0),), {}) == ("obs_1",)
    with pytest.raises(ValueError):
        module.validate_history_policy(POLICY)


def test_review_distinguishes_actual_and_preview_and_escapes():
    from evaluation.benchmarks.core.review import history_projection_review
    audit = {"policies": {"image": {"selected_observation_ids": ["<script>"]}}}
    assert "Captured request image selection" in history_projection_review(audit)
    preview = history_projection_review(audit, preview=True)
    assert "NOT sent to model" in preview
    assert "<script>" not in preview and "&lt;script&gt;" in preview
    assert history_projection_review(None) == ""


def test_planner_uses_injected_projector_and_hides_selection_audit(tmp_path):
    from tests.test_model_action_planner import StubPlanner, response_with_calls
    planner = StubPlanner(response_with_calls(("h1", "uav_hover", {})))
    planner.history_projector = projector()
    planner.plan_turn(history(tmp_path, 25), default_registry())
    audit = planner.last_exchange["history_projection"]
    assert audit["observation_image_count"] == 20
    assert audit["policies"]["image"]["omitted_count"] == 5
    assert "selected_observation_ids" not in json.dumps(planner.payloads[0])


def test_json_configuration_binds_selector_and_rejects_disabled_slot():
    registry = ContextModuleRegistry.from_dict({
        "context_modules": {"image": "default"}, "context_history": {"image": POLICY},
    })
    assert registry.history_bindings()["image"][0] is registry.modules["image"]
    with pytest.raises(ContextModuleConfigError):
        ContextModuleRegistry.from_dict({
            "context_modules": {"image": False}, "context_history": {"image": POLICY},
        })
    with pytest.raises(ContextModuleConfigError):
        ContextModuleRegistry.from_dict({"context_history": []})


def test_old_reference_cannot_escape_sampling(tmp_path):
    transcript = history(tmp_path, 25)
    current = transcript.latest_context()
    current["fragments"][0]["model_inputs"].insert(0, {
        **current["fragments"][0]["model_inputs"][0],
        "observation_id": "obs_2", "frame_id": "frame_2",
    })
    transcript.append_state(current, "obs_25")
    result = projector().project(transcript, default_registry())
    assert result.audit["observation_image_count"] == 20
    assert json.dumps(result.messages).count("CURRENT observation_id=obs_25") == 1


def test_offline_asset_containment_and_sha_checks(tmp_path):
    from tests.manual.history_image_sampling_probe import _captured_media
    output = tmp_path / "output"
    (output / "review_assets").mkdir(parents=True)
    source = tmp_path / "run"
    source.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (2, 2)).save(outside)
    model_round = {"request_media": [{"asset_ref": "../outside.png"}]}
    _, _, errors = _captured_media(model_round, source, output)
    assert errors == ["missing or unsafe captured asset"]
    assert not list((output / "review_assets").iterdir())


def test_offline_probe_refuses_existing_output_before_writes(tmp_path):
    from tests.manual.history_image_sampling_probe import run_probe
    case = tmp_path / "case"
    case.mkdir()
    (case / "audit.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(ValueError, match="must be new"):
        run_probe(case, out, tmp_path / "unused.json")
