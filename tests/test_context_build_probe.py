"""Offline experiment checks; constructed inputs are not historical replay."""
from types import SimpleNamespace

import pytest
from PIL import Image

from src.agent.planning.model_action_planner import ModelActionPlanner
from src.agent.transcript import AgentTranscript
from src.perception.context_builder import ObservationContextBuilder
from src.runtime.configs.model_config import ModelConfig
from src.tools.registry import default_registry
from tests.test_context_builder import make_memory, make_state
from tests.manual.context_build_probe import compare_builds, checkpoint_blockers


def test_full_build_and_intercepted_request_equal_with_fewer_reads(tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (64, 48), (20, 80, 130)).save(image)
    state = make_state()
    observations = [{"source": "after_action", "frames": {"rgb": {"path": str(image),
                     "width": 64, "height": 48}}, "modalities": ["rgb"]} for _ in range(12)]
    memory = make_memory(tmp_path, state, observations=observations)
    builder = ObservationContextBuilder()
    captured = []
    class Captured(Exception):
        pass
    class OfflinePlanner(ModelActionPlanner):
        def _request(self, payload):
            captured.append(payload)
            raise Captured
    planner = OfflinePlanner(ModelConfig(api_key="offline-test", model_name="offline",
                                        base_url="https://invalid.test"), client=object())
    registry = default_registry()
    def capture(context):
        transcript = AgentTranscript(SimpleNamespace(write_transcript_event=lambda **kw: None))
        transcript.start_instruction(state.user_instruction, "inst_1")
        transcript.append_state(context, "obs_0012")
        with pytest.raises(Captured):
            planner.plan_turn(transcript, registry, decision_index=6)
        return captured[-1]
    result = compare_builds(builder, state, memory, capture, tmp_path / "comparison", repeats=10)
    assert result["state"] == "passed"
    assert len(result["rows"]) == 20
    baseline, candidate = (result["diagnostics"][name] for name in ("baseline", "candidate"))
    assert candidate["file_reads"] < baseline["file_reads"]
    assert all(count == 1 for count in candidate["files"].values())
    assert candidate["json_parses"] == baseline["json_parses"]
    assert len({r["request_sha256"] for r in result["rows"]}) == 1
    assert any("data:image/png;base64," in str(m) for m in captured[0]["messages"])


def test_comparison_rejects_request_difference(tmp_path):
    state = make_state()
    memory = make_memory(tmp_path, state)
    count = 0
    def capture(context):
        nonlocal count
        count += 1
        return {"changed": count}
    with pytest.raises(ValueError, match="differs"):
        compare_builds(ObservationContextBuilder(), state, memory, capture,
                       tmp_path / "difference", repeats=2)
    assert (tmp_path / "difference/difference.json").exists()


def test_saved_output_or_final_state_is_not_a_historical_input_checkpoint():
    assert checkpoint_blockers({"model_context": {"task_plan": {}},
                                "model_rounds": [], "latest_state": {}, "flight_memory": {}})
