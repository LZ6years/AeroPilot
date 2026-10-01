"""Evidence projection and diagnostic invariants without a model or simulator."""
import copy

from PIL import Image

from tests.manual.airsim_action_regression_probe import collision_change
from tests.manual.navigation_decision_probe import historical_transcript, SYNTHETIC
from tests.test_model_action_planner import StubPlanner, make_transcript, response_with_calls
from src.plugins import PluginRegistry
from src.tools.registry import default_registry
from src.memory.context.contracts import (
    INSTRUCTION_ENVELOPE_SCHEMA,
    STATE_CHECKPOINT_SCHEMA,
    TRANSCRIPT_SCHEMA,
)


def test_new_grounding_contract_reaches_provider(tmp_path):
    planner = StubPlanner(response_with_calls(("c", "skill_visual_grounding_measure_visible_depth",
        {"observation_id": "obs_1", "query": "object"})))
    planner.plan_turn(
        make_transcript(tmp_path),
        default_registry(PluginRegistry.from_config()),
    )
    tool = next(t["function"] for t in planner.last_request["tools"]
                if t["function"]["name"] == "skill_visual_grounding_measure_visible_depth")
    assert set(tool["parameters"]["properties"]) == {"observation_id", "query"}
    assert tool["parameters"]["additionalProperties"] is False
    assert "surface" in tool["description"]


def test_historical_probe_attaches_exact_original_and_preserves_source(tmp_path):
    path = tmp_path / "original.png"
    Image.new("RGB", (8, 8), "red").save(path)
    sample = {"frames": {"obs_1": str(path)}, "events": [
        {"schema_version": TRANSCRIPT_SCHEMA, "kind": "instruction_envelope", "payload": {
            "schema_version": INSTRUCTION_ENVELOPE_SCHEMA, "instruction_id": "source",
            "protocol_context": "", "identity": {}, "user_preferences": {},
            "knowledge_catalog": [], "static_module_guidance": []}},
        {"schema_version": TRANSCRIPT_SCHEMA, "kind": "user_instruction", "payload": {"instruction": "Inspect"}},
        {"schema_version": TRANSCRIPT_SCHEMA, "kind": "state_checkpoint", "payload": {
            "schema_version": STATE_CHECKPOINT_SCHEMA, "checkpoint_id": "checkpoint_0001",
            "reason": "instruction_start", "observation_id": "obs_1",
            "task_plan": None, "loaded_knowledge": [], "fingerprint": "source",
            "fragments": [{"name": "image", "model_content": {}, "model_inputs": []}]}}],
        "evaluation_truth": {"hidden_target": "must not be projected"}}
    before = copy.deepcopy(sample)
    transcript = historical_transcript(sample, tmp_path / "probe")
    planner = StubPlanner(response_with_calls(("c", "uav_turn_relative", {"yaw_deg": 10})))
    planner.plan_turn(transcript, default_registry())
    assert planner.last_exchange["media"][0]["data"] == path.read_bytes()
    assert "hidden_target" not in str(planner.last_request)
    assert sample == before
    assert len({s[0] for s in SYNTHETIC}) == len(SYNTHETIC)


def test_collision_name_alone_is_not_new_contact():
    before = {"collision": {"time_stamp": 20, "object_name": "old", "has_collided": False}}
    after = copy.deepcopy(before)
    assert not collision_change(before, after)["new_contact_timestamp"]
    after["collision"]["time_stamp"] = 30
    assert collision_change(before, after)["new_contact_timestamp"]
    assert collision_change(before, after)["reported_contact"] is False
