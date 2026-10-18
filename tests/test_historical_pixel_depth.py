"""Exercise source-time measurements, task authorization and per-source quotas."""
import json

import numpy as np
from PIL import Image
import pytest

from plugins.capabilities.pixel_depth.measure_pixel_depth.scripts.handler import handle
from src.memory.stores import MemoryEventWriter, MemoryRecordReader
from src.tools.schemas import ToolRuntimePolicy, ToolSpec
from src.tools.registry import ToolRegistry
from src.agent.transcript import AssistantTurn, AgentToolCall
from src.agent.planning.tool_contract_validator import validate_native_turn, NativeTurnValidationError


def archived_task(tmp_path, count=61):
    writer = MemoryEventWriter(tmp_path)
    writer.write_instruction("Find the described vehicle")
    ids = []
    for index in range(count):
        rgb = tmp_path / f"rgb_{index}.png"
        depth = tmp_path / f"depth_{index}.npy"
        Image.new("RGB", (8, 6), (index, 20, 30)).save(rgb)
        np.save(depth, np.full((6, 8), index + 7.25, dtype=np.float32))
        record = writer.write_observation({"timestamp": f"2026-10-18T00:00:{index:02d}Z",
            "scene_instance_id": "scene1", "agent_id": "uav1", "frames": {
                "rgb": {"path": str(rgb), "camera_name": "front"},
                "depth": {"path": str(depth), "camera_name": "front",
                          "depth_unit": "meter", "capture_source": "depth_perspective"}}})
        ids.append(record["observation_ref"]["id"])
    reader = MemoryRecordReader(tmp_path)
    context = {"latest_observation": reader.read_observation(ids[-1]),
               "run_dir": str(tmp_path), "instruction_id": "inst_0001",
               "evidence_refs": ids, "tool_call_id": "depth1"}
    return ids, context


def measure(context, observation_id):
    return handle({"runtime_context": context, "args": {
        "observation_id": observation_id, "pixel_x": 3, "pixel_y": 2}})


def test_oldest_frame_uses_its_own_depth_and_never_changes_current_state(tmp_path):
    ids, context = archived_task(tmp_path)
    source_bytes = (tmp_path / "memory/observations.jsonl").read_bytes()
    result = measure(context, ids[0])
    assert result.data["distance_m"] == 7.2
    assert result.data["observation_role"] == "HISTORY"
    assert result.data["evidence_origin"] == "historical_memory"
    assert result.data["observation_timestamp"] == "2026-10-18T00:00:00Z"
    media = result.model_inputs[0]["media_ref"]
    assert media["observation_id"] == ids[0]
    assert "HISTORY" in media["usage"] and "current distance" in media["usage"]
    assert context["latest_observation"]["observation_id"] == ids[-1]
    assert (tmp_path / "memory/observations.jsonl").read_bytes() == source_bytes
    assert measure(context, ids[-1]).data["observation_role"] == "CURRENT"


def test_unavailable_and_other_instruction_observations_are_rejected(tmp_path):
    ids, context = archived_task(tmp_path, 2)
    context["evidence_refs"] = [ids[-1]]
    with pytest.raises(ValueError, match="active instruction"):
        measure(context, ids[0])
    with pytest.raises(ValueError, match="active instruction"):
        measure(context, "obs_unknown")


def test_missing_old_depth_does_not_fall_back_to_current_depth(tmp_path):
    ids, context = archived_task(tmp_path, 2)
    (tmp_path / "depth_0.npy").unlink()
    result = measure(context, ids[0])
    assert result.data["distance_m"] is None
    assert result.data["distance_status"] == "metric_depth_unavailable"
    assert result.data["observation_role"] == "HISTORY"


def test_mismatched_persisted_frame_is_rejected(tmp_path):
    ids, context = archived_task(tmp_path, 2)
    path = tmp_path / "memory/frames.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[1]["observation_id"] = ids[-1]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatched"):
        measure(context, ids[0])


def test_quota_binds_to_requested_source_including_same_turn_calls():
    policy = ToolRuntimePolicy("read_only", "immediate", "parallel_safe", True,
        max_calls_per_observation=2, observation_id_parameter="observation_id")
    spec = ToolSpec.from_input_schema("measure", "measure", "skill_capability", "measure", {
        "type": "object", "properties": {"observation_id": {"type": "string", "minLength": 1}},
        "required": ["observation_id"]}, policy)
    registry = ToolRegistry()
    registry.register(spec)
    def validate(sources, counts):
        calls = [AgentToolCall(f"call{i}", "measure", {"observation_id": oid}) for i, oid in enumerate(sources)]
        return validate_native_turn(AssistantTurn("turn", "", calls), registry,
            allowed_names={"measure"}, latest_observation={"observation_id": "new"}, executed_read_counts=counts)
    assert validate(["old", "new"], {("old", "measure"): 1}) == "read_only"
    with pytest.raises(NativeTurnValidationError, match="call limit"):
        validate(["old", "old"], {("old", "measure"): 1})
    with pytest.raises(NativeTurnValidationError, match="call limit"):
        validate(["old"], {("old", "measure"): 2})
