"""Offline regression for manual diagnostic tooling."""
from plugins.capabilities.place_memory.store import PlaceStore
from plugins.capabilities.place_memory.inspect_place.scripts.handler import handle as inspect_place
from tests.test_pixel_place_memory import flight
from tests.test_proactive_place_memory import save_context


def test_pose_audit_checks_receipt_against_original_observation(tmp_path):
    from tests.manual.aerial_vln_pixel_place_audit import audit_capability_records

    memory = flight(tmp_path, [(0, 0, 0), (1, 2, 3)])
    store = PlaceStore(tmp_path, memory.active_instruction_id)
    store.update("obs_0001", "obs_0002")
    result = inspect_place({"runtime_context": save_context(tmp_path, memory), "args": {"place_id": "P001"}})
    events = [
        {"kind": "state_checkpoint", "payload": {"observation_id": "obs_0001"}},
        {"kind": "assistant_turn", "payload": {"tool_calls": [{"tool_call_id": "inspect",
            "tool_name": "skill.place_memory.inspect_place", "args": {"place_id": "P001"}}]}},
        {"kind": "tool_result", "payload": {"tool_call_id": "inspect", "tool_name": "skill.place_memory.inspect_place",
            "status": "ok", "data": result.data}},
    ]
    audit = audit_capability_records(tmp_path, events)
    assert not audit["failures"]
    assert all(p["historical_pose_consistent"] for p in audit["photo_checks"])
    result.data["photos"][1]["uav_yaw_relative_initial_rad"] = 1.5
    assert audit_capability_records(tmp_path, events)["failures"] == ["place_photo_historical_pose_mismatch"]
