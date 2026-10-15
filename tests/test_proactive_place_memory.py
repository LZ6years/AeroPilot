"""Proactive place annotations, immutable request evidence and preview auditing.

These checks use existing production plugin/registry paths without external
models or simulators. Model decisions remain unforced; only state and evidence
boundaries are deterministic here.
"""
import json
from pathlib import Path

from plugins.capabilities.place_memory.store import PlaceStore, annotation_status, atomic_json
from plugins.capabilities.place_memory.render import render_map
from plugins.capabilities.place_memory.layout import draw_map
from plugins.capabilities.place_memory.save_place.scripts.handler import handle as save_place
from plugins.capabilities.place_memory.context_modules.map import PluginContextModule
from src.plugins import PluginRegistry
from src.tools.registry import default_registry
from src.memory.context.contracts import MemorySourceBundle, TaskContextInput
from src.agent.transcript import AgentToolCall, AssistantTurn, ToolResult
from evaluation.benchmarks.aerial_vln.pixel_map_campaign import select_preview_cases, write_place_check
from tests.test_pixel_place_memory import flight


def save_args(oid="obs_0001"):
    return {"place_id": "P001", "summary": "红色入口旁有路口；右侧尚未检查。",
            "landmarks": ["红色入口", "路口"], "evidence_refs": [oid],
            "photo_observation_ids": [oid]}


def save_context(path, memory, oid="obs_0001", call_id="save"):
    return {"run_dir": str(path), "instruction_id": memory.active_instruction_id,
            "tool_call_id": call_id, "evidence_refs": [oid]}


def test_provider_tool_carries_applicability_and_parameter_meaning():
    tools = default_registry(PluginRegistry.from_config("configs/plugins.pixel_map.json")).model_specs()
    wire = next(t.to_provider_tool()["function"] for t in tools if t.name == "skill.place_memory.save_place")
    assert all(word in wire["description"] for word in ("Proactively", "junctions", "revisits", "unsuccessful searches", "older checkpoint"))
    schema = wire["parameters"]
    assert set(schema["required"]) == set(save_args())
    assert all(p.get("description") for p in schema["properties"].values())


def test_annotation_boundary_revisit_idempotence_and_photo_retention(tmp_path):
    memory = flight(tmp_path, [(0, 0, 0), (1, 0, 0), (20, 0, 0), (0, 0, 0)])
    store = PlaceStore(tmp_path, memory.active_instruction_id)
    graph = store.update("obs_0001", "obs_0002")
    assert annotation_status(graph["nodes"][0])["is_annotated"] is False
    args, context = save_args("obs_0002"), save_context(tmp_path, memory, "obs_0002")
    graph = store.annotate(args, context)
    assert annotation_status(graph["nodes"][0]) == {
        "is_annotated": True, "last_saved_observation_id": "obs_0002", "new_observations_since_save": 0}
    graph = store.update("obs_0001", "obs_0003")
    assert annotation_status(graph["nodes"][0])["new_observations_since_save"] == 0
    graph = store.update("obs_0001", "obs_0004")
    assert annotation_status(graph["nodes"][0])["new_observations_since_save"] == 1
    assert graph == store.annotate(args, context)  # same call never moves its save boundary
    changed = {**args, "summary": "入口再次确认，左侧已检查。", "photo_observation_ids": []}
    graph = store.annotate(changed, {**context, "tool_call_id": "update"})
    assert graph["nodes"][0]["photos"][0]["observation_id"] == "obs_0002"
    graph = store.update("obs_0001", "obs_0004")
    assert graph["nodes"][0]["photos"][0]["observation_id"] == "obs_0002"
    result = PluginContextModule().build(MemorySourceBundle(memory), TaskContextInput("inspect"))
    assert result.fragment.content == {}
    assert annotation_status(store.load()["nodes"][0])["last_saved_observation_id"] == "obs_0004"


def test_legacy_annotations_do_not_invent_save_coverage(tmp_path):
    memory = flight(tmp_path, [(0, 0, 0)])
    store = PlaceStore(tmp_path, memory.active_instruction_id)
    graph = store.update("obs_0001", "obs_0001")
    node = graph["nodes"][0]
    node["summary"] = "Older note"
    node.pop("last_saved_observation_id")
    node.pop("annotation_visit_count")
    atomic_json(store.path, graph)
    node = store.update("obs_0001", "obs_0001")["nodes"][0]
    assert annotation_status(node) == {
        "is_annotated": True, "last_saved_observation_id": None, "new_observations_since_save": None}


def test_save_immediate_feedback_and_map_usage_replace_pre_save_state(tmp_path):
    memory = flight(tmp_path, [(0, 0, 0)])
    module = PluginContextModule()
    old = module.build(MemorySourceBundle(memory), TaskContextInput("inspect"))
    old_ref = old.fragment.media_refs[0]["media_ref"]
    result = save_place({"runtime_context": save_context(tmp_path, memory), "args": save_args()})
    assert old.fragment.content == {}
    assert result.data["summary"] == save_args()["summary"]
    assert result.data["current_place"]["is_annotated"] is True
    assert result.data["new_observations_since_save"] == 0
    latest = result.model_inputs[0]["media_ref"]
    assert latest["ref"] != old_ref["ref"]
    assert "is_annotated" not in latest["usage"]
    assert "红色入口" not in latest["usage"]
    assert "historical map" in latest["usage"]
    assert "inspect_place" not in latest["usage"]
    assert (tmp_path / old_ref["ref"]).exists()
    # The plugin reports state; it never appends another observation or action.
    assert len(PlaceStore(tmp_path, memory.active_instruction_id).load()["observations"]) == 1


def test_revisited_current_site_is_pinned_in_long_map(tmp_path):
    memory = flight(tmp_path, [(i * 20, 0, 0) for i in range(12)] + [(0, 0, 0)])
    store = PlaceStore(tmp_path, memory.active_instruction_id)
    graph = store.update("obs_0001", "obs_0013")
    assert graph["current_place_id"] == "P001"
    assert graph["nodes"][-8:][0]["id"] != "P001"
    assert "P001" in draw_map(graph).label_boxes[0]["ids"]
    render_map(store, graph)


def test_preview_selection_has_four_fixed_cases_without_outcome_filtering():
    tasks = [{"episode_id": str(i), "difficulty": ("high", "high", "low", "medium")[i % 4],
              "split": "val_seen", "scene_id": "2", "old_success": bool(i % 2)} for i in range(100)]
    selected = select_preview_cases(tasks)
    assert [t["episode_id"] for t in selected] == ["2", "0", "3", "1"]
    assert [t["difficulty"] for t in selected] == ["low", "high", "medium", "high"]
    assert [t["episode_id"] for t in select_preview_cases([{**t, "old_success": not t["old_success"]} for t in tasks])] == ["2", "0", "3", "1"]


def test_report_requires_model_calls_and_consistent_original_evidence(tmp_path):
    state = {"tasks": {}}
    for i in range(4):
        run = tmp_path / f"case_{i}/runs/task"
        memory = flight(run, [(0, 0, 0)])
        PluginContextModule().build(MemorySourceBundle(memory), TaskContextInput("inspect"))
        events = []
        if i < 2:
            saved = save_place({"runtime_context": save_context(run, memory), "args": save_args()})
            turn = AssistantTurn("t", "Retain this recognizable junction", [AgentToolCall("save", "skill.place_memory.save_place", save_args())])
            result = ToolResult("save", "skill.place_memory.save_place", "ok", data=saved.data, model_inputs=list(saved.model_inputs))
            events = [{"kind": "assistant_turn", "payload": turn.to_dict()}, {"kind": "tool_result", "payload": result.to_dict()}]
        (run / "memory/agent_transcript.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events), encoding="utf-8")
        state["tasks"][str(i)] = {"task": {"episode_id": str(i)}, "status": "scored", "result": {"run_dir": str(run)}}
    report = write_place_check(tmp_path, state)
    assert report["passed"] and report["verified_save_episodes"] == 2
    path = Path(state["tasks"]["0"]["result"]["run_dir"]) / "memory/agent_transcript.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    events[0]["payload"]["tool_calls"][0]["args"]["summary"] = "Unsupported replacement"
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    assert write_place_check(tmp_path, state)["passed"] is False
    # A saved artifact without its matching assistant call is not autonomous.
    path.write_text(json.dumps(events[1]), encoding="utf-8")
    assert write_place_check(tmp_path, state)["cases"][0]["autonomous_saves"] == 0
