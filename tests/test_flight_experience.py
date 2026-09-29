"""Evidence provenance and fail-closed offline review without live model calls."""
import ast
import base64
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import pytest

from plugins.research.flight_experience import ExperimentSettings, read_run, run_experiment
from src.llm import ModelResponse, ProviderRequestError


def _jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    image = BytesIO()
    Image.new("RGB", (4, 4), "brown").save(image, format="PNG")
    _jsonl(root / "frames.jsonl", [{"id": "f1", "observation_id": "o1", "modality": "rgb",
        "metadata": {"base64": base64.b64encode(image.getvalue()).decode()}}])
    _jsonl(root / "observations.jsonl", [{"id": "o1", "timestamp": "2026-01-01", "target_coordinates": "SECRET"}])
    events = [
        ("user_instruction", {"instruction": "Follow the road"}),
        ("state_checkpoint", {"observation_id": "o1", "context": {"reference_path": "SECRET",
            "fragments": [{"name": "action", "content": {"last_window_uav_motion": {"forward_m": 2},
                "ground_truth": "SECRET"}}, {"name": "map", "content": {"answer": "SECRET"}}]}}),
        ("assistant_turn", {"public_intent": "I think I reached the end", "tool_calls": []}),
        ("tool_result", {"tool_name": "uav.move_body", "tool_call_id": "c1", "status": "ok",
            "requested_args": {"vx": 5, "duration": 4}, "executed_args": {"vx": 1, "duration": 2},
            "data": {"pose": {"position": "SECRET"}}, "safety_interventions": [{"reason": "clipped"}]}),
    ]
    _jsonl(root / "agent_transcript.jsonl", [{"id": f"t{i}", "instruction_id": "i1", "kind": kind,
        "payload": payload} for i, (kind, payload) in enumerate(events)])
    return root


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def request(self, request):
        self.requests.append(request)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        text = value if isinstance(value, str) else json.dumps(value)
        return ModelResponse({"model": "fake", "choices": [{"message": {"content": text}}],
                              "usage": {"prompt_tokens": 100, "completion_tokens": 10}})


def _candidate(**changes):
    return {"id": "e1", "title":"Actual motion feedback", "decision_question":"How should the next movement window be estimated?",
        "tags":{k:[] for k in ('decision','action','observation','anomaly','platform','control','sensor')}, "applicability": "After clipped forward motion",
        "observed_evidence": "Request differs from executed motion at t3",
        "recommendation": "Re-estimate from actual motion before extending a window",
        "limitations": "One scene; clearance and different geometry need new observations",
        "evidence_refs": ["t3", "o1"], "observation_ids": ["o1"], **changes}


def _selection(source_id='run'):
    return {'selections':[{'source_id':source_id,'observation_ids':['o1']}],'reason':'Inspect before and after'}


def _review(decision='approve'):
    return {'decision':decision,'reason':'Actual motion is explicitly different','evidence_support':'t3 execution differs',
        'condition_boundaries':'Only after clipping','counterexamples':'No conflicting supplied event',
        'decision_value':'Use actual movement feedback','revision_requests':['Narrow applicability'] if decision=='revise' else []}


def _run(source, output, candidates=None, reviews=None):
    reflect = FakeClient([_selection(), {"candidates": candidates or [_candidate()]}])
    review = FakeClient([_selection(), (reviews or [_review()])[0]])
    result = run_experiment(read_run(source), output, reflection=reflect, reviewer=review)
    return result, reflect, review


def test_adapter_keeps_original_images_actual_feedback_and_omits_truth(source):
    bundle = read_run(source)
    assert "SECRET" not in json.dumps(bundle.index())
    assert bundle.timeline[0]["measured_self_motion"][0]["last_window_uav_motion"]["forward_m"] == 2
    assert bundle.timeline[-1]["requested_args"] != bundle.timeline[-1]["executed_args"]
    assert "unverified_intent" in bundle.timeline[1]
    assert bundle.observations["o1"]["image_status"] == "available"
    assert len(bundle.sources) == 3


def test_review_has_fresh_context_original_images_audit_and_automatic_publication(source, tmp_path):
    before = (source / "agent_transcript.jsonl").read_bytes()
    output = tmp_path / "experiment"
    result, reflect, review = _run(source, output)
    assert result["status"] == "completed" and len(result["accepted"]) == 1
    assert result["estimated_cost_usd"] is None
    assert result['published']
    assert len(review.requests) == 2
    assert [m["role"] for m in review.requests[-1].messages] == ["system", "user"]
    assert any(p["type"] == "image_url" for p in review.requests[-1].messages[-1]["content"])
    assert result["calls"][1]["request_media"][0]["asset_ref"].startswith("media/")
    assert (output / result["calls"][1]["request_media"][0]["asset_ref"]).is_file()
    assert "Applicable needs:" in json.loads((output / "catalog.json").read_text())[0]["description"]
    assert (source / "agent_transcript.jsonl").read_bytes() == before
    with pytest.raises(FileExistsError):
        _run(source, output)
    assert _run(source, tmp_path / "second")[0]["status"] == "completed"


@pytest.mark.parametrize("change,issue", [
    ({"evidence_refs": ["invented", "o1"]}, "unknown_evidence_reference"),
    ({"observation_ids": ["missing"]}, "missing_original_image"),
    ({"limitations": ""}, "missing_limitations"),
])
def test_invalid_candidate_is_deferred_before_model_audit(source, tmp_path, change, issue):
    result, _, reviewer = _run(source, tmp_path / "experiment", [_candidate(**change)])
    assert not result["accepted"] and not reviewer.requests
    assert issue in result["reviews"][0]["validator_issues"]


def test_invalid_revision_cannot_be_adopted(source, tmp_path):
    result = run_experiment(read_run(source),tmp_path/'experiment',reflection=FakeClient([
        _selection(),{'candidates':[_candidate()]},{'candidates':[_candidate(evidence_refs=['fabricated','o1'])]}]),
        reviewer=FakeClient([_selection(),_review('revise')]))
    assert not result['accepted'] and result['reviews'][-1]['decision']=='defer'


@pytest.mark.parametrize("reply", ["not json", {"reviews": []},
    ProviderRequestError("Do not persist SECRET", category="http", attempts=1, retry_exhausted=False, status_code=401)])
def test_review_failure_has_durable_report_and_no_accepted_experience(source, tmp_path, reply):
    output = tmp_path / "experiment"
    result = run_experiment(read_run(source), output,
        reflection=FakeClient([_selection(), {"candidates": [_candidate()]}]),
        reviewer=FakeClient([_selection(), reply]))
    assert result["status"] == "failed" and not result["accepted"]
    assert "SECRET" not in (output / "experiment.json").read_text()
    assert (output / "report.html").is_file()


def test_missing_and_external_images_are_not_loaded(source, tmp_path):
    _jsonl(source / "frames.jsonl", [{"id": "f1", "observation_id": "o1", "modality": "rgb",
                                    "path": "../secret.png"}])
    assert not read_run(source).images
    assert read_run(source).observations["o1"]["image_status"] == "invalid"
    (source / "frames.jsonl").unlink()
    assert read_run(source).observations["o1"]["image_status"] == "missing"
    result = run_experiment(read_run(source), tmp_path / "experiment",
        reflection=FakeClient([{"observation_ids": ["o1"]}]), reviewer=FakeClient([]))
    assert result["status"] == "failed" and not result["accepted"]


def test_task_isolation_legacy_and_source_output_boundaries(source, tmp_path):
    path = source / "agent_transcript.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write('\n' + json.dumps({"id": "other", "instruction_id": "i2", "kind": "user_instruction",
                                       "payload": {"instruction": "Other task"}}))
    with pytest.raises(ValueError, match="multi-task"):
        read_run(source)
    assert read_run(source, instruction_id="i1").instruction == "Follow the road"
    with pytest.raises(ValueError, match="differs"):
        read_run(source, instruction_id="i1", instruction="replacement")
    bundle = read_run(source, instruction_id="i1")
    with pytest.raises(ValueError, match="separate"):
        run_experiment(bundle, source / "output", reflection=FakeClient([]), reviewer=FakeClient([]))
    path.unlink()
    _jsonl(source / "actions.jsonl", [{"id": "a1", "tool": "uav.move_body", "args": {"vx": 1},
                                       "result": {"ok": True, "data": {"target": "SECRET"}}}])
    with pytest.raises(ValueError, match="original instruction"):
        read_run(source)
    legacy = read_run(source, instruction="Original")
    assert legacy.timeline[0]["id"] == "a1"
    assert "SECRET" not in json.dumps(legacy.index())


def test_module_depends_only_on_shared_model_layer_not_core():
    root = Path("plugins/research/flight_experience")
    for path in root.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            imports = [node.module or ""] if isinstance(node, ast.ImportFrom) else (
                [n.name for n in node.names] if isinstance(node, ast.Import) else [])
            assert all(not name.startswith("src.") or name.startswith("src.llm") for name in imports)
    assert not (root / "capability.json").exists()
    assert not (root / "SKILL.md").exists()


def test_prices_are_optional_and_invalid_budgets_rejected():
    with pytest.raises(ValueError):
        ExperimentSettings(max_images=0)
    with pytest.raises(ValueError):
        ExperimentSettings(review_input_price=-1)


def test_temporal_index_fetches_original_neighbor_windows_and_cited_rows(source):
    bundle = read_run(source)
    bundle.timeline = []
    for i in range(6):
        bundle.timeline.extend([
            {"id": f"s{i}", "kind": "state_checkpoint", "observation_id": f"o{i}"},
            {"id": f"a{i}", "kind": "assistant_turn", "unverified_intent": "claim" * 100},
            {"id": f"r{i}", "kind": "tool_result", "status": "ok", "requested_args": {"vx": i}},
        ])
        bundle.observations[f"o{i}"] = {"image_status": "available"}
    index = bundle.temporal_index()
    assert index["timeline_index"][1]["excerpt_truncated"]
    assert len(index["timeline_index"][1]["unverified_intent_excerpt"]) == 160
    excerpt = bundle.excerpt(["o2"], ["r5"])
    ids = [r["id"] for r in excerpt["timeline"]]
    assert ids == ["s1", "a1", "r1", "s2", "a2", "r2", "s3", "r5"]
    assert excerpt["timeline"][1]["unverified_intent"] == "claim" * 100
    assert excerpt["omitted_event_count"] == 10
    assert len(bundle.timeline) == 18


def test_explicit_observation_scope_skips_selection_but_keeps_independent_audit(source, tmp_path):
    reflection = FakeClient([{"candidates": [_candidate()]}])
    reviewer = FakeClient([_selection(),_review()])
    result = run_experiment(read_run(source), tmp_path / "experiment", reflection=reflection, reviewer=reviewer,
                            settings=ExperimentSettings(observation_ids=("o1",)))
    assert result["status"] == "completed" and len(result["accepted"]) == 1
    assert result["frame_selection"] == "explicit_observation_ids"
    assert len(reflection.requests) == 1 and len(reviewer.requests)==2
    assert [c["stage"] for c in result["calls"]] == ["reflection", "review_select_e1_0", "review_e1_0"]
