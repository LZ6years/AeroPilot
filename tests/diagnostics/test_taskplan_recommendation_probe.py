"""Offline contracts for the paired gpt-6.1-sol TaskPlan A/B runner."""

from __future__ import annotations

from collections import namedtuple
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agent.config import ModelConfig
from tests.manual import taskplan_recommendation_probe as probe


def _case(index: int, *, status: str = "completed") -> dict:
    benchmark = "aerial" if index < 5 else "travel"
    return {
        "benchmark": benchmark,
        "case_id": f"case_{index + 1:02d}",
        "difficulty": "low" if benchmark == "aerial" else "easy",
        "scene": str(index),
        "historical": True,
        "status": status,
        "benchmark_status": "ok" if status == "completed" else None,
        "cleanup_verified": status in probe.TERMINAL_CASE_STATUSES,
        "metrics": {
            "success": index < 4 if status == "completed" else None,
            "oracle_success": index < 6 if status == "completed" else None,
            "ndtw": 0.1 * index if benchmark == "aerial" else None,
            "sdtw": 0.05 * index if benchmark == "aerial" else None,
            "distance_to_goal": float(index) if benchmark == "aerial" else None,
            "navigation_error": float(index) if benchmark == "travel" else None,
            "spl": 0.01 * index if benchmark == "travel" else None,
            "path_length": float(index + 10),
            "uav_action_steps": index + 1,
        },
        "elapsed_seconds": float(index + 1),
        "plan_calls": 0,
        "first_flight_preceded_by_plan": False,
        "protocol_rejections": 0,
        "invalid_argument_rejections": 0,
        "infrastructure_error": None,
    }


def _cases(*, status: str = "completed") -> list[dict]:
    return [_case(index, status=status) for index in range(10)]


def _provider_profile() -> dict:
    return ModelConfig(
        provider="openai_compatible",
        model_name=probe.MODEL_NAME,
        api_key="sk-never-persist-this-secret",
        base_url=probe.IKUNCODE_BASE_URL,
        api_protocol="responses",
        reasoning_effort="low",
    ).redacted_summary()


def _tool_contracts() -> dict:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "runtime_update_plan",
                "description": "plan",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
            },
        }
    ]
    return {
        "schema_version": "taskplan_ab_tool_contracts_v1",
        "count": 1,
        "names": ["runtime_update_plan"],
        "tools_sha256": probe._json_sha256(tools),
        "tools": tools,
    }


def test_treatment_replaces_exactly_one_planning_paragraph():
    base = f"before\n\n{probe.CONTROL_PLANNING_PARAGRAPH}\n\nafter\n"
    treatment = probe.build_treatment_prompt(base)
    assert treatment.count(probe.TREATMENT_PLANNING_PARAGRAPH) == 1
    assert probe.CONTROL_PLANNING_PARAGRAPH not in treatment
    assert treatment.replace(
        probe.TREATMENT_PLANNING_PARAGRAPH,
        probe.CONTROL_PLANNING_PARAGRAPH,
        1,
    ) == base
    assert "Before any physical action" in treatment


def test_treatment_rejects_missing_duplicate_or_existing_contract():
    with pytest.raises(ValueError, match="one expected Planning paragraph"):
        probe.build_treatment_prompt("no planning paragraph")
    duplicate = probe.CONTROL_PLANNING_PARAGRAPH * 2
    with pytest.raises(ValueError, match="one expected Planning paragraph"):
        probe.build_treatment_prompt(duplicate)
    existing = (
        probe.CONTROL_PLANNING_PARAGRAPH
        + probe.TREATMENT_PLANNING_PARAGRAPH
    )
    with pytest.raises(ValueError, match="already contains"):
        probe.build_treatment_prompt(existing)


def test_execution_order_is_paired_counterbalanced_and_complete():
    schedule = probe.execution_order(_cases())
    assert len(schedule) == 20
    assert [row["sequence"] for row in schedule] == list(range(1, 21))
    assert [row["condition"] for row in schedule[:4]] == [
        "control",
        "treatment",
        "treatment",
        "control",
    ]
    for index in range(10):
        pair = schedule[index * 2 : index * 2 + 2]
        assert {row["condition"] for row in pair} == set(probe.CONDITIONS)
        assert {row["case_id"] for row in pair} == {f"case_{index + 1:02d}"}


def test_profiles_may_differ_only_by_prompt_path():
    base = _provider_profile()
    profiles = {
        "control": {**base, "action_planner_prompt_path": "control"},
        "treatment": {**base, "action_planner_prompt_path": "treatment"},
    }
    probe._assert_profiles_equivalent(profiles)
    profiles["treatment"]["reasoning_effort"] = "high"
    with pytest.raises(ValueError, match="outside the prompt path"):
        probe._assert_profiles_equivalent(profiles)


def test_prepare_freezes_two_current_run_arms_without_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    case_source = tmp_path / "case_source"
    case_source.mkdir()
    cases = _cases()
    source_manifest = {
        "model": probe.MODEL_NAME,
        "model_config": _provider_profile(),
        "dataset_sha256": {"aerial": "a", "travel": "b"},
        "cases": cases,
    }
    (case_source / "manifest.json").write_text(
        json.dumps(source_manifest), encoding="utf-8"
    )
    (case_source / "summary.json").write_text(
        json.dumps({"cases": cases}), encoding="utf-8"
    )
    provider_manifest = tmp_path / "provider.json"
    provider_manifest.write_text(
        json.dumps({"model_config": _provider_profile()}), encoding="utf-8"
    )
    production_prompt = tmp_path / "agent_tool_loop_system.txt"
    production_prompt.write_text(
        f"before\n\n{probe.CONTROL_PLANNING_PARAGRAPH}\n\nafter\n",
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setattr(probe, "PRODUCTION_PROMPT", production_prompt)
    monkeypatch.setattr(probe.frozen_source, "hashes", lambda: {"src/a.py": "a"})
    monkeypatch.setattr(probe.batch, "source_hashes", lambda: {"src/a.py": "a"})
    monkeypatch.setattr(probe, "_tool_contract_snapshot", _tool_contracts)
    monkeypatch.setattr(probe, "_copy_source_snapshot", lambda *_args: None)
    monkeypatch.setattr(probe, "report", lambda *_args, **_kwargs: {})
    usage = namedtuple("usage", "total free used")
    monkeypatch.setattr(
        probe.shutil,
        "disk_usage",
        lambda _path: usage(20 * 1024**3, 20 * 1024**3, 0),
    )

    root = tmp_path / "experiment"
    manifest = probe.prepare(
        root,
        case_source_root=case_source,
        provider_manifest_path=provider_manifest,
    )

    assert manifest["formal_runs"] == 20
    assert manifest["cases"] == cases
    assert manifest["dataset_sha256"] == {"aerial": "a", "travel": "b"}
    assert manifest["historical_appendix"]["metrics"]["successes"] == 4
    assert len(manifest["execution_order"]) == 20
    control = Path(manifest["prompts"]["control_path"]).read_bytes()
    assert control == production_prompt.read_bytes()
    treatment = Path(manifest["prompts"]["treatment_path"]).read_text(
        encoding="utf-8"
    )
    assert treatment == probe.build_treatment_prompt(control.decode("utf-8"))
    condition_manifests = [
        json.loads((root / condition / "manifest.json").read_text(encoding="utf-8"))
        for condition in probe.CONDITIONS
    ]
    assert condition_manifests[0]["cases"] == condition_manifests[1]["cases"]
    assert (
        condition_manifests[0]["tool_contracts_sha256"]
        == condition_manifests[1]["tool_contracts_sha256"]
    )
    persisted = (root / "manifest.json").read_text(encoding="utf-8")
    assert "sk-never-persist-this-secret" not in persisted


def test_recovery_never_overwrites_and_allows_only_one_attempt(tmp_path: Path):
    item = {"benchmark": "aerial", "case_id": "case", "scene": "1"}
    original = probe.batch.case_dir(tmp_path, 0, item)
    original.mkdir(parents=True)
    (original / "status.json").write_text(
        json.dumps(
            {"status": "infrastructure_error", "cleanup_verified": True}
        ),
        encoding="utf-8",
    )
    recovery = probe._next_case_directory(tmp_path, 0, item, resume=True)
    assert recovery is not None and recovery.name.endswith("_recovery_01")
    recovery.mkdir()
    (recovery / "status.json").write_text(
        json.dumps(
            {"status": "infrastructure_error", "cleanup_verified": True}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Only one"):
        probe._next_case_directory(tmp_path, 0, item, resume=True)


def test_incomplete_or_unclean_attempt_cannot_resume(tmp_path: Path):
    item = {"benchmark": "aerial", "case_id": "case", "scene": "1"}
    original = probe.batch.case_dir(tmp_path, 0, item)
    original.mkdir(parents=True)
    (original / "status.json").write_text(
        json.dumps({"status": "incomplete", "cleanup_verified": False}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="verified recovery"):
        probe._next_case_directory(tmp_path, 0, item, resume=True)


def test_final_paired_report_uses_only_current_arms():
    control = _cases()
    treatment = json.loads(json.dumps(control))
    treatment[4]["metrics"]["success"] = True
    for row in treatment[:8]:
        row["plan_calls"] = 2
        row["first_flight_preceded_by_plan"] = True
    comparison = probe.paired_comparison(
        Path("control"),
        {"cases": control},
        Path("treatment"),
        {"cases": treatment},
        historical_appendix={"metrics": {"successes": 4}},
    )
    assert comparison["status"] == "complete"
    assert comparison["completed_pairs"] == 10
    assert comparison["control"]["successes"] == 4
    assert comparison["treatment"]["successes"] == 5
    assert comparison["primary_results"]["delta"]["successes"] == 1
    assert comparison["treatment"]["preflight_plan_cases"] == 8
    assert comparison["treatment"]["plan_updates"] == 8
    assert comparison["interpretation_warnings"] == []
    assert comparison["historical_appendix"]["metrics"]["successes"] == 4


def test_adoption_and_separation_thresholds_are_explicit():
    control = _cases()
    treatment = json.loads(json.dumps(control))
    for row in control[:3]:
        row["plan_calls"] = 1
        row["first_flight_preceded_by_plan"] = True
    for row in treatment[:7]:
        row["plan_calls"] = 1
        row["first_flight_preceded_by_plan"] = True
    comparison = probe.paired_comparison(
        Path("control"),
        {"cases": control},
        Path("treatment"),
        {"cases": treatment},
    )
    assert comparison["interpretation_warnings"] == [
        "treatment_plan_requirement_not_reliably_adopted",
        "condition_separation_insufficient",
    ]


def test_partial_report_does_not_publish_primary_result():
    control = _cases()
    treatment = _cases(status="pending")
    comparison = probe.paired_comparison(
        Path("control"),
        {"cases": control},
        Path("treatment"),
        {"cases": treatment},
    )
    assert comparison["status"] == "partial"
    assert comparison["completed_pairs"] == 0
    assert comparison["primary_results"] is None
    assert comparison["interpretation_warnings"] == []


def test_audit_metrics_capture_first_plan_and_upstream_token_attribution(
    tmp_path: Path,
):
    attempt = tmp_path / "cases/01"
    audit_path = attempt / "evaluation_output/evaluation/cases/case_01/audit.json"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(
        json.dumps(
            {
                "agent_transcript": [
                    {
                        "kind": "assistant_turn",
                        "payload": {
                            "tool_calls": [
                                {"tool_name": "runtime.update_plan"}
                            ]
                        },
                    }
                ],
                "planner_debug": [
                    {
                        "model_rounds": [
                            {
                                "response": {
                                    "usage": {
                                        "input_tokens": 100,
                                        "output_tokens": 20,
                                        "total_tokens": 120,
                                        "attribution": {
                                            "request_fields": {
                                                "instructions": {"input_tokens": 40},
                                                "tools": {"input_tokens": 30},
                                            }
                                        },
                                    }
                                },
                                "history_projection": {
                                    "background_compaction": {"status": "ready"}
                                },
                            }
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = probe._audit_extra(
        tmp_path,
        {"attempt_dir": "cases/01", "case_id": "case_01"},
    )
    assert result == {
        "planner_rounds": 1,
        "decision_count": 1,
        "first_plan_round": 1,
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "instructions_tokens": 40,
        "tools_tokens": 30,
        "compactor_statuses": ["ready"],
    }


def test_status_reports_frozen_20_run_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cases = _cases(status="pending")
    manifest = {"execution_order": probe.execution_order(cases), "cases": cases}
    for condition in probe.CONDITIONS:
        (tmp_path / condition).mkdir()
    first = probe.batch.case_dir(tmp_path / "control", 0, cases[0])
    first.mkdir(parents=True)
    (first / "status.json").write_text(
        json.dumps({"status": "completed", "cleanup_verified": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(probe, "_load", lambda *_args, **_kwargs: (manifest, {}))
    snapshot = probe.status_snapshot(tmp_path)
    assert len(snapshot["runs"]) == 20
    assert snapshot["counts"] == {"completed": 1, "pending": 19}


def test_condition_admission_validates_native_plan_arguments(
    monkeypatch: pytest.MonkeyPatch,
):
    arguments = {
        "explanation": "retain sequence",
        "steps": [
            {
                "id": "inspect",
                "description": "inspect target",
                "status": "pending",
                "evidence_refs": [],
            },
            {
                "id": "finish",
                "description": "finish task",
                "status": "pending",
                "evidence_refs": [],
            },
        ],
    }
    response = SimpleNamespace(
        tool_calls=[
            {
                "id": "call_1",
                "function": {
                    "name": "runtime_update_plan",
                    "arguments": json.dumps(arguments),
                },
            }
        ],
        data={"instructions": "You are Codex"},
        transport_trace={"request_id": "req_1"},
        usage={"total_tokens": 1},
        text="",
    )
    client = SimpleNamespace(request=lambda *_args, **_kwargs: response)
    monkeypatch.setattr(probe, "create_model_client", lambda *_args: client)
    config = SimpleNamespace(
        to_provider_settings=lambda: object(),
        generation_parameters=lambda: {"reasoning_effort": "low"},
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["explanation", "steps"],
        "properties": {
            "explanation": {"type": "string"},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "description", "status", "evidence_refs"],
                    "properties": {
                        "id": {"type": "string"},
                        "description": {"type": "string"},
                        "status": {"type": "string"},
                        "evidence_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
        },
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "runtime_update_plan",
                "description": "plan",
                "parameters": schema,
            },
        }
    ]
    record = probe._condition_admission("control", config, "prompt", tools)
    assert record["accepted"] is True
    assert record["response_instructions_codex_marker"] is True


def test_secret_and_inline_image_checks_cover_report_artifacts(tmp_path: Path):
    (tmp_path / "safe.json").write_text("{}", encoding="utf-8")
    (tmp_path / "bad.log").write_text("sk-test-secret", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"sk-test-secret")
    assert probe._secret_leaks(tmp_path, ("sk-test-secret",)) == ["bad.log"]
    (tmp_path / "summary.json").write_text(
        '{"image":"data:image/png;base64,AAAA"}', encoding="utf-8"
    )
    assert probe._inline_image_report_leaks(tmp_path) == ["summary.json"]


def test_review_validation_rejects_overlong_paths(tmp_path: Path):
    for condition, length in (("control", 180), ("treatment", 241)):
        directory = tmp_path / condition
        directory.mkdir()
        (directory / "export_summary.json").write_text(
            json.dumps(
                {
                    "validation": {
                        "image_references": 1,
                        "files": 2,
                        "max_path_length": length,
                    }
                }
            ),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="exceeds 240"):
        probe._review_validation(tmp_path)


def test_default_review_root_stays_outside_experiment_tree(tmp_path: Path):
    root = tmp_path / "runs/experiments/taskplan_ab"
    assert probe._review_root(root) == (
        tmp_path / "runs/reviews/taskplan_ab"
    ).resolve()


def test_ssh_key_is_loaded_from_local_env_without_being_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    key = tmp_path / "id_test"
    key.write_text("private", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text(f"FLYAGENT_SSH_KEY={key}\n", encoding="utf-8")
    monkeypatch.delenv("FLYAGENT_SSH_KEY", raising=False)

    resolved = probe._ensure_ssh_key_environment(env_path)

    assert resolved == str(key.resolve())
    assert os.environ["FLYAGENT_SSH_KEY"] == str(key.resolve())


def test_explicit_ssh_key_environment_takes_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    process_key = tmp_path / "process_key"
    process_key.write_text("private", encoding="utf-8")
    env_key = tmp_path / "env_key"
    env_key.write_text("private", encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text(f"FLYAGENT_SSH_KEY={env_key}\n", encoding="utf-8")
    monkeypatch.setenv("FLYAGENT_SSH_KEY", str(process_key))

    resolved = probe._ensure_ssh_key_environment(env_path)

    assert resolved == str(process_key.resolve())
