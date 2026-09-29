"""Tests for the manual visual instruction evaluation helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import base64


REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def load_eval_module():
    path = REPO_ROOT / "tests" / "manual" / "visual_instruction_eval.py"
    spec = importlib.util.spec_from_file_location("visual_instruction_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_visual_instruction_cases_are_visual_and_action_free():
    module = load_eval_module()

    cases = module.load_cases()

    assert len(cases) >= 6
    for case in cases:
        assert Path(case["image"]).suffix == ".png"
        assert (REPO_ROOT / case["image"]).exists()
        assert "phase" in case
        instruction = case["instruction"]
        assert "5m" not in instruction
        assert "5米" not in instruction
        assert "30度" not in instruction


def test_default_image_context_passes_latest_real_rgb_to_model_parts(tmp_path):
    module = load_eval_module()
    case = module.load_cases()[0]

    run_dir = tmp_path / "case"
    context = module.build_case_context(case, work_dir=run_dir)
    user_content = module.build_model_user_content(context, run_dir=run_dir)
    result = module.evaluate_case_context(case, context, user_content)

    metrics = result["metrics"]
    assert metrics["has_image_model_input"] is True
    assert metrics["image_input_is_data_url"] is True
    assert metrics["json_text_omits_base64"] is True
    assert metrics["planner_saw_image_part"] is True
    assert result["image_fragment_summary"]["has_visual_description"] is False
    assert result["image_fragment_summary"]["visual_evidence"]["interpretation_status"] == (
        "image_attached_not_interpreted"
    )


def test_visual_eval_results_are_redacted(tmp_path):
    module = load_eval_module()
    case = module.load_cases()[0]
    run_dir = tmp_path / "case"
    context = module.build_case_context(case, work_dir=run_dir)
    user_content = module.build_model_user_content(context, run_dir=run_dir)
    result = module.evaluate_case_context(case, context, user_content)

    paths = module.write_outputs(
        tmp_path / "out",
        results=[result],
        console_lines=["ok"],
        redacted_payloads=[module._redacted_payload_summary(case, context, user_content)],
        write_redacted_payloads=True,
    )

    output_text = paths["json"].read_text(encoding="utf-8")
    redacted_text = (paths["redacted_payloads"] / f"{case['id']}.json").read_text(encoding="utf-8")
    assert "data:image/png;base64," not in output_text
    assert "data:image/png;base64," not in redacted_text
    assert '"sha256"' in output_text


def test_run_dir_can_generate_ad_hoc_visual_cases(tmp_path):
    module = load_eval_module()
    rgb_dir = tmp_path / "run" / "artifacts" / "rgb"
    rgb_dir.mkdir(parents=True)
    (rgb_dir / "frame_000001.png").write_bytes(b"fake-png")

    cases = module.cases_from_run_dir(tmp_path / "run", limit=1)

    assert len(cases) == 1
    assert cases[0]["image"] == str(rgb_dir / "frame_000001.png")
    assert cases[0]["phase"]["name"] == "real_engine_visual_check"


def test_explicit_past_now_pair_projects_only_latest_checkpoint_image(tmp_path):
    module = load_eval_module()
    past = tmp_path / "past.png"
    now = tmp_path / "now.png"
    past.write_bytes(TINY_PNG)
    now.write_bytes(TINY_PNG)
    case = module.case_from_image_pair(
        past,
        now,
        instruction="Describe only what is currently visible.",
    )

    run_dir = tmp_path / "pair"
    context = module.build_case_context(case, work_dir=run_dir)
    image = next(
        fragment for fragment in context["fragments"]
        if fragment["name"] == "image"
    )
    assert [
        item["media_ref"]["role"] for item in image["model_inputs"]
    ] == ["current_observation"]

    content = module.build_model_user_content(context, run_dir=run_dir)
    assert isinstance(content, list)
    assert [item["type"] for item in content] == ["text", "image_url", "text"]
    assert json.loads(content[2]["text"])["observation_id"] == "obs_0002"


def test_summary_counts_structural_passes():
    module = load_eval_module()
    results = [
        {
            "metrics": {
                "has_image_model_input": True,
                "image_input_is_data_url": True,
                "json_text_omits_base64": True,
                "planner_saw_image_part": True,
                "no_fake_visual_claim_when_image_missing": True,
                "visual_grounding_score": 1.0,
                "semantic_checked": False,
            }
        }
    ]

    summary = module.summarize_results(results)

    assert summary["case_count"] == 1
    assert summary["passed_structural_cases"] == 1
    assert summary["metric_pass_counts"]["planner_saw_image_part"] == 1
    assert json.dumps(summary)
