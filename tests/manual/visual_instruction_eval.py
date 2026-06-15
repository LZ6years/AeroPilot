"""Evaluate default image-module visual inputs with natural-language cases.

This manual probe verifies that the default ``image`` context module can pass
the latest real RGB frame to the planner as an encoded multimodal input while
remaining lightweight: no captioning and no model call inside the module.

PowerShell:

    $env:PYTHONIOENCODING='utf-8'
    python tests/manual/visual_instruction_eval.py --cases tests/fixtures/visual_instruction_cases.json

Add ``--call-model`` to make a real OpenAI-compatible multimodal request using
the configured FLYAGENT model. The default path only inspects the assembled
planner payload, so it is safe for normal local checks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.planning.model_inputs import build_model_user_content
from src.agent.semantic.task_parser import _loads_json_object
from src.memory.context.composer import ContextComposer
from src.memory.context.registry import ContextModuleRegistry
from src.memory.context.schemas import MemorySourceBundle, StageContextInput
from src.memory.flight_memory import FlightMemory
from src.memory.stores import MapMemoryStore, MemoryEventWriter, MemoryRecordReader
from src.runtime.configs.model_config import ModelConfig, load_model_config


DEFAULT_CASES = REPO_ROOT / "tests" / "fixtures" / "visual_instruction_cases.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tests" / "manual_outputs" / "visual_instruction_eval"
DEFAULT_CONTEXT_CONFIG = REPO_ROOT / "configs" / "plugins.json"
DEFAULT_LIMIT = 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate default image model_inputs with visual instruction cases.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--run-dir", default="", help="Optional run directory; uses artifacts/rgb/*.png as ad hoc cases.")
    parser.add_argument("--context-config", default=str(DEFAULT_CONTEXT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--call-model", action="store_true", help="Call the configured multimodal model for semantic checks.")
    parser.add_argument("--write-redacted-payloads", action="store_true")
    return parser


def load_cases(path: str | Path = DEFAULT_CASES) -> list[dict[str, Any]]:
    data = json.loads(_repo_path(path).read_text(encoding="utf-8-sig"))
    cases = data.get("cases", data)
    if not isinstance(cases, list):
        raise ValueError("visual instruction cases must be a list or contain a cases list")
    return [_normalise_case(item, index) for index, item in enumerate(cases, start=1)]


def cases_from_run_dir(run_dir: str | Path, limit: int = DEFAULT_LIMIT) -> list[dict[str, Any]]:
    root = _repo_path(run_dir)
    image_dir = root / "artifacts" / "rgb"
    images = sorted(image_dir.glob("*.png"))[: max(0, int(limit))]
    if not images:
        raise ValueError(f"No PNG images found in {image_dir}")
    return [
        _normalise_case(
            {
                "id": f"run_rgb_{index:03d}",
                "image": str(path),
                "instruction": "Describe visual facts related to building entrances, vehicle references, and open paved areas in the current real RGB frame. Do not output movement advice.",
                "phase": {
                    "name": "real_engine_visual_check",
                    "goal": "extract visual facts relevant to the current phase from a real RGB frame",
                    "done_when": ["relevant visual facts are described"],
                    "observe": ["latest_observation"],
                    "continue_when": ["relevant visual facts are unclear"],
                    "metrics": [],
                },
                "expected": {
                    "must_mention_any": ["building", "vehicle", "tree", "paving", "entrance"],
                    "landmark_any": ["building", "vehicle", "tree"],
                    "forbidden_terms": _default_forbidden_terms(),
                },
            },
            index,
        )
        for index, path in enumerate(images, start=1)
    ]


def build_case_context(
    case: dict[str, Any],
    *,
    context_config: str | Path = DEFAULT_CONTEXT_CONFIG,
    work_dir: str | Path,
) -> dict[str, Any]:
    image_path = _repo_path(case["image"])
    run_dir = Path(work_dir)
    writer = MemoryEventWriter(run_dir)
    reader = MemoryRecordReader(run_dir)
    memory = FlightMemory(
        "visual_instruction_eval",
        "real_engine_fixture",
        str(case["id"]),
        "uav_1",
        active_instruction=str(case["instruction"]),
        memory_reader=reader,
    )
    memory.update_instruction_ref(
        writer.write_instruction(str(case["instruction"]), {"phase": case.get("phase", {})}),
        str(case["instruction"]),
    )
    observation_write = writer.write_observation(
        {
            "source": "visual_instruction_eval",
            "modalities": ["rgb"],
            "frames": {"rgb": {"path": str(image_path), "mock": False}},
        }
    )
    memory.append_observation_ref(observation_write["observation_ref"])
    memory.append_frame_refs(observation_write["frame_refs"])
    stage_context = StageContextInput(
        active_instruction=str(case["instruction"]),
        current_phase=dict(case.get("phase", {})),
        semantic_goal={"source": "visual_instruction_eval", "case_id": case["id"]},
    )
    registry = ContextModuleRegistry.from_config(_repo_path(context_config))
    return ContextComposer(registry).compose(MemorySourceBundle(memory, MapMemoryStore()), stage_context)


def evaluate_case_context(
    case: dict[str, Any],
    context: dict[str, Any],
    user_content: str | list[dict[str, Any]],
    model_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image_fragment = _fragment(context, "image")
    image_inputs = [
        item for item in image_fragment.get("model_inputs", [])
        if isinstance(item, dict) and item.get("type") == "image"
    ]
    image_input = image_inputs[0] if image_inputs else {}
    text_part = user_content[0]["text"] if isinstance(user_content, list) and user_content else user_content
    expected = case.get("expected", {}) if isinstance(case.get("expected"), dict) else {}
    forbidden_terms = [str(item) for item in expected.get("forbidden_terms", [])]
    output_text = json.dumps(model_result.get("parsed", {}), ensure_ascii=False) if model_result else ""
    forbidden_terms_hit = [term for term in forbidden_terms if term and term in output_text]
    must_mentions = [str(item) for item in expected.get("must_mention_any", [])]
    must_mentions_hit = [term for term in must_mentions if term and term in output_text]
    metrics = {
        "has_image_model_input": bool(image_inputs),
        "image_input_is_data_url": isinstance(image_input.get("data"), str)
        and image_input.get("data", "").startswith("data:image/"),
        "json_text_omits_base64": isinstance(text_part, str)
        and "data:image/" not in text_part
        and "<omitted encoded model input data" in text_part,
        "planner_saw_image_part": _has_image_part(user_content),
        "must_mentions_hit": must_mentions_hit,
        "forbidden_terms_hit": forbidden_terms_hit,
        "semantic_checked": model_result is not None,
        "no_fake_visual_claim_when_image_missing": True,
    }
    metrics["visual_grounding_score"] = _visual_grounding_score(metrics)
    return {
        "case_id": case["id"],
        "image": str(_repo_path(case["image"])),
        "instruction": case["instruction"],
        "phase": dict(case.get("phase", {})),
        "metrics": metrics,
        "image_fragment_summary": _image_fragment_summary(image_fragment),
        "model_result": model_result or {},
    }


def call_visual_eval_model(config: ModelConfig, context: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    if not config.api_key:
        raise RuntimeError("Missing FLYAGENT_MODEL_API_KEY; API key will not be printed.")
    system_prompt = (
        "You are a UAV visual-input pipeline evaluator. Return JSON only, based on the provided context and image. "
        "Check whether the model can use image content to answer the current visual instruction. Do not output movement advice."
    )
    expected = case.get("expected", {}) if isinstance(case.get("expected"), dict) else {}
    eval_context = dict(context)
    eval_context["visual_eval_request"] = {
        "case_id": case["id"],
        "must_mention_any": expected.get("must_mention_any", []),
        "forbidden_terms": expected.get("forbidden_terms", []),
        "response_schema": {
            "visual_answer": "short visual answer",
            "mentions": ["matched visual keywords"],
            "forbidden_terms": ["movement advice or forbidden terms if present"],
            "uses_image": True,
        },
    }
    payload = {
        "model": config.model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_model_user_content(eval_context)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    started = time.perf_counter()
    request = urllib.request.Request(
        config.chat_completions_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_sec) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"visual eval request failed: HTTP {exc.code} {exc.reason}; body={body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"visual eval request failed: {exc}") from exc
    duration_sec = round(time.perf_counter() - started, 3)
    content = data["choices"][0]["message"]["content"]
    return {
        "parsed": _loads_json_object(content),
        "duration_sec": duration_sec,
        "usage": data.get("usage", {}),
        "model": config.model_name,
    }


def write_outputs(
    output_dir: str | Path,
    *,
    results: list[dict[str, Any]],
    console_lines: list[str],
    redacted_payloads: list[dict[str, Any]],
    write_redacted_payloads: bool = False,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(results)
    json_path = out / "results.json"
    md_path = out / "results.md"
    console_path = out / "console.txt"
    json_path.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(_markdown_report(summary, results), encoding="utf-8")
    console_path.write_text("\n".join(console_lines) + "\n", encoding="utf-8")
    paths = {"json": json_path, "markdown": md_path, "console": console_path}
    if write_redacted_payloads:
        payload_dir = out / "redacted_payloads"
        payload_dir.mkdir(parents=True, exist_ok=True)
        for payload in redacted_payloads:
            (payload_dir / f"{payload['case_id']}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        paths["redacted_payloads"] = payload_dir
    return paths


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"case_count": 0}
    metric_keys = [
        "has_image_model_input",
        "image_input_is_data_url",
        "json_text_omits_base64",
        "planner_saw_image_part",
        "no_fake_visual_claim_when_image_missing",
    ]
    return {
        "case_count": len(results),
        "passed_structural_cases": sum(1 for item in results if _structural_pass(item["metrics"])),
        "average_visual_grounding_score": round(
            sum(float(item["metrics"]["visual_grounding_score"]) for item in results) / len(results),
            3,
        ),
        "semantic_checked_cases": sum(1 for item in results if item["metrics"].get("semantic_checked")),
        "metric_pass_counts": {
            key: sum(1 for item in results if item["metrics"].get(key) is True)
            for key in metric_keys
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    cases = cases_from_run_dir(args.run_dir, args.limit) if args.run_dir else load_cases(args.cases)[: args.limit]
    output_dir = Path(args.output_dir)
    work_root = output_dir / "memory_cases"
    model_config = load_model_config(args.env_path) if args.call_model else None
    console_lines = [
        f"cases={len(cases)}",
        f"context_config={_repo_path(args.context_config)}",
        f"call_model={args.call_model}",
    ]
    results: list[dict[str, Any]] = []
    redacted_payloads: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        context = build_case_context(
            case,
            context_config=args.context_config,
            work_dir=work_root / str(case["id"]),
        )
        user_content = build_model_user_content(context)
        model_result = call_visual_eval_model(model_config, context, case) if model_config else None
        result = evaluate_case_context(case, context, user_content, model_result)
        results.append(result)
        redacted_payloads.append(_redacted_payload_summary(case, context, user_content))
        line = (
            f"[{index}/{len(cases)}] {case['id']}: "
            f"image_input={result['metrics']['has_image_model_input']}, "
            f"image_part={result['metrics']['planner_saw_image_part']}, "
            f"score={result['metrics']['visual_grounding_score']}"
        )
        if model_result:
            line += f", duration={model_result.get('duration_sec')}s, tokens={model_result.get('usage', {}).get('total_tokens')}"
        console_lines.append(line)
        print(line)
    paths = write_outputs(
        output_dir,
        results=results,
        console_lines=console_lines,
        redacted_payloads=redacted_payloads,
        write_redacted_payloads=args.write_redacted_payloads,
    )
    print("\nSummary:")
    print(json.dumps(summarize_results(results), indent=2, ensure_ascii=False))
    print("\nWrote outputs:")
    for key, path in paths.items():
        print(f"{key}={path}")


def _normalise_case(item: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"case #{index} must be an object")
    case = dict(item)
    case.setdefault("id", f"case_{index:03d}")
    if "image" not in case:
        raise ValueError(f"case {case['id']} missing image")
    case.setdefault("instruction", "Describe visual facts relevant to the current phase. Do not output movement advice.")
    case.setdefault(
        "phase",
        {
            "name": "visual_check",
            "goal": "describe visual facts relevant to the current phase",
            "done_when": ["relevant visual facts are described"],
            "observe": ["latest_observation"],
            "continue_when": ["relevant visual facts are unclear"],
            "metrics": [],
        },
    )
    expected = dict(case.get("expected", {})) if isinstance(case.get("expected"), dict) else {}
    expected.setdefault("must_mention_any", [])
    expected.setdefault("landmark_any", [])
    expected.setdefault("forbidden_terms", _default_forbidden_terms())
    case["expected"] = expected
    return case


def _default_forbidden_terms() -> list[str]:
    return ["forward", "approach", "fly toward", "hover", "scan", "continue", "recommended_next_observation", "5m", "30 degrees"]


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _fragment(context: dict[str, Any], name: str) -> dict[str, Any]:
    return next(fragment for fragment in context.get("fragments", []) if fragment.get("name") == name)


def _has_image_part(user_content: str | list[dict[str, Any]]) -> bool:
    return isinstance(user_content, list) and any(
        isinstance(item, dict)
        and item.get("type") == "image_url"
        and isinstance(item.get("image_url"), dict)
        and str(item["image_url"].get("url", "")).startswith("data:image/")
        for item in user_content
    )


def _visual_grounding_score(metrics: dict[str, Any]) -> float:
    score = 0.0
    for key in ["has_image_model_input", "image_input_is_data_url", "json_text_omits_base64", "planner_saw_image_part"]:
        if metrics.get(key):
            score += 0.2
    if metrics.get("semantic_checked"):
        if metrics.get("must_mentions_hit"):
            score += 0.15
        if not metrics.get("forbidden_terms_hit"):
            score += 0.05
    else:
        score += 0.2
    return round(score, 3)


def _structural_pass(metrics: dict[str, Any]) -> bool:
    return all(
        metrics.get(key)
        for key in [
            "has_image_model_input",
            "image_input_is_data_url",
            "json_text_omits_base64",
            "planner_saw_image_part",
        ]
    )


def _image_fragment_summary(fragment: dict[str, Any]) -> dict[str, Any]:
    content = fragment.get("content", {}) if isinstance(fragment.get("content"), dict) else {}
    model_inputs = fragment.get("model_inputs", []) if isinstance(fragment.get("model_inputs"), list) else []
    summaries = []
    for item in model_inputs:
        if not isinstance(item, dict):
            continue
        summary = {
            "type": item.get("type"),
            "source": item.get("source"),
            "role": item.get("role"),
            "mime_type": item.get("mime_type"),
            "encoding": item.get("encoding"),
            "metadata": dict(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {},
        }
        if isinstance(item.get("data"), str):
            summary["data_chars"] = len(item["data"])
            summary["data_summary"] = "data:image/*;base64,<omitted>"
        summaries.append(summary)
    return {
        "source": fragment.get("source"),
        "visual_evidence": content.get("visual_evidence", {}),
        "has_visual_description": "visual_description" in content,
        "model_inputs": summaries,
        "artifact_refs": list(fragment.get("artifact_refs", [])),
        "warnings": list(fragment.get("warnings", [])),
    }


def _redacted_payload_summary(
    case: dict[str, Any],
    context: dict[str, Any],
    user_content: str | list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "case_id": case["id"],
        "image": str(_repo_path(case["image"])),
        "fragment_count": len(context.get("fragments", [])),
        "user_content_type": "parts" if isinstance(user_content, list) else "text",
        "part_types": [item.get("type") for item in user_content] if isinstance(user_content, list) else [],
        "has_base64_in_text": (
            "data:image/" in user_content
            if isinstance(user_content, str)
            else bool(user_content and "data:image/" in user_content[0].get("text", ""))
        ),
    }


def _markdown_report(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines = [
        "# Visual Instruction Eval",
        "",
        "This checks the default image context module path: metadata plus latest real RGB `model_inputs[type=image]`.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    for item in results:
        metrics = item["metrics"]
        image = Path(item["image"]).resolve().as_posix()
        lines.extend(
            [
                f"## {item['case_id']}",
                "",
                f"![{item['case_id']}]({image})",
                "",
                f"- instruction: {item['instruction']}",
                f"- image_input: {metrics['has_image_model_input']}",
                f"- image_part: {metrics['planner_saw_image_part']}",
                f"- json_text_omits_base64: {metrics['json_text_omits_base64']}",
                f"- score: {metrics['visual_grounding_score']}",
                "",
            ]
        )
        if item.get("model_result"):
            lines.extend(
                [
                    "Model result:",
                    "",
                    "```json",
                    json.dumps(item["model_result"], indent=2, ensure_ascii=False),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
