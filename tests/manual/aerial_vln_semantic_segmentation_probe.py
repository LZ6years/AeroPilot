"""Run a real-model, parser-only AerialVLN-S segmentation diagnostic.

This manual probe deliberately stops at semantic parsing.  It passes only the
official instruction text to :class:`TaskParser`; it never constructs a scene,
Runtime, planner, EnvClient, or simulator connection.  The fixed ten-case set
is intended to make future parser prompt changes comparable across low,
medium, and high route-difficulty examples.

Generated files live under ``tests/manual_outputs`` and are ignored by git.
They are human diagnostic artifacts, not benchmark scores or stable fixtures.

For a paired prompt experiment, supply --instructions-file (ordered objects
with id/instruction/protocol), --before-prompts (the three pre-edit text files)
and a new --output-dir. This mode bypasses dataset loading and keyword review
flags, records public parser exchanges, and stops on the first request failure.
Add --structure-only to stop after the production structure contract, without
skill selection or criteria generation. This is a probe mode, not a new
production parser mode. The two criteria prompt snapshots must remain equal.
Example:
  python tests/manual/aerial_vln_semantic_segmentation_probe.py --env-path .env \
    --instructions-file tests/manual/phase_prompt_comparison_instructions.json \
    --before-prompts BEFORE_DIRECTORY --output-dir NEW_DIRECTORY
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmarks.aerial_vln.cases import load_aerial_vln_dataset
from evaluation.benchmarks.aerial_vln.protocol import (
    AERIAL_VLN_PARSER_PROTOCOL_CONTEXT,
    AERIAL_VLN_PROTOCOL,
)
from src.agent.semantic.task_parser import TaskParser
from src.agent.semantic.parser_contracts import SemanticPlanStructure
from src.runtime.configs.model_config import ModelConfig, load_model_config


SAMPLE_CASES: tuple[tuple[str, str], ...] = (
    ("3AAJC4I4FPQO2SQW3E7VJW64CWCZJJ", "low"),
    ("3G0WWMR1U4IB7QGQVGGALABHGGXNQO", "low"),
    ("3NL0RFNU0OL3AF8P06HHQFT8T3F4KA", "low"),
    ("39GHHAVOMOP8ZV27ILH5P0TV5F6J4A", "low"),
    ("37ZHEEHM65KUX4I1HYNPC5WGYC537F", "medium"),
    ("37Q970SNZN6K6ZOEIX7HJ3DFVRM1S4", "medium"),
    ("3GDTJDAPV39ZJ4VUXLKAM41HGS98ML", "medium"),
    ("32ZKVD547OLHZGLRUFZ5H7ICUWMB30", "high"),
    ("3IGI0VL64GI8ME3AG3O6W5GIJ4VONV", "high"),
    ("3L4D84MIL8Q2B7Q2XS1OJP2Z8KEJHG", "high"),
)

_EVALUATION_TRUTH_KEYS = (
    "goal_position",
    "reference_path",
    "official_actions",
)

# This is intentionally a review heuristic rather than a validator.  It helps
# a person find phase goals worth inspecting without modifying parser output or
# teaching benchmark examples to production parsing code.
_RESULT_FAMILY_TERMS: dict[str, tuple[str, ...]] = {
    "orientation": ("turn", "face", "look", "orient", "adjust heading"),
    "route_translation": (
        "fly",
        "move",
        "go ",
        "proceed",
        "follow",
        "approach",
        "reach",
    ),
    "altitude_transition": (
        "ascend",
        "descend",
        "lift",
        "rise",
        "lower",
        "fly up",
        "fly down",
    ),
    "passage": ("cross", "pass", "through", "beyond", "over the"),
    "episode_termination": ("stop", "end the episode", "hover", "hold"),
}


class RecordingTaskParser(TaskParser):
    """Record validated parser contracts without changing parser behavior."""

    def __init__(self, model_config: ModelConfig, instruction: str, *,
                 protocol_context: str = AERIAL_VLN_PARSER_PROTOCOL_CONTEXT,
                 prompts: dict[str, str] | None = None) -> None:
        super().__init__(
            mode="model",
            model_config=model_config,
            protocol_context=protocol_context,
        )
        for attribute, content in (prompts or {}).items():
            if attribute not in PROMPT_FILES:
                raise ValueError(f"Unknown prompt slot: {attribute}")
            setattr(self, attribute, content)
        self._diagnostic_instruction = instruction
        self.contract_calls: list[dict[str, Any]] = []
        self.model_input_checks: list[dict[str, Any]] = []
        self.exchanges: list[dict[str, Any]] = []
        self.stop_on_selection_failure = prompts is not None

    def parse_structure(self, instruction: str) -> SemanticPlanStructure:
        """Exercise only the existing first-stage contract, including repairs."""
        clean_instruction = instruction.strip()
        if not clean_instruction:
            raise ValueError("instruction must not be empty")
        return self._call_json_contract(
            self._build_structure_messages(clean_instruction),
            SemanticPlanStructure.from_dict,
            contract_name="semantic structure",
        )

    def _resolve_phase_guidance(self, instruction, structure, phase):
        guidance, trace = super()._resolve_phase_guidance(instruction, structure, phase)
        if self.stop_on_selection_failure and trace.get("selection_status") in {"model_error", "resolver_error"}:
            self.contract_calls.append({"stage": "skill_selection", "status": "error", "trace": trace})
            raise RuntimeError("Parser skill selection failed; see recorded contract trace")
        return guidance, trace

    def _request_model_content(self, messages):
        # Capture only public text contracts, never provider-private reasoning
        # or HTTP headers. Preserve ordinary contract-repair attempts as well.
        exchange = {"messages": copy.deepcopy(messages)}
        self.exchanges.append(exchange)
        try:
            content = super()._request_model_content(messages)
            exchange["content"] = content
            return content
        except Exception as exc:
            exchange["error"] = f"{type(exc).__name__}: {exc}"
            raise

    def _call_json_contract(
        self,
        messages: list[dict[str, str]],
        validator: Callable[[dict[str, Any]], Any],
        *,
        contract_name: str,
    ) -> Any:
        stage = self._stage_name(contract_name)
        self.model_input_checks.append(
            _model_input_check(
                stage,
                messages,
                original_instruction=self._diagnostic_instruction,
            )
        )
        started = time.perf_counter()
        try:
            result = super()._call_json_contract(
                messages,
                validator,
                contract_name=contract_name,
            )
        except Exception as exc:
            self.contract_calls.append(
                {
                    "stage": stage,
                    "contract_name": contract_name,
                    "status": "error",
                    "duration_sec": round(time.perf_counter() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        self.contract_calls.append(
            {
                "stage": stage,
                "contract_name": contract_name,
                "status": "ok",
                "duration_sec": round(time.perf_counter() - started, 3),
                "result": _json_ready(result),
            }
        )
        return result

    def _stage_name(self, contract_name: str) -> str:
        if contract_name == "semantic structure":
            return "single_pass_structure"
        if contract_name.startswith("phase criteria "):
            return contract_name.replace("phase criteria ", "criteria_", 1)
        return contract_name.replace(" ", "_")


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    if args.instructions_file:
        return run_comparison(args)
    if args.structure_only:
        raise ValueError("--structure-only requires --instructions-file and --before-prompts")
    if args.before_prompts:
        raise ValueError("--before-prompts requires --instructions-file")
    model_config = load_model_config(args.env_path)
    _validate_model_config(model_config)
    cases, provenance = load_aerial_vln_dataset(
        args.dataset_root,
        "val_seen",
        dataset_family="aerialvln-s",
        dataset_version="aerialvln-s-official",
    )
    cases_by_id = {case.episode_id: case for case in cases}
    missing = [case_id for case_id, _ in SAMPLE_CASES if case_id not in cases_by_id]
    if missing:
        raise SystemExit(
            "Fixed segmentation sample is missing from val_seen: "
            + ", ".join(missing)
        )

    output_dir = _resolve_output_dir(args.output_dir)
    case_dir = output_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []

    print(f"output_dir={output_dir}", flush=True)
    for index, (episode_id, difficulty) in enumerate(SAMPLE_CASES, start=1):
        case = cases_by_id[episode_id]
        print(
            f"[{index}/{len(SAMPLE_CASES)}] parsing {episode_id} "
            f"difficulty={difficulty} scene={case.scene_id}",
            flush=True,
        )
        result = _parse_case(
            model_config=model_config,
            episode_id=episode_id,
            scene_id=case.scene_id,
            difficulty=difficulty,
            instruction=case.instruction,
        )
        _write_json(case_dir / f"{episode_id}.json", result)
        results.append(result)
        print(
            f"[{index}/{len(SAMPLE_CASES)}] {result['status']} "
            f"phases={result.get('final_phase_count')} "
            f"duration_sec={result['duration_sec']}",
            flush=True,
        )

    summary = _build_summary(
        created_at=created_at,
        provenance=provenance.to_dict(),
        model_config=model_config,
        results=results,
    )
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "review.html").write_text(
        _render_html(summary, results),
        encoding="utf-8",
    )
    print(f"review_html={output_dir / 'review.html'}", flush=True)
    print(
        f"completed={summary['counts']['ok']} failed={summary['counts']['error']}",
        flush=True,
    )
    return 0 if summary["counts"]["error"] == 0 else 1


def _parse_case(
    *,
    model_config: ModelConfig,
    episode_id: str,
    scene_id: str,
    difficulty: str,
    instruction: str,
) -> dict[str, Any]:
    parser = RecordingTaskParser(model_config, instruction)
    started = time.perf_counter()
    error: str | None = None
    plan: dict[str, Any] | None = None
    try:
        parse_result = parser.parse(instruction)
        plan = parse_result.plan.to_dict()
        status = "ok"
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    structure_call = next(
        (
            {
                "stage": item["stage"],
                "duration_sec": item["duration_sec"],
                "status": item["status"],
                "structure": item.get("result"),
                **({"error": item["error"]} if item.get("error") else {}),
            }
            for item in parser.contract_calls
            if item["stage"] == "single_pass_structure"
        ),
        None,
    )
    review_flags = _phase_review_flags(plan)
    input_audit = {
        "parser_received_only_instruction_argument": True,
        "original_instruction_sha256": _sha256_text(instruction),
        "contract_call_checks": parser.model_input_checks,
        "evaluation_truth_keys_detected": sorted(
            {
                key
                for item in parser.model_input_checks
                for key in item["evaluation_truth_keys_detected"]
            }
        ),
    }
    return {
        "diagnostic_kind": "parser_only_semantic_segmentation",
        "not_a_benchmark_result": True,
        "episode_id": episode_id,
        "scene_id": scene_id,
        "difficulty": difficulty,
        "protocol": AERIAL_VLN_PROTOCOL,
        "instruction": instruction,
        "instruction_sha256": _sha256_text(instruction),
        "status": status,
        "duration_sec": round(time.perf_counter() - started, 3),
        "final_phase_count": len(plan.get("phases", [])) if plan else None,
        "single_pass_structure": structure_call,
        "final_semantic_plan": plan,
        "parser_trace": dict(parser.last_parse_trace),
        "phase_review_flags": review_flags,
        "model_input_audit": input_audit,
        "error": error,
    }


def _model_input_check(
    stage: str,
    messages: list[dict[str, str]],
    *,
    original_instruction: str,
) -> dict[str, Any]:
    contents = [str(message.get("content") or "") for message in messages]
    joined = "\n".join(contents)
    return {
        "stage": stage,
        "message_count": len(messages),
        "original_instruction_present": original_instruction in joined,
        "evaluation_truth_keys_detected": [
            key for key in _EVALUATION_TRUTH_KEYS if key in joined
        ],
    }


def _phase_review_flags(plan: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not plan:
        return []
    flags = []
    for phase in plan.get("phases", []):
        if not isinstance(phase, dict):
            continue
        text = f"{phase.get('name', '')} {phase.get('goal', '')}".lower()
        families = [
            family
            for family, terms in _RESULT_FAMILY_TERMS.items()
            if any(term in text for term in terms)
        ]
        if len(families) > 1:
            flags.append(
                {
                    "phase_index": phase.get("index"),
                    "phase_name": phase.get("name"),
                    "detected_result_families": families,
                    "review_reason": (
                        "heuristic_only: multiple motion/result families appear in "
                        "one phase; inspect whether they share one stable navigable checkpoint"
                    ),
                }
            )
    return flags


def _build_summary(
    *,
    created_at: str,
    provenance: dict[str, Any],
    model_config: ModelConfig,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    ok = sum(item["status"] == "ok" for item in results)
    error = len(results) - ok
    by_difficulty = {
        difficulty: sum(item["difficulty"] == difficulty for item in results)
        for difficulty in ("low", "medium", "high")
    }
    return {
        "schema_version": 2,
        "diagnostic_kind": "parser_only_semantic_segmentation",
        "not_a_benchmark_result": True,
        "created_at": created_at,
        "protocol": AERIAL_VLN_PROTOCOL,
        "dataset": provenance,
        "model": {
            "provider": model_config.provider,
            "model_name": model_config.model_name,
            "timeout_sec": model_config.timeout_sec,
            "has_api_key": bool(model_config.api_key),
        },
        "counts": {
            "requested": len(SAMPLE_CASES),
            "ok": ok,
            "error": error,
            "by_difficulty": by_difficulty,
        },
        "cases": [
            {
                "episode_id": item["episode_id"],
                "scene_id": item["scene_id"],
                "difficulty": item["difficulty"],
                "status": item["status"],
                "final_phase_count": item["final_phase_count"],
                "structure_phase_count": len(
                    ((item.get("single_pass_structure") or {}).get("structure") or {}).get(
                        "phases", []
                    )
                ),
                "review_flag_count": len(item["phase_review_flags"]),
                "duration_sec": item["duration_sec"],
                "error": item["error"],
            }
            for item in results
        ],
    }


def _render_html(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    grouped = []
    for difficulty in ("low", "medium", "high"):
        cases = [item for item in results if item["difficulty"] == difficulty]
        grouped.append(
            f'<section><h2>{_e(difficulty.title())}</h2>'
            + "".join(_case_html(item) for item in cases)
            + "</section>"
        )
    dataset = summary["dataset"]
    model = summary["model"]
    counts = summary["counts"]
    body = (
        "<h1>AerialVLN-S Single-Pass Semantic Segmentation Diagnostic</h1>"
        '<p class="warning"><strong>Parser diagnostic only.</strong> '
        "No simulator, flight execution, benchmark Success, nDTW, or trajectory is included.</p>"
        '<div class="summary">'
        f"<div><strong>Split</strong><br>{_e(dataset['split'])}</div>"
        f"<div><strong>SHA256</strong><br><code>{_e(dataset['split_sha256'])}</code></div>"
        f"<div><strong>Protocol</strong><br><code>{_e(summary['protocol'])}</code></div>"
        f"<div><strong>Model</strong><br>{_e(model['provider'])} / {_e(model['model_name'])}</div>"
        f"<div><strong>Results</strong><br>{counts['ok']} ok / {counts['error']} error</div>"
        "</div>"
        + "".join(grouped)
    )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>AerialVLN-S Semantic Segmentation Diagnostic</title>"
        f"<style>{_STYLE}</style></head><body>{body}</body></html>"
    )


def _case_html(case: dict[str, Any]) -> str:
    structure_card = _structure_html(case.get("single_pass_structure") or {})
    plan = case.get("final_semantic_plan") or {}
    phases = plan.get("phases", []) if isinstance(plan, dict) else []
    phase_traces = {
        int(item["index"]): item
        for item in (case.get("parser_trace") or {}).get("phases", [])
        if isinstance(item, dict) and isinstance(item.get("index"), int)
    }
    phase_rows = "".join(
        _phase_row(phase, phase_traces.get(int(phase.get("index") or 0), {}))
        for phase in phases
        if isinstance(phase, dict)
    )
    flags = case["phase_review_flags"]
    flags_html = (
        "<ul>"
        + "".join(
            "<li>Phase "
            f"{_e(flag.get('phase_index'))} <code>{_e(flag.get('phase_name'))}</code>: "
            f"{_e(', '.join(flag['detected_result_families']))}</li>"
            for flag in flags
        )
        + "</ul>"
        if flags
        else '<p class="muted">No multi-motion local leg requires heuristic review.</p>'
    )
    error = (
        f'<pre class="error">{_e(case["error"])}</pre>' if case.get("error") else ""
    )
    return (
        '<details class="case" open><summary>'
        f'<span class="status {case["status"]}">{_e(case["status"])}</span> '
        f"{_e(case['episode_id'])} · scene {_e(case['scene_id'])} · "
        f"{_e(case.get('final_phase_count'))} phases</summary>"
        f'<p class="instruction">{_e(case["instruction"])}</p>{error}'
        "<h3>Single-pass structure</h3>"
        f'<div class="revisions">{structure_card}</div>'
        "<h3>Final semantic phases</h3>"
        '<div class="table-wrap"><table><thead><tr>'
        "<th>#</th><th>Name</th><th>Goal</th><th>Done when</th>"
        "<th>Continue when</th><th>Metrics</th><th>Selected skills</th>"
        f"</tr></thead><tbody>{phase_rows}</tbody></table></div>"
        "<h3>Human-review flags</h3>"
        f"{flags_html}"
        f'<p><a href="cases/{_e(case["episode_id"])}.json">Open raw case JSON</a></p>'
        "</details>"
    )


def _structure_html(structure_call: dict[str, Any]) -> str:
    structure = structure_call.get("structure") or {}
    phases = structure.get("phases", []) if isinstance(structure, dict) else []
    items = "".join(
        "<li>"
        f"<code>{_e(phase.get('index'))}. {_e(phase.get('name'))}</code>"
        f"<span>{_e(phase.get('goal'))}</span>"
        "</li>"
        for phase in phases
        if isinstance(phase, dict)
    )
    return (
        '<article class="revision">'
        f"<h4>{_e(structure_call.get('stage', 'single_pass_structure'))} · {len(phases)} phases</h4>"
        f'<p class="muted">{_e(structure_call.get("duration_sec"))} sec</p>'
        f"<ol>{items}</ol></article>"
    )


def _phase_row(phase: dict[str, Any], phase_trace: dict[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{_e(phase.get('index'))}</td>"
        f"<td><code>{_e(phase.get('name'))}</code></td>"
        f"<td>{_e(phase.get('goal'))}</td>"
        f"<td>{_list_html(phase.get('done_when'))}</td>"
        f"<td>{_list_html(phase.get('continue_when'))}</td>"
        f"<td><pre>{_e(json.dumps(phase.get('metrics', []), ensure_ascii=False, indent=2))}</pre></td>"
        f"<td>{_list_html(phase_trace.get('selected_skills'))}</td>"
        "</tr>"
    )


def _list_html(value: Any) -> str:
    items = value if isinstance(value, list) else []
    return "<ul>" + "".join(f"<li>{_e(item)}</li>" for item in items) + "</ul>"


def _json_ready(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _validate_model_config(config: ModelConfig) -> None:
    if config.provider != "openai_compatible":
        raise SystemExit(f"Unsupported model provider: {config.provider}")
    if not config.model_name:
        raise SystemExit("FLYAGENT_MODEL_NAME is required")
    if not config.api_key:
        raise SystemExit("FLYAGENT_MODEL_API_KEY is required")


def _resolve_output_dir(value: str | None) -> Path:
    if value:
        return Path(value)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(
        f"tests/manual_outputs/aerial_vln_semantic_segmentation_10_single_pass_{stamp}"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse ten fixed AerialVLN-S val_seen instructions with the real "
            "TaskParser and generate a standalone segmentation review."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default=".external_refs/AirVLN/DATA/data/aerialvln-s",
    )
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--instructions-file", type=Path,
                        help="Ordered instruction-only JSON list; compare before/current prompts without loading the dataset.")
    parser.add_argument("--before-prompts", type=Path,
                        help="Directory containing the three pre-edit prompt snapshots; required for comparison.")
    parser.add_argument("--structure-only", action="store_true",
                        help="Explicit-list comparison only: no skill selection or completion criteria.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional exact output directory; defaults to a timestamped manual_outputs path.",
    )
    return parser


# The comparison path deliberately bypasses legacy dataset loading and keyword
# review flags. This is an experiment adapter, not a production parser mode.
PROMPT_FILES = {
    "structure_prompt": "task_parser_structure_system.txt",
    "done_when_rules": "done_when_generation_system.txt",
    "criteria_prompt": "task_parser_phase_criteria_system.txt",
}


def load_instruction_list(path: Path) -> list[dict[str, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty ordered instruction list")
    seen, result = set(), []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "instruction", "protocol"}:
            raise ValueError("Each row contains only id, instruction, protocol")
        if any(not isinstance(v, str) or not v.strip() for v in row.values()):
            raise ValueError("Instruction fields must be non-empty strings")
        if row["id"] in seen or row["protocol"] not in {"aerial_vln", "generic"}:
            raise ValueError("Duplicate ID or unknown protocol")
        seen.add(row["id"])
        result.append(dict(row))
    return result


def _redacted(value: Any, api_key: str) -> Any:
    """Defense in depth for provider error strings; no config object is saved."""
    if isinstance(value, str):
        return value.replace(api_key, "[REDACTED]") if api_key else value
    if isinstance(value, dict):
        return {k: _redacted(v, api_key) for k, v in value.items()}
    if isinstance(value, list):
        return [_redacted(v, api_key) for v in value]
    return value


def _comparison_html(rows, results, mode="full_parse"):
    """Render structures as structures, never as synthetic completed plans."""
    structure_only = mode == "structure_only"
    cards = []
    for row in rows:
        columns = []
        for version in ("before", "after"):
            item = next((r for r in results if r["id"] == row["id"] and r["version"] == version), None)
            if item is None:
                body = "Not run"
            elif item["status"] != "ok":
                body = f'<pre class="error">{_e(item.get("error"))}</pre>'
            else:
                parsed = item["structure" if structure_only else "plan"]
                task_goal = parsed.get("task_goal", {})
                body = ("<h4>Task goal</h4><pre>"
                        + _e(json.dumps(task_goal, ensure_ascii=False, indent=2))
                        + "</pre><ol>" + "".join(
                    f'<li><strong>{_e(p["name"])}</strong><p>{_e(p["goal"])}</p>'
                    + ("" if structure_only else _list_html(p["done_when"])) + "</li>"
                    for p in parsed["phases"]) + "</ol>")
            link = f'<a href="cases/{item["artifact"]}">Raw request/result</a>' if item else ""
            columns.append(f'<article class="revision"><h3>{version}</h3>{body}{link}</article>')
        cards.append(f'<section class="case"><h2>{_e(row["id"])}</h2>'
                     f'<p class="instruction">{_e(row["instruction"])}</p>'
                     '<div class="revisions">' + ''.join(columns) + '</div></section>')
    title = "Phase structure only" if structure_only else "Phase + done_when"
    return ('<!doctype html><html><meta charset="utf-8"><title>Parser prompt comparison</title>'
            f'<style>{_STYLE}</style><h1>{title} · before / after</h1>'
            '<p>Parser-only model outputs. No simulator, semantic grading or navigation result. '
            'Each version is attempted once; phase count is not a success metric.</p>'
            + ''.join(cards) + '</html>')


def run_comparison(args) -> int:
    """Run paired parses, persisting every result before starting the next one."""
    if not args.before_prompts or not args.output_dir:
        raise ValueError("Comparison requires --before-prompts and a new --output-dir")
    mode = "structure_only" if getattr(args, "structure_only", False) else "full_parse"
    result_key = "structure" if mode == "structure_only" else "plan"
    rows = load_instruction_list(args.instructions_file)
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Comparison output must be new; never overwrite or resume silently")
    config = load_model_config(args.env_path)
    _validate_model_config(config)
    current = REPO_ROOT / "src/agent/semantic/prompts"
    versions = {name: {slot: (directory / filename).read_text(encoding="utf-8").strip()
                       for slot, filename in PROMPT_FILES.items()}
                for name, directory in (("before", args.before_prompts), ("after", current))}
    if mode == "structure_only":
        for slot in ("done_when_rules", "criteria_prompt"):
            if versions["before"][slot] != versions["after"][slot]:
                raise ValueError(f"Structure-only comparison requires unchanged {slot}")
    output.mkdir(parents=True)
    _write_json(output / "instructions.json", rows)
    _write_json(output / "prompts.json", {
        version: {slot: {"text": text, "sha256": _sha256_text(text)} for slot, text in prompts.items()}
        for version, prompts in versions.items()})
    results, aborted = [], False
    for index, row in enumerate(rows, 1):
        for version, prompts in versions.items():
            print(f'[{index}/{len(rows)}] {version} {row["id"]}', flush=True)
            parser = RecordingTaskParser(config, row["instruction"], prompts=prompts,
                protocol_context=AERIAL_VLN_PARSER_PROTOCOL_CONTEXT if row["protocol"] == "aerial_vln" else "")
            item = {**row, "mode": mode, "version": version, "artifact": f"{index:02d}_{version}.json"}
            started = time.perf_counter()
            try:
                parsed = (parser.parse_structure(row["instruction"]) if mode == "structure_only"
                          else parser.parse(row["instruction"]).plan)
                item.update(status="ok", **{result_key: parsed.to_dict()})
            except Exception as exc:
                item.update(status="error", error=f"{type(exc).__name__}: {exc}")
                aborted = True
            item.update(duration_sec=round(time.perf_counter() - started, 3),
                        exchanges=parser.exchanges, contracts=parser.contract_calls,
                        parser_trace=parser.last_parse_trace)
            item = _redacted(item, config.api_key)
            _write_json(output / "cases" / item["artifact"], item)
            results.append(item)
            _write_json(output / "summary.json", {
                "status": "aborted" if aborted else "running",
                "mode": mode, "model": config.redacted_summary(), "temperature": 0,
                "requested_pairs": len(rows), "results": results})
            (output / "review.html").write_text(_comparison_html(rows, results, mode), encoding="utf-8")
            print(f'  {item["status"]}: {len(item.get(result_key, {}).get("phases", []))} phases', flush=True)
            if aborted:
                return 1
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    summary["status"] = "completed"
    _write_json(output / "summary.json", summary)
    return 0


_STYLE = """
:root { color-scheme: light; font-family: Inter, Segoe UI, sans-serif; }
body { max-width: 1600px; margin: 0 auto; padding: 28px; color: #172033; background: #f5f7fb; }
h1, h2, h3, h4 { color: #10213d; }
code, pre { font-family: Consolas, ui-monospace, monospace; }
.warning { padding: 14px 18px; border-left: 5px solid #d97706; background: #fff7ed; }
.summary { display: grid; grid-template-columns: repeat(auto-fit,minmax(190px,1fr)); gap: 10px; margin: 20px 0; }
.summary > div { background: white; border: 1px solid #dbe3ef; border-radius: 8px; padding: 12px; overflow-wrap: anywhere; }
.case { background: white; border: 1px solid #cfd9e8; border-radius: 10px; margin: 14px 0; padding: 12px 16px; }
.case > summary { cursor: pointer; font-weight: 700; }
.instruction { font-size: 1.02rem; line-height: 1.55; padding: 12px; background: #eef4ff; border-radius: 6px; }
.status { display: inline-block; min-width: 42px; padding: 2px 8px; border-radius: 999px; color: white; text-align: center; }
.status.ok { background: #15803d; } .status.error { background: #b91c1c; }
.revisions { display: grid; grid-template-columns: repeat(auto-fit,minmax(280px,1fr)); gap: 12px; }
.revision { border: 1px solid #dbe3ef; border-radius: 8px; padding: 10px 14px; background: #fafcff; }
.revision ol { padding-left: 22px; } .revision li { margin-bottom: 9px; }
.revision li span { display: block; margin-top: 3px; color: #40516c; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { border: 1px solid #d9e1ec; padding: 8px; vertical-align: top; text-align: left; }
th { background: #edf2f9; position: sticky; top: 0; }
td ul { min-width: 220px; margin: 0; padding-left: 18px; }
td pre { max-width: 340px; white-space: pre-wrap; margin: 0; }
.muted { color: #66758c; } .error { color: #991b1b; background: #fef2f2; padding: 10px; white-space: pre-wrap; }
a { color: #1d4ed8; }
"""


if __name__ == "__main__":
    raise SystemExit(main())
