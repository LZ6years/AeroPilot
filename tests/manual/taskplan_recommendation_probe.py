"""Legacy single-paragraph TaskPlan A/B experiment for ``gpt-6.1-sol``.

This probe requires the original optional-planning paragraph and refuses newer
guidance whose applicability lives in the tool description. It remains usable
with its historical prompt contract; do not reinterpret it as the newer plan
stage experiment. Use plan_stage_behavior_probe.py for current saved-context
checks and aerial_vln_plan_context_probe.py for current navigation.

This evaluation-only orchestrator runs the same five AerialVLN and five
TravelUAV cases twice on the current executable source. The control condition
uses a byte-for-byte copy of the production planner prompt. The treatment
condition replaces only the first ``Planning And Knowledge`` paragraph with a
strong pre-flight TaskPlan requirement. Runtime, Context, tools and benchmark
semantics are not modified.

The two conditions are interleaved per case to reduce time and simulator drift.
Every attempt remains immutable, and only a cleanup-verified infrastructure
interruption may use one recovery directory. The Provider credential is read
from the process environment; the SSH key path may come from the process or the
ignored local ``.env`` file. Neither is written to the experiment.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm import ModelRequest, create_model_client
from src.llm.audit import redact_provider_value
from src.agent.config import read_env_file
from src.runtime.factory import RuntimeFactory
from src.tools.registry import default_registry
from src.tools.validation import validate_tool_args
from tests.manual import benchmark_20_case_probe as batch
from tests.manual import flight_experience_batch_probe as frozen_source
from tests.manual import model_matrix_benchmark_probe as matrix
from tests.manual import provider_instruction_probe


SCHEMA_VERSION = "taskplan_paired_ab_v2"
MODEL_NAME = "gpt-6.1-sol"
IKUNCODE_BASE_URL = "https://api.ikuncode.cc/v1"
CONDITIONS = ("control", "treatment")
TERMINAL_CASE_STATUSES = frozenset({"completed", "model_error"})
EXPECTED_PYTHON = Path(r"F:\conda_env\flyagent\python.exe")
DEFAULT_ROOT = ROOT / "runs/experiments/gpt_6_1_sol_taskplan_ab_20261006"
CASE_SOURCE_ROOT = ROOT / (
    "runs/experiments/model_matrix_cf_20261003_full/stage2/01_gpt_6_1_sol"
)
PROVIDER_MANIFEST = ROOT / (
    "runs/diagnostics/ikuncode_sdk_pilots_20261006_02/manifest.json"
)
PRODUCTION_PROMPT = ROOT / (
    "src/agent/planning/prompts/agent_tool_loop_system.txt"
)

# Only this production preference changes in the experimental treatment. Exact
# argument semantics stay in the shared ToolSpec rather than a copied prompt.
CONTROL_PLANNING_PARAGRAPH = "When a plan is useful, keep its recorded stage aligned with observed progress."

TREATMENT_PLANNING_PARAGRAPH = """Before any physical action in a navigation task, call runtime.update_plan once
to establish a concise, outcome-focused plan with stable step IDs. Use pending
or in_progress steps containing only id, description and status. Do
not invent environmental facts or movement parameters for unseen areas. Revise the plan
only when observed progress or the approach changes, using operation=patch and steps for existing IDs.
Completing an explicit movement does not by itself complete a separate landmark,
route-relation or arrival requirement."""

TEXT_EXTENSIONS = frozenset(
    {".json", ".jsonl", ".html", ".txt", ".log", ".md", ".patch", ".csv"}
)
REPORT_FILENAMES = frozenset(
    {
        "manifest.json",
        "summary.json",
        "comparison.json",
        "comparison.html",
        "index.html",
        "status.json",
        "admission_status.json",
        "review_export.json",
        "export_summary.json",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _ensure_ssh_key_environment(env_path: Path = ROOT / ".env") -> str:
    """Resolve the remote-simulator key without persisting it in artifacts."""

    configured = os.environ.get("FLYAGENT_SSH_KEY", "").strip()
    if not configured:
        configured = read_env_file(env_path).get("FLYAGENT_SSH_KEY", "").strip()
    if not configured:
        raise ValueError(
            "FLYAGENT_SSH_KEY must be set in the process or the local .env file"
        )
    resolved = Path(os.path.expandvars(configured)).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("Configured FLYAGENT_SSH_KEY does not exist")
    os.environ["FLYAGENT_SSH_KEY"] = str(resolved)
    return str(resolved)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def build_treatment_prompt(base_prompt: str) -> str:
    """Replace exactly one production paragraph and preserve all other text."""

    if base_prompt.count(CONTROL_PLANNING_PARAGRAPH) != 1:
        raise ValueError("Production prompt must contain one expected Planning paragraph")
    if TREATMENT_PLANNING_PARAGRAPH in base_prompt:
        raise ValueError("Production prompt already contains the treatment contract")
    treatment = base_prompt.replace(
        CONTROL_PLANNING_PARAGRAPH,
        TREATMENT_PLANNING_PARAGRAPH,
        1,
    )
    if (
        treatment.replace(
            TREATMENT_PLANNING_PARAGRAPH,
            CONTROL_PLANNING_PARAGRAPH,
            1,
        )
        != base_prompt
    ):
        raise AssertionError("Treatment changed text outside the Planning paragraph")
    return treatment


def execution_order(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the frozen counterbalanced 20-run schedule."""

    schedule: list[dict[str, Any]] = []
    for index, item in enumerate(cases):
        conditions = CONDITIONS if index % 2 == 0 else tuple(reversed(CONDITIONS))
        for condition in conditions:
            schedule.append(
                {
                    "sequence": len(schedule) + 1,
                    "case_index": index + 1,
                    "condition": condition,
                    "benchmark": item["benchmark"],
                    "case_id": item["case_id"],
                }
            )
    return schedule


def _review_root(root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    if root.parent.name == "experiments":
        return (root.parent.parent / "reviews" / root.name).resolve()
    return (root.parent / "reviews" / root.name).resolve()


def _profile_summary(
    provider_summary: Mapping[str, Any], prompt_path: Path
) -> dict[str, Any]:
    summary = dict(provider_summary)
    summary["action_planner_prompt_path"] = str(prompt_path.resolve())
    summary["has_api_key"] = True
    return summary


def _profile_without_prompt(summary: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(summary)
    result.pop("action_planner_prompt_path", None)
    return result


def _assert_profiles_equivalent(profiles: Mapping[str, Mapping[str, Any]]) -> None:
    if set(profiles) != set(CONDITIONS):
        raise ValueError("Both control and treatment profiles are required")
    if _profile_without_prompt(profiles["control"]) != _profile_without_prompt(
        profiles["treatment"]
    ):
        raise ValueError("Condition model profiles differ outside the prompt path")


def _historical_metrics(summary: Mapping[str, Any]) -> dict[str, int]:
    rows = list(summary.get("cases") or [])
    return {
        "cases": len(rows),
        "successes": sum(
            bool((row.get("metrics") or {}).get("success")) for row in rows
        ),
        "oracle_successes": sum(
            bool((row.get("metrics") or {}).get("oracle_success")) for row in rows
        ),
        "plan_calls": sum(int(row.get("plan_calls") or 0) for row in rows),
        "preflight_plan_cases": sum(
            row.get("first_flight_preceded_by_plan") is True for row in rows
        ),
    }


def _tool_contract_snapshot() -> dict[str, Any]:
    """Freeze the exact provider-visible tools shared by both conditions."""

    factory = RuntimeFactory()
    registry = default_registry(factory.plugin_registry)
    tools = [spec.to_provider_tool() for spec in registry.model_specs()]
    names = [item["function"]["name"] for item in tools]
    if "runtime_update_plan" not in names:
        raise ValueError("runtime.update_plan is missing from the model tool surface")
    return {
        "schema_version": "taskplan_ab_tool_contracts_v1",
        "count": len(tools),
        "names": names,
        "tools_sha256": _json_sha256(tools),
        "tools": tools,
    }


def _copy_source_snapshot(root: Path, hashes: Mapping[str, str]) -> None:
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    for name in hashes:
        source = ROOT / name
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for filename, command in (
        ("git_head.txt", ["git", "rev-parse", "HEAD"]),
        ("git_status.txt", ["git", "status", "--short"]),
        ("working_tree.patch", ["git", "diff", "--binary", "HEAD"]),
    ):
        (snapshot / filename).write_bytes(
            subprocess.check_output(command, cwd=ROOT)
        )


def prepare(
    root: Path,
    *,
    case_source_root: Path = CASE_SOURCE_ROOT,
    provider_manifest_path: Path = PROVIDER_MANIFEST,
    review_root: Path | None = None,
) -> dict[str, Any]:
    """Freeze both conditions without making a Provider or simulator request."""

    root = root.resolve()
    case_source_root = case_source_root.resolve()
    provider_manifest_path = provider_manifest_path.resolve()
    if root.exists():
        raise ValueError(f"Output exists; refusing to replace evidence: {root}")
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise ValueError("Less than 10 GiB free before experiment preparation")

    source_manifest_path = case_source_root / "manifest.json"
    historical_summary_path = case_source_root / "summary.json"
    for path in (
        source_manifest_path,
        historical_summary_path,
        provider_manifest_path,
        PRODUCTION_PROMPT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_manifest = _read_json(source_manifest_path)
    historical_summary = _read_json(historical_summary_path)
    provider_manifest = _read_json(provider_manifest_path)
    cases = [dict(item) for item in source_manifest.get("cases") or []]
    if len(cases) != 10 or len({item.get("case_id") for item in cases}) != 10:
        raise ValueError("Case source must contain ten distinct cases")
    if Counter(item.get("benchmark") for item in cases) != {
        "aerial": 5,
        "travel": 5,
    }:
        raise ValueError("Case source must contain five Aerial and five Travel cases")
    historical = _historical_metrics(historical_summary)
    if historical != {
        "cases": 10,
        "successes": 4,
        "oracle_successes": 6,
        "plan_calls": 0,
        "preflight_plan_cases": 0,
    }:
        raise ValueError(f"Unexpected historical appendix metrics: {historical}")

    provider_summary = provider_manifest.get("model_config")
    if not isinstance(provider_summary, Mapping):
        raise ValueError("Provider source has no redacted model_config")
    if (
        provider_summary.get("provider") != "openai_compatible"
        or provider_summary.get("model_name") != MODEL_NAME
        or str(provider_summary.get("base_url", "")).rstrip("/")
        != IKUNCODE_BASE_URL
        or provider_summary.get("api_protocol") != "responses"
        or provider_summary.get("reasoning_effort") != "low"
    ):
        raise ValueError("Provider source is not the approved iKunCode SDK profile")

    base_prompt_bytes = PRODUCTION_PROMPT.read_bytes()
    base_prompt = base_prompt_bytes.decode("utf-8")
    treatment_prompt = build_treatment_prompt(base_prompt)
    current_hashes = frozen_source.hashes()
    benchmark_hashes = batch.source_hashes()
    tool_contracts = _tool_contract_snapshot()

    root.mkdir(parents=True)
    config_dir = root / "config"
    config_dir.mkdir()
    control_prompt_path = config_dir / "control_system.txt"
    treatment_prompt_path = config_dir / "treatment_system.txt"
    control_prompt_path.write_bytes(base_prompt_bytes)
    treatment_prompt_path.write_text(
        treatment_prompt,
        encoding="utf-8",
        newline="\n",
    )
    tool_contract_path = config_dir / "provider_tools.json"
    batch.write_json(tool_contract_path, tool_contracts)

    profiles = {
        "control": _profile_summary(provider_summary, control_prompt_path),
        "treatment": _profile_summary(provider_summary, treatment_prompt_path),
    }
    _assert_profiles_equivalent(profiles)
    dataset_sha256 = dict(source_manifest["dataset_sha256"])
    schedule = execution_order(cases)
    condition_manifests: dict[str, str] = {}
    for condition in CONDITIONS:
        condition_root = root / condition
        condition_root.mkdir()
        condition_manifest = {
            "schema_version": "taskplan_ab_condition_v1",
            "condition": condition,
            "model": MODEL_NAME,
            "model_config": profiles[condition],
            "dataset_sha256": dataset_sha256,
            "source_hashes": benchmark_hashes,
            "tool_contracts_sha256": _sha256(tool_contract_path),
            "cases": cases,
        }
        condition_path = condition_root / "manifest.json"
        batch.write_json(condition_path, condition_manifest)
        condition_manifests[condition] = str(condition_path.resolve())

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now(),
        "model": MODEL_NAME,
        "provider": "openai_python_sdk_via_ikuncode",
        "analysis": "paired_intention_to_treat",
        "formal_runs": 20,
        "case_source": {
            "root": str(case_source_root),
            "manifest_path": str(source_manifest_path),
            "manifest_sha256": _sha256(source_manifest_path),
            "summary_path": str(historical_summary_path),
            "summary_sha256": _sha256(historical_summary_path),
        },
        "provider_source": {
            "manifest_path": str(provider_manifest_path),
            "manifest_sha256": _sha256(provider_manifest_path),
        },
        "historical_appendix": {
            "label": "2026-10-03 CF historical context; excluded from A/B deltas",
            "metrics": historical,
        },
        "dataset_sha256": dataset_sha256,
        "cases": cases,
        "execution_order": schedule,
        "condition_manifests": condition_manifests,
        "model_configs": profiles,
        "prompts": {
            "production_path": str(PRODUCTION_PROMPT.resolve()),
            "production_sha256": _sha256(PRODUCTION_PROMPT),
            "control_path": str(control_prompt_path.resolve()),
            "control_sha256": _sha256(control_prompt_path),
            "treatment_path": str(treatment_prompt_path.resolve()),
            "treatment_sha256": _sha256(treatment_prompt_path),
            "control_planning_paragraph": CONTROL_PLANNING_PARAGRAPH,
            "treatment_planning_paragraph": TREATMENT_PLANNING_PARAGRAPH,
        },
        "tool_contracts": {
            "path": str(tool_contract_path.resolve()),
            "sha256": _sha256(tool_contract_path),
            "tools_sha256": tool_contracts["tools_sha256"],
            "count": tool_contracts["count"],
            "names": tool_contracts["names"],
        },
        "complete_source_hashes": current_hashes,
        "source_hashes": benchmark_hashes,
        "review_root": str(_review_root(root, review_root)),
        "interpretation": {
            "treatment_preflight_plan_threshold": 0.8,
            "control_spontaneous_plan_threshold": 0.2,
            "subgroup_analysis": "descriptive_only",
            "statistical_significance_claim": False,
        },
    }
    batch.write_json(root / "manifest.json", manifest)
    _copy_source_snapshot(root, current_hashes)
    report(root, review_root=review_root)
    return manifest


def _validate_frozen_inputs(manifest: Mapping[str, Any]) -> None:
    if frozen_source.hashes() != manifest["complete_source_hashes"]:
        raise ValueError("Executable source changed after experiment preparation")
    if batch.source_hashes() != manifest["source_hashes"]:
        raise ValueError("Benchmark execution source changed after preparation")
    for section, fields in (
        ("case_source", ("manifest_path", "summary_path")),
        ("provider_source", ("manifest_path",)),
    ):
        record = manifest[section]
        for field in fields:
            path = Path(record[field])
            digest_field = field.replace("_path", "_sha256")
            if not path.is_file() or _sha256(path) != record[digest_field]:
                raise ValueError(f"Frozen input changed: {section}.{field}")
    prompts = manifest["prompts"]
    for name in ("production", "control", "treatment"):
        path = Path(prompts[f"{name}_path"])
        if not path.is_file() or _sha256(path) != prompts[f"{name}_sha256"]:
            raise ValueError(f"Frozen prompt changed: {name}")
    control = Path(prompts["control_path"]).read_bytes()
    production = Path(prompts["production_path"]).read_bytes()
    if control != production:
        raise ValueError("Control prompt no longer matches the production snapshot")
    treatment = Path(prompts["treatment_path"]).read_text(encoding="utf-8")
    if build_treatment_prompt(control.decode("utf-8")) != treatment:
        raise ValueError("Treatment prompt contains changes outside its one paragraph")

    tool_record = manifest["tool_contracts"]
    tool_path = Path(tool_record["path"])
    if not tool_path.is_file() or _sha256(tool_path) != tool_record["sha256"]:
        raise ValueError("Frozen provider tool contracts changed")
    if _read_json(tool_path) != _tool_contract_snapshot():
        raise ValueError("Current provider tool contracts differ from the frozen snapshot")


def _load(
    root: Path,
    *,
    require_key: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected TaskPlan A/B experiment schema")
    if Path(sys.executable).resolve() != EXPECTED_PYTHON.resolve():
        raise ValueError(f"Unexpected Python interpreter: {sys.executable}")
    _validate_frozen_inputs(manifest)
    profiles = manifest["model_configs"]
    _assert_profiles_equivalent(profiles)
    configs: dict[str, Any] = {}
    if require_key:
        for condition in CONDITIONS:
            config = matrix._profile_from_summary(profiles[condition])
            if (
                config.model_name != MODEL_NAME
                or config.base_url.rstrip("/") != IKUNCODE_BASE_URL
                or config.api_protocol != "responses"
                or config.reasoning_effort != "low"
            ):
                raise ValueError(f"Frozen Provider profile changed: {condition}")
            configs[condition] = config
    return manifest, configs


def _secret_leaks(root: Path, secrets: Sequence[str]) -> list[str]:
    needles = [value.encode("utf-8") for value in secrets if value]
    if not needles:
        return []
    leaks: list[str] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS:
            content = path.read_bytes()
            if any(needle in content for needle in needles):
                leaks.append(path.relative_to(root).as_posix())
    return sorted(leaks)


def _inline_image_report_leaks(root: Path) -> list[str]:
    leaks: list[str] = []
    for path in root.rglob("*"):
        if path.is_file() and path.name in REPORT_FILENAMES:
            if b"data:image/" in path.read_bytes().lower():
                leaks.append(path.relative_to(root).as_posix())
    return sorted(leaks)


def _assert_no_secrets(root: Path, api_key: str) -> None:
    ssh_key = os.environ.get("FLYAGENT_SSH_KEY", "")
    leaks = _secret_leaks(root, (api_key, ssh_key))
    if leaks:
        raise ValueError(f"Credential leaked into experiment artifacts: {leaks}")
    image_leaks = _inline_image_report_leaks(root)
    if image_leaks:
        raise ValueError(f"Inline image leaked into report artifacts: {image_leaks}")


def _condition_admission(
    condition: str,
    config: Any,
    prompt: str,
    tools: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Send one no-flight request through the exact condition prompt and tools."""

    started = time.monotonic()
    client = create_model_client(config.to_provider_settings())
    parameters = {"tool_choice": "auto", "temperature": 0}
    parameters.update(config.generation_parameters())
    response = client.request(
        ModelRequest(
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "No flight will be executed. Call runtime_update_plan exactly once "
                        "with operation=replace, explanation=null and a concise two-step inspection plan, both steps pending and "
                        "containing only id, description and status. Do not call any other function."
                    ),
                },
            ],
            tools=[dict(item) for item in tools],
            parameters=parameters,
        ),
        error_prefix=f"{condition} prompt admission failed",
    )
    errors: list[str] = []
    calls = response.tool_calls
    if len(calls) != 1:
        errors.append(f"expected one tool call, received {len(calls)}")
    elif calls[0].get("function", {}).get("name") != "runtime_update_plan":
        errors.append("expected runtime_update_plan")
    else:
        try:
            arguments = json.loads(calls[0]["function"]["arguments"])
            schema = next(
                item["function"]["parameters"]
                for item in tools
                if item["function"]["name"] == "runtime_update_plan"
            )
            validate_tool_args(arguments, schema)
            from src.agent.planning.task_plan import TaskPlan
            TaskPlan().updated(arguments)
        except Exception as exc:
            errors.append(f"invalid plan arguments: {type(exc).__name__}: {exc}")
    public_instructions = response.data.get("instructions")
    instructions_text = (
        public_instructions.lower() if isinstance(public_instructions, str) else ""
    )
    return {
        "condition": condition,
        "accepted": not errors,
        "errors": errors,
        "elapsed_seconds": time.monotonic() - started,
        "request_id": response.transport_trace.get("request_id"),
        "usage": response.usage,
        "tool_calls": calls,
        "public_text": response.text,
        "response_instructions_visibility": (
            "visible" if public_instructions is not None else "not_returned"
        ),
        "response_instructions_codex_marker": "you are codex" in instructions_text,
        "response_instructions": redact_provider_value(public_instructions),
        "transport": response.transport_trace,
    }


def admit(root: Path) -> dict[str, Any]:
    """Run the shared Provider gate and one exact-prompt request per condition."""

    root = root.resolve()
    manifest, configs = _load(root, require_key=True)
    status_path = root / "admission_status.json"
    if status_path.exists():
        status = _read_json(status_path)
        if status.get("status") == "passed":
            return status
        raise ValueError("Prior admission failed; preserve it and prepare a new root")

    admission_root = root / "admission"
    admission_root.mkdir(exist_ok=False)
    status: dict[str, Any] = {
        "status": "running",
        "started_at": _now(),
        "model_configs": {
            condition: configs[condition].redacted_summary()
            for condition in CONDITIONS
        },
    }
    batch.write_json(status_path, status)
    api_key = configs["control"].api_key
    try:
        shared = provider_instruction_probe.run_probe(
            configs["control"],
            admission_root / "provider_protocol",
            expected_base_url=IKUNCODE_BASE_URL,
            expected_model=MODEL_NAME,
        )
        tool_contracts = _read_json(Path(manifest["tool_contracts"]["path"]))
        condition_records = []
        for condition in CONDITIONS:
            prompt = Path(manifest["prompts"][f"{condition}_path"]).read_text(
                encoding="utf-8"
            )
            condition_records.append(
                _condition_admission(
                    condition,
                    configs[condition],
                    prompt,
                    tool_contracts["tools"],
                )
            )
        blockers = list(shared.get("blockers") or [])
        blockers.extend(
            f"{record['condition']}_prompt_admission"
            for record in condition_records
            if not record["accepted"]
        )
        codex_visible = bool(shared.get("codex_instruction_visible")) or any(
            record.get("response_instructions_codex_marker") is True
            for record in condition_records
        )
        warnings = list(shared.get("warnings") or [])
        if codex_visible and "upstream_codex_instruction_visible" not in warnings:
            warnings.append("upstream_codex_instruction_visible")
        status.update(
            status="passed" if not blockers else "failed",
            blockers=sorted(set(blockers)),
            warnings=sorted(set(warnings)),
            codex_instruction_visible=codex_visible,
            shared_protocol={
                "status": shared.get("status"),
                "report": str(
                    admission_root
                    / "provider_protocol/provider_instruction_probe.json"
                ),
                "max_reported_instruction_tokens": shared.get(
                    "max_reported_instruction_tokens"
                ),
            },
            condition_requests=condition_records,
        )
        return status
    except Exception as exc:
        status.update(
            status="failed",
            blockers=["admission_exception"],
            error={
                "type": type(exc).__name__,
                "message": matrix._redact(exc, api_key=api_key),
            },
        )
        return status
    finally:
        status["finished_at"] = _now()
        batch.write_json(status_path, status)
        _assert_no_secrets(root, api_key)


def _next_case_directory(
    condition_root: Path,
    index: int,
    item: Mapping[str, Any],
    *,
    resume: bool,
) -> Path | None:
    current = batch.latest_case_dir(condition_root, index, dict(item))
    if not current.exists():
        return current
    status_path = current / "status.json"
    if not status_path.exists():
        raise ValueError(f"Unclassified existing case output: {current}")
    status = _read_json(status_path)
    if status.get("status") in TERMINAL_CASE_STATUSES:
        return None
    if not resume or status.get("cleanup_verified") is not True:
        raise ValueError(f"Case requires verified recovery: {current}")
    original = batch.case_dir(condition_root, index, dict(item))
    if list(original.parent.glob(original.name + "_recovery_*")):
        raise ValueError("Only one infrastructure recovery is permitted per case")
    return original.with_name(original.name + "_recovery_01")


def _condition_summary(root: Path, condition: str) -> dict[str, Any]:
    condition_root = root / condition
    manifest = _read_json(condition_root / "manifest.json")
    return batch.report(condition_root, manifest)


def _usage_request_tokens(usage: Mapping[str, Any], field: str) -> int:
    attribution = usage.get("attribution")
    if not isinstance(attribution, Mapping):
        return 0
    request_fields = attribution.get("request_fields")
    if not isinstance(request_fields, Mapping):
        return 0
    record = request_fields.get(field)
    return int(record.get("input_tokens") or 0) if isinstance(record, Mapping) else 0


def _audit_extra(batch_root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    attempt = row.get("attempt_dir")
    empty = {
        "planner_rounds": 0,
        "decision_count": 0,
        "first_plan_round": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "instructions_tokens": 0,
        "tools_tokens": 0,
        "compactor_statuses": [],
    }
    if not attempt:
        return empty
    audit_path = (
        batch_root
        / str(attempt)
        / "evaluation_output/evaluation/cases"
        / str(row["case_id"])
        / "audit.json"
    )
    if not audit_path.is_file():
        return empty
    audit = _read_json(audit_path)
    decisions = list(audit.get("planner_debug") or [])
    planner_rounds = 0
    input_tokens = output_tokens = total_tokens = 0
    instructions_tokens = tools_tokens = 0
    compactor_statuses: list[str] = []
    for decision in decisions:
        for model_round in decision.get("model_rounds") or []:
            planner_rounds += 1
            response = model_round.get("response") or {}
            usage = response.get("usage") or {}
            input_tokens += int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
            output_tokens += int(
                usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
            )
            total_tokens += int(usage.get("total_tokens") or 0)
            instructions_tokens += _usage_request_tokens(usage, "instructions")
            tools_tokens += _usage_request_tokens(usage, "tools")
            projection = model_round.get("history_projection") or {}
            background = projection.get("background_compaction") or {}
            status = background.get("status")
            if isinstance(status, str) and status and status != "idle":
                compactor_statuses.append(status)
    first_plan_round = None
    assistant_round = 0
    for event in audit.get("agent_transcript") or []:
        if event.get("kind") != "assistant_turn":
            continue
        assistant_round += 1
        if any(
            call.get("tool_name") == "runtime.update_plan"
            for call in (event.get("payload") or {}).get("tool_calls") or []
        ):
            first_plan_round = assistant_round
            break
    return {
        "planner_rounds": planner_rounds,
        "decision_count": len(decisions),
        "first_plan_round": first_plan_round,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "instructions_tokens": instructions_tokens,
        "tools_tokens": tools_tokens,
        "compactor_statuses": sorted(set(compactor_statuses)),
    }


def _case_view(batch_root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(row.get("metrics") or {})
    plan_calls = int(row.get("plan_calls") or 0)
    return {
        "status": row.get("status"),
        "benchmark_status": row.get("benchmark_status"),
        "cleanup_verified": row.get("cleanup_verified"),
        "success": metrics.get("success"),
        "oracle_success": metrics.get("oracle_success"),
        "ndtw": metrics.get("ndtw"),
        "sdtw": metrics.get("sdtw"),
        "distance_to_goal": metrics.get("distance_to_goal"),
        "navigation_error": metrics.get("navigation_error"),
        "spl": metrics.get("spl"),
        "path_length": metrics.get("path_length"),
        "uav_action_steps": metrics.get("uav_action_steps"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "plan_calls": plan_calls,
        "plan_updates": max(plan_calls - 1, 0),
        "first_flight_preceded_by_plan": row.get(
            "first_flight_preceded_by_plan"
        ),
        "protocol_rejections": row.get("protocol_rejections"),
        "invalid_argument_rejections": row.get("invalid_argument_rejections"),
        "infrastructure_error": row.get("infrastructure_error"),
        **_audit_extra(batch_root, row),
    }


def _numeric_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
        and not isinstance(row.get(key), bool)
    ]
    if not values:
        return {"count": 0, "mean": None, "median": None, "sum": 0.0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sum": sum(values),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    terminal = [row for row in rows if row.get("status") in TERMINAL_CASE_STATUSES]
    completed = [row for row in rows if row.get("status") == "completed"]
    compactor = Counter(
        status
        for row in rows
        for status in row.get("compactor_statuses") or []
    )
    result = {
        "cases": len(rows),
        "terminal": len(terminal),
        "completed": len(completed),
        "successes": sum(row.get("success") is True for row in rows),
        "oracle_successes": sum(row.get("oracle_success") is True for row in rows),
        "plan_calls": sum(int(row.get("plan_calls") or 0) for row in rows),
        "plan_updates": sum(int(row.get("plan_updates") or 0) for row in rows),
        "cases_with_plan": sum(int(row.get("plan_calls") or 0) > 0 for row in rows),
        "preflight_plan_cases": sum(
            row.get("first_flight_preceded_by_plan") is True for row in rows
        ),
        "protocol_rejections": sum(
            int(row.get("protocol_rejections") or 0) for row in rows
        ),
        "invalid_argument_rejections": sum(
            int(row.get("invalid_argument_rejections") or 0) for row in rows
        ),
        "provider_or_infrastructure_errors": sum(
            bool(row.get("infrastructure_error"))
            or row.get("status") not in TERMINAL_CASE_STATUSES | {"pending"}
            for row in rows
        ),
        "compactor_statuses": dict(compactor),
    }
    for key in (
        "ndtw",
        "sdtw",
        "distance_to_goal",
        "navigation_error",
        "spl",
        "path_length",
        "uav_action_steps",
        "elapsed_seconds",
        "planner_rounds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "instructions_tokens",
        "tools_tokens",
    ):
        result[key] = _numeric_summary(rows, key)
    result["by_benchmark"] = {
        benchmark: {
            "cases": len(subset),
            "terminal": sum(
                row.get("status") in TERMINAL_CASE_STATUSES for row in subset
            ),
            "successes": sum(row.get("success") is True for row in subset),
            "oracle_successes": sum(
                row.get("oracle_success") is True for row in subset
            ),
            "path_length": _numeric_summary(subset, "path_length"),
            "uav_action_steps": _numeric_summary(subset, "uav_action_steps"),
            "ndtw": _numeric_summary(subset, "ndtw"),
            "sdtw": _numeric_summary(subset, "sdtw"),
            "distance_to_goal": _numeric_summary(subset, "distance_to_goal"),
            "navigation_error": _numeric_summary(subset, "navigation_error"),
            "spl": _numeric_summary(subset, "spl"),
        }
        for benchmark in ("aerial", "travel")
        for subset in [[row for row in rows if row.get("benchmark") == benchmark]]
    }
    return result


def _paired_transitions(pairs: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    result = Counter()
    for pair in pairs:
        control = pair["control"].get(key) is True
        treatment = pair["treatment"].get(key) is True
        if treatment and not control:
            result["treatment_win"] += 1
        elif control and not treatment:
            result["control_win"] += 1
        else:
            result["tie"] += 1
    return {
        "treatment_win": result["treatment_win"],
        "control_win": result["control_win"],
        "tie": result["tie"],
    }


def paired_comparison(
    control_root: Path,
    control_summary: Mapping[str, Any],
    treatment_root: Path,
    treatment_summary: Mapping[str, Any],
    *,
    historical_appendix: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one partial-or-final paired ITT report from current-run arms."""

    control_rows = {
        str(row["case_id"]): row for row in control_summary.get("cases") or []
    }
    treatment_rows = {
        str(row["case_id"]): row for row in treatment_summary.get("cases") or []
    }
    if list(control_rows) != list(treatment_rows):
        raise ValueError("Control and treatment cases differ or changed order")
    pairs = []
    for index, (case_id, control_row) in enumerate(control_rows.items(), 1):
        treatment_row = treatment_rows[case_id]
        pairs.append(
            {
                "index": index,
                "case_id": case_id,
                "benchmark": control_row["benchmark"],
                "difficulty": control_row.get("difficulty"),
                "control": {
                    "benchmark": control_row["benchmark"],
                    **_case_view(control_root, control_row),
                },
                "treatment": {
                    "benchmark": treatment_row["benchmark"],
                    **_case_view(treatment_root, treatment_row),
                },
            }
        )
    control_view = [pair["control"] for pair in pairs]
    treatment_view = [pair["treatment"] for pair in pairs]
    control_aggregate = _aggregate(control_view)
    treatment_aggregate = _aggregate(treatment_view)
    complete_pairs = [
        pair
        for pair in pairs
        if pair["control"]["status"] in TERMINAL_CASE_STATUSES
        and pair["treatment"]["status"] in TERMINAL_CASE_STATUSES
    ]
    is_final = len(complete_pairs) == len(pairs) == 10
    interpretation_warnings: list[str] = []
    if is_final:
        if treatment_aggregate["preflight_plan_cases"] / 10 < 0.8:
            interpretation_warnings.append("treatment_plan_requirement_not_reliably_adopted")
        if control_aggregate["preflight_plan_cases"] / 10 > 0.2:
            interpretation_warnings.append("condition_separation_insufficient")
    primary_results = None
    if is_final:
        primary_results = {
            "delta": {
                "successes": treatment_aggregate["successes"]
                - control_aggregate["successes"],
                "oracle_successes": treatment_aggregate["oracle_successes"]
                - control_aggregate["oracle_successes"],
                "elapsed_seconds": treatment_aggregate["elapsed_seconds"]["sum"]
                - control_aggregate["elapsed_seconds"]["sum"],
                "total_tokens": treatment_aggregate["total_tokens"]["sum"]
                - control_aggregate["total_tokens"]["sum"],
                "planner_rounds": treatment_aggregate["planner_rounds"]["sum"]
                - control_aggregate["planner_rounds"]["sum"],
            },
            "success_transitions": _paired_transitions(complete_pairs, "success"),
            "oracle_success_transitions": _paired_transitions(
                complete_pairs, "oracle_success"
            ),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _now(),
        "status": "complete" if is_final else "partial",
        "analysis": "paired_intention_to_treat",
        "completed_pairs": len(complete_pairs),
        "expected_pairs": len(pairs),
        "control": control_aggregate,
        "treatment": treatment_aggregate,
        "primary_results": primary_results,
        "interpretation_warnings": interpretation_warnings,
        "historical_appendix": dict(historical_appendix or {}),
        "limitations": [
            "The treatment is a strong prompt requirement, not a Runtime-forced plan call.",
            "Adopted versus non-adopted subgroups are descriptive and self-selected.",
            "Ten paired cases do not support a statistical-significance claim.",
            "The historical CF result is context only and is excluded from A/B deltas.",
        ],
        "pairs": pairs,
    }


def _review_validation(review_root: Path) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    complete = True
    for condition in CONDITIONS:
        path = review_root / condition / "export_summary.json"
        if not path.is_file():
            conditions[condition] = {"status": "pending"}
            complete = False
            continue
        validation = dict(_read_json(path).get("validation") or {})
        if int(validation.get("max_path_length") or 0) > 240:
            raise ValueError(f"Review path exceeds 240 characters: {condition}")
        conditions[condition] = {"status": "ready", **validation}
    return {"complete": complete, "conditions": conditions}


def _comparison_html(
    comparison: Mapping[str, Any], review_root: Path
) -> str:
    rows = []
    for pair in comparison["pairs"]:
        index = int(pair["index"])
        control = pair["control"]
        treatment = pair["treatment"]
        links = []
        for condition in CONDITIONS:
            path = review_root / condition / "cases" / f"{index:02d}" / "review.html"
            links.append(
                f'<a href="{html.escape(path.as_uri())}">{condition}</a>'
                if path.is_file()
                else f"{condition}: pending"
            )
        values = (
            pair["case_id"],
            pair["benchmark"],
            control.get("success"),
            treatment.get("success"),
            control.get("oracle_success"),
            treatment.get("oracle_success"),
            control.get("plan_calls"),
            treatment.get("plan_calls"),
            control.get("first_flight_preceded_by_plan"),
            treatment.get("first_flight_preceded_by_plan"),
            control.get("uav_action_steps"),
            treatment.get("uav_action_steps"),
            " / ".join(links),
        )
        rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(value))}</td>" for value in values[:-1])
            + f"<td>{values[-1]}</td></tr>"
        )
    aggregate = {
        "status": comparison["status"],
        "completed_pairs": comparison["completed_pairs"],
        "control": comparison["control"],
        "treatment": comparison["treatment"],
        "primary_results": comparison["primary_results"],
        "interpretation_warnings": comparison["interpretation_warnings"],
        "limitations": comparison["limitations"],
    }
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>gpt-6.1-sol TaskPlan paired A/B</title>"
        "<style>body{font-family:system-ui;margin:24px}table{border-collapse:collapse}"
        "th,td{border:1px solid #ccc;padding:6px;vertical-align:top}pre{white-space:pre-wrap}</style>"
        "</head><body><h1>gpt-6.1-sol TaskPlan paired A/B</h1>"
        f"<p>Status: <strong>{html.escape(str(comparison['status']))}</strong>; "
        f"complete pairs: {comparison['completed_pairs']}/{comparison['expected_pairs']}</p>"
        "<p>Partial reports do not declare a winner.</p><h2>Aggregate</h2><pre>"
        + html.escape(json.dumps(aggregate, ensure_ascii=False, indent=2))
        + "</pre><h2>Paired cases</h2><table><tr>"
        "<th>Case</th><th>Benchmark</th><th>C success</th><th>T success</th>"
        "<th>C oracle</th><th>T oracle</th><th>C plans</th><th>T plans</th>"
        "<th>C preflight plan</th><th>T preflight plan</th><th>C actions</th>"
        "<th>T actions</th><th>Reviews</th></tr>"
        + "".join(rows)
        + "</table></body></html>"
    )


def report(
    root: Path,
    *,
    review_root: Path | None = None,
) -> dict[str, Any]:
    """Refresh partial summaries without making Provider or simulator calls."""

    root = root.resolve()
    manifest = _read_json(root / "manifest.json")
    control_summary = _condition_summary(root, "control")
    treatment_summary = _condition_summary(root, "treatment")
    comparison = paired_comparison(
        root / "control",
        control_summary,
        root / "treatment",
        treatment_summary,
        historical_appendix=manifest.get("historical_appendix"),
    )
    destination = _review_root(root, review_root)
    comparison["review_validation"] = _review_validation(destination)
    batch.write_json(root / "comparison.json", comparison)
    rendered = _comparison_html(comparison, destination)
    (root / "comparison.html").write_text(
        rendered,
        encoding="utf-8",
        newline="\n",
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "index.html").write_text(
        rendered,
        encoding="utf-8",
        newline="\n",
    )
    return comparison


def status_snapshot(root: Path) -> dict[str, Any]:
    """Read the immutable schedule and current attempt states without running work."""

    root = root.resolve()
    manifest, _ = _load(root, require_key=False)
    rows = []
    counts = Counter()
    for entry in manifest["execution_order"]:
        index = int(entry["case_index"]) - 1
        item = dict(manifest["cases"][index])
        condition_root = root / str(entry["condition"])
        directory = batch.latest_case_dir(condition_root, index, item)
        status_path = directory / "status.json"
        if status_path.is_file():
            record = _read_json(status_path)
            state = str(record.get("status") or "unknown")
            cleanup = record.get("cleanup_verified")
        elif directory.exists():
            state = "unclassified"
            cleanup = None
        else:
            state = "pending"
            cleanup = None
        counts[state] += 1
        rows.append(
            {
                **entry,
                "status": state,
                "cleanup_verified": cleanup,
                "attempt_dir": (
                    str(directory.relative_to(root)) if directory.exists() else None
                ),
            }
        )
    admission_path = root / "admission_status.json"
    return {
        "schema_version": SCHEMA_VERSION,
        "admission": (
            _read_json(admission_path).get("status")
            if admission_path.is_file()
            else "pending"
        ),
        "counts": dict(counts),
        "runs": rows,
    }


def run_formal(
    root: Path,
    *,
    resume: bool = False,
    review_root: Path | None = None,
) -> dict[str, Any]:
    """Run the 20 frozen attempts in counterbalanced paired order."""

    root = root.resolve()
    _ensure_ssh_key_environment()
    manifest, configs = _load(root, require_key=True)
    admission_path = root / "admission_status.json"
    if not admission_path.is_file() or _read_json(admission_path).get("status") != "passed":
        raise ValueError("Provider admission must pass before formal cases")
    destination = _review_root(root, review_root)
    api_key = configs["control"].api_key

    for condition in CONDITIONS:
        condition_root = root / condition
        condition_manifest = _read_json(condition_root / "manifest.json")
        if not (condition_root / "preflight.json").exists():
            with matrix._scoped_environment(configs[condition]):
                batch.preflight(condition_root, condition_manifest)

    for entry in manifest["execution_order"]:
        _load(root, require_key=False)
        condition = str(entry["condition"])
        index = int(entry["case_index"]) - 1
        item = dict(manifest["cases"][index])
        condition_root = root / condition
        condition_manifest = _read_json(condition_root / "manifest.json")
        directory = _next_case_directory(
            condition_root,
            index,
            item,
            resume=resume,
        )
        if directory is None:
            continue
        if shutil.disk_usage(ROOT).free < 5 * 1024**3:
            raise ValueError("Less than 5 GiB free before next formal case")
        print(
            f"START {entry['sequence']}/20 {condition} "
            f"{item['benchmark']} {item['case_id']}",
            flush=True,
        )
        with matrix._scoped_environment(configs[condition]):
            status = batch.run_one(
                condition_root,
                index,
                item,
                directory=directory,
            )
        batch.report(condition_root, condition_manifest)
        batch.refresh_reviews(condition_root, destination / condition)
        comparison = report(root, review_root=destination)
        print(
            json.dumps(
                {
                    "sequence": entry["sequence"],
                    "condition": condition,
                    "case_id": item["case_id"],
                    **status,
                    "completed_pairs": comparison["completed_pairs"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if status.get("cleanup_verified") is not True:
            raise RuntimeError(f"Cleanup not verified: {directory}")
        if status.get("status") not in TERMINAL_CASE_STATUSES:
            raise RuntimeError(
                f"Formal run paused: {condition}/{item['case_id']} "
                f"status={status.get('status')}"
            )
        _assert_no_secrets(root, api_key)

    for condition in CONDITIONS:
        condition_root = root / condition
        condition_manifest = _read_json(condition_root / "manifest.json")
        batch.report(condition_root, condition_manifest)
        batch.refresh_reviews(condition_root, destination / condition)
    result = report(root, review_root=destination)
    if result["status"] == "complete" and not result["review_validation"]["complete"]:
        raise ValueError("Final Review export validation is incomplete")
    _assert_no_secrets(root, api_key)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "admit", "run", "status", "report"),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--case-source-root", type=Path, default=CASE_SOURCE_ROOT)
    parser.add_argument("--provider-manifest", type=Path, default=PROVIDER_MANIFEST)
    parser.add_argument("--review-root", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retry one cleanup-verified infrastructure interruption",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            args.root,
            case_source_root=args.case_source_root,
            provider_manifest_path=args.provider_manifest,
            review_root=args.review_root,
        )
        output: Mapping[str, Any] = {
            "manifest": str(args.root.resolve() / "manifest.json"),
            "cases": len(result["cases"]),
            "formal_runs": result["formal_runs"],
            "review_root": result["review_root"],
        }
    elif args.command == "admit":
        output = admit(args.root)
    elif args.command == "run":
        output = run_formal(
            args.root,
            resume=args.resume,
            review_root=args.review_root,
        )
    elif args.command == "status":
        output = status_snapshot(args.root)
    else:
        output = report(args.root, review_root=args.review_root)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
