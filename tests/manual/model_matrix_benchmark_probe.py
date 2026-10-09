"""Freeze and supervise the two-stage five-model benchmark matrix.

This module is evaluation-only orchestration.  It never changes production
prompts, Runtime behavior, benchmark semantics, or provider credentials.  The
first stage proves that each configured model can use the real multimodal tool
protocol, history compactor, and one non-scoring simulator pilot.  The second
stage, intentionally invoked separately, runs the same frozen five AerialVLN
and five TravelUAV cases for each qualified model.

Credentials are inherited from the parent process.  They are never accepted as
CLI arguments and never serialized.  All child commands use ``sys.executable``
so launching this file with the repository's ``flyagent`` interpreter also
controls every diagnostic and simulator child.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.benchmarks.aerial_vln.runner import AerialVlnBenchmarkRunner
from evaluation.benchmarks.travel_uav.dataset import load_travel_uav_index
from src.agent.config import ModelConfig, load_model_config
from src.llm import ModelRequest, create_model_client
from src.llm.providers.openai_compatible import public_response_snapshot
from tests.manual import benchmark_20_case_probe as batch
from tests.manual import flight_experience_batch_probe as frozen_source
from tests.manual.experiment_sources import source_hashes


SCHEMA_VERSION = "model_matrix_benchmark_v1"
BASE_URL = "https://cf.api.fan/v1"
MODEL_ORDER = (
    "gpt-6.1-sol",
    "gpt-5.6-terra",
    "gpt-6-sol",
    "qwen3.8-flash",
    "gpt-5.6-sol",
)
PILOT_CASE_ID = "3AAJC4I4FPQO2SQW3E7VJW64CWCZJJ"
AERIAL_CASE_IDS = (
    "39GHHAVOMOP8ZV27ILH5P0TV5F6J4A",
    "3NL0RFNU0OL3AF8P06HHQFT8T3F4KA",
    "3OWEPKL08IA118BBZH74LJUM4SMN74",
    "3GDTJDAPV39ZJ4VUXLKAM41HGS98ML",
    "3L4D84MIL8Q2B7Q2XS1OJP2Z8KEJHG",
)
TRAVEL_CASE_IDS = (
    "Carla_Town10HD__53e4e305-2f7f-46ef-ad07-554ba86b380f",
    "ModernCityMap__a2625ae8-d423-4708-a6d3-cc33606a7e6e",
    "TropicalIsland__b2665c19-3db6-45ff-be2d-795004aaeb5b",
    "Carla_Town01__5a7c4254-ad4f-4fda-a3c7-569094b61fc1",
    "NYCEnvironmentMegapa__f69693c6-bf6e-4d27-a349-6c9726e93053",
)

# These saved inputs are diagnostic fixtures only.  ``prepare`` records their
# hashes, and every later stage refuses to continue if they drift.
ADMISSION_RUN = ROOT / (
    "runs/aerial_vln_3L4D84MIL8Q2B7Q2XS1OJP2Z8KEJHG_"
    "20261001_202048_8fee1b"
)
AERIAL_COMPACTION_TRANSCRIPT = ADMISSION_RUN / "memory/agent_transcript.jsonl"
TRAVEL_COMPACTION_TRANSCRIPT = ROOT / (
    "runs/travel_uav_NYCEnvironmentMegapa__bc86986b-c79a-43a0-"
    "ae6b-6408e568589f_20260929_175502_8e1e86/"
    "memory/agent_transcript.jsonl"
)

_TERMINAL_CASE_STATUSES = frozenset({"completed", "model_error"})
_SECRET_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _redact(value: object, *, api_key: str = "") -> str:
    text = str(value)
    if api_key:
        text = text.replace(api_key, "***REDACTED***")
    return _SECRET_PATTERN.sub("sk-***REDACTED***", text)


def _require_process_api_key() -> str:
    """Require an explicit process secret instead of silently using ``.env``."""

    value = os.environ.get("FLYAGENT_MODEL_API_KEY", "").strip()
    if not value:
        raise ValueError(
            "FLYAGENT_MODEL_API_KEY must be injected into the matrix process"
        )
    return value


def _admission_source() -> Path:
    matches = sorted(ADMISSION_RUN.glob("debug/tool_loop/*/decision_0001.json"))
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one frozen admission decision under "
            f"{ADMISSION_RUN}, found {len(matches)}"
        )
    return matches[0]


def _profile(
    base: ModelConfig,
    model_name: str,
    *,
    stream: bool = False,
    structured_output_format: str = "json_schema",
    thinking_wire_format: str = "thinking_object",
) -> ModelConfig:
    """Return one explicit provider profile without changing Agent behavior."""

    return replace(
        base,
        provider="openai_compatible",
        model_name=model_name,
        base_url=BASE_URL,
        api_protocol="responses",
        stream=stream,
        thinking_mode="",
        thinking_wire_format=thinking_wire_format,
        reasoning_effort="low",
        structured_output_format=structured_output_format,
        timeout_sec=120.0,
        provider_timeout_retries=1,
        provider_connection_retries=2,
        output_language="en",
    )


def _profile_from_summary(summary: Mapping[str, object]) -> ModelConfig:
    """Rehydrate a frozen redacted profile with only the current process key."""

    return ModelConfig(
        provider=str(summary["provider"]),
        model_name=str(summary["model_name"]),
        api_key=_require_process_api_key(),
        base_url=str(summary["base_url"]),
        api_protocol=str(summary["api_protocol"]),
        stream=bool(summary["stream"]),
        thinking_mode=str(summary["thinking_mode"]),
        thinking_wire_format=str(summary["thinking_wire_format"]),
        reasoning_effort=str(summary["reasoning_effort"]),
        structured_output_format=str(summary["structured_output_format"]),
        timeout_sec=float(summary["timeout_sec"]),
        provider_timeout_retries=int(summary["provider_timeout_retries"]),
        provider_connection_retries=int(summary["provider_connection_retries"]),
        max_tool_rounds=int(summary["max_tool_rounds"]),
        max_consecutive_protocol_rejections=int(
            summary["max_consecutive_protocol_rejections"]
        ),
        action_planner_prompt_path=str(summary["action_planner_prompt_path"]),
        task_progress_reminder_interval=int(
            summary["task_progress_reminder_interval"]
        ),
        output_language=str(summary["output_language"]),  # type: ignore[arg-type]
    )


def _profile_environment(
    config: ModelConfig, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Build the exact child environment; the returned mapping is never logged."""

    env = dict(base or os.environ)
    env.update(
        {
            # Windows child stdout is captured as UTF-8 below. Force the child
            # interpreter to use the same encoding even when its parent shell
            # locale is GBK and model-authored summaries contain other scripts.
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "FLYAGENT_MODEL_PROVIDER": config.provider,
            "FLYAGENT_MODEL_NAME": config.model_name,
            "FLYAGENT_MODEL_API_KEY": config.api_key,
            "FLYAGENT_MODEL_BASE_URL": config.base_url,
            "FLYAGENT_MODEL_API_PROTOCOL": config.api_protocol,
            "FLYAGENT_MODEL_STREAM": "true" if config.stream else "false",
            "FLYAGENT_MODEL_THINKING_MODE": config.thinking_mode,
            "FLYAGENT_MODEL_THINKING_WIRE_FORMAT": config.thinking_wire_format,
            "FLYAGENT_MODEL_REASONING_EFFORT": config.reasoning_effort,
            "FLYAGENT_MODEL_STRUCTURED_OUTPUT_FORMAT": (
                config.structured_output_format
            ),
            "FLYAGENT_MODEL_TIMEOUT_SEC": str(config.timeout_sec),
            "FLYAGENT_MODEL_TIMEOUT_RETRIES": str(
                config.provider_timeout_retries
            ),
            "FLYAGENT_MODEL_CONNECTION_RETRIES": str(
                config.provider_connection_retries
            ),
            "FLYAGENT_AGENT_MAX_TOOL_ROUNDS": str(config.max_tool_rounds),
            "FLYAGENT_AGENT_MAX_CONSECUTIVE_PROTOCOL_REJECTIONS": str(
                config.max_consecutive_protocol_rejections
            ),
            "FLYAGENT_AGENT_TASK_PROGRESS_REMINDER_INTERVAL": str(
                config.task_progress_reminder_interval
            ),
            "FLYAGENT_AGENT_OUTPUT_LANGUAGE": config.output_language,
        }
    )
    if config.action_planner_prompt_path:
        env["FLYAGENT_ACTION_PLANNER_PROMPT_PATH"] = (
            config.action_planner_prompt_path
        )
    else:
        env.pop("FLYAGENT_ACTION_PLANNER_PROMPT_PATH", None)
    return env


@contextmanager
def _scoped_environment(config: ModelConfig) -> Iterator[None]:
    """Expose one model profile to existing in-process benchmark helpers."""

    changed = _profile_environment(config)
    old = dict(os.environ)
    os.environ.clear()
    os.environ.update(changed)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _source_inputs() -> dict[str, dict[str, str]]:
    inputs = {
        "admission": _admission_source(),
        "aerial_compaction": AERIAL_COMPACTION_TRANSCRIPT,
        "travel_compaction": TRAVEL_COMPACTION_TRANSCRIPT,
    }
    result: dict[str, dict[str, str]] = {}
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = {
            "path": str(path.resolve()),
            "sha256": batch.sha256(path),
        }
    return result


def _frozen_cases() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Resolve fixed IDs against current official data and verify metadata."""

    prepared = AerialVlnBenchmarkRunner(
        batch.aerial_config()
    ).prepare_evaluation(require_execution_ready=False)
    if prepared.provenance.split_sha256 != batch.AERIAL_HASH:
        raise ValueError("AerialVLN dataset SHA changed")
    assignments = prepared.selection.assignment_by_episode

    def aerial_item(case_id: str) -> dict[str, Any]:
        item = assignments.get(case_id)
        if item is None:
            raise ValueError(f"Missing frozen AerialVLN case: {case_id}")
        return {
            "benchmark": "aerial",
            "case_id": case_id,
            "difficulty": item.difficulty,
            "scene": item.scene_id,
            "reference_path_length_m": item.reference_path_length_m,
            "historical": True,
        }

    pilot = aerial_item(PILOT_CASE_ID)
    aerial = [aerial_item(case_id) for case_id in AERIAL_CASE_IDS]
    expected_aerial = ["low", "low", "medium", "medium", "high"]
    if [item["difficulty"] for item in aerial] != expected_aerial:
        raise ValueError("Frozen AerialVLN difficulty metadata changed")

    travel, provenance = load_travel_uav_index(
        batch.travel_config().index_path, split="test_seen"
    )
    if provenance.index_sha256 != batch.TRAVEL_HASH:
        raise ValueError("TravelUAV dataset SHA changed")
    by_id = {item.episode_id: item for item in travel}
    selected_travel: list[dict[str, Any]] = []
    for case_id in TRAVEL_CASE_IDS:
        item = by_id.get(case_id)
        if item is None:
            raise ValueError(f"Missing frozen TravelUAV case: {case_id}")
        selected_travel.append(
            {
                "benchmark": "travel",
                "case_id": case_id,
                "difficulty": item.difficulty,
                "scene": item.map_name,
                "reference_path_length_m": item.reference_path_length_m,
                "historical": item.difficulty == "easy",
            }
        )
    if [item["difficulty"] for item in selected_travel] != [
        "easy",
        "easy",
        "easy",
        "hard",
        "hard",
    ]:
        raise ValueError("Frozen TravelUAV difficulty metadata changed")
    cases = [*aerial, *selected_travel]
    if len(cases) != 10 or len({item["case_id"] for item in cases}) != 10:
        raise ValueError("Formal matrix must contain ten distinct cases")
    if pilot["case_id"] in {item["case_id"] for item in cases}:
        raise ValueError("The non-scoring pilot cannot enter the formal ten")
    return pilot, cases


def prepare(root: Path) -> dict[str, Any]:
    """Freeze all implementation, data, case, and provider identities once."""

    if root.exists():
        raise ValueError(f"Output exists; refusing to replace evidence: {root}")
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise ValueError("Less than 10 GiB free before matrix preparation")
    _require_process_api_key()
    pilot, cases = _frozen_cases()
    base = load_model_config(ROOT / ".env")
    profiles = {
        model: _profile(base, model).redacted_summary() for model in MODEL_ORDER
    }
    complete_hashes = frozen_source.hashes()
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now(),
        "base_url": BASE_URL,
        "model_order": list(MODEL_ORDER),
        "initial_profiles": profiles,
        "dataset_sha256": {
            "aerial": batch.AERIAL_HASH,
            "travel": batch.TRAVEL_HASH,
        },
        "pilot": pilot,
        "cases": cases,
        "source_inputs": _source_inputs(),
        "source_hashes": source_hashes(),
        "complete_source_hashes": complete_hashes,
        "git": {
            "head": _git_output("rev-parse", "HEAD"),
            "status": _git_output("status", "--short"),
            "dirty_diff_sha256": hashlib.sha256(
                subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
            ).hexdigest(),
        },
        "stage1_real_pilot_count": len(MODEL_ORDER),
        "stage2_formal_case_count": len(MODEL_ORDER) * len(cases),
    }
    root.mkdir(parents=True)
    snapshot = root / "source_snapshot"
    snapshot.mkdir()
    for name in complete_hashes:
        source = ROOT / name
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (snapshot / "git_status.txt").write_text(
        manifest["git"]["status"], encoding="utf-8"
    )
    (snapshot / "working_tree.patch").write_bytes(
        subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
    )
    batch.write_json(root / "manifest.json", manifest)
    refresh_stage1_summary(root, manifest)
    return manifest


def _guard(root: Path, manifest: Mapping[str, Any]) -> None:
    """Reject source, data, fixture, interpreter, or secret drift."""

    _require_process_api_key()
    if Path(sys.executable).resolve() != Path(
        r"F:\conda_env\flyagent\python.exe"
    ).resolve():
        raise ValueError(f"Unexpected Python interpreter: {sys.executable}")
    if manifest.get("base_url") != BASE_URL:
        raise ValueError("Frozen Provider base URL changed")
    if tuple(manifest.get("model_order") or ()) != MODEL_ORDER:
        raise ValueError("Frozen model order changed")
    if manifest.get("complete_source_hashes") != frozen_source.hashes():
        raise ValueError("Executable source changed after matrix preparation")
    if batch.sha256(Path(batch.aerial_config().dataset_root) / "val_seen.json") != (
        manifest["dataset_sha256"]["aerial"]
    ):
        raise ValueError("AerialVLN dataset changed after matrix preparation")
    if batch.sha256(Path(batch.travel_config().index_path)) != (
        manifest["dataset_sha256"]["travel"]
    ):
        raise ValueError("TravelUAV dataset changed after matrix preparation")
    for item in manifest.get("source_inputs") or {}:
        record = manifest["source_inputs"][item]
        path = Path(record["path"])
        if not path.is_file() or batch.sha256(path) != record["sha256"]:
            raise ValueError(f"Frozen diagnostic input changed: {item}")


def _run_command(
    command: Sequence[str], directory: Path, config: ModelConfig
) -> dict[str, Any]:
    """Run one diagnostic child while persisting only redacted metadata."""

    directory.mkdir(parents=True, exist_ok=False)
    batch.write_json(
        directory / "command.json",
        {
            "argv": list(command),
            "config": config.redacted_summary(),
            "started_at": _now(),
        },
    )
    started = time.monotonic()
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        env=_profile_environment(config),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    record = {
        "exit_code": completed.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "stdout": _redact(completed.stdout, api_key=config.api_key),
        "stderr": _redact(completed.stderr, api_key=config.api_key),
        "finished_at": _now(),
    }
    batch.write_json(directory / "process.json", record)
    return record


def _connectivity_probe(config: ModelConfig, output: Path) -> dict[str, Any]:
    """Use the production client for one minimal identity-checked response."""

    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record: dict[str, Any] = {
        "status": "running",
        "config": config.redacted_summary(),
        "started_at": _now(),
    }
    try:
        response = create_model_client(config.to_provider_settings()).request(
            ModelRequest(
                messages=[
                    {
                        "role": "user",
                        "content": "Reply with exactly CONNECTED and no other text.",
                    }
                ],
                parameters=config.generation_parameters(),
            )
        )
        returned_model = response.data.get("model")
        if returned_model != config.model_name:
            raise ValueError(f"Provider returned unexpected model: {returned_model}")
        choices = response.data.get("choices") or []
        message = choices[0].get("message") if choices else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Provider returned no assistant text")
        record.update(
            status="passed",
            returned_model=returned_model,
            response=public_response_snapshot(response.data),
            transport=response.transport_trace,
        )
    except Exception as exc:
        record.update(
            status="failed",
            error={
                "type": type(exc).__name__,
                "message": _redact(exc, api_key=config.api_key),
            },
        )
    record.update(elapsed_seconds=time.monotonic() - started, finished_at=_now())
    batch.write_json(output, record)
    return record


def _suggests_stream_mismatch(record: Mapping[str, Any]) -> bool:
    diagnostic: list[object] = []

    def collect(value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in {"error", "stderr"}:
                    diagnostic.append(item)
                else:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(record)
    text = json.dumps(diagnostic, ensure_ascii=False).lower()
    return any(
        marker in text
        for marker in (
            "stream",
            "event-stream",
            "sse",
            "invalid_response",
            "jsondecode",
            "no assistant text",
            "choices",
        )
    )


def _admit_transport(
    model_dir: Path, initial: ModelConfig, admission_source: Path
) -> tuple[ModelConfig | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    stream_candidates = (
        (initial.stream,)
        if initial.api_protocol == "responses"
        else (initial.stream, not initial.stream)
    )
    for attempt_index, stream in enumerate(stream_candidates, 1):
        config = replace(initial, stream=stream)
        directory = model_dir / "admission" / f"attempt_{attempt_index:02d}"
        directory.mkdir(parents=True, exist_ok=False)
        connectivity = _connectivity_probe(config, directory / "connectivity.json")
        row: dict[str, Any] = {
            "config": config.redacted_summary(),
            "connectivity": connectivity,
        }
        attempts.append(row)
        if connectivity["status"] != "passed":
            batch.write_json(directory / "attempt.json", row)
            if attempt_index == 1 and _suggests_stream_mismatch(connectivity):
                continue
            break
        process = _run_command(
            [
                sys.executable,
                str(ROOT / "tests/manual/model_admission_probe.py"),
                "--source",
                str(admission_source),
                "--output",
                str(directory / "protocol"),
                "--expected-base-url",
                BASE_URL,
                "--expected-model",
                config.model_name,
            ],
            directory / "protocol_process",
            config,
        )
        admission_path = directory / "protocol/admission.json"
        admission = _read_json(admission_path) if admission_path.exists() else {}
        row.update(process=process, admission=admission)
        batch.write_json(directory / "attempt.json", row)
        if process["exit_code"] == 0 and admission.get("status") == "passed":
            return config, attempts
        if attempt_index == 1 and _suggests_stream_mismatch(row):
            continue
        break
    return None, attempts


def _compaction_error_text(row: Mapping[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False).lower()


def _admit_compactor(
    model_dir: Path,
    transport: ModelConfig,
    source_inputs: Mapping[str, Mapping[str, str]],
) -> tuple[ModelConfig | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    candidates = [(transport.structured_output_format, transport.thinking_wire_format)]
    attempted: set[tuple[str, str]] = set()
    while candidates:
        structured, wire = candidates.pop(0)
        key = (structured, wire)
        if key in attempted:
            continue
        attempted.add(key)
        config = replace(
            transport,
            structured_output_format=structured,
            thinking_wire_format=wire,
        )
        index = len(attempts) + 1
        directory = model_dir / "compaction" / f"attempt_{index:02d}"
        report_path = directory / "report.json"
        process = _run_command(
            [
                sys.executable,
                str(ROOT / "tests/manual/history_compaction_probe.py"),
                "--sample",
                f"aerial={source_inputs['aerial_compaction']['path']}",
                "--sample",
                f"travel={source_inputs['travel_compaction']['path']}",
                "--invoke-model",
                "--output",
                str(report_path),
            ],
            directory / "process",
            config,
        )
        report = _read_json(report_path) if report_path.exists() else {}
        samples = report.get("samples") or {}
        accepted = (
            process["exit_code"] == 0
            and set(samples) == {"aerial", "travel"}
            and all(
                item.get("status") == "ready"
                and item.get("validation_accepted") is True
                for item in samples.values()
            )
        )
        row = {
            "config": config.redacted_summary(),
            "process": process,
            "report": report,
            "accepted": accepted,
        }
        attempts.append(row)
        batch.write_json(directory / "attempt.json", row)
        if accepted:
            return config, attempts
        text = _compaction_error_text(row)
        if structured == "json_schema" and any(
            marker in text
            for marker in ("response_format", "json_schema", "structured output")
        ):
            candidates.append(("json_object", wire))
        if transport.api_protocol != "responses" and wire == "thinking_object" and any(
            marker in text
            for marker in ("thinking", "enable_thinking", "unknown parameter")
        ):
            candidates.append((structured, "enable_thinking"))
    return None, attempts


def _pilot_manifest(
    model: str, config: ModelConfig, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "model": model,
        "model_config": config.redacted_summary(),
        "dataset_sha256": manifest["dataset_sha256"],
        "source_hashes": source_hashes(),
        "cases": [manifest["pilot"]],
    }


def _run_pilot(
    model_dir: Path, config: ModelConfig, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    pilot_root = model_dir / "pilot"
    pilot_root.mkdir(parents=True, exist_ok=False)
    pilot_manifest = _pilot_manifest(config.model_name, config, manifest)
    batch.write_json(pilot_root / "manifest.json", pilot_manifest)
    with _scoped_environment(config):
        batch.preflight(pilot_root, pilot_manifest)
        item = dict(manifest["pilot"])
        status = batch.run_one(pilot_root, 0, item)
        if (
            status["status"] == "infrastructure_error"
            and status.get("cleanup_verified")
        ):
            status = batch.run_one(
                pilot_root,
                0,
                item,
                directory=batch.case_dir(pilot_root, 0, item).with_name(
                    batch.case_dir(pilot_root, 0, item).name + "_recovery_01"
                ),
            )
        batch.report(pilot_root, pilot_manifest)
    return status


def _stage1_model_dir(root: Path, model: str) -> Path:
    return root / "stage1" / f"{MODEL_ORDER.index(model) + 1:02d}_{_slug(model)}"


def run_stage1_model(
    root: Path, manifest: Mapping[str, Any], model: str
) -> dict[str, Any]:
    """Run one independent model admission and non-scoring pilot."""

    if model not in MODEL_ORDER:
        raise ValueError(f"Unknown matrix model: {model}")
    _guard(root, manifest)
    model_dir = _stage1_model_dir(root, model)
    status_path = model_dir / "status.json"
    if status_path.exists():
        return _read_json(status_path)
    if model_dir.exists():
        raise ValueError(
            f"Interrupted Stage 1 attempt needs inspection: {model_dir}"
        )
    model_dir.mkdir(parents=True)
    initial = _profile_from_summary(manifest["initial_profiles"][model])
    status: dict[str, Any] = {
        "model": model,
        "status": "running",
        "started_at": _now(),
        "qualified": False,
    }
    phase = "admission"
    batch.write_json(status_path, status)
    try:
        transport, admission_attempts = _admit_transport(
            model_dir,
            initial,
            Path(manifest["source_inputs"]["admission"]["path"]),
        )
        status["admission_attempts"] = admission_attempts
        if transport is None:
            status.update(status="admission_failed", blocking_reason="transport")
            return status
        compactor, compaction_attempts = _admit_compactor(
            model_dir, transport, manifest["source_inputs"]
        )
        status["compaction_attempts"] = compaction_attempts
        if compactor is None:
            status.update(status="admission_failed", blocking_reason="compactor")
            return status
        status["resolved_profile"] = compactor.redacted_summary()
        phase = "pilot"
        pilot = _run_pilot(model_dir, compactor, manifest)
        status["pilot"] = pilot
        if not pilot.get("cleanup_verified"):
            status.update(status="cleanup_failed", blocking_reason="pilot_cleanup")
            return status
        if pilot.get("status") != "completed":
            status.update(status="pilot_failed", blocking_reason=pilot.get("status"))
            return status
        status.update(status="qualified", qualified=True)
        return status
    except Exception as exc:
        # Admission and compaction have no simulator side effects.  An
        # unexpected exception after pilot startup is different: ownership of
        # the remote process must be inspected before another model is tried.
        failed_status = "cleanup_failed" if phase == "pilot" else "infrastructure_failed"
        status.update(
            status=failed_status,
            blocking_reason=f"{phase}_orchestrator_exception",
            error={
                "type": type(exc).__name__,
                "message": _redact(exc, api_key=_require_process_api_key()),
            },
        )
        return status
    finally:
        status["finished_at"] = _now()
        batch.write_json(status_path, status)
        refresh_stage1_summary(root, manifest)


def refresh_stage1_summary(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        path = _stage1_model_dir(root, model) / "status.json"
        row = _read_json(path) if path.exists() else {
            "model": model,
            "status": "pending",
            "qualified": False,
        }
        rows.append(row)
    ready = all(row.get("qualified") is True for row in rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _now(),
        "source_digest": hashlib.sha256(
            json.dumps(
                manifest.get("complete_source_hashes") or {}, sort_keys=True
            ).encode("utf-8")
        ).hexdigest(),
        "model_order": list(MODEL_ORDER),
        "models": rows,
        "eligible_models": [
            row["model"] for row in rows if row.get("qualified") is True
        ],
        "blocking_reasons": [
            {
                "model": row["model"],
                "status": row["status"],
                "reason": row.get("blocking_reason"),
            }
            for row in rows
            if row["status"] not in {"pending", "running", "qualified"}
        ],
        "stage2_ready": ready,
        "formal_case_count": len(MODEL_ORDER) * len(manifest["cases"]),
    }
    batch.write_json(root / "stage1_summary.json", summary)
    table = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(row["model"]),
            html.escape(str(row["status"])),
            "yes" if row.get("qualified") else "no",
        )
        for row in rows
    )
    (root / "stage1_report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Model matrix Stage 1</title>"
        "<h1>Model matrix Stage 1</h1>"
        f"<p>Stage 2 ready: <strong>{str(ready).lower()}</strong></p>"
        "<table border='1'><tr><th>Model</th><th>Status</th>"
        f"<th>Qualified</th></tr>{table}</table>",
        encoding="utf-8",
    )
    return summary


def stage1(root: Path, *, only_model: str | None = None) -> dict[str, Any]:
    manifest = _read_json(root / "manifest.json")
    _guard(root, manifest)
    models = (only_model,) if only_model else MODEL_ORDER
    for model in models:
        result = run_stage1_model(root, manifest, model)
        print(
            json.dumps(
                {
                    "model": model,
                    "status": result["status"],
                    "qualified": result.get("qualified", False),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if result["status"] == "cleanup_failed":
            raise RuntimeError(f"Cleanup not verified for {model}")
    return refresh_stage1_summary(root, manifest)


def _stage2_model_root(root: Path, model: str) -> Path:
    return root / "stage2" / f"{MODEL_ORDER.index(model) + 1:02d}_{_slug(model)}"


def _next_case_directory(
    model_root: Path,
    index: int,
    item: Mapping[str, Any],
    *,
    resume: bool,
) -> Path | None:
    current = batch.latest_case_dir(model_root, index, dict(item))
    if not current.exists():
        return current
    path = current / "status.json"
    if not path.exists():
        raise ValueError(f"Unclassified existing case output: {current}")
    status = _read_json(path)
    if status["status"] in _TERMINAL_CASE_STATUSES:
        return None
    if not resume or not status.get("cleanup_verified"):
        raise ValueError(f"Case requires verified recovery: {current}")
    original = batch.case_dir(model_root, index, dict(item))
    attempts = list(original.parent.glob(original.name + "_recovery_*"))
    if attempts:
        raise ValueError("Only one infrastructure recovery is permitted per case")
    return original.with_name(original.name + "_recovery_01")


def stage2(root: Path, *, resume: bool = False) -> dict[str, Any]:
    """Run five independent, consecutive ten-case model batches."""

    manifest = _read_json(root / "manifest.json")
    _guard(root, manifest)
    admission = refresh_stage1_summary(root, manifest)
    if admission.get("stage2_ready") is not True:
        raise ValueError("all five Stage 1 models must qualify before Stage 2")
    resolved = {
        row["model"]: row["resolved_profile"] for row in admission["models"]
    }
    for model in MODEL_ORDER:
        _guard(root, manifest)
        config = _profile_from_summary(resolved[model])
        model_root = _stage2_model_root(root, model)
        model_manifest = {
            "model": model,
            "model_config": config.redacted_summary(),
            "dataset_sha256": manifest["dataset_sha256"],
            "source_hashes": source_hashes(),
            "cases": manifest["cases"],
        }
        if not model_root.exists():
            model_root.mkdir(parents=True)
            batch.write_json(model_root / "manifest.json", model_manifest)
        elif _read_json(model_root / "manifest.json") != model_manifest:
            raise ValueError(f"Frozen Stage 2 manifest changed for {model}")
        with _scoped_environment(config):
            if not (model_root / "preflight.json").exists():
                batch.preflight(model_root, model_manifest)
            for index, item in enumerate(manifest["cases"]):
                _guard(root, manifest)
                directory = _next_case_directory(
                    model_root, index, item, resume=resume
                )
                if directory is None:
                    continue
                if shutil.disk_usage(ROOT).free < 5 * 1024**3:
                    raise ValueError("Less than 5 GiB free before next formal case")
                status = batch.run_one(
                    model_root, index, dict(item), directory=directory
                )
                batch.report(model_root, model_manifest)
                if not status.get("cleanup_verified"):
                    raise RuntimeError(f"Cleanup not verified: {model}/{item['case_id']}")
                if status["status"] not in _TERMINAL_CASE_STATUSES:
                    raise RuntimeError(
                        f"Stage 2 paused: {model}/{item['case_id']} "
                        f"status={status['status']}"
                    )
        batch.report(model_root, model_manifest)
    return report(root)


def report(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "manifest.json")
    stage1_result = refresh_stage1_summary(root, manifest)
    model_rows = []
    for model in MODEL_ORDER:
        path = _stage2_model_root(root, model) / "summary.json"
        model_rows.append(
            {
                "model": model,
                "summary": _read_json(path) if path.exists() else None,
            }
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _now(),
        "stage1": stage1_result,
        "stage2_models": model_rows,
        "formal_case_count": sum(
            len((row["summary"] or {}).get("cases") or []) for row in model_rows
        ),
    }
    batch.write_json(root / "comparison.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "stage1", "stage2", "report"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_ORDER)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "prepare":
        result = prepare(root)
        output = {
            "manifest": str(root / "manifest.json"),
            "models": result["model_order"],
            "stage1_pilots": result["stage1_real_pilot_count"],
            "stage2_cases": result["stage2_formal_case_count"],
        }
    elif args.command == "stage1":
        output = stage1(root, only_model=args.model)
    elif args.command == "stage2":
        if args.model:
            raise ValueError("Stage 2 always preserves the full frozen model order")
        output = stage2(root, resume=args.resume)
    else:
        output = report(root)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
