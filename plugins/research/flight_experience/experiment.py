"""Public experiment settings, provenance validation and compatible HTML reports.

V2 execution lives in generation.py; only independently approved original-evidence
experiences may be published to an isolated library. No flight/runtime hook exists.
"""
from __future__ import annotations

from dataclasses import dataclass
import html
import json
from pathlib import Path
import time
from typing import Any

from src.llm import ModelClient, ModelRequest
from src.llm.audit import snapshot_request

from .evidence import EvidenceBundle


@dataclass(frozen=True)
class ExperimentSettings:
    """Image budget per independent reader; prices are optional user-supplied USD.

    Token prices are per million tokens. Unset prices yield unknown cost, never
    a fabricated zero. Human correction counts remain pending for manual review.
    """
    max_images: int = 8
    max_candidates: int = 3
    checkpoint_directory: Path | None = None
    library_path: Path | None = None
    reflection_model: str = "caller-configured"
    review_model: str = "caller-configured"
    reflection_input_price: float | None = None
    reflection_output_price: float | None = None
    review_input_price: float | None = None
    review_output_price: float | None = None
    observation_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.max_images <= 8 or not 1 <= self.max_candidates <= 3:
            raise ValueError("Image budget must be 1–8 and candidate budget 1–3")
        if self.observation_ids is not None and (
            not self.observation_ids or len(self.observation_ids) > self.max_images
            or any(not isinstance(obs, str) or not obs for obs in self.observation_ids)
            or len(set(self.observation_ids)) != len(self.observation_ids)
        ):
            raise ValueError("Explicit observation IDs must be unique and fit the image budget")
        for value in (self.reflection_input_price, self.reflection_output_price,
                      self.review_input_price, self.review_output_price):
            if value is not None and value < 0:
                raise ValueError("Token prices cannot be negative")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_output(bundle: EvidenceBundle, output: Path) -> Path:
    """Refuse overwrite or source overlap, including symlink-resolved paths."""
    output = output.resolve()
    if output.is_relative_to(bundle.source) or bundle.source.is_relative_to(output):
        raise ValueError("Experiment output must be separate from the source run")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "evidence_index.json", bundle.index())
    return output


def _validate_candidate(value: Any, bundle: EvidenceBundle, images: set[str],
                        supplied_refs: set[str]) -> list[str]:
    """Return machine-checkable reasons to defer, not an automated semantic verdict."""
    if not isinstance(value, dict):
        return ["candidate_not_object"]
    issues = []
    for key in ("id", "applicability", "observed_evidence", "recommendation", "limitations"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            issues.append(f"missing_{key}")
    refs, obs = value.get("evidence_refs"), value.get("observation_ids")
    if not isinstance(refs, list) or not refs or not all(isinstance(x, str) for x in refs):
        issues.append("invalid_evidence_refs")
        refs = []
    if not isinstance(obs, list) or not obs or not all(isinstance(x, str) for x in obs):
        issues.append("missing_observation_ids")
        obs = []
    if set(refs) - bundle.evidence_ids:
        issues.append("unknown_evidence_reference")
    if set(refs) - supplied_refs:
        issues.append("full_evidence_not_supplied")
    if set(obs) - set(refs):
        issues.append("image_not_cited")
    if set(obs) - images:
        issues.append("image_not_seen_by_this_reader")
    if set(obs) - set(bundle.images):
        issues.append("missing_original_image")
    if not (set(refs) & {r["id"] for r in bundle.timeline}):
        issues.append("missing_timeline_evidence")
    return issues


def run_experiment(bundle: EvidenceBundle, output: Path, *, reflection: ModelClient,
                   reviewer: ModelClient, settings: ExperimentSettings | None = None,
                   parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the V2 evidence-backed pipeline; old artifact readers remain compatible."""
    from .generation import run_generation
    return run_generation(bundle, output, reflection=reflection, reviewer=reviewer,
        settings=settings or ExperimentSettings(), parameters=parameters)


def _write_report(output: Path, bundle: EvidenceBundle, result: dict[str, Any],
                  image_links: dict[str, str]) -> None:
    """Plain UTF-8 docs and escaped HTML are inspection artifacts, never knowledge."""
    lines = ["# Offline candidate experience", "", "Experimental only; no verified flight improvement.", ""]
    catalog = []
    for index, candidate in enumerate(result["accepted"], 1):
        name = f"experience_{index:03d}"
        description = (f"Applicable needs: {candidate['applicability']}; "
            f"Decision questions: {candidate['recommendation']}; Contents: {candidate['observed_evidence']}; "
            f"Limits: {candidate['limitations']}")
        catalog.append({"name": name, "description": description, "evidence_refs": candidate["evidence_refs"]})
        lines.extend([f"## {name}", "", f"Applicable conditions: {candidate['applicability']}", "",
            f"Observed evidence: {candidate['observed_evidence']}", "",
            f"Advice: {candidate['recommendation']}", "", f"Limits: {candidate['limitations']}", "",
            "Sources: " + ", ".join(candidate["evidence_refs"]), ""])
    if not catalog:
        lines.append("No accepted experience. Inspect experiment.json for deferrals or failure.")
    (output / "experience.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "catalog.json", catalog)
    # Exclude long requests from the reading report; exact snapshots remain in JSON.
    report = {k: v for k, v in result.items() if k != "calls"}
    report["calls"] = [{k: c.get(k) for k in ("stage", "returned_model", "status", "elapsed_seconds",
        "usage", "estimated_cost_usd", "image_ids", "http_status")} for c in result["calls"]]
    body = '<!doctype html><meta charset="utf-8"><title>Offline flight experience</title>'
    body += '<h1>Offline experiment — no flight benefit established</h1><p>' + html.escape(bundle.instruction) + '</p>'
    body += '<pre style="white-space:pre-wrap">' + html.escape(json.dumps(report, ensure_ascii=False, indent=2)) + '</pre>'
    for obs, relative in image_links.items():
        body += f'<figure><img src="{relative}"><figcaption>{html.escape(obs)}</figcaption></figure>'
    (output / "report.html").write_text(body, encoding="utf-8")
