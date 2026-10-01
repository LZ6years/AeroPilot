"""Non-blocking orchestration for bounded history projection and compaction.

The manager runs only on the planner thread. It selects boundaries, validates
completed model output, commits task-local state, and renders the final request.
The coordinator's daemon worker performs only the stateless provider call.
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

from src.agent.context.compression.contracts import (
    CompressedHistorySection,
    CompressionCandidate,
    CompressionRequest,
    HistoryCompactor,
    HistoryCompressionConfig,
    HistoryProjectionState,
)
from src.agent.context.compression.coordinator import (
    CompactionJobSnapshot,
    HistoryCompactionCoordinator,
)
from src.agent.context.compression.validator import CompressionValidator
from src.agent.context.projection import HistoryProjector, ProjectionPlan, ProjectionResult

if TYPE_CHECKING:
    from src.agent.transcript import AgentTranscript
    from src.tools.registry import ToolRegistry


class ContextWindowManager:
    """Coordinate one pure projector with an optional semantic compactor."""

    def __init__(
        self,
        projector: HistoryProjector,
        config: HistoryCompressionConfig,
        *,
        compactor: HistoryCompactor | None = None,
    ) -> None:
        self.projector = projector
        self.config = config
        self.compactor = compactor
        self.validator = CompressionValidator(config.summary_max_tokens)

    def project(
        self,
        transcript: "AgentTranscript",
        registry: "ToolRegistry",
        *,
        state: HistoryProjectionState | None,
        coordinator: HistoryCompactionCoordinator | None = None,
        fixed_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        parameters: dict[str, Any],
        current_task_state: dict[str, Any] | None = None,
    ) -> ProjectionResult:
        """Return immediately with a full, staged, or compacted request view."""

        task_state = copy.deepcopy(current_task_state or {})
        # Standalone callers intentionally remain stateless and preserve full
        # history. Production supplies both objects from its TaskSession.
        if state is None or not self.config.enabled:
            return self.projector.project(transcript, registry)

        transition: dict[str, Any] = {}
        if coordinator is not None:
            transition = self._poll_completed_job(state, coordinator)

        result = self._current_projection(transcript, registry, state)
        estimate_before = self._estimate(
            fixed_messages, result.messages, tools, parameters, state
        )
        state.last_estimated_tokens = estimate_before
        self._add_budget_audit(result, estimate_before, coordinator, transition)
        if (
            coordinator is None
            or self.compactor is None
            or estimate_before < self.config.prefetch_trigger_tokens
        ):
            if coordinator is None and estimate_before >= self.config.prefetch_trigger_tokens:
                result.audit["compaction_status"] = "coordinator_unavailable"
            return result

        snapshot = coordinator.poll()
        candidate: CompressionCandidate | None = None
        preparation_status = "existing_job"
        # A failed job may be replaced only after the eligible boundary changes;
        # submit() owns the per-digest retry prohibition.
        if snapshot.status in {"idle", "failed"}:
            candidate, preparation_status = self._prepare_candidate(
                transcript, registry, state, task_state
            )
            if candidate is not None and (
                candidate.estimated_input_tokens
                <= self.config.max_compactor_input_tokens
            ):
                if coordinator.submit(candidate, self.compactor):
                    result.audit["compaction_status"] = "prefetch_started"
                    result.audit["prefetch_candidate"] = self._candidate_audit(candidate)
                    snapshot = coordinator.poll()
                else:
                    snapshot = coordinator.poll()
            elif candidate is not None:
                result.audit["compaction_status"] = "compactor_input_oversized"
                result.audit["prefetch_candidate"] = self._candidate_audit(candidate)

        # Prefetch never changes the request in the turn that starts it. This is
        # what keeps 48k work fully overlapped with the ordinary planner call.
        if estimate_before < self.config.soft_trigger_tokens:
            self._add_job_audit(result, snapshot)
            if preparation_status != "existing_job":
                result.audit["candidate_preparation_status"] = preparation_status
            return result

        snapshot = coordinator.poll()
        activation_candidate = candidate
        if (
            activation_candidate is None
            and snapshot.candidate is not None
            and self._candidate_matches_unactivated(state, snapshot.candidate)
        ):
            activation_candidate = snapshot.candidate
        if activation_candidate is None:
            activation_candidate, preparation_status = self._prepare_candidate(
                transcript, registry, state, task_state
            )
        if activation_candidate is None:
            result.audit.setdefault("compaction_status", preparation_status)
            self._add_job_audit(result, snapshot)
            return result

        errors: tuple[str, ...] = snapshot.errors
        draft = None
        status = snapshot.status
        if (
            status == "ready"
            and snapshot.result is not None
            and self._candidate_matches_unactivated(state, activation_candidate)
        ):
            validation = self.validator.validate(
                snapshot.result.draft, activation_candidate.request
            )
            if validation.accepted and validation.normalized_draft is not None:
                draft = validation.normalized_draft
                status = "accepted"
            else:
                errors = validation.errors
                coordinator.reject(
                    activation_candidate.request.source_digest, validation.errors
                )
                status = "prefetch_invalid"
        if draft is None:
            # An unvalidated summary can never justify removing raw history.
            # Keep the current accepted section (if any) and its complete tail;
            # the next planner turn may accept a completed background result.
            if status == "pending":
                status = "prefetch_pending"
            elif status == "failed":
                status = "prefetch_failed"
            elif status == "idle" and preparation_status == "single_window_oversized":
                status = "blocked_oversized_window"
            elif status == "ready":
                status = "prefetch_invalid"
                errors = errors or ("compaction_result_missing",)
            result.audit.update(
                {
                    "compaction_status": status,
                    "compaction_errors": list(errors),
                    "target_unreachable": estimate_before > self.config.target_tokens,
                    "staged_candidate": self._candidate_audit(activation_candidate),
                }
            )
            self._add_job_audit(result, coordinator.poll())
            return result

        self._activate(state, activation_candidate, draft)
        coordinator.consume(activation_candidate.request.source_digest)

        compacted = self._current_projection(transcript, registry, state)
        estimate_after = self._estimate(
            fixed_messages, compacted.messages, tools, parameters, state
        )
        state.last_estimated_tokens = estimate_after
        compacted.audit.update(
            {
                "estimated_input_tokens_before_compaction": estimate_before,
                "estimated_input_tokens": estimate_after,
                "prefetch_trigger_tokens": self.config.prefetch_trigger_tokens,
                "soft_trigger_tokens": self.config.soft_trigger_tokens,
                "target_tokens": self.config.target_tokens,
                "max_compactor_input_tokens": self.config.max_compactor_input_tokens,
                "compactor_timeout_sec": self.config.compactor_timeout_sec,
                "compactor_timeout_retries": self.config.compactor_timeout_retries,
                "compactor_connection_retries": (
                    self.config.compactor_connection_retries
                ),
                "compaction_status": "accepted",
                "compaction_errors": list(errors),
                "target_unreachable": estimate_after > self.config.target_tokens,
                "activated_candidate": self._candidate_audit(activation_candidate),
            }
        )
        self._add_job_audit(compacted, coordinator.poll())
        return compacted

    def observe_usage(
        self,
        state: HistoryProjectionState | None,
        usage: dict[str, Any] | None,
    ) -> None:
        """Calibrate a provider-independent estimate from reported input usage."""

        if state is None or not isinstance(usage, dict):
            return
        value = usage.get("prompt_tokens", usage.get("input_tokens"))
        if type(value) is not int or value < 1 or state.last_estimated_tokens < 1:
            return
        state.last_actual_prompt_tokens = value
        error_ratio = value / state.last_estimated_tokens
        updated = state.calibration_ratio * (0.8 + 0.2 * error_ratio)
        state.calibration_ratio = min(3.0, max(0.5, updated))

    def _poll_completed_job(
        self,
        state: HistoryProjectionState,
        coordinator: HistoryCompactionCoordinator,
    ) -> dict[str, Any]:
        snapshot = coordinator.poll()
        candidate = snapshot.candidate
        if snapshot.status != "ready" or candidate is None or snapshot.result is None:
            return {}
        if candidate.base_generation != state.capsule_generation:
            coordinator.consume(candidate.request.source_digest)
            return {"compaction_status": "stale_result_discarded"}
        # A valid ready result at the current generation remains staged until
        # the main request reaches the 64k activation threshold.
        return {"compaction_status": "prefetch_ready"}

    def _prepare_candidate(
        self,
        transcript: "AgentTranscript",
        registry: "ToolRegistry",
        state: HistoryProjectionState,
        current_task_state: dict[str, Any],
    ) -> tuple[CompressionCandidate | None, str]:
        base = max(2, state.compacted_through_event_index)
        new_events = transcript.snapshot_range(base)
        initial = self.projector.analyze(
            new_events,
            base_event_index=base,
            keep_recent_decision_windows=self.config.keep_recent_decision_windows,
            registry=registry,
            current_task_state=current_task_state,
        )
        if initial.compactable_end_event_index is None:
            return None, "no_complete_prefix"

        # Full history is copied only on prefetch/activation turns. It supplies
        # exact facts and knowledge independently from prior model summaries.
        full_events = transcript.snapshot()
        plan = self.projector.analyze(
            new_events,
            base_event_index=base,
            keep_recent_decision_windows=self.config.keep_recent_decision_windows,
            all_events_for_evidence=full_events,
            registry=registry,
            current_task_state=current_task_state,
        )
        if plan.source_digest is None or plan.source_digest == state.source_digest:
            return None, "unchanged_boundary"
        candidate = self._candidate_from_plan(plan, state)
        if candidate.estimated_input_tokens <= self.config.max_compactor_input_tokens:
            return candidate, "all_eligible_windows"

        # Find the largest oldest contiguous prefix that fits. Each trial ends
        # only at a complete DecisionWindow, so assistant/tool pairing is never
        # split even when one pathological window exceeds the whole budget.
        low, high = 1, len(plan.windows)
        best: CompressionCandidate | None = None
        while low <= high:
            count = (low + high) // 2
            limited_plan = self.projector.analyze(
                new_events,
                base_event_index=base,
                keep_recent_decision_windows=self.config.keep_recent_decision_windows,
                all_events_for_evidence=full_events,
                max_compactable_windows=count,
                registry=registry,
                current_task_state=current_task_state,
            )
            trial = self._candidate_from_plan(limited_plan, state)
            if trial.estimated_input_tokens <= self.config.max_compactor_input_tokens:
                best = trial
                low = count + 1
            else:
                high = count - 1
        if best is not None:
            return best, "limited_by_compactor_input_budget"
        # Keep the one-window candidate only to audit the blocking boundary.
        # It is never submitted, and raw history remains visible because an
        # oversized window cannot be replaced by an unvalidated summary.
        one = self.projector.analyze(
            new_events,
            base_event_index=base,
            keep_recent_decision_windows=self.config.keep_recent_decision_windows,
            all_events_for_evidence=full_events,
            max_compactable_windows=1,
            registry=registry,
            current_task_state=current_task_state,
        )
        return self._candidate_from_plan(one, state), "single_window_oversized"

    def _candidate_from_plan(
        self, plan: ProjectionPlan, state: HistoryProjectionState
    ) -> CompressionCandidate:
        if plan.compactable_end_event_index is None or plan.source_digest is None:
            raise ValueError("cannot build a candidate without a complete boundary")
        request = CompressionRequest(
            source_digest=plan.source_digest,
            instruction=plan.instruction,
            windows=plan.windows,
            evidence_catalog=plan.evidence_catalog,
            current_state_hint=plan.current_state_hint,
            deterministic_facts=plan.deterministic_facts,
            previous_summary=(state.section.summary if state.section else None),
        )
        estimator = getattr(self.compactor, "estimate_input_tokens", None)
        if callable(estimator):
            estimate = int(estimator(request))
        else:
            estimate = max(
                1,
                round(
                    len(
                        json.dumps(
                            {
                                "instruction": request.instruction,
                                "windows": [item.to_dict() for item in request.windows],
                                "evidence_catalog": [
                                    {
                                        "alias": item.alias,
                                        "canonical_ref": item.canonical_ref,
                                        "kind": item.kind,
                                    }
                                    for item in request.evidence_catalog
                                ],
                                "current": request.current_state_hint,
                                "facts": request.deterministic_facts,
                                "previous": (
                                    request.previous_summary.to_dict()
                                    if request.previous_summary
                                    else None
                                ),
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                    / 3.5
                ),
            )
        return CompressionCandidate(
            request=request,
            through_event_index=plan.compactable_end_event_index,
            base_generation=state.capsule_generation,
            deterministic_facts=plan.deterministic_facts,
            exact_loaded_knowledge=plan.exact_loaded_knowledge,
            prefix_media=plan.prefix_media,
            estimated_input_tokens=estimate,
        )

    @staticmethod
    def _candidate_matches_unactivated(
        state: HistoryProjectionState, candidate: CompressionCandidate
    ) -> bool:
        return candidate.base_generation == state.capsule_generation

    @staticmethod
    def _activate(
        state: HistoryProjectionState,
        candidate: CompressionCandidate,
        draft,
    ) -> None:
        section = CompressedHistorySection(
            generation=candidate.target_generation,
            through_event_index=candidate.through_event_index,
            source_digest=candidate.request.source_digest,
            summary=draft,
            deterministic_facts=candidate.deterministic_facts,
            exact_loaded_knowledge=candidate.exact_loaded_knowledge,
            prefix_media=candidate.prefix_media,
        )
        state.section = section
        state.compacted_through_event_index = section.through_event_index
        state.capsule_generation = section.generation
        state.source_digest = section.source_digest

    def _add_budget_audit(
        self,
        result: ProjectionResult,
        estimate: int,
        coordinator: HistoryCompactionCoordinator | None,
        transition: dict[str, Any],
    ) -> None:
        result.audit.update(
            {
                "estimated_input_tokens": estimate,
                "prefetch_trigger_tokens": self.config.prefetch_trigger_tokens,
                "soft_trigger_tokens": self.config.soft_trigger_tokens,
                "target_tokens": self.config.target_tokens,
                "max_compactor_input_tokens": self.config.max_compactor_input_tokens,
                "compactor_timeout_sec": self.config.compactor_timeout_sec,
                "compactor_timeout_retries": self.config.compactor_timeout_retries,
                "compactor_connection_retries": (
                    self.config.compactor_connection_retries
                ),
                **transition,
            }
        )
        if coordinator is not None:
            self._add_job_audit(result, coordinator.poll())

    @staticmethod
    def _candidate_audit(candidate: CompressionCandidate) -> dict[str, Any]:
        return {
            "source_digest": candidate.request.source_digest,
            "through_event_index": candidate.through_event_index,
            "base_generation": candidate.base_generation,
            "target_generation": candidate.target_generation,
            "decision_window_count": len(candidate.request.windows),
            "estimated_input_tokens": candidate.estimated_input_tokens,
            "compactor_image_count": 0,
            "indexed_prefix_media_count": len(candidate.prefix_media),
        }

    def _add_job_audit(
        self, result: ProjectionResult, snapshot: CompactionJobSnapshot
    ) -> None:
        job: dict[str, Any] = {"status": snapshot.status}
        if snapshot.candidate is not None:
            job.update(self._candidate_audit(snapshot.candidate))
        if snapshot.errors:
            job["errors"] = list(snapshot.errors)
        if snapshot.started_at is not None and snapshot.finished_at is not None:
            job["elapsed_seconds"] = max(
                0.0, snapshot.finished_at - snapshot.started_at
            )
        if snapshot.result is not None:
            job.update(
                {
                    "elapsed_seconds": snapshot.result.elapsed_seconds,
                    "finish_reason": snapshot.result.finish_reason,
                    "usage": copy.deepcopy(snapshot.result.usage),
                    "transport_trace": copy.deepcopy(snapshot.result.transport_trace),
                }
            )
        result.audit["background_compaction"] = job

    def _current_projection(
        self,
        transcript: "AgentTranscript",
        registry: "ToolRegistry",
        state: HistoryProjectionState,
    ) -> ProjectionResult:
        if state.section is None:
            return self.projector.project(transcript, registry)
        prelude = transcript.snapshot_range(0, 2)
        tail = transcript.snapshot_range(state.compacted_through_event_index)
        return self.projector.project_compacted(prelude, tail, state.section, registry)

    @staticmethod
    def _estimate(
        fixed_messages: list[dict[str, Any]],
        history_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        parameters: dict[str, Any],
        state: HistoryProjectionState,
    ) -> int:
        image_count = 0

        def binary_free(value: Any) -> Any:
            nonlocal image_count
            if isinstance(value, str) and value.startswith("data:image/"):
                image_count += 1
                return "<image>"
            if isinstance(value, dict):
                return {key: binary_free(item) for key, item in value.items()}
            if isinstance(value, list):
                return [binary_free(item) for item in value]
            return value

        payload = binary_free(
            {
                "messages": [*copy.deepcopy(fixed_messages), *copy.deepcopy(history_messages)],
                "tools": copy.deepcopy(tools),
                "parameters": copy.deepcopy(parameters),
            }
        )
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        raw = len(encoded) / 3.5 + image_count * 1_024
        return max(1, round(raw * state.calibration_ratio))
