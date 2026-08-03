"""Simulator-neutral benchmark execution, audit, and reporting contracts."""

from evaluation.benchmarks.core.audit import (
    BenchmarkAuditLog,
    BenchmarkAuditReader,
    EpisodeAuditCursor,
    FrameAssetResolver,
    ResolvedFrameAsset,
    write_audit_json,
)
from evaluation.benchmarks.core.coordinates import (
    ned_to_canonical_position,
    quaternion_wxyz_to_euler_deg,
)
from evaluation.benchmarks.core.executor import (
    BenchmarkEpisodeExecutor,
    BenchmarkEpisodeResult,
    BenchmarkExecutionOutcome,
)
from evaluation.benchmarks.core.models import PreparedBenchmarkEpisode
from evaluation.benchmarks.core.protocols import (
    BenchmarkCaseAdapter,
    BenchmarkMetricSuite,
    BenchmarkScenarioController,
    NoopScenarioController,
)
from evaluation.benchmarks.core.review import (
    BenchmarkCaseReview,
    BenchmarkReviewRenderer,
    BenchmarkSummaryReview,
    ReviewSection,
)
from evaluation.benchmarks.core.trace import (
    EpisodeTrace,
    EpisodeTraceReader,
    EpisodeTraceSample,
    TraceCursor,
)

__all__ = [
    "BenchmarkCaseAdapter",
    "BenchmarkAuditLog",
    "BenchmarkAuditReader",
    "BenchmarkCaseReview",
    "BenchmarkEpisodeExecutor",
    "BenchmarkEpisodeResult",
    "BenchmarkExecutionOutcome",
    "BenchmarkMetricSuite",
    "BenchmarkScenarioController",
    "BenchmarkReviewRenderer",
    "BenchmarkSummaryReview",
    "EpisodeAuditCursor",
    "EpisodeTrace",
    "EpisodeTraceReader",
    "EpisodeTraceSample",
    "NoopScenarioController",
    "PreparedBenchmarkEpisode",
    "FrameAssetResolver",
    "ResolvedFrameAsset",
    "ReviewSection",
    "TraceCursor",
    "ned_to_canonical_position",
    "quaternion_wxyz_to_euler_deg",
    "write_audit_json",
]
