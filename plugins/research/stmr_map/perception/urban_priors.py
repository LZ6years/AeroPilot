"""Stable landmark query priors for urban UAV STMR experiments.

Priors are detector queries, not observations. A query from this module only
asks the perception stack to look for a concept; it must never be interpreted
as evidence that the concept exists in the scene.
"""

from __future__ import annotations

from collections.abc import Iterable
import re

from .landmark_extractor import Landmark


DEFAULT_UAV_PRIOR_NAME = "default_uav_prior"
URBAN_UAV_PRIOR_V1_NAME = "urban_uav_prior_v1"
MAX_LANDMARK_QUERIES = 16

default_uav_prior: tuple[str, ...] = (
    "road",
    "building",
    "tree",
    "vehicle",
    "person",
)

urban_uav_prior_v1: tuple[str, ...] = (
    "building",
    "rooftop",
    "entrance",
    "road",
    "sidewalk",
    "traffic light",
    "traffic sign",
    "pole",
    "tree",
    "vehicle",
    "person",
    "facade",
    "tower",
    "bridge",
)

PRIOR_SETS: dict[str, tuple[str, ...]] = {
    DEFAULT_UAV_PRIOR_NAME: default_uav_prior,
    URBAN_UAV_PRIOR_V1_NAME: urban_uav_prior_v1,
}

_ALIASES: dict[str, str] = {
    "automobile": "vehicle",
    "bus": "vehicle",
    "car": "vehicle",
    "lorry": "vehicle",
    "truck": "vehicle",
    "van": "vehicle",
    "doorway": "entrance",
    "entry": "entrance",
    "entryway": "entrance",
    "roof": "rooftop",
    "roof top": "rooftop",
    "road sign": "traffic sign",
    "signboard": "traffic sign",
    "stop light": "traffic light",
    "street light": "traffic light",
    "traffic signal": "traffic light",
    "vegetation": "tree",
}


def normalize_landmark_query(value: str) -> str:
    """Normalize one detector query and collapse exact semantic aliases."""

    text = value.lower().replace("_", " ").replace("-", " ").replace("/", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    normalized = re.sub(r"\s+", " ", text).strip()
    return _ALIASES.get(normalized, normalized)


def prior_queries(prior: str | Iterable[str]) -> tuple[str, ...]:
    """Resolve a named prior or normalize an explicit prior sequence."""

    if isinstance(prior, str):
        try:
            return PRIOR_SETS[prior]
        except KeyError as exc:
            choices = ", ".join(sorted(PRIOR_SETS))
            raise ValueError(f"Unknown STMR prior {prior!r}; expected one of: {choices}") from exc
    return tuple(normalize_landmark_query(item) for item in prior if normalize_landmark_query(item))


def assemble_landmark_queries(
    task_landmarks: Iterable[Landmark | str],
    *,
    prior: str | Iterable[str] = DEFAULT_UAV_PRIOR_NAME,
    max_queries: int = MAX_LANDMARK_QUERIES,
) -> list[Landmark]:
    """Put task landmarks first, then fill remaining slots from a fixed prior."""

    if max_queries <= 0:
        return []
    prior_name = prior if isinstance(prior, str) else "custom_uav_prior"
    assembled: list[Landmark] = []
    seen: set[str] = set()

    def append(item: Landmark | str, source: str) -> None:
        if len(assembled) >= max_queries:
            return
        raw_label = item.label if isinstance(item, Landmark) else str(item)
        label = normalize_landmark_query(raw_label)
        if not label or label in seen:
            return
        seen.add(label)
        if isinstance(item, Landmark):
            assembled.append(
                Landmark(
                    label=label,
                    source=item.source,
                    matched_text=item.matched_text,
                    reason=item.reason,
                )
            )
        else:
            assembled.append(
                Landmark(
                    label=label,
                    source=source,
                    matched_text=raw_label,
                    reason="fixed detector query prior; not scene evidence",
                )
            )

    for item in task_landmarks:
        append(item, "task_landmark")
    for query in prior_queries(prior):
        append(query, str(prior_name))
    return assembled


__all__ = [
    "DEFAULT_UAV_PRIOR_NAME",
    "MAX_LANDMARK_QUERIES",
    "PRIOR_SETS",
    "URBAN_UAV_PRIOR_V1_NAME",
    "assemble_landmark_queries",
    "default_uav_prior",
    "normalize_landmark_query",
    "prior_queries",
    "urban_uav_prior_v1",
]
