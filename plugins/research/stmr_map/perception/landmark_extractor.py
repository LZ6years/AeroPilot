"""Deterministic landmark extraction for STMR open-vocabulary probes.

The STMR paper uses an LLM landmark extractor before GroundingDINO. For the
standalone research probe we keep that boundary explicit but deterministic, so
the perception stack can be tested offline and manually overridden with
``--landmarks`` when a benchmark instruction needs a specific phrase.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Landmark:
    """A text concept that should be grounded in the RGB observation."""

    label: str
    source: str
    matched_text: str | None = None
    reason: str | None = None


_LANDMARK_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("traffic light", ("traffic light", "traffic signal", "stop light")),
    ("traffic sign", ("traffic sign", "road sign", "signboard", "sign")),
    ("crosswalk", ("crosswalk", "zebra crossing")),
    ("sidewalk", ("sidewalk", "pavement", "walkway")),
    ("road", ("road", "street", "lane", "route", "path")),
    ("building", ("building", "house", "skyscraper", "facade", "wall")),
    ("rooftop", ("rooftop", "roof top", "roof")),
    ("tower", ("tower",)),
    ("bridge", ("bridge", "overpass")),
    ("sunshade", ("sunshade", "umbrella", "parasol")),
    ("river", ("river", "water", "lake", "pool")),
    ("tree", ("tree", "vegetation", "plant", "grass", "bush")),
    ("person", ("person", "people", "pedestrian", "human", "man", "woman")),
    ("car", ("car", "vehicle", "truck", "bus", "van")),
    ("bicycle", ("bicycle", "bike", "motorcycle", "scooter")),
    ("fence", ("fence", "railing")),
    ("pole", ("pole", "lamp", "streetlight")),
)


def extract_landmarks(
    instruction: str | None,
    *,
    manual_landmarks: list[str] | tuple[str, ...] | None = None,
    include_defaults: bool = True,
) -> list[Landmark]:
    """Return stable open-vocabulary landmark prompts for an instruction."""

    landmarks: list[Landmark] = []
    seen: set[str] = set()
    for raw in manual_landmarks or ():
        label = _normalize_phrase(raw)
        if label and label not in seen:
            seen.add(label)
            landmarks.append(Landmark(label=label, source="manual_override", matched_text=raw))

    normalized_instruction = _normalize_text(instruction or "")
    for label, aliases in _LANDMARK_PATTERNS:
        for alias in aliases:
            if _contains_phrase(normalized_instruction, alias):
                if label not in seen:
                    seen.add(label)
                    landmarks.append(Landmark(label=label, source="instruction_keyword", matched_text=alias))
                break

    if include_defaults:
        for label in ("road", "building", "tree", "car", "person"):
            if label not in seen:
                seen.add(label)
                landmarks.append(Landmark(label=label, source="default_uav_prior"))
    return landmarks


def _contains_phrase(text: str, phrase: str) -> bool:
    return f" {_normalize_phrase(phrase)} " in f" {text} "


def _normalize_phrase(value: str) -> str:
    return _normalize_text(value).strip()


def _normalize_text(value: str) -> str:
    text = value.lower().replace("_", " ").replace("-", " ").replace("/", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
