"""Small TF-IDF style text filter for STMR open-vocabulary masks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Protocol

from .landmark_extractor import Landmark


class PhraseDetection(Protocol):
    phrase: str
    score: float


@dataclass(frozen=True)
class TfidfMatch:
    """A detection kept or rejected by landmark-phrase text similarity."""

    detection_index: int
    phrase: str
    landmark: str
    similarity: float
    keep: bool
    score: float


def filter_detections_by_tfidf(
    detections: list[PhraseDetection],
    landmarks: list[Landmark],
    *,
    threshold: float = 0.8,
) -> list[TfidfMatch]:
    """Match each detection phrase against the closest landmark phrase."""

    matches: list[TfidfMatch] = []
    landmark_labels = [item.label for item in landmarks]
    for index, detection in enumerate(detections):
        best_label = ""
        best_score = 0.0
        for label in landmark_labels:
            score = tfidf_similarity(detection.phrase, label, corpus=[*landmark_labels, detection.phrase])
            if score > best_score:
                best_score = score
                best_label = label
        matches.append(
            TfidfMatch(
                detection_index=index,
                phrase=detection.phrase,
                landmark=best_label,
                similarity=round(best_score, 4),
                keep=best_score >= threshold,
                score=float(detection.score),
            )
        )
    return matches


def tfidf_similarity(left: str, right: str, *, corpus: list[str] | None = None) -> float:
    """Compute cosine similarity over a minimal TF-IDF token vector."""

    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    if left_tokens == right_tokens:
        return 1.0
    docs = [_tokens(item) for item in (corpus or [left, right])]
    vocab = sorted(set(left_tokens) | set(right_tokens))
    left_vec = _tfidf_vector(left_tokens, vocab, docs)
    right_vec = _tfidf_vector(right_tokens, vocab, docs)
    numerator = sum(left_vec[token] * right_vec[token] for token in vocab)
    left_norm = math.sqrt(sum(value * value for value in left_vec.values()))
    right_norm = math.sqrt(sum(value * value for value in right_vec.values()))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _tfidf_vector(tokens: list[str], vocab: list[str], docs: list[list[str]]) -> dict[str, float]:
    counts = Counter(tokens)
    total = float(sum(counts.values()) or 1)
    doc_count = float(len(docs) or 1)
    vector = {}
    for token in vocab:
        tf = float(counts.get(token, 0)) / total
        df = sum(1 for doc in docs if token in doc)
        idf = math.log((doc_count + 1.0) / (float(df) + 1.0)) + 1.0
        vector[token] = tf * idf
    return vector


def _tokens(value: str) -> list[str]:
    normalized = value.lower().replace("_", " ").replace("-", " ").replace("/", " ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return [token for token in normalized.split() if token]
