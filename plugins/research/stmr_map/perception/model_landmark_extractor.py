"""LLM-backed landmark extraction for standalone STMR probes.

This module is deliberately scoped to the research STMR manual probe. It uses
the same OpenAI-compatible model configuration as FlyAgent, but it does not
change planner prompts, default runtime behavior, or model-facing context.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import urllib.error
import urllib.request
from typing import Any

from src.agent.model_support import loads_json_object as _loads_json_object
from src.agent.config import ModelConfig, load_model_config

from .landmark_extractor import Landmark


SYSTEM_PROMPT = """You extract visual landmarks for UAV navigation perception.

Return only a JSON object:
{
  "landmarks": [
    {"label": "short noun phrase", "reason": "brief visual reason"}
  ]
}

Rules:
- Output only physical things or scene regions that can be grounded in an RGB image.
- Prefer concise nouns such as road, building, traffic light, rooftop, tower, car, person, tree.
- Do not output actions, directions, phases, coordinates, scores, benchmark truth, hidden labels, or path answers.
- Do not include more landmarks than requested.
"""


@dataclass(frozen=True)
class ModelLandmarkExtractionResult:
    """Structured result plus raw LLM response for debug artifacts."""

    landmarks: list[Landmark]
    raw_response: dict[str, Any]
    prompt: dict[str, Any]
    model_summary: dict[str, Any]


class ModelLandmarkExtractorError(RuntimeError):
    """Raised when the LLM landmark extractor cannot produce a valid result."""


class ModelLandmarkExtractor:
    """Call an OpenAI-compatible model to extract STMR landmark prompts."""

    def __init__(self, model_config: ModelConfig | None = None) -> None:
        self.model_config = model_config or load_model_config()

    def extract(
        self,
        instruction: str,
        *,
        task_class: str | None = None,
        frame_context: str | None = None,
        max_landmarks: int = 12,
    ) -> ModelLandmarkExtractionResult:
        if self.model_config.provider != "openai_compatible":
            raise ModelLandmarkExtractorError(f"Unsupported model provider: {self.model_config.provider}")
        if not self.model_config.api_key:
            raise ModelLandmarkExtractorError("LLM landmark extraction requires FLYAGENT_MODEL_API_KEY")
        prompt_payload = {
            "instruction": instruction,
            "task_class": task_class or "",
            "frame_context": frame_context or "",
            "max_landmarks": int(max_landmarks),
        }
        response = self._call_json(prompt_payload)
        landmarks = parse_landmarks_response(response, max_landmarks=max_landmarks)
        return ModelLandmarkExtractionResult(
            landmarks=landmarks,
            raw_response=response,
            prompt=prompt_payload,
            model_summary=self.model_config.redacted_summary(),
        )

    def _call_json(self, user_payload: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model_config.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.model_config.chat_completions_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.model_config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.model_config.timeout_sec) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ModelLandmarkExtractorError(
                f"LLM landmark request failed: HTTP {exc.code} {exc.reason}; body={body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ModelLandmarkExtractorError(f"LLM landmark request failed: {exc}") from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelLandmarkExtractorError("LLM landmark response missing choices[0].message.content") from exc
        parsed = _loads_json_object(str(content))
        parsed["_raw_content"] = content
        return parsed


def parse_landmarks_response(response: dict[str, Any], *, max_landmarks: int = 12) -> list[Landmark]:
    """Parse and sanitize LLM landmark JSON."""

    raw_landmarks = response.get("landmarks")
    if not isinstance(raw_landmarks, list):
        raise ModelLandmarkExtractorError("LLM landmark response must include a landmarks list")
    landmarks: list[Landmark] = []
    seen: set[str] = set()
    for item in raw_landmarks:
        if not isinstance(item, dict):
            continue
        label = _sanitize_label(str(item.get("label") or ""))
        if not label or label in seen:
            continue
        seen.add(label)
        landmarks.append(
            Landmark(
                label=label,
                source="model_landmark_extractor",
                matched_text=label,
                reason=str(item.get("reason") or "")[:240],
            )
        )
        if len(landmarks) >= max(1, int(max_landmarks)):
            break
    if not landmarks:
        raise ModelLandmarkExtractorError("LLM landmark response produced no usable landmarks")
    return landmarks


def _sanitize_label(value: str) -> str:
    normalized = value.lower().replace("_", " ").replace("-", " ").replace("/", " ")
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    words = normalized.split()
    if len(words) > 5:
        words = words[:5]
    return " ".join(words)
