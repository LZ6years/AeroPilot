"""Load UAV-Flow task JSON files as evaluation-only benchmark cases.

The loader intentionally separates fields that may be used by UAV-Agent
(`instruction` and initial scene setup) from fields that are evaluation-only
(`target_pos` and `reference_path_preprocessed`). This keeps benchmark truth out
of planner context while still preserving it for metrics and reports.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_INSTRUCTION_FIELD = "instruction_unified"


@dataclass(frozen=True)
class UavFlowCase:
    """One UAV-Flow benchmark case normalized for UAV-Agent evaluation."""

    case_id: str
    file_name: str
    source_path: str
    class_name: str | None
    instruction: str
    instruction_unified: str
    selected_instruction: str
    initial_pos: list[float]
    end_pos: list[float] = field(default_factory=list)
    use_obj: int | None = None
    obj_id: int | None = None
    target_pos: list[float] | None = None
    reference_path_raw: list[list[float]] = field(default_factory=list)
    reference_path_preprocessed: list[list[float]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        class_name: str | None = None,
        instruction_field: str = DEFAULT_INSTRUCTION_FIELD,
    ) -> "UavFlowCase":
        case_path = Path(path)
        data = json.loads(case_path.read_text(encoding="utf-8-sig"))
        initial_pos = _float_list(data.get("initial_pos"))
        if len(initial_pos) < 6:
            raise ValueError(f"UAV-Flow case {case_path} missing valid initial_pos")
        instruction = str(data.get("instruction", "")).strip()
        unified = str(data.get("instruction_unified", "")).strip()
        selected = str(data.get(instruction_field, "")).strip() or unified or instruction
        if not selected:
            raise ValueError(f"UAV-Flow case {case_path} missing instruction")
        return cls(
            case_id=case_path.stem,
            file_name=case_path.name,
            source_path=str(case_path),
            class_name=class_name,
            instruction=instruction,
            instruction_unified=unified,
            selected_instruction=selected,
            initial_pos=initial_pos,
            end_pos=_float_list(data.get("end_pos")),
            use_obj=_optional_int(data.get("use_obj")),
            obj_id=_optional_int(data.get("obj_id")),
            target_pos=_optional_float_list(data.get("target_pos")),
            reference_path_raw=_float_path(data.get("reference_path_raw")),
            reference_path_preprocessed=_float_path(data.get("reference_path_preprocessed")),
            raw=dict(data),
        )

    def to_dict(self, *, include_evaluation_truth: bool = True) -> dict[str, Any]:
        """Return a JSON-safe representation of this case.

        Set ``include_evaluation_truth=False`` for any planner-facing metadata.
        """

        data = asdict(self)
        if not include_evaluation_truth:
            for key in (
                "end_pos",
                "target_pos",
                "reference_path_raw",
                "reference_path_preprocessed",
                "raw",
            ):
                data.pop(key, None)
        return data

    def planner_safe_metadata(self) -> dict[str, Any]:
        """Return case metadata that is safe to expose to runtime logs/context."""

        return self.to_dict(include_evaluation_truth=False)

    def evaluation_truth(self) -> dict[str, Any]:
        """Return fields reserved for metric and report generation."""

        return {
            "end_pos": list(self.end_pos),
            "target_pos": list(self.target_pos) if self.target_pos else None,
            "reference_path_raw": list(self.reference_path_raw),
            "reference_path_preprocessed": list(self.reference_path_preprocessed),
        }


def load_classification(path: str | Path) -> dict[str, list[str]]:
    """Load ``classified_instr.json`` as class name -> task file names."""

    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return {
        str(class_name): [str(item) for item in file_names]
        for class_name, file_names in data.items()
        if isinstance(file_names, list)
    }


def load_cases(
    uav_flow_repo_path: str | Path,
    *,
    case_ids: tuple[str, ...] | list[str] | None = None,
    classes: list[str] | None = None,
    limit_per_class: int | None = None,
    instruction_field: str = DEFAULT_INSTRUCTION_FIELD,
    test_jsons_path: str | Path | None = None,
    classified_instr_path: str | Path | None = None,
) -> list[UavFlowCase]:
    """Load benchmark cases from a UAV-Flow-Eval checkout."""

    repo = Path(uav_flow_repo_path)
    test_dir = Path(test_jsons_path) if test_jsons_path else repo / "test_jsons"
    classified_path = Path(classified_instr_path) if classified_instr_path else repo / "classified_instr.json"
    by_class = (
        load_classification(classified_path)
        if classified_path.exists()
        else {}
    )
    if case_ids is not None:
        normalized_ids = _validate_case_ids(case_ids)
        missing = [
            case_id
            for case_id in normalized_ids
            if not (test_dir / f"{case_id}.json").is_file()
        ]
        if missing:
            raise ValueError(
                "UAV-Flow case_ids not found under "
                f"{test_dir}: {', '.join(missing)}"
            )
        class_by_file = {
            file_name: class_name
            for class_name, file_names in by_class.items()
            for file_name in file_names
        }
        return [
            UavFlowCase.from_file(
                test_dir / f"{case_id}.json",
                class_name=class_by_file.get(f"{case_id}.json"),
                instruction_field=instruction_field,
            )
            for case_id in normalized_ids
        ]

    if classified_path.exists():
        wanted_classes = classes or list(by_class)
        cases: list[UavFlowCase] = []
        for class_name in wanted_classes:
            file_names = by_class.get(class_name, [])
            if limit_per_class is not None:
                file_names = file_names[: max(0, int(limit_per_class))]
            for file_name in file_names:
                path = test_dir / file_name
                if path.exists():
                    cases.append(
                        UavFlowCase.from_file(
                            path,
                            class_name=class_name,
                            instruction_field=instruction_field,
                        )
                    )
        return cases

    file_paths = sorted(test_dir.glob("*.json"))
    if limit_per_class is not None:
        file_paths = file_paths[: max(0, int(limit_per_class))]
    return [UavFlowCase.from_file(path, instruction_field=instruction_field) for path in file_paths]


def _validate_case_ids(
    case_ids: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Keep the loader safe when called without ``UavFlowEvalConfig``."""

    normalized = tuple(str(item).strip() for item in case_ids)
    if not normalized or any(not item for item in normalized):
        raise ValueError("case_ids must contain non-empty dataset case stems")
    invalid = [
        item
        for item in normalized
        if item.endswith(".json") or "/" in item or "\\" in item
    ]
    if invalid:
        raise ValueError(
            "case_ids must use file-name stems without paths or .json: "
            + ", ".join(invalid)
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("case_ids must be unique")
    return normalized


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            return []
    return result


def _optional_float_list(value: Any) -> list[float] | None:
    result = _float_list(value)
    return result or None


def _float_path(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        row = _float_list(item)
        if row:
            result.append(row)
    return result


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
