"""Parse UAV-Agent folder skills with minimal frontmatter and optional capability."""

from __future__ import annotations

import json
from pathlib import Path

from src.plugins.contracts.skill import Skill, SkillCapability


class SkillMarkdownError(ValueError):
    """Raised when a skill markdown file does not match the minimal schema."""


def parse_skill_markdown(path: str | Path, plugin_id: str) -> Skill:
    skill_path = Path(path)
    skill_dir = skill_path.parent
    text = skill_path.read_text(encoding="utf-8-sig")
    frontmatter, body = _split_frontmatter(text, skill_path)
    name = _required_field(frontmatter, "name", skill_path)
    description = _required_field(frontmatter, "description", skill_path)
    return Skill(
        plugin_id=plugin_id,
        name=name,
        description=description,
        skill_dir=skill_dir,
        skill_path=skill_path,
        reference_paths=_reference_paths(skill_dir),
        script_paths=_script_paths(skill_dir),
        body=body.strip(),
        capability=_capability(skill_dir),
    )


def _capability(skill_dir: Path) -> SkillCapability | None:
    path = skill_dir / "capability.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise SkillMarkdownError(f"Invalid capability.json in {skill_dir}: {exc}") from exc
    if not isinstance(data, dict):
        raise SkillMarkdownError(f"Capability must be a JSON object: {path}")
    call_name = str(data.get("call_name", "")).strip()
    if not call_name.startswith("skill."):
        raise SkillMarkdownError(f"Capability call_name must start with 'skill.': {path}")
    description = str(data.get("description", "")).strip()
    if not description:
        raise SkillMarkdownError(f"Capability missing description: {path}")
    input_schema = data.get("input_schema")
    if not isinstance(input_schema, dict):
        raise SkillMarkdownError(f"Capability input_schema must be an object: {path}")
    handler = str(data.get("handler", "")).strip()
    if not handler:
        raise SkillMarkdownError(f"Capability missing handler: {path}")
    handler_path = (skill_dir / handler).resolve()
    skill_root = skill_dir.resolve()
    try:
        handler_path.relative_to(skill_root)
    except ValueError as exc:
        raise SkillMarkdownError(f"Capability handler must stay inside skill dir: {path}") from exc
    if not handler_path.exists() or not handler_path.is_file():
        raise SkillMarkdownError(f"Capability handler does not exist: {handler_path}")
    return SkillCapability(
        call_name=call_name,
        description=description,
        input_schema=dict(input_schema),
        handler=handler,
        handler_path=handler_path,
        mock=bool(data.get("mock", False)),
    )


def _reference_paths(skill_dir: Path) -> tuple[Path, ...]:
    paths = []
    default_reference = skill_dir / "reference.md"
    if default_reference.exists():
        paths.append(default_reference)
    references_dir = skill_dir / "references"
    if references_dir.exists():
        paths.extend(sorted(path for path in references_dir.glob("*.md") if path.is_file()))
    return tuple(paths)


def _script_paths(skill_dir: Path) -> tuple[Path, ...]:
    scripts_dir = skill_dir / "scripts"
    if not scripts_dir.exists():
        return ()
    return tuple(sorted(path for path in scripts_dir.iterdir() if path.is_file()))


def _split_frontmatter(text: str, path: Path) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillMarkdownError(f"Skill requires frontmatter: {path}")
    end_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break
    if end_index is None:
        raise SkillMarkdownError(f"Skill frontmatter is not closed: {path}")
    fields: dict[str, str] = {}
    for raw_line in lines[1:end_index]:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise SkillMarkdownError(f"Invalid frontmatter line in {path}: {raw_line!r}")
        fields[key.strip()] = _unquote(value.strip())
    body = "\n".join(lines[end_index + 1 :])
    return fields, body


def _required_field(frontmatter: dict[str, str], key: str, path: Path) -> str:
    value = frontmatter.get(key, "").strip()
    if not value:
        raise SkillMarkdownError(f"Skill frontmatter missing {key!r}: {path}")
    return value


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value
