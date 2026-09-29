"""Read-only experience capability; database selection is administrator-owned."""
from pathlib import Path

from plugins.research.flight_experience.store import Store
from src.plugins.contracts.skill import SkillCapabilityResult


def handle(payload):
    settings, args = dict(payload.get("settings", {})), dict(payload.get("args", {}))
    if set(args) - {"query", "filters", "limit"}:
        raise ValueError("Unknown search arguments")
    path = settings.get("database_path")
    if not isinstance(path, str) or not path.strip() or not Path(path).is_absolute():
        raise ValueError("Configure an absolute experience database_path before enabling the plugin")
    cards = Store(path).search(args.get("query"), filters=args.get("filters"), limit=args.get("limit", 3),
        budget=settings.get("character_budget", 6000), scope=settings.get("scope", "production"))
    return SkillCapabilityResult(message=f"Retrieved {len(cards)} historical guidance items; not current completion evidence.",
        data={"experiences": cards, "evidence_role": "historical_guidance"})
