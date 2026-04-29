"""Test that FlyAgent does not import AirNavigation map/navigation internals."""

from __future__ import annotations

from pathlib import Path


def test_runtime_code_does_not_import_airnavigation_map_controller():
    # Scan source code for forbidden AirNavigation architecture dependencies.
    source = "\n".join(path.read_text(encoding="utf-8") for path in Path("src").rglob("*.py"))

    assert "MapController" not in source
    assert "UAV_navigater" not in source
