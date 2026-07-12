"""Stable semantic palette for RGB segmentation masks used by custom_map.

SegFormer ADE20K predicts dataset-specific labels. The map plugin needs a small
stable set of UAV-useful classes, encoded as RGB colors so the existing
mask_label_map and projection code can consume them without knowing about the
model backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class CustomSemanticClass:
    semantic_id: int
    label: str
    category: str
    color: tuple[int, int, int]

    @property
    def color_key(self) -> str:
        return f"{self.color[0]},{self.color[1]},{self.color[2]}"


SEMANTIC_CLASSES: dict[str, CustomSemanticClass] = {
    "unknown": CustomSemanticClass(-2, "unknown", "unknown", (148, 163, 184)),
    "free": CustomSemanticClass(0, "free", "free_space", (0, 0, 0)),
    "person": CustomSemanticClass(10, "person", "dynamic_actor", (220, 20, 60)),
    "vehicle": CustomSemanticClass(11, "vehicle", "dynamic_actor", (0, 0, 142)),
    "two_wheeler_or_small_vehicle": CustomSemanticClass(12, "two_wheeler_or_small_vehicle", "dynamic_actor", (245, 158, 11)),
    "route_surface": CustomSemanticClass(20, "route_surface", "navigation_surface", (128, 64, 128)),
    "building_structure": CustomSemanticClass(30, "building_structure", "structure", (70, 70, 70)),
    "vegetation": CustomSemanticClass(40, "vegetation", "natural", (107, 142, 35)),
    "street_fixture": CustomSemanticClass(50, "street_fixture", "fixture", (153, 153, 153)),
    "movable_obstacle": CustomSemanticClass(60, "movable_obstacle", "obstacle", (234, 179, 8)),
    "overhead_structure": CustomSemanticClass(61, "overhead_structure", "overhead_structure", (100, 116, 139)),
    "water_or_hazard_surface": CustomSemanticClass(70, "water_or_hazard_surface", "hazard_surface", (0, 119, 182)),
    "visual_clutter": CustomSemanticClass(80, "visual_clutter", "low_value_visual_context", (168, 85, 247)),
}


_KEYWORD_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("person", ("person", "people", "human", "man", "woman", "child", "rider", "pedestrian")),
    ("two_wheeler_or_small_vehicle", ("bicycle", "bike", "motorbike", "motorcycle", "scooter")),
    (
        "vehicle",
        (
            "car",
            "bus",
            "truck",
            "van",
            "vehicle",
            "boat",
            "ship",
            "airplane",
            "train",
        ),
    ),
    (
        "movable_obstacle",
        (
            "chair",
            "table",
            "bench",
            "box",
            "pot",
            "armchair",
            "bag",
            "basket",
            "bottle",
            "plaything",
            "vase",
            "cabinet",
            "case",
            "barrel",
        ),
    ),
    (
        "overhead_structure",
        (
            "ceiling",
            "canopy",
            "awning",
            "tunnel",
            "arch",
            "overpass",
            "bridge underside",
            "roof overhang",
        ),
    ),
    ("water_or_hazard_surface", ("water", "river", "sea", "pool", "fountain", "lake")),
    (
        "visual_clutter",
        (
            "painting",
            "curtain",
            "food",
            "poster",
            "monitor",
            "crt screen",
            "screen",
            "tray",
            "mirror",
            "blind",
            "clock",
            "stage",
            "conveyer belt",
        ),
    ),
    (
        "route_surface",
        (
            "road",
            "sidewalk",
            "path",
            "runway",
            "dirt track",
            "lane",
            "street",
            "crosswalk",
            "floor",
            "pavement",
            "walkway",
            "ground",
            "stair",
            "earth",
            "mountain",
            "rock",
            "sand",
            "hill",
            "land",
            "field",
            "terrain",
            "curb",
            "plaza",
        ),
    ),
    (
        "building_structure",
        (
            "building",
            "house",
            "wall",
            "window",
            "windowpane",
            "door",
            "entrance",
            "skyscraper",
            "bridge",
            "tower",
            "roof",
            "fence",
            "railing",
            "facade",
            "balcony",
            "column",
            "stairs",
            "stairway",
            "booth",
        ),
    ),
    ("vegetation", ("tree", "grass", "plant", "flower", "palm", "bush", "shrub", "vegetation", "foliage", "flora", "leaves")),
    (
        "street_fixture",
        (
            "pole",
            "sign",
            "signboard",
            "streetlight",
            "lamp",
            "light",
            "traffic",
            "ashcan",
            "trash",
            "sculpture",
            "bannister",
            "bollard",
            "parking meter",
            "fire hydrant",
            "bulletin board",
        ),
    ),
    ("free", ("sky", "cloud")),
)


def semantic_class_for_ade_label(label: str) -> CustomSemanticClass:
    """Return the stable custom_map class for a closed-set segmentation label."""

    normalized = _normalize_label(label)
    for semantic_key, keywords in _KEYWORD_GROUPS:
        if any(keyword in normalized for keyword in keywords):
            return SEMANTIC_CLASSES[semantic_key]
    return SEMANTIC_CLASSES["unknown"]


def semantic_label_map(source: str = "segformer_ade20k") -> dict[str, dict[str, Any]]:
    """Build a color-keyed label map compatible with mask_semantics.py."""

    return {
        item.color_key: {
            "id": item.semantic_id,
            "label": item.label,
            "category": item.category,
            "source": source,
            "confidence": "closed_set_semantic_segmentation",
        }
        for item in SEMANTIC_CLASSES.values()
    }


def color_key(color: tuple[int, int, int]) -> str:
    return f"{color[0]},{color[1]},{color[2]}"


def _normalize_label(label: str) -> str:
    normalized = label.lower().replace("_", " ").replace("-", " ").replace("/", " ")
    normalized = re.sub(r"\([^)]*\)", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return f" {normalized} "
