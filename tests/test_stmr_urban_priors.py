"""Tests for STMR urban detector-query priors."""

from plugins.research.custom_map.perception.semantic_palette import SEMANTIC_CLASSES, semantic_class_for_ade_label
from plugins.research.stmr_map.perception.landmark_extractor import Landmark
from plugins.research.stmr_map.perception.landmark_extractor import extract_landmarks
from plugins.research.stmr_map.perception.urban_priors import (
    DEFAULT_UAV_PRIOR_NAME,
    URBAN_UAV_PRIOR_V1_NAME,
    assemble_landmark_queries,
    default_uav_prior,
    normalize_landmark_query,
    urban_uav_prior_v1,
)


def test_task_landmarks_stay_before_fixed_prior_queries():
    task = [
        Landmark("tower", "instruction_keyword"),
        Landmark("traffic signal", "manual_override"),
    ]

    queries = assemble_landmark_queries(task, prior=URBAN_UAV_PRIOR_V1_NAME)

    assert [item.label for item in queries[:2]] == ["tower", "traffic light"]
    assert queries[0].source == "instruction_keyword"
    assert queries[1].source == "manual_override"
    assert queries[2].label == "building"


def test_default_and_urban_prior_order_is_stable():
    default = assemble_landmark_queries([], prior=DEFAULT_UAV_PRIOR_NAME)
    urban = assemble_landmark_queries([], prior=URBAN_UAV_PRIOR_V1_NAME)

    assert [item.label for item in default] == list(default_uav_prior)
    assert [item.label for item in urban] == list(urban_uav_prior_v1)
    assert all(item.reason == "fixed detector query prior; not scene evidence" for item in urban)


def test_query_aliases_deduplicate_and_respect_maximum():
    task = ["car", "vehicle", "roof", "rooftop", "road sign"]

    queries = assemble_landmark_queries(task, prior=URBAN_UAV_PRIOR_V1_NAME, max_queries=6)
    labels = [item.label for item in queries]

    assert labels == ["vehicle", "rooftop", "traffic sign", "building", "entrance", "road"]
    assert len(labels) == len(set(labels)) == 6
    assert normalize_landmark_query("Traffic_Signal") == "traffic light"


def test_urban_building_parts_compress_to_building_structure():
    expected_id = SEMANTIC_CLASSES["building_structure"].semantic_id

    for label in ("entrance", "facade", "rooftop", "tower", "bridge"):
        assert semantic_class_for_ade_label(label).semantic_id == expected_id


def test_approach_sunshade_is_a_task_landmark_before_prior_queries():
    task = extract_landmarks(
        "Approach the red sunshade and hold nearby.",
        include_defaults=False,
    )

    queries = assemble_landmark_queries(task, prior=URBAN_UAV_PRIOR_V1_NAME)

    assert queries[0].label == "sunshade"
    assert queries[0].source == "instruction_keyword"
    assert queries[1].label == "building"
