"""Visual contract tests for STMR historical travel-direction overlays."""

from __future__ import annotations

from PIL import Image

from plugins.research.stmr_map.core import StmrMap
from plugins.research.stmr_map.renderer import (
    GRID_ORIGIN,
    CELL_PX,
    TRACE_COLOR,
    UAV_COLOR,
    _arrow_placements,
    render_stmr_map,
)


def test_arrow_placements_follow_old_to_new_point_order():
    forward = _arrow_placements([[(10.0, 10.0), (100.0, 10.0)]], spacing_px=30.0, max_arrows=3)
    reverse = _arrow_placements([[(100.0, 10.0), (10.0, 10.0)]], spacing_px=30.0, max_arrows=3)

    assert len(forward) == len(reverse) == 3
    assert all(ux > 0 and uy == 0 for _x, _y, ux, uy in forward)
    assert all(ux < 0 and uy == 0 for _x, _y, ux, uy in reverse)


def test_renderer_draws_direction_overlay_and_current_heading(tmp_path):
    stmr_map = _map_with_trace(
        [
            {"sequence_index": 0, "segment_index": 0, "row": 12.0, "col": 10.0},
            {"sequence_index": 1, "segment_index": 0, "row": 11.2, "col": 10.1},
            {"sequence_index": 2, "segment_index": 0, "row": 10.0, "col": 10.0},
        ],
        translation_status="actual_translation",
    )

    path = render_stmr_map(stmr_map, tmp_path)

    with Image.open(path) as image:
        grid = image.crop(
            (
                GRID_ORIGIN[0],
                GRID_ORIGIN[1],
                GRID_ORIGIN[0] + 20 * CELL_PX,
                GRID_ORIGIN[1] + 20 * CELL_PX,
            )
        )
        colors = list(grid.getdata())
    assert colors.count(TRACE_COLOR) > 0
    assert colors.count(UAV_COLOR) > 0


def test_renderer_does_not_draw_false_trace_for_yaw_only_history(tmp_path):
    stmr_map = _map_with_trace(
        [{"sequence_index": 0, "segment_index": 0, "row": 10.0, "col": 10.0}],
        translation_status="stationary_or_yaw_only",
    )

    path = render_stmr_map(stmr_map, tmp_path)

    with Image.open(path) as image:
        grid = image.crop(
            (
                GRID_ORIGIN[0],
                GRID_ORIGIN[1],
                GRID_ORIGIN[0] + 20 * CELL_PX,
                GRID_ORIGIN[1] + 20 * CELL_PX,
            )
        )
        colors = list(grid.getdata())
    assert TRACE_COLOR not in colors
    assert UAV_COLOR in colors


def _map_with_trace(trace: list[dict], *, translation_status: str) -> StmrMap:
    return StmrMap(
        semantic_matrix=[[0 for _ in range(20)] for _ in range(20)],
        semantic_legend={
            -2: {"label": "unknown", "color": [148, 163, 184]},
            -1: {"label": "visited_path", "color": [37, 99, 235]},
            0: {"label": "free", "color": [255, 255, 255]},
        },
        path_cells=[{"row": 12, "col": 10}, {"row": 11, "col": 10}, {"row": 10, "col": 10}],
        trajectory_trace=trace,
        trajectory_summary={
            "pose_count": len(trace),
            "visible_trace_point_count": len(trace),
            "visible_segment_count": 1 if trace else 0,
            "visible_start_marker": "START",
            "recent_motion_local_m": {"forward_delta_m": 0.0, "right_delta_m": 0.0, "distance_m": 0.0},
            "translation_status": translation_status,
        },
        current_cell={"row": 10, "col": 10},
        center_pose={"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0},
        map_config={"grid_size": 20, "meters_per_cell": 5.0},
        coordinate_frame={},
        projection_debug={"point_count": 0, "voxel_count": 0, "source_observation_count": 0},
    )
