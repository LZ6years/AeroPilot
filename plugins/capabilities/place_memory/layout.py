"""Measured-place map layout shared by live requests and offline previews.

This module only projects and draws recorded geometry. Labels are excerpts of
existing notes, never inferred landmarks, plans, free-space or goal positions.
Hit regions are returned in image pixels so HTML can share the exact layout.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RENDER_VERSION = "simple-place-map-v1.2"
IMAGE_SIZE = (1600, 1100)
MAP_RECT = (45, 35, 1555, 1065)
INSET_RECT = (1190, 785, 1535, 1045)
BLUE = "#1878f5"
INK = "#17324f"


@dataclass(frozen=True)
class Projection:
    """Uniform metre-to-pixel projection; initial forward is always screen up."""

    centre_m: tuple[float, float]
    centre_px: tuple[float, float]
    pixels_per_m: float

    def project(self, position):
        return (self.centre_px[0] + (position[0] - self.centre_m[0]) * self.pixels_per_m,
                self.centre_px[1] - (position[1] - self.centre_m[1]) * self.pixels_per_m)


@dataclass
class MapLayout:
    """Image and geometry for tests, accessible HTML and exact click targets."""

    image: Image.Image
    projection: Projection
    inset_projection: Projection
    node_positions: dict
    hit_regions: list
    label_boxes: list
    current_position: tuple
    current_tip: tuple
    grid_spacing_m: float


def _font(size):
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/segoeui.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _nice_step(minimum):
    """Choose a readable 1/2/5 metre spacing, never finer than ten metres."""
    minimum = max(10.0, minimum)
    power = 10 ** math.floor(math.log10(minimum))
    return next(multiplier * power for multiplier in (1, 2, 5, 10)
                if multiplier * power >= minimum)


def _intersects(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _inside(box, outer):
    return box[0] >= outer[0] and box[1] >= outer[1] and box[2] <= outer[2] and box[3] <= outer[3]


def _projection(points):
    # The complete measured path is retained, so extent never depends only on
    # the latest view or on a merged node's old anchor. Reserve the inset by
    # fitting segment bounds, rather than hiding part of the global route.
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    centre = ((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2)
    scale = min(1330 / max(max(xs) - min(xs), 40), 740 / max(max(ys) - min(ys), 40))
    reserved = (INSET_RECT[0] - 36, INSET_RECT[1] - 36, INSET_RECT[2] + 36, INSET_RECT[3] + 36)
    for factor in (1, .9, .8, .7, .6):
        for offset in ((0, 0), (0, -90), (-120, -90)):
            view = Projection(centre, (800 + offset[0], 585 + offset[1]), scale * factor)
            projected = [view.project(p) for p in points]
            pairs = list(zip(projected, projected[1:])) or [(projected[0], projected[0])]
            if all(not _intersects((min(a[0], b[0]) - 24, min(a[1], b[1]) - 24,
                                    max(a[0], b[0]) + 24, max(a[1], b[1]) + 24), reserved)
                   for a, b in pairs):
                return view
    return Projection(centre, (680, 475), scale * .6)


def heading_tip(position, yaw_rad, length):
    """Positive measured yaw points right in the fixed initial-heading frame."""
    return (position[0] + length * math.sin(yaw_rad), position[1] - length * math.cos(yaw_rad))


def _uav(draw, position, yaw_rad, size):
    x, y = position
    draw.ellipse((x - size, y - size, x + size, y + size), fill="#dcecff")
    forward = (math.sin(yaw_rad), -math.cos(yaw_rad))
    right = (math.cos(yaw_rad), math.sin(yaw_rad))
    def at(f, r):
        return (x + size * (f * forward[0] + r * right[0]),
                y + size * (f * forward[1] + r * right[1]))
    draw.polygon([at(1, 0), at(-.65, .65), at(-.30, 0), at(-.65, -.65)], fill=BLUE)
    draw.line([at(1, 0), at(-.65, .65), at(-.30, 0), at(-.65, -.65), at(1, 0)], fill="white", width=3)


def _grid(draw, view, rect, font):
    step = _nice_step(34 / view.pixels_per_m)
    lower = (view.centre_m[0] + (rect[0] - view.centre_px[0]) / view.pixels_per_m,
             view.centre_m[1] - (rect[3] - view.centre_px[1]) / view.pixels_per_m)
    upper = (view.centre_m[0] + (rect[2] - view.centre_px[0]) / view.pixels_per_m,
             view.centre_m[1] - (rect[1] - view.centre_px[1]) / view.pixels_per_m)
    for index in range(math.ceil(lower[0] / step), math.floor(upper[0] / step) + 1):
        x, _ = view.project((index * step, 0))
        draw.line((x, rect[1], x, rect[3]), fill="#e3ebf2", width=1)
    for index in range(math.ceil(lower[1] / step), math.floor(upper[1] / step) + 1):
        _, y = view.project((0, index * step))
        draw.line((rect[0], y, rect[2], y), fill="#e3ebf2", width=1)
    draw.text((rect[0] + 18, rect[1] + 12), f"Grid {step:g} m", font=font, fill="#597089")
    return step


def _scale_bar(draw, view, position, maximum_width, font):
    length_m = _nice_step(maximum_width / view.pixels_per_m / 5)
    length_px = length_m * view.pixels_per_m
    x, y = position
    draw.line((x, y, x + length_px, y), fill=INK, width=3)
    for end in (x, x + length_px):
        draw.line((end, y - 5, end, y + 5), fill=INK, width=2)
    draw.text((x, y - 30), f"{length_m:g} m", fill=INK, font=font)


def _path(draw, points, view, rect):
    # Clip each straight segment to the viewport. Never connect clipped-away
    # points: doing so would manufacture a shortcut in the local inset.
    travelled_px = 0.0
    for index, (a, b) in enumerate(zip(points, points[1:])):
        a, b = view.project(a), view.project(b)
        dx, dy = b[0] - a[0], b[1] - a[1]
        low, high = 0., 1.
        visible = True
        for p, q in ((-dx, a[0] - rect[0]), (dx, rect[2] - a[0]),
                     (-dy, a[1] - rect[1]), (dy, rect[3] - a[1])):
            if p == 0:
                if q < 0:
                    visible = False
                    break
            elif p < 0:
                low = max(low, q / p)
            else:
                high = min(high, q / p)
        if not visible or low > high:
            continue
        start, end = (a[0] + low * dx, a[1] + low * dy), (a[0] + high * dx, a[1] + high * dy)
        color = BLUE if index >= len(points) - 5 else "#9cb2c8"
        draw.line([start, end], fill=color, width=5 if color == BLUE else 3)
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        travelled_px += distance
        if distance > 18 and travelled_px >= 95:
            ux, uy = (end[0] - start[0]) / distance, (end[1] - start[1]) / distance
            x, y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
            draw.polygon([(x + 7 * ux, y + 7 * uy), (x - 5 * ux - 4 * uy, y - 5 * uy + 4 * ux),
                          (x - 5 * ux + 4 * uy, y - 5 * uy - 4 * ux)], fill=color)
            travelled_px = 0


def _excerpt(draw, text, font, maximum_width):
    text = " ".join(text.split())
    if draw.textlength(text, font=font) <= maximum_width:
        return text
    while text and draw.textlength(text + "…", font=font) > maximum_width:
        text = text[:-1]
    # Avoid breaking English words; Chinese remains naturally character based.
    if " " in text and text[-1:].isascii():
        text = text.rsplit(" ", 1)[0]
    return text.rstrip(" ,;，；") + "…"


def draw_map(graph):
    """Draw one graph without touching persistent state or original photos."""
    if not graph.get("trajectory") or not graph.get("nodes"):
        raise ValueError("map needs measured trajectory and visited nodes")
    image = Image.new("RGB", IMAGE_SIZE, "#f7fafc")
    draw = ImageDraw.Draw(image)
    font, small = _font(20), _font(17)
    draw.rounded_rectangle(MAP_RECT, radius=20, fill="#f0f5f9", outline="#dbe5ed", width=2)
    points = [t["position"] for t in graph["trajectory"]]
    view = _projection(points + [n["position"] for n in graph["nodes"]])
    spacing = _grid(draw, view, MAP_RECT, small)
    _path(draw, points, view, MAP_RECT)
    node_positions = {n["id"]: view.project(n["position"]) for n in graph["nodes"]}
    current = view.project(points[-1])
    # Height-separated nodes at one XY remain distinct memory IDs. A shared
    # click target opens an ID group instead of hiding one beneath another.
    groups = []
    for node in graph["nodes"]:
        point = node_positions[node["id"]]
        group = next((g for g in groups if math.dist(point, g["anchor"]) < 22), None)
        if group is None:
            group = {"anchor": point, "nodes": []}
            groups.append(group)
        group["nodes"].append(node)
    groups.sort(key=lambda g: (not any(n["id"] == graph["current_place_id"] for n in g["nodes"]),
                               not any(n == graph["nodes"][0] for n in g["nodes"]),
                               not any(n.get("summary") or n.get("landmarks") for n in g["nodes"])))
    current_text = f"YOU · {graph['current_place_id']}"
    text_x = min(current[0] + 38, MAP_RECT[2] - draw.textlength(current_text, font=font) - 12)
    occupied = [INSET_RECT, (current[0] - 44, current[1] - 44, current[0] + 44, current[1] + 44),
                (text_x - 4, current[1] - 49, text_x + draw.textlength(current_text, font=font) + 4, current[1] - 15),
                (MAP_RECT[0], MAP_RECT[1], MAP_RECT[0] + 165, MAP_RECT[1] + 48),
                (MAP_RECT[2] - 205, MAP_RECT[1], MAP_RECT[2], MAP_RECT[1] + 65),
                (MAP_RECT[0], MAP_RECT[3] - 70, MAP_RECT[0] + 220, MAP_RECT[3])]
    occupied.extend((x-12, y-12, x+12, y+12) for x,y in node_positions.values())
    label_boxes, hits = [], []
    for group in groups:
        x, y = group["anchor"]
        ids = [n["id"] for n in group["nodes"]]
        origin = graph["nodes"][0]["id"] in ids
        annotated = any(n.get("summary") or n.get("landmarks") for n in group["nodes"])
        color = "#17a66b" if origin else BLUE if annotated else "#6986a4"
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color, outline="white", width=2)
        label = " / ".join(ids)
        node = next((n for n in group["nodes"] if n["id"] == graph["current_place_id"]), group["nodes"][0])
        note = next(iter(node.get("landmarks", [])), "") or node.get("summary", "")
        if origin:
            label += " · START"
        note = _excerpt(draw, note, small, 230) if note else ""
        width = max(draw.textlength(label, font=font), draw.textlength(note, font=small)) + 28
        height = 65 if note else 38
        # Try nearby positions first, then use a leader to an empty area. No
        # label movement changes the observation-site coordinate itself.
        candidates = [(x + 16, y - height / 2), (x - width - 16, y - height / 2),
                      (x - width / 2, y - height - 18), (x - width / 2, y + 18)]
        for gap in (45, 75, 110):
            candidates.extend(((x + gap, y - height / 2), (x - width - gap, y - height / 2),
                               (x - width / 2, y - height - gap), (x - width / 2, y + gap)))
        candidates += [(lx, ly) for ly in range(MAP_RECT[1] + 65, MAP_RECT[3] - 65, 76)
                       for lx in range(MAP_RECT[0] + 30, MAP_RECT[2] - 85, 260)]
        box = None
        for lx, ly in candidates:
            candidate = (lx, ly, lx + width, ly + height)
            if _inside(candidate, MAP_RECT) and not any(_intersects(candidate, other) for other in occupied):
                box = candidate
                break
        if box:
            occupied.append(box)
            label_boxes.append({"ids": ids, "box": box, "text": label, "excerpt": note})
            end = (min(max(x, box[0]), box[2]), min(max(y, box[1]), box[3]))
            draw.line((x, y, *end), fill="#a6b8c9", width=1)
            draw.rounded_rectangle(box, radius=7, fill="white", outline="#ccdae7", width=1)
            draw.text((box[0] + 10, box[1] + 5), label, fill=INK, font=font)
            if note:
                draw.text((box[0] + 10, box[1] + 33), note, fill="#597089", font=small)
            if annotated:
                draw.rectangle((box[2] - 12, box[1] + 8, box[2] - 6, box[1] + 19), fill=BLUE)
            hits.append({"ids": ids, "box": box, "anchor": (x, y)})
        else:
            # Extremely dense maps retain a human-clickable pin and every ID
            # in archive, even if there is no room for another full callout.
            hits.append({"ids": ids, "box": (x - 11, y - 11, x + 11, y + 11), "anchor": (x, y)})
    yaw = graph["trajectory"][-1]["yaw_rad"]
    _uav(draw, current, yaw, 32)
    draw.text((text_x, current[1] - 45), current_text, fill=BLUE, font=font)
    draw.text((MAP_RECT[2] - 180, MAP_RECT[1] + 18), "Initial heading ↑", font=small, fill=INK)
    _scale_bar(draw, view, (MAP_RECT[0] + 28, MAP_RECT[3] - 30), 150, small)

    # Inset spans exactly 80 x 80 m. Equal scales preserve the main map's
    # orientation; only the origin and zoom change, never the heading frame.
    draw.rounded_rectangle(INSET_RECT, radius=12, fill="white", outline="#bfd1e2", width=2)
    draw.text((INSET_RECT[0] + 12, INSET_RECT[1] + 10), "LOCAL · 80 × 80 m", font=small, fill=INK)
    local_rect = (INSET_RECT[0] + 10, INSET_RECT[1] + 42, INSET_RECT[2] - 10, INSET_RECT[3] - 12)
    side = min(local_rect[2] - local_rect[0], local_rect[3] - local_rect[1])
    centre_px = ((local_rect[0] + local_rect[2]) / 2, (local_rect[1] + local_rect[3]) / 2)
    local_rect = (centre_px[0] - side / 2, centre_px[1] - side / 2,
                  centre_px[0] + side / 2, centre_px[1] + side / 2)
    local = Projection(tuple(points[-1][:2]), centre_px, side / 80)
    _grid(draw, local, local_rect, _font(12))
    _path(draw, points, local, local_rect)
    for node in graph["nodes"]:
        x, y = local.project(node["position"])
        if _inside((x - 4, y - 4, x + 4, y + 4), local_rect):
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#6986a4")
    _uav(draw, centre_px, yaw, 22)
    _scale_bar(draw, local, (local_rect[0] + 8, local_rect[3] - 8), 70, _font(12))
    return MapLayout(image, view, local, node_positions, hits, label_boxes, current,
                     heading_tip(current, yaw, 32), spacing)
