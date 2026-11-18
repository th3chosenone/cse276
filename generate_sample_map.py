"""
generate_sample_map.py — Procedural City Map Generator with Dark Roads & Buildings
Last-Mile Delivery Route Optimiser
---------------------------------------------------------------------------------
Features:
  • Generates realistic procedural city grids with dark asphalt roads and building blocks.
  • Dynamically places a RED depot and BLUE delivery houses directly along road networks.
  • Can be invoked via CLI or imported into Streamlit/Kivy apps to generate fresh maps on demand.
"""

from __future__ import annotations

import os
import random
from typing import Optional
import cv2
import numpy as np


def create_city_map(
    out_path: Optional[str] = "sample_map.png",
    width: int = 860,
    height: int = 620,
    n_houses: int = 8,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """
    Generate a procedural city map with dark roads, distinct buildings,
    a red depot, and blue delivery houses.

    Returns
    -------
    image_bgr   : Synthesized color map image.
    road_mask   : Binary mask where road pixels = 255, buildings = 0.
    depot_pos   : (cx, cy) pixel of the depot.
    house_pos   : List of (cx, cy) pixels of the delivery houses.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
        random.seed(seed)
    else:
        rng = np.random.default_rng()

    # 1. Base Canvas — light urban pavement / terrain background
    canvas = np.full((height, width, 3), (228, 230, 226), dtype=np.uint8)
    road_mask = np.zeros((height, width), dtype=np.uint8)

    # 2. Procedural Road Grid Layout
    road_width = 30
    x_spacing = int(rng.integers(130, 160))
    y_spacing = int(rng.integers(110, 140))

    x_roads = list(range(40, width - 30, x_spacing))
    y_roads = list(range(40, height - 30, y_spacing))

    # Fill City Blocks with Buildings (between road corridors)
    building_palettes = [
        (190, 195, 185),  # light slate
        (175, 180, 175),  # muted cement
        (205, 200, 195),  # warm sandstone
        (185, 175, 170),  # brick tint
        (210, 215, 215),  # modern glass tint
    ]

    for i in range(len(x_roads) - 1):
        x1 = x_roads[i] + road_width // 2 + 6
        x2 = x_roads[i + 1] - road_width // 2 - 6
        if x2 <= x1:
            continue

        for j in range(len(y_roads) - 1):
            y1 = y_roads[j] + road_width // 2 + 6
            y2 = y_roads[j + 1] - road_width // 2 - 6
            if y2 <= y1:
                continue

            # Subdivide each block into 2-4 individual building footprints
            bw = x2 - x1
            bh = y2 - y1

            sub_div_x = 2 if bw > 70 else 1
            sub_div_y = 2 if bh > 70 else 1

            dx_step = bw // sub_div_x
            dy_step = bh // sub_div_y

            for sx in range(sub_div_x):
                for sy in range(sub_div_y):
                    bx1 = x1 + sx * dx_step + 2
                    by1 = y1 + sy * dy_step + 2
                    bx2 = bx1 + dx_step - 4
                    by2 = by1 + dy_step - 4

                    color = building_palettes[int(rng.integers(0, len(building_palettes)))]
                    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, -1)
                    # Building border / roof shadow
                    cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (150, 150, 150), 1)

                    # Interior rooftop accent
                    if (bx2 - bx1 > 20) and (by2 - by1 > 20):
                        inner_color = tuple(max(0, c - 20) for c in color)
                        cv2.rectangle(canvas, (bx1 + 4, by1 + 4), (bx2 - 4, by2 - 4), inner_color, -1)

    # 3. Draw Dark Asphalt Roads
    road_color = (48, 50, 52)  # Dark charcoal BGR

    # Draw horizontal road strips
    for y in y_roads:
        cv2.line(canvas, (0, y), (width, y), road_color, road_width)
        cv2.line(road_mask, (0, y), (width, y), 255, road_width)

    # Draw vertical road strips
    for x in x_roads:
        cv2.line(canvas, (x, 0), (x, height), road_color, road_width)
        cv2.line(road_mask, (x, 0), (x, height), 255, road_width)

    # Clean intersections & draw dashed white centerlines
    centerline_color = (210, 210, 210)
    for y in y_roads:
        for x in range(0, width, 18):
            cv2.line(canvas, (x, y), (x + 8, y), centerline_color, 1)

    for x in x_roads:
        for y in range(0, height, 18):
            cv2.line(canvas, (x, y), (x, y + 8), centerline_color, 1)

    # 4. Place Red Depot on an Intersection
    depot_x = x_roads[min(1, len(x_roads) - 1)]
    depot_y = y_roads[min(len(y_roads) // 2, len(y_roads) - 1)]

    # Draw Depot: Bold Red circle with white border
    cv2.circle(canvas, (depot_x, depot_y), 17, (0, 0, 235), -1)
    cv2.circle(canvas, (depot_x, depot_y), 19, (255, 255, 255), 2)
    cv2.circle(road_mask, (depot_x, depot_y), 18, 255, -1)

    # 5. Place Blue Delivery Houses along road segments
    houses: list[tuple[int, int]] = []
    road_pixels = np.argwhere(road_mask > 0)  # [y, x]

    attempts = 0
    min_dist_between_nodes = 70

    while len(houses) < n_houses and attempts < 3000:
        attempts += 1
        idx = int(rng.integers(0, len(road_pixels)))
        py, px = road_pixels[idx]

        # Don't place on extreme borders
        if px < 40 or px > width - 40 or py < 40 or py > height - 40:
            continue

        # Keep distance from depot and already placed houses
        all_pts = [(depot_x, depot_y)] + houses
        if all(np.hypot(px - ox, py - oy) >= min_dist_between_nodes for ox, oy in all_pts):
            houses.append((int(px), int(py)))
            # Draw blue house circle with white border
            cv2.circle(canvas, (int(px), int(py)), 13, (230, 85, 10), -1)
            cv2.circle(canvas, (int(px), int(py)), 15, (255, 255, 255), 2)
            cv2.circle(road_mask, (int(px), int(py)), 15, 255, -1)

    # 6. Draw Map Info / Watermark in top corner (text only to avoid false color detections)
    leg_x, leg_y = width - 180, 12
    cv2.rectangle(canvas, (leg_x, leg_y), (leg_x + 170, leg_y + 36), (255, 255, 255), -1)
    cv2.rectangle(canvas, (leg_x, leg_y), (leg_x + 170, leg_y + 36), (180, 180, 180), 1)
    cv2.putText(canvas, "City Delivery Network", (leg_x + 8, leg_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Red = Depot | Blue = Stops", (leg_x + 8, leg_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1, cv2.LINE_AA)

    if out_path:
        cv2.imwrite(out_path, canvas)

    return canvas, road_mask, (depot_x, depot_y), houses


if __name__ == "__main__":
    import sys
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    img, mask, d_pos, h_pos = create_city_map("sample_map.png", n_houses=count)
    print(f"[gen] Generated new city map with {len(h_pos)} houses and depot at {d_pos}.")
