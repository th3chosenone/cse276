"""
pathfinding.py — Road Extraction & A* Pathfinding Engine
Last-Mile Delivery Route Optimiser
-------------------------------------------------------
Responsibilities:
  • Convert map to grayscale and threshold to isolate road/street pixels.
  • Snap off-road or building coordinates to the nearest walkable road pixel.
  • Fast A* search on 2D road grids using 8-connectivity and NumPy acceleration.
  • Calculate true pairwise A* road distance matrix strictly between the primary Red Depot
    (Index 0) and all selected Blue Houses.
  • Render road-following polylines with turn indicators, arrows, and step badges.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Optional
import cv2
import numpy as np


class RoadGrid:
    """
    Manages the 2D walkable road grid, coordinate snapping, and fast A* search.
    """

    def __init__(self, road_mask: np.ndarray, grid_scale: int = 2):
        """
        Parameters
        ----------
        road_mask  : 2D uint8 numpy array where >0 is walkable road, 0 is obstacle.
        grid_scale : Downsample factor for A* grid (default 2 for high speed & fidelity).
        """
        self.orig_h, self.orig_w = road_mask.shape[:2]
        self.grid_scale = max(1, int(grid_scale))

        if self.grid_scale > 1:
            sw = max(1, self.orig_w // self.grid_scale)
            sh = max(1, self.orig_h // self.grid_scale)
            resized = cv2.resize(road_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
            self.mask = (resized > 0).astype(np.uint8) * 255
        else:
            self.mask = (road_mask > 0).astype(np.uint8) * 255

        self.h, self.w = self.mask.shape[:2]
        self.walkable = (self.mask > 0)

        # 8-connected moves: (dx, dy, step_cost)
        sqrt2 = math.sqrt(2.0)
        self.moves = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, sqrt2),
            (1, -1, sqrt2),
            (-1, 1, sqrt2),
            (-1, -1, sqrt2),
        ]

    def snap_to_road(self, px: int, py: int) -> tuple[int, int]:
        """
        Map original pixel (px, py) to the nearest walkable grid cell (gx, gy).
        """
        gx = min(max(0, int(px / self.grid_scale)), self.w - 1)
        gy = min(max(0, int(py / self.grid_scale)), self.h - 1)

        if self.walkable[gy, gx]:
            return (gx, gy)

        # Expanding window search for nearest walkable road pixel
        max_r = max(self.w, self.h)
        for radius in range(1, max_r):
            x_min = max(0, gx - radius)
            x_max = min(self.w - 1, gx + radius)
            y_min = max(0, gy - radius)
            y_max = min(self.h - 1, gy + radius)

            sub = self.walkable[y_min : y_max + 1, x_min : x_max + 1]
            if np.any(sub):
                ys, xs = np.where(sub)
                dists = (xs + x_min - gx) ** 2 + (ys + y_min - gy) ** 2
                best_idx = int(np.argmin(dists))
                return (int(xs[best_idx] + x_min), int(ys[best_idx] + y_min))

        return (gx, gy)

    def find_path(
        self, start_pt: tuple[int, int], goal_pt: tuple[int, int]
    ) -> tuple[float, list[tuple[int, int]]]:
        """
        Compute the shortest road path between start_pt and goal_pt.

        Parameters
        ----------
        start_pt : (x, y) in original pixel coordinates
        goal_pt  : (x, y) in original pixel coordinates

        Returns
        -------
        (road_distance_px, pixel_path)
            road_distance_px : cumulative pixel length along roads
            pixel_path       : list of (x, y) in original image coordinates
        """
        sx, sy = self.snap_to_road(start_pt[0], start_pt[1])
        gx, gy = self.snap_to_road(goal_pt[0], goal_pt[1])

        if (sx, sy) == (gx, gy):
            direct_dist = math.hypot(start_pt[0] - goal_pt[0], start_pt[1] - goal_pt[1])
            return direct_dist, [start_pt, goal_pt]

        w = self.w
        h = self.h
        start_idx = sy * w + sx
        goal_idx = gy * w + gx

        # Octile heuristic for 8-connected grid
        D = 1.0
        D2 = math.sqrt(2.0)

        def heuristic(x: int, y: int) -> float:
            dx = abs(x - gx)
            dy = abs(y - gy)
            return D * (dx + dy) + (D2 - 2.0 * D) * min(dx, dy)

        # Flat NumPy arrays for fastest C-speed lookups
        g_scores = np.full(h * w, np.inf, dtype=np.float32)
        g_scores[start_idx] = 0.0

        came_from: dict[int, int] = {}
        pq = [(heuristic(sx, sy), 0.0, sx, sy)]
        found = False

        while pq:
            f, g, cx, cy = heapq.heappop(pq)
            c_idx = cy * w + cx

            if c_idx == goal_idx:
                found = True
                break

            if g > g_scores[c_idx]:
                continue

            for dx, dy, step_cost in self.moves:
                nx = cx + dx
                ny = cy + dy

                if 0 <= nx < w and 0 <= ny < h and self.walkable[ny, nx]:
                    n_idx = ny * w + nx
                    ng = g + step_cost
                    if ng < g_scores[n_idx]:
                        g_scores[n_idx] = ng
                        came_from[n_idx] = c_idx
                        heapq.heappush(pq, (ng + heuristic(nx, ny), ng, nx, ny))

        if not found:
            # Fallback if nodes belong to disconnected road segments (e.g. island roads)
            manhattan = abs(start_pt[0] - goal_pt[0]) + abs(start_pt[1] - goal_pt[1])
            return float(manhattan), [start_pt, (goal_pt[0], start_pt[1]), goal_pt]

        # Reconstruct path in grid coordinates
        path_grid = []
        curr = goal_idx
        while curr in came_from:
            path_grid.append((curr % w, curr // w))
            curr = came_from[curr]
        path_grid.append((sx, sy))
        path_grid.reverse()

        # Map back to original image pixel coordinates
        scale = self.grid_scale
        pixel_path: list[tuple[int, int]] = [start_pt]
        for gx_node, gy_node in path_grid:
            orig_x = int(gx_node * scale + scale // 2)
            orig_y = int(gy_node * scale + scale // 2)
            pixel_path.append((orig_x, orig_y))
        pixel_path.append(goal_pt)

        # Compute accurate road distance
        road_distance = 0.0
        for i in range(len(pixel_path) - 1):
            p1 = pixel_path[i]
            p2 = pixel_path[i + 1]
            road_distance += math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        return road_distance, pixel_path


# ──────────────────────────────────────────────────────────────────────────────
# Road Extraction & Masking
# ──────────────────────────────────────────────────────────────────────────────

def extract_road_mask(
    image: np.ndarray,
    road_type: str = "dark",
    threshold: int = 140,
    node_coords: Optional[list[tuple[int, int]]] = None,
    unblock_radius: int = 20,
) -> np.ndarray:
    """
    Extract a binary mask where road pixels are 255 (walkable) and buildings are 0.

    Parameters
    ----------
    image          : Loaded BGR or RGB map image.
    road_type      : "dark" (asphalt/dark roads on light map) or "light" (bright corridors).
    threshold      : Grayscale cutoff threshold (0..255).
    node_coords    : Optional list of (cx, cy) centroids for depot/houses.
                     These regions are stamped as walkable so colored dot markers
                     do not act as obstacles blocking road access.
    unblock_radius : Radius around each stop to guarantee road connectivity.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if road_type == "dark":
        # Road pixels are darker than surrounding buildings
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    else:
        # Road pixels are lighter than surroundings
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Morphological closing to seal small road cracks/dashes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Ensure all delivery stops and the primary depot have walkable access paths
    if node_coords:
        for cx, cy in node_coords:
            cv2.circle(mask, (int(cx), int(cy)), unblock_radius, 255, -1)

    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Pairwise Road Distance Matrix Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_road_distance_matrix(
    nodes: list,
    road_mask: np.ndarray,
    grid_scale: int = 2,
) -> tuple[list[list[float]], dict[tuple[int, int], list[tuple[int, int]]]]:
    """
    Computes true shortest path along road pixels between every pair of nodes.
    Guarantees Index 0 is the primary Red Depot.

    Returns
    -------
    dist_matrix : n x n symmetric matrix of road distances in pixels.
    path_cache  : Dictionary mapping (i, j) -> list of pixel coordinates along the road.
    """
    n = len(nodes)
    grid = RoadGrid(road_mask, grid_scale=grid_scale)

    dist_matrix = [[0.0] * n for _ in range(n)]
    path_cache: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for i in range(n):
        pt_i = (int(nodes[i].cx), int(nodes[i].cy))
        for j in range(i + 1, n):
            pt_j = (int(nodes[j].cx), int(nodes[j].cy))
            dist, path = grid.find_path(pt_i, pt_j)

            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist

            path_cache[(i, j)] = path
            path_cache[(j, i)] = list(reversed(path))

    return dist_matrix, path_cache


# ──────────────────────────────────────────────────────────────────────────────
# Road-Following Route Drawing
# ──────────────────────────────────────────────────────────────────────────────

LEG_PALETTE: list[tuple[int, int, int]] = [
    (235, 175, 40),   # Leg 1: Sky Blue / Cyan (BGR)
    (40, 215, 90),    # Leg 2: Emerald Green
    (210, 50, 230),   # Leg 3: Bright Magenta / Violet
    (20, 150, 255),   # Leg 4: Amber / Orange
    (240, 80, 160),   # Leg 5: Electric Purple
    (40, 220, 240),   # Leg 6: Golden Yellow
    (70, 100, 245),   # Leg 7: Coral Red
    (160, 230, 50),   # Leg 8: Lime Green
]
RETURN_LEG_COLOR: tuple[int, int, int] = (0, 210, 255)  # Gold/Amber for return to depot


def draw_road_following_route(
    base_image: np.ndarray,
    route_indices: list[int],
    nodes: list,
    path_cache: dict[tuple[int, int], list[tuple[int, int]]],
    color: tuple[int, int, int] = (0, 235, 110),
    line_thickness: int = 4,
    show_steps: bool = True,
    use_leg_colors: bool = True,
) -> np.ndarray:
    """
    Overlay the exact A* path coordinates onto the map image, following road bends.
    Draws each leg in a distinct color/gradient so overlapping paths are clearly separable.
    Ensures route lines are drawn UNDER node markers and text labels, and labels display
    both the visit order and the house ID (e.g. 'Stop 1: H2') for complete clarity.
    """
    canvas = base_image.copy()
    n = len(route_indices)
    if n < 2:
        return canvas

    # Map each route stop to its visit sequence (1, 2, ...)
    visit_order_map: dict[int, int] = {}
    house_counter = 0
    for idx in route_indices:
        if not getattr(nodes[idx], "is_depot", False):
            house_counter += 1
            visit_order_map[idx] = house_counter

    # 1. Draw dark under-glow for all legs (clean contrast on both dark and light asphalt)
    glow_color = (15, 15, 20)
    for i in range(n):
        from_idx = route_indices[i]
        to_idx = route_indices[(i + 1) % n]
        path = path_cache.get((from_idx, to_idx))

        if path and len(path) >= 2:
            pts = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], isClosed=False, color=glow_color, thickness=line_thickness + 4, lineType=cv2.LINE_AA)

    # 2. Draw colored road-following route polylines per leg
    for i in range(n):
        from_idx = route_indices[i]
        to_idx = route_indices[(i + 1) % n]
        path = path_cache.get((from_idx, to_idx))

        if path and len(path) >= 2:
            pts = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
            if use_leg_colors:
                leg_col = RETURN_LEG_COLOR if i == n - 1 else LEG_PALETTE[i % len(LEG_PALETTE)]
            else:
                leg_col = color

            cv2.polylines(canvas, [pts], isClosed=False, color=leg_col, thickness=line_thickness, lineType=cv2.LINE_AA)

            # Draw directional arrow on the midpoint of the leg
            mid = len(path) // 2
            if mid + 2 < len(path):
                p_from = path[mid]
                p_to = path[mid + 2]
                cv2.arrowedLine(canvas, p_from, p_to, (255, 255, 255), 2, tipLength=0.45)

    # 3. Draw Stop Badges (circles) on top of route lines
    if show_steps:
        for step, idx in enumerate(route_indices):
            node = nodes[idx]
            is_depot = getattr(node, "is_depot", False)

            badge_center = (int(node.cx), int(node.cy))
            badge_radius = 16 if is_depot else 13
            badge_color = (0, 0, 220) if is_depot else (220, 80, 10)

            # Outer ring and filled badge
            cv2.circle(canvas, badge_center, badge_radius + 2, (255, 255, 255), 2)
            cv2.circle(canvas, badge_center, badge_radius, badge_color, -1)

            if is_depot:
                cv2.putText(
                    canvas, "D",
                    (badge_center[0] - 6, badge_center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
                )
            else:
                step_num = visit_order_map.get(idx, step)
                step_text = str(step_num)
                x_off = 5 if len(step_text) == 1 else 9
                cv2.putText(
                    canvas, step_text,
                    (badge_center[0] - x_off, badge_center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA,
                )

    # 4. Draw crisp text labels ON TOP of route lines (showing Visit Order alongside House ID)
    for node_idx, node in enumerate(nodes):
        cx, cy = int(node.cx), int(node.cy)
        is_depot = getattr(node, "is_depot", False)
        tw = getattr(node, "time_window", (540, 1080))
        tw_str = f"{tw[0]//60:02d}:{tw[0]%60:02d}-{tw[1]//60:02d}:{tw[1]%60:02d}"

        tx, ty = cx + 14, cy - 8

        if is_depot:
            # Depot label pill
            cv2.rectangle(canvas, (tx - 3, ty - 13), (tx + 105, ty + 16), (20, 20, 26), -1)
            cv2.rectangle(canvas, (tx - 3, ty - 13), (tx + 105, ty + 16), (80, 80, 90), 1)
            cv2.putText(
                canvas, "DEPOT (Hub)", (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 190, 255), 2, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, tw_str, (tx, ty + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 210, 210), 1, cv2.LINE_AA,
            )
        elif node_idx in visit_order_map:
            # Visited Delivery House: show visit order alongside house ID
            step_num = visit_order_map[node_idx]
            order_label = f"Stop {step_num}: {node.node_id}"

            # Measure label size for perfect pill sizing
            cv2.rectangle(canvas, (tx - 3, ty - 13), (tx + 105, ty + 16), (18, 18, 24), -1)
            cv2.rectangle(canvas, (tx - 3, ty - 13), (tx + 105, ty + 16), (60, 160, 240), 1)
            cv2.putText(
                canvas, order_label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 60), 2, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, tw_str, (tx, ty + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA,
            )
        else:
            # Unvisited House: muted label
            cv2.rectangle(canvas, (tx - 3, ty - 13), (tx + 86, ty + 16), (25, 25, 30), -1)
            cv2.rectangle(canvas, (tx - 3, ty - 13), (tx + 86, ty + 16), (70, 70, 75), 1)
            cv2.putText(
                canvas, node.node_id, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 140, 140), 1, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, tw_str, (tx, ty + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 130), 1, cv2.LINE_AA,
            )

    return canvas
