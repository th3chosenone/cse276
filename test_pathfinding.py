"""
test_pathfinding.py — Unit tests for Road Extraction and A* Pathfinding
Run with: python -m pytest test_pathfinding.py -v
"""

import math
import numpy as np
import cv2
import pytest

from pathfinding import (
    RoadGrid,
    extract_road_mask,
    build_road_distance_matrix,
    draw_road_following_route,
)


@pytest.fixture
def sample_grid_mask():
    """
    Creates a simple 200x200 road grid:
    - Vertical road at x=50 (width 10)
    - Vertical road at x=150 (width 10)
    - Horizontal road at y=100 (width 10)
    """
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[:, 45:55] = 255
    mask[:, 145:155] = 255
    mask[95:105, :] = 255
    return mask


class TestRoadGrid:
    def test_init_grid(self, sample_grid_mask):
        grid = RoadGrid(sample_grid_mask, grid_scale=2)
        assert grid.h == 100
        assert grid.w == 100
        assert np.any(grid.walkable)

    def test_snap_to_road_on_road(self, sample_grid_mask):
        grid = RoadGrid(sample_grid_mask, grid_scale=1)
        # Point (50, 50) is directly inside the vertical road
        gx, gy = grid.snap_to_road(50, 50)
        assert gx == 50
        assert gy == 50

    def test_snap_to_road_off_road(self, sample_grid_mask):
        grid = RoadGrid(sample_grid_mask, grid_scale=1)
        # Point (40, 50) is 5px away from road at x=45..55
        gx, gy = grid.snap_to_road(40, 50)
        assert 45 <= gx <= 55
        assert grid.walkable[gy, gx]

    def test_find_path_along_roads(self, sample_grid_mask):
        grid = RoadGrid(sample_grid_mask, grid_scale=1)
        # Start at (50, 20), Goal at (150, 20)
        # Direct Euclidean line crosses building obstacle at y=20 from x=55..145
        # Road path MUST go down to y=100, across to x=150, and up to y=20
        dist, path = grid.find_path((50, 20), (150, 20))
        assert len(path) > 2
        # Expected road distance is approx (100 - 20) + (150 - 50) + (100 - 20) = 80 + 100 + 80 = 260
        # Euclidean straight line would be 100. Road distance must be significantly greater than 100!
        assert dist > 200.0

    def test_find_path_same_point(self, sample_grid_mask):
        grid = RoadGrid(sample_grid_mask, grid_scale=1)
        dist, path = grid.find_path((50, 50), (50, 50))
        assert dist == 0.0


class TestRoadExtraction:
    def test_extract_dark_roads(self):
        # Create image with light background and dark road strip
        img = np.full((100, 100, 3), 240, dtype=np.uint8)
        img[40:60, :] = 40  # dark road

        mask = extract_road_mask(img, road_type="dark", threshold=120)
        assert np.all(mask[45:55, :] == 255)
        assert np.all(mask[10:30, :] == 0)

    def test_node_unblocking(self):
        # Even if a pixel is obstacle color, node_coords stamps it as walkable
        img = np.full((100, 100, 3), 240, dtype=np.uint8)  # all light
        mask = extract_road_mask(img, road_type="dark", threshold=120, node_coords=[(50, 50)], unblock_radius=10)
        assert mask[50, 50] == 255


class TestRoadDistanceMatrixAndDrawing:
    def test_build_road_distance_matrix(self, sample_grid_mask):
        class MockNode:
            def __init__(self, cx, cy, nid="H1"):
                self.cx = cx
                self.cy = cy
                self.node_id = nid
                self.is_depot = False

        nodes = [MockNode(50, 20, "DEPOT"), MockNode(150, 20, "H1"), MockNode(50, 180, "H2")]
        dist_matrix, path_cache = build_road_distance_matrix(nodes, sample_grid_mask, grid_scale=1)

        assert len(dist_matrix) == 3
        assert len(dist_matrix[0]) == 3
        assert dist_matrix[0][1] == dist_matrix[1][0]
        assert dist_matrix[0][1] > 200.0  # routed via y=100
        assert (0, 1) in path_cache
        assert (1, 0) in path_cache

    def test_draw_road_following_route(self, sample_grid_mask):
        class MockNode:
            def __init__(self, cx, cy, nid="H1", is_depot=False):
                self.cx = cx
                self.cy = cy
                self.node_id = nid
                self.is_depot = is_depot

        nodes = [MockNode(50, 20, "DEPOT", is_depot=True), MockNode(150, 20, "H1")]
        _, path_cache = build_road_distance_matrix(nodes, sample_grid_mask, grid_scale=1)

        base_img = np.full((200, 200, 3), 200, dtype=np.uint8)
        route_img = draw_road_following_route(base_img, [0, 1], nodes, path_cache)
        assert route_img.shape == base_img.shape
        # Image has been drawn on
        assert not np.array_equal(route_img, base_img)

