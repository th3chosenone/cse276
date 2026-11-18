"""
test_vision.py — Unit tests for the Computer Vision module
Run with: python -m pytest test_vision.py -v
Requires: opencv-python, numpy
"""

import os
import cv2
import numpy as np
import pytest

from vision import MapVisionProcessor, DetectedNode, DetectionConfig


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_map(
    depot_pos: tuple[int, int] = (100, 300),
    house_positions: list[tuple[int, int]] | None = None,
    width: int = 800,
    height: int = 600,
    bg: tuple[int, int, int] = (240, 240, 240),
) -> np.ndarray:
    """Create an in-memory BGR test map."""
    if house_positions is None:
        house_positions = [(200, 150), (400, 400), (600, 200)]
    canvas = np.full((height, width, 3), bg, dtype=np.uint8)
    cv2.circle(canvas, depot_pos, 15, (0, 0, 255), -1)
    for px, py in house_positions:
        cv2.circle(canvas, (px, py), 12, (255, 70, 0), -1)
    return canvas


def _save_tmp(img: np.ndarray, tmp_path, name: str = "map.png") -> str:
    path = str(tmp_path / name)
    cv2.imwrite(path, img)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# MapVisionProcessor tests
# ──────────────────────────────────────────────────────────────────────────────

class TestMapVisionProcessor:
    def test_process_returns_nodes(self, tmp_path):
        img  = _make_map(house_positions=[(250, 200), (450, 350)])
        path = _save_tmp(img, tmp_path)
        processor = MapVisionProcessor()
        nodes = processor.process(path)
        assert len(nodes) >= 1          # at least depot

    def test_depot_detected(self, tmp_path):
        img  = _make_map()
        path = _save_tmp(img, tmp_path)
        processor = MapVisionProcessor()
        nodes = processor.process(path)
        depots = [n for n in nodes if n.is_depot]
        assert len(depots) == 1
        assert depots[0].node_id == "DEPOT"

    def test_house_ids_sequential(self, tmp_path):
        house_pos = [(200, 150), (400, 300), (600, 450)]
        img  = _make_map(house_positions=house_pos)
        path = _save_tmp(img, tmp_path)
        processor = MapVisionProcessor()
        nodes = processor.process(path)
        houses = [n for n in nodes if not n.is_depot]
        for i, h in enumerate(houses):
            assert h.node_id == f"H{i+1}"

    def test_labeled_image_shape(self, tmp_path):
        img  = _make_map()
        path = _save_tmp(img, tmp_path)
        processor = MapVisionProcessor()
        processor.process(path)
        labeled = processor.get_labeled_image()
        assert labeled is not None
        assert labeled.shape == img.shape

    def test_rgb_image_returned(self, tmp_path):
        img  = _make_map()
        path = _save_tmp(img, tmp_path)
        processor = MapVisionProcessor()
        processor.process(path)
        rgb = processor.get_labeled_image_rgb()
        assert rgb is not None
        assert rgb.shape[2] == 3

    def test_missing_file_raises(self):
        processor = MapVisionProcessor()
        with pytest.raises(FileNotFoundError):
            processor.process("/nonexistent/path/map.png")

    def test_no_depot_map(self, tmp_path):
        """Image with only blue dots – depot should be absent."""
        canvas = np.full((400, 600, 3), 230, dtype=np.uint8)
        cv2.circle(canvas, (200, 200), 12, (255, 70, 0), -1)
        path = _save_tmp(canvas, tmp_path, "no_depot.png")
        processor = MapVisionProcessor()
        nodes = processor.process(path)
        depots = [n for n in nodes if n.is_depot]
        assert len(depots) == 0

    def test_time_windows_assigned(self, tmp_path):
        img  = _make_map(house_positions=[(200, 150), (400, 300)])
        path = _save_tmp(img, tmp_path)
        processor = MapVisionProcessor()
        nodes = processor.process(path)
        for node in nodes:
            lo, hi = node.time_window
            assert hi > lo

    def test_no_modulo_wrap_around_in_time_windows(self, tmp_path):
        """Verify that houses > 8 get standard flexible windows (09:00-18:00) without modulo wrap-around."""
        # Create 10 house positions
        positions = [(100 + (i % 5) * 80, 100 + (i // 5) * 120) for i in range(10)]
        img = _make_map(house_positions=positions)
        path = _save_tmp(img, tmp_path)
        processor = MapVisionProcessor()
        nodes = processor.process(path)
        houses = [n for n in nodes if not n.is_depot]
        assert len(houses) == 10

        # Houses beyond index 3 should get standard full-day window (540, 1080)
        for h in houses[4:]:
            assert h.time_window == (540, 1080)

    def test_coordinates_within_image(self, tmp_path):
        w, h = 800, 600
        img  = _make_map(width=w, height=h)
        path = _save_tmp(img, tmp_path)
        processor = MapVisionProcessor()
        nodes = processor.process(path)
        for node in nodes:
            assert 0 <= node.cx < w
            assert 0 <= node.cy < h

    def test_draw_route_returns_image(self, tmp_path):
        img  = _make_map(house_positions=[(250, 200), (500, 350)])
        path = _save_tmp(img, tmp_path)
        processor = MapVisionProcessor()
        nodes = processor.process(path)
        route_img = processor.draw_route([0, 1, 2], nodes)
        assert route_img is not None
        assert route_img.ndim == 3


# ──────────────────────────────────────────────────────────────────────────────
# DetectionConfig tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectionConfig:
    def test_default_config_values(self):
        cfg = DetectionConfig()
        assert cfg.min_area_red  > 0
        assert cfg.min_area_blue > 0
        assert cfg.morph_k       > 0

    def test_custom_config_propagates(self, tmp_path):
        cfg = DetectionConfig(min_area_red=1, min_area_blue=1, morph_k=3)
        img  = _make_map()
        path = _save_tmp(img, tmp_path)
        processor = MapVisionProcessor(config=cfg)
        nodes = processor.process(path)
        assert isinstance(nodes, list)


# ──────────────────────────────────────────────────────────────────────────────
# DetectedNode tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectedNode:
    def test_default_time_window(self):
        node = DetectedNode("H1", 100, 200)
        assert node.time_window == (0, 1440)

    def test_depot_flag(self):
        depot = DetectedNode("DEPOT", 50, 50, is_depot=True)
        assert depot.is_depot
        house = DetectedNode("H1", 100, 100, is_depot=False)
        assert not house.is_depot

