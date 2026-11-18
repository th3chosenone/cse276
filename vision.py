"""
vision.py — Robust Computer Vision Module
Last-Mile Delivery Route Optimiser
------------------------------------------
Responsibilities:
  • Color-space normalization (handles both BGR and RGB seamlessly).
  • Dual HSV range masking for RED (0-10° and 170-180°).
  • Tight HSV range masking for BLUE (100-140°).
  • Contour area and circularity filtering to reject text, legend headers, and noise.
  • Explicitly isolates the single primary Red Depot as Index 0 (Hub).
  • Sequential auto-labeling of Blue delivery houses (H1, H2, H3...).
  • Labeled map generation with clean annotations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional
import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Detection Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionConfig:
    """Tunable HSV ranges and contour morphology parameters."""

    # Dual HSV ranges for RED (handles 0°/180° wrap-around)
    red_lower1: tuple = (0, 120, 70)
    red_upper1: tuple = (10, 255, 255)
    red_lower2: tuple = (170, 120, 70)
    red_upper2: tuple = (180, 255, 255)

    # Tight HSV range for BLUE houses
    blue_lower: tuple = (100, 150, 50)
    blue_upper: tuple = (140, 255, 255)

    # Morphology kernel size
    morph_k: int = 5

    # Contour area thresholds in px² (filters out tiny text, dots, or huge backgrounds)
    min_area_red: int = 60
    max_area_red: int = 4000

    min_area_blue: int = 45
    max_area_blue: int = 3000

    # Minimum circularity (4*pi*Area / P^2) to reject text, legend boxes, or straight lines
    min_circularity: float = 0.35

    # Visual annotation styling
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.55
    font_thickness: int = 2
    label_offset_x: int = 12
    label_offset_y: int = -10

    radius_depot: int = 14
    radius_house: int = 12


@dataclass
class DetectedNode:
    """A detected stop on the map (Depot or Delivery House)."""
    node_id: str                          # "DEPOT", "H1", "H2", ...
    cx: int                               # Centroid X (pixels)
    cy: int                               # Centroid Y (pixels)
    is_depot: bool = False
    area: float = 0.0                     # Contour area in px²
    time_window: tuple = (0, 1440)        # [earliest_min, deadline_min]
    service_time: int = 5                 # Service duration at stop (minutes)
    address: str = ""                     # Human-readable street address or location name


# ──────────────────────────────────────────────────────────────────────────────
# Core Vision Processor
# ──────────────────────────────────────────────────────────────────────────────

class MapVisionProcessor:
    """
    OpenCV detector with robust HSV color masking and contour filtering.
    Guarantees the primary Red Depot is always Index 0.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.cfg = config or DetectionConfig()
        self._original_bgr: Optional[np.ndarray] = None
        self._labeled_bgr: Optional[np.ndarray] = None
        self.nodes: list[DetectedNode] = []
        self.primary_depot: Optional[DetectedNode] = None
        self.all_detected_depots: list[DetectedNode] = []
        self.houses: list[DetectedNode] = []

    @property
    def _original(self) -> Optional[np.ndarray]:
        return self._original_bgr

    @property
    def _labeled(self) -> Optional[np.ndarray]:
        return self._labeled_bgr

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, image_path: str) -> list[DetectedNode]:
        """Load image from disk and process."""
        img = self._load(image_path)
        return self.process_image(img)

    def process_image(self, img: np.ndarray) -> list[DetectedNode]:
        """
        Process an in-memory image array.
        Automatically checks and normalizes color format (RGB vs BGR) to prevent channel swapping.
        """
        # Ensure BGR color order
        img_bgr = self._normalize_to_bgr(img)

        # Scale down large maps to keep pathfinding and UI snappy
        h, w = img_bgr.shape[:2]
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

        self._original_bgr = img_bgr
        self._labeled_bgr = img_bgr.copy()

        # Convert normalized BGR to HSV
        hsv = cv2.cvtColor(self._original_bgr, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.cfg.morph_k, self.cfg.morph_k))

        # Detect primary Red Depot and Blue Houses
        self.all_detected_depots = self._detect_red_depots(hsv, kernel)
        self.houses = self._detect_blue_houses(hsv, kernel)

        # Ensure single primary depot is assigned to Index 0
        self.nodes = []
        if self.all_detected_depots:
            # Primary depot is the one with the largest circular area
            self.primary_depot = self.all_detected_depots[0]
            self.nodes.append(self.primary_depot)
        else:
            self.primary_depot = None

        self.nodes.extend(self.houses)

        # Draw annotations on _labeled_bgr
        self._annotate(self.primary_depot, self.houses)
        return self.nodes

    def get_labeled_image(self) -> Optional[np.ndarray]:
        """Return annotated BGR image."""
        return self._labeled_bgr

    def get_labeled_image_rgb(self) -> Optional[np.ndarray]:
        """Return annotated RGB image (for Streamlit or Kivy textures)."""
        if self._labeled_bgr is None:
            return None
        return cv2.cvtColor(self._labeled_bgr, cv2.COLOR_BGR2RGB)

    def draw_route(
        self,
        route_indices: list[int],
        node_subset: list[DetectedNode],
        color: tuple = (0, 235, 110),
        thickness: int = 4,
        path_cache: Optional[dict[tuple[int, int], list[tuple[int, int]]]] = None,
    ) -> np.ndarray:
        """
        Draw optimized route onto the labeled image.
        If path_cache is provided, traces along road pixels using A* paths.
        Otherwise, draws direct arrow lines.
        """
        # Start from original clean map if available so route lines sit UNDER node labels and badges
        if self._original_bgr is not None:
            canvas = self._original_bgr.copy()
        elif self._labeled_bgr is not None:
            canvas = self._labeled_bgr.copy()
        else:
            canvas = np.zeros((600, 800, 3), np.uint8)

        if len(route_indices) < 2:
            return canvas

        if path_cache is not None:
            from pathfinding import draw_road_following_route
            return draw_road_following_route(
                canvas,
                route_indices,
                node_subset,
                path_cache,
                color=color,
                line_thickness=thickness,
                show_steps=True,
            )

        # Direct line fallback
        n = len(route_indices)
        for i in range(n):
            a = node_subset[route_indices[i]]
            b = node_subset[route_indices[(i + 1) % n]]
            cv2.arrowedLine(canvas, (a.cx, a.cy), (b.cx, b.cy), color, thickness, tipLength=0.04)

        for step, idx in enumerate(route_indices):
            node = node_subset[idx]
            step_str = "D" if getattr(node, "is_depot", False) and step == 0 else str(step)
            cv2.putText(
                canvas,
                step_str,
                (node.cx + 5, node.cy - 12),
                self.cfg.font, 0.5, (255, 255, 255), 2, cv2.LINE_AA,
            )
        return canvas

    # ── Color & Contour Detection Helpers ─────────────────────────────────────

    @staticmethod
    def _normalize_to_bgr(img: np.ndarray) -> np.ndarray:
        """
        Inspect color distribution to detect if image is RGB or BGR,
        and return guaranteed BGR representation.
        """
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # In BGR: Red has high Ch2 & low Ch0; Blue has high Ch0 & low Ch2
        red_in_ch2 = np.sum((img[:, :, 2] > 160) & (img[:, :, 0] < 80) & (img[:, :, 1] < 80))
        blue_in_ch0 = np.sum((img[:, :, 0] > 160) & (img[:, :, 2] < 80) & (img[:, :, 1] < 120))

        # In RGB: Red has high Ch0 & low Ch2; Blue has high Ch2 & low Ch0
        red_in_ch0 = np.sum((img[:, :, 0] > 160) & (img[:, :, 2] < 80) & (img[:, :, 1] < 80))
        blue_in_ch2 = np.sum((img[:, :, 2] > 160) & (img[:, :, 0] < 80) & (img[:, :, 1] < 120))

        bgr_score = red_in_ch2 + blue_in_ch0
        rgb_score = red_in_ch0 + blue_in_ch2

        if rgb_score > bgr_score * 1.5:
            # Input was RGB — convert to BGR
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img.copy()

    def _extract_filtered_blobs(
        self,
        mask: np.ndarray,
        min_area: int,
        max_area: int,
        kernel: np.ndarray,
    ) -> list[tuple[int, int, float]]:
        """
        Morphologically cleans mask, extracts external contours, and filters
        by area and circularity to discard text strings and rectangular borders.
        Returns list of (cx, cy, area).
        """
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_blobs = []

        for c in contours:
            area = float(cv2.contourArea(c))
            if not (min_area <= area <= max_area):
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                circularity = 4.0 * math.pi * (area / (perimeter * perimeter))
                if circularity < self.cfg.min_circularity:
                    # Ignore non-circular shapes (text letters, lines, legend borders)
                    continue

            # Bounding box aspect ratio filter
            bx, by, bw, bh = cv2.boundingRect(c)
            aspect_ratio = float(bw) / float(bh) if bh > 0 else 0
            if not (0.4 <= aspect_ratio <= 2.5):
                continue

            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                valid_blobs.append((cx, cy, area))

        return valid_blobs

    def _detect_red_depots(self, hsv: np.ndarray, kernel: np.ndarray) -> list[DetectedNode]:
        """
        Detects red dots using dual HSV ranges.
        Ranks by area descending so the primary depot is always first.
        """
        # Range 1: 0..10 deg
        m1 = cv2.inRange(hsv, np.array(self.cfg.red_lower1), np.array(self.cfg.red_upper1))
        # Range 2: 170..180 deg
        m2 = cv2.inRange(hsv, np.array(self.cfg.red_lower2), np.array(self.cfg.red_upper2))
        mask_red = cv2.bitwise_or(m1, m2)

        blobs = self._extract_filtered_blobs(
            mask_red, self.cfg.min_area_red, self.cfg.max_area_red, kernel
        )

        if not blobs:
            return []

        # Sort by area descending — the largest circular red dot is the true depot
        blobs.sort(key=lambda b: b[2], reverse=True)

        depots = []
        for i, (cx, cy, area) in enumerate(blobs):
            node_id = "DEPOT" if i == 0 else f"DEPOT_{i+1}"
            depots.append(
                DetectedNode(
                    node_id=node_id,
                    cx=cx,
                    cy=cy,
                    is_depot=True,
                    area=area,
                    time_window=(0, 1440),
                )
            )
        return depots

    def _detect_blue_houses(self, hsv: np.ndarray, kernel: np.ndarray) -> list[DetectedNode]:
        """
        Detects blue delivery houses using tight HSV bounds.
        Sorts deterministically (top-to-bottom, left-to-right) and assigns H1, H2, ...
        """
        mask_blue = cv2.inRange(
            hsv, np.array(self.cfg.blue_lower), np.array(self.cfg.blue_upper)
        )

        blobs = self._extract_filtered_blobs(
            mask_blue, self.cfg.min_area_blue, self.cfg.max_area_blue, kernel
        )

        # Sort top-to-bottom, left-to-right for consistent labeling
        blobs.sort(key=lambda b: (b[1], b[0]))

        # Realistic business delivery schedule (09:00 - 18:00 standard depot day):
        # Default standard delivery window is 09:00 - 18:00 (540 - 1080 min).
        # A realistic subset of customers have scheduled priority appointment slots:
        #   Customer 1 (i=0): 09:00 - 12:00 (540, 720) [Morning priority]
        #   Customer 2 (i=1): 11:00 - 14:00 (660, 840) [Midday]
        #   Customer 3 (i=2): 13:00 - 17:00 (780, 1020) [Afternoon priority]
        #   Customer 4 (i=3): 14:00 - 18:00 (840, 1080) [Late afternoon]
        # All other customers (and any houses beyond i=3): Standard flexible 09:00 - 18:00 (540, 1080).
        # This eliminates the rigid 1-hour staggered slots and eliminates the modulo wrap-around bug (i % 8).
        PRIORITY_WINDOWS: dict[int, tuple[int, int]] = {
            0: (540, 720),   # 09:00 - 12:00 (Morning)
            1: (660, 840),   # 11:00 - 14:00 (Midday)
            2: (780, 1020),  # 13:00 - 17:00 (Afternoon)
            3: (840, 1080),  # 14:00 - 18:00 (Late Afternoon)
        }

        houses = []
        for i, (cx, cy, area) in enumerate(blobs):
            tw = PRIORITY_WINDOWS.get(i, (540, 1080))
            houses.append(
                DetectedNode(
                    node_id=f"H{i + 1}",
                    cx=cx,
                    cy=cy,
                    is_depot=False,
                    area=area,
                    time_window=tw,
                    service_time=5,
                )
            )
        return houses

    def _annotate(self, depot: Optional[DetectedNode], houses: list[DetectedNode]):
        """Annotate map with distinct depot badge and house labels with high-contrast text pill."""
        cfg = self.cfg
        canvas = self._labeled_bgr

        if depot:
            # Prominent Red Hub marker with white border
            cv2.circle(canvas, (depot.cx, depot.cy), cfg.radius_depot + 2, (255, 255, 255), 3)
            cv2.circle(canvas, (depot.cx, depot.cy), cfg.radius_depot, (0, 0, 230), -1)
            # High-contrast label pill for Depot
            tx, ty = depot.cx + cfg.label_offset_x, depot.cy + cfg.label_offset_y
            cv2.rectangle(canvas, (tx - 3, ty - 14), (tx + 95, ty + 4), (20, 20, 25), -1)
            cv2.putText(
                canvas,
                "DEPOT (Hub)",
                (tx, ty),
                cfg.font, cfg.font_scale, (0, 180, 255), cfg.font_thickness, cv2.LINE_AA,
            )

        for h in houses:
            # Blue Delivery House marker with white border
            cv2.circle(canvas, (h.cx, h.cy), cfg.radius_house + 2, (255, 255, 255), 2)
            cv2.circle(canvas, (h.cx, h.cy), cfg.radius_house, (230, 85, 10), -1)

            tw_str = f"{h.time_window[0]//60:02d}:{h.time_window[0]%60:02d}-{h.time_window[1]//60:02d}:{h.time_window[1]%60:02d}"
            hx, hy = h.cx + cfg.label_offset_x, h.cy + cfg.label_offset_y
            # Draw semi-transparent / solid dark backing pill for text legibility
            cv2.rectangle(canvas, (hx - 3, hy - 13), (hx + 86, hy + 18), (18, 18, 22), -1)
            cv2.rectangle(canvas, (hx - 3, hy - 13), (hx + 86, hy + 18), (70, 70, 80), 1)

            cv2.putText(
                canvas,
                h.node_id,
                (hx, hy),
                cfg.font, cfg.font_scale, (255, 190, 50), cfg.font_thickness, cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                tw_str,
                (hx, hy + 14),
                cfg.font, 0.38, (210, 210, 210), 1, cv2.LINE_AA,
            )

        self._labeled_bgr = canvas

    @staticmethod
    def _load(path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {path}")
        return img
