"""
main.py — Cross-Platform Mobile App Entry Point
Last-Mile Delivery Route Optimiser
-----------------------------------------
Framework : Kivy  (runs on Windows / Linux / macOS / Android / iOS)
Requires  : kivy, opencv-python, numpy

Run locally:
    python main.py

Build for Android:
    buildozer android debug   (see buildozer.spec)
"""

# ── Kivy must be imported BEFORE any kivy.* submodules ──────────────────────
import os
os.environ.setdefault("KIVY_NO_ENV_CONFIG", "1")

import kivy
kivy.require("2.3.0")

from kivy.app import App
from kivy.uix.boxlayout   import BoxLayout
from kivy.uix.gridlayout  import GridLayout
from kivy.uix.scrollview  import ScrollView
from kivy.uix.button      import Button
from kivy.uix.label       import Label
from kivy.uix.image       import Image as KivyImage
from kivy.uix.spinner     import Spinner
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.progressbar import ProgressBar
from kivy.uix.popup       import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.graphics.texture import Texture
from kivy.clock           import Clock
from kivy.core.window     import Window
from kivy.metrics         import dp

import time
import random
import threading
import numpy as np
import cv2

from vision import MapVisionProcessor, DetectedNode
from solver import TSPSolver, build_graph_from_detected_nodes, render_convergence_chart
from pathfinding import extract_road_mask, build_road_distance_matrix
from generate_sample_map import create_city_map

# ── Colour palette ────────────────────────────────────────────────────────────
C_PRIMARY   = (0.13, 0.59, 0.95, 1)   # Kivy blue
C_ACCENT    = (0.0,  0.78, 0.44, 1)   # emerald
C_WARN      = (0.95, 0.61, 0.07, 1)   # amber
C_DARK      = (0.11, 0.11, 0.14, 1)   # near-black
C_SURFACE   = (0.17, 0.17, 0.22, 1)   # dark card
C_ON_DARK   = (0.95, 0.95, 0.95, 1)   # light text
C_GRID_ALT  = (0.14, 0.14, 0.18, 1)

ALGO_NAMES = {
    "nn":     "Nearest Neighbour",
    "greedy": "Greedy Insertion",
    "2opt":   "2-opt",
    "hill":   "Hill Climbing",
    "sa":     "Simulated Annealing",
    "exact":  "Exact (Brute Force)",
}

# ──────────────────────────────────────────────────────────────────────────────
# Helper: numpy array → Kivy Texture
# ──────────────────────────────────────────────────────────────────────────────

def numpy_to_texture(img_rgb: np.ndarray) -> Texture:
    """Convert an H×W×3 uint8 RGB array to a Kivy Texture."""
    h, w = img_rgb.shape[:2]
    # Kivy expects bottom-up rows
    flipped = np.flipud(img_rgb)
    texture = Texture.create(size=(w, h), colorfmt="rgb")
    texture.blit_buffer(flipped.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
    return texture


# ──────────────────────────────────────────────────────────────────────────────
# Custom Widgets
# ──────────────────────────────────────────────────────────────────────────────

class StyledButton(Button):
    def __init__(self, bg_color=C_PRIMARY, **kwargs):
        super().__init__(**kwargs)
        self.background_normal  = ""
        self.background_color   = bg_color
        self.color              = C_ON_DARK
        self.bold               = True
        self.font_size          = dp(13)


class SectionLabel(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bold       = True
        self.color      = C_PRIMARY
        self.font_size  = dp(14)
        self.size_hint_y= None
        self.height     = dp(30)


class MetricsTable(GridLayout):
    """Renders a list[BenchmarkResult] as a styled grid with time accounting and optimality gaps."""

    HEADERS = ["Algorithm", "Dist(px)", "Drive(m)", "Wait(m)", "Late(m)", "Elapsed(m)", "Score", "CPU(ms)", "Gap(%)"]

    def __init__(self, **kwargs):
        super().__init__(cols=len(self.HEADERS), **kwargs)
        self.spacing  = [dp(1), dp(1)]
        self.padding  = dp(3)
        self.size_hint_y = None
        self.bind(minimum_height=self.setter("height"))
        self._draw_headers()

    def _cell(self, text, bold=False, bg=C_SURFACE, color=C_ON_DARK):
        lbl = Label(
            text       = str(text),
            bold       = bold,
            color      = color,
            font_size  = dp(10),
            size_hint_y= None,
            height     = dp(24),
        )
        with lbl.canvas.before:
            from kivy.graphics import Color, Rectangle
            Color(*bg)
            lbl._rect = Rectangle(pos=lbl.pos, size=lbl.size)
        lbl.bind(pos=lambda w, v: setattr(w._rect, "pos", v),
                 size=lambda w, v: setattr(w._rect, "size", v))
        return lbl

    def _draw_headers(self):
        for h in self.HEADERS:
            self.add_widget(self._cell(h, bold=True, bg=C_DARK, color=C_PRIMARY))

    def populate(self, results: dict):
        # Clear old rows (keep headers)
        children_snapshot = list(self.children)
        for w in children_snapshot[:-len(self.HEADERS)]:
            self.remove_widget(w)

        for i, (key, r) in enumerate(results.items()):
            bg = C_SURFACE if i % 2 == 0 else C_GRID_ALT
            gap_str = (
                f"{r.optimality_gap_pct:.1f}%"
                if r.optimality_gap_pct is not None
                else ("0.0% (Ref)" if key == "exact" and r.score < float("inf") else "N/A")
            )
            vals = [
                r.algorithm,
                f"{r.total_distance:.1f}",
                f"{r.travel_time_min:.1f}",
                f"{r.waiting_time_min:.1f}",
                f"{r.tw_penalty:.1f}",
                f"{r.total_elapsed_min:.1f}",
                f"{r.score:.1f}" if r.score < float("inf") else "Refused",
                f"{r.cpu_time_ms:.3f}",
                gap_str,
            ]
            for v in vals:
                self.add_widget(self._cell(v, bg=bg))


# ──────────────────────────────────────────────────────────────────────────────
# Main layout
# ──────────────────────────────────────────────────────────────────────────────

class DeliveryOptimizerLayout(BoxLayout):
    """Root widget — left panel (controls) + right panel (map display)."""

    def __init__(self, **kwargs):
        super().__init__(orientation="horizontal", **kwargs)
        self.spacing = dp(6)
        self.padding = dp(6)

        # State
        self._vision      = MapVisionProcessor()
        self._all_nodes:  list[DetectedNode] = []
        self._house_nodes: list[DetectedNode] = []
        self._toggle_btns: dict[str, ToggleButton] = {}
        self._results     = {}
        self._selected_algos: list[str] = list(ALGO_NAMES.keys())
        self._route_img: np.ndarray | None = None

        # Build sub-layouts
        self._build_left_panel()
        self._build_right_panel()

    # ── Left panel ────────────────────────────────────────────────────────────

    def _build_left_panel(self):
        left = BoxLayout(orientation="vertical", size_hint_x=0.38,
                         spacing=dp(5))

        # ── Upload section
        left.add_widget(SectionLabel(text="[1] Map Management"))
        map_btn_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(4))
        self._btn_upload = StyledButton(text="Upload Map", size_hint_x=0.5)
        self._btn_upload.bind(on_press=self._open_file_chooser)
        self._btn_random = StyledButton(text="Random Map", bg_color=C_ACCENT, size_hint_x=0.5)
        self._btn_random.bind(on_press=self._on_generate_random_map)
        map_btn_row.add_widget(self._btn_upload)
        map_btn_row.add_widget(self._btn_random)
        left.add_widget(map_btn_row)

        self._lbl_file = Label(text="sample_map.png (loaded)", color=C_WARN,
                               size_hint_y=None, height=dp(20),
                               font_size=dp(10))
        left.add_widget(self._lbl_file)

        # ── House selection
        left.add_widget(SectionLabel(text="[2] Select Delivery Houses"))
        # 4 columns and dedicated height ensures all houses (H1..H8/H12) are fully visible without clipping
        self._house_scroll = ScrollView(size_hint=(1, None), height=dp(80))
        self._house_grid   = GridLayout(cols=4, spacing=dp(3), padding=dp(2),
                                        size_hint_y=None)
        self._house_grid.bind(minimum_height=self._house_grid.setter("height"))
        self._house_scroll.add_widget(self._house_grid)
        left.add_widget(self._house_scroll)

        row = BoxLayout(size_hint_y=None, height=dp(34), spacing=dp(4))
        btn_all  = StyledButton(text="All",  bg_color=C_ACCENT)
        btn_none = StyledButton(text="None", bg_color=C_WARN)
        btn_all.bind( on_press=lambda *_: self._toggle_all(True))
        btn_none.bind(on_press=lambda *_: self._toggle_all(False))
        row.add_widget(btn_all)
        row.add_widget(btn_none)
        left.add_widget(row)

        # ── Algorithm selection
        left.add_widget(SectionLabel(text="[3] Algorithms"))
        self._algo_grid = GridLayout(cols=2, spacing=dp(3), padding=dp(2),
                                     size_hint_y=None, height=dp(110))
        self._algo_toggle_btns: dict[str, ToggleButton] = {}
        for key, label in ALGO_NAMES.items():
            tb = ToggleButton(text=label, state="down", font_size=dp(10),
                              size_hint_y=None, height=dp(28))
            # Use on_state which fires AFTER Kivy flips the state (fixes inverted bug)
            tb.bind(on_state=lambda w, val, k=key: self._on_algo_toggle(k, val))
            self._algo_toggle_btns[key] = tb
            self._algo_grid.add_widget(tb)
        left.add_widget(self._algo_grid)

        # ── Run button
        self._btn_run = StyledButton(text="Optimise Route",
                                     size_hint_y=None, height=dp(44),
                                     bg_color=C_ACCENT)
        self._btn_run.bind(on_press=self._on_run)
        left.add_widget(self._btn_run)

        # ── Progress
        self._progress = ProgressBar(max=100, value=0,
                                     size_hint_y=None, height=dp(12))
        left.add_widget(self._progress)
        self._lbl_status = Label(text="Ready.", color=C_ON_DARK,
                                 font_size=dp(10), size_hint_y=None,
                                 height=dp(16))
        left.add_widget(self._lbl_status)

        # ── Metrics table
        left.add_widget(SectionLabel(text="[4] Benchmark Results"))
        scroll = ScrollView(size_hint=(1, 1))
        self._metrics = MetricsTable()
        scroll.add_widget(self._metrics)
        left.add_widget(scroll)

        self.add_widget(left)

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right_panel(self):
        right = BoxLayout(orientation="vertical", size_hint_x=0.62)

        self._map_display = KivyImage(
            allow_stretch=True,
            keep_ratio=True,
        )
        # Placeholder texture
        placeholder = np.full((600, 800, 3), 30, dtype=np.uint8)
        cv2.putText(placeholder, "Upload a map to begin",
                    (160, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (120, 120, 120), 2, cv2.LINE_AA)
        self._map_display.texture = numpy_to_texture(placeholder)
        right.add_widget(self._map_display)

        # Route label below map
        self._lbl_route = Label(
            text="Optimized route will appear here.",
            color=C_ON_DARK, font_size=dp(11),
            size_hint_y=None, height=dp(40),
            halign="center", valign="middle",
        )
        self._lbl_route.bind(size=self._lbl_route.setter("text_size"))
        right.add_widget(self._lbl_route)

        self.add_widget(right)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _open_file_chooser(self, *_):
        content = BoxLayout(orientation="vertical")
        chooser = FileChooserListView(
            filters=["*.png", "*.jpg", "*.jpeg", "*.bmp"],
            path=os.path.expanduser("~"),
        )
        btn_row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        btn_ok     = StyledButton(text="Open",   bg_color=C_ACCENT)
        btn_cancel = StyledButton(text="Cancel", bg_color=C_WARN)
        btn_row.add_widget(btn_ok)
        btn_row.add_widget(btn_cancel)
        content.add_widget(chooser)
        content.add_widget(btn_row)

        popup = Popup(title="Select Map Image", content=content,
                      size_hint=(0.9, 0.9))

        def _on_open(*_):
            if chooser.selection:
                popup.dismiss()
                self._load_image(chooser.selection[0])

        btn_ok.bind(on_press=_on_open)
        btn_cancel.bind(on_press=popup.dismiss)
        popup.open()

    def _on_generate_random_map(self, *_):
        self._lbl_status.text = "Generating fresh random city map..."
        self._progress.value = 15
        self._btn_random.disabled = True
        # Save to generated_map.png so sample_map.png is never overwritten
        out_path = os.path.join(os.path.dirname(__file__), "generated_map.png")
        seed = int(time.time() * 1000) % 100000

        def _worker():
            try:
                create_city_map(out_path=out_path, n_houses=8, seed=seed)
                Clock.schedule_once(lambda dt: self._load_image(out_path))
            except Exception as e:
                Clock.schedule_once(lambda dt: self._show_error(f"Map generation failed: {e}"))
            finally:
                Clock.schedule_once(lambda dt: setattr(self._btn_random, "disabled", False))

        threading.Thread(target=_worker, daemon=True).start()

    def _load_image(self, path: str):
        self._current_map_path = path
        self._lbl_file.text = os.path.basename(path)
        self._lbl_status.text = "Processing image..."
        self._progress.value = 20

        def _worker():
            try:
                nodes = self._vision.process(path)
                Clock.schedule_once(lambda dt: self._on_vision_done(nodes))
            except Exception as e:
                Clock.schedule_once(lambda dt: self._show_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_vision_done(self, nodes: list[DetectedNode]):
        self._all_nodes   = nodes
        self._house_nodes = [n for n in nodes if not n.is_depot]
        self._rebuild_house_toggles()

        img_rgb = self._vision.get_labeled_image_rgb()
        if img_rgb is not None:
            self._map_display.texture = numpy_to_texture(img_rgb)

        depot_count = sum(1 for n in nodes if n.is_depot)
        house_count = len(self._house_nodes)
        self._lbl_status.text = (
            f"Detected {depot_count} depot + {house_count} houses. Ready."
        )
        self._progress.value = 40

    def _rebuild_house_toggles(self):
        self._house_grid.clear_widgets()
        self._toggle_btns.clear()
        for node in self._house_nodes:
            tb = ToggleButton(text=node.node_id, state="down",
                              font_size=dp(10),
                              size_hint_y=None, height=dp(28))
            self._toggle_btns[node.node_id] = tb
            self._house_grid.add_widget(tb)

    def _toggle_all(self, select: bool):
        for tb in self._toggle_btns.values():
            tb.state = "down" if select else "normal"

    def _on_algo_toggle(self, key: str, state: str):
        """Called by on_state AFTER Kivy has already flipped the toggle."""
        if state == "down":
            if key not in self._selected_algos:
                self._selected_algos.append(key)
        else:
            self._selected_algos = [a for a in self._selected_algos if a != key]

    def _on_run(self, *_):
        # Collect selected houses – read state at solve time (authoritative)
        selected_ids = [nid for nid, tb in self._toggle_btns.items()
                        if tb.state == "down"]
        if not selected_ids:
            self._show_error("Please select at least one delivery house.")
            return

        depot_nodes = [n for n in self._all_nodes if n.is_depot]
        house_nodes = [n for n in self._house_nodes if n.node_id in selected_ids]
        run_nodes   = depot_nodes + house_nodes

        if len(run_nodes) < 2:
            self._show_error("Need at least 2 nodes (depot + 1 house).")
            return

        # Re-derive selected algos from button states at solve time (authoritative)
        selected_algos = [
            k for k, tb in self._algo_toggle_btns.items() if tb.state == "down"
        ]
        if not selected_algos:
            self._show_error("Please select at least one algorithm.")
            return

        self._lbl_status.text = "Extracting roads & computing A* shortest paths..."
        self._progress.value  = 45
        self._btn_run.disabled = True

        def _worker():
            try:
                # 1. Load base image and extract road network mask
                base_img = getattr(self._vision, "_original", None)
                if base_img is None and hasattr(self, "_current_map_path"):
                    base_img = cv2.imread(self._current_map_path)

                node_coords = [(n.cx, n.cy) for n in run_nodes]
                road_mask = extract_road_mask(base_img, road_type="dark", threshold=130, node_coords=node_coords)

                # 2. Compute true A* road distance matrix & paths
                dist_matrix, path_cache = build_road_distance_matrix(run_nodes, road_mask, grid_scale=2)

                # 3. Solve TSP with A* distance matrix and reproducible seed
                graph  = build_graph_from_detected_nodes(run_nodes, dist_matrix=dist_matrix)
                solver = TSPSolver(graph, seed=42)
                results = solver.run_all(selected_algos)

                Clock.schedule_once(
                    lambda dt: self._on_solve_done(results, run_nodes, path_cache)
                )
            except Exception as e:
                Clock.schedule_once(lambda dt: self._show_error(str(e)))
            finally:
                Clock.schedule_once(lambda dt: setattr(self._btn_run, "disabled", False))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_solve_done(self, results: dict, run_nodes: list[DetectedNode], path_cache: dict):
        self._results = results
        self._progress.value = 90

        # Pick best heuristic result (excluding exact solver) to display and compute gap
        if results:
            heuristic_results = {k: r for k, r in results.items() if k != "exact" and r.score < float("inf")}
            if heuristic_results:
                best_heur_key = min(heuristic_results.keys(), key=lambda k: heuristic_results[k].score)
                best_heur = heuristic_results[best_heur_key]
            else:
                best_heur_key = min(results.keys(), key=lambda k: results[k].score)
                best_heur = results[best_heur_key]

            # Route to draw: best heuristic or exact if only exact was chosen
            best = best_heur if best_heur is not None else results.get("exact")

            route_img = self._vision.draw_route(best.route, run_nodes, path_cache=path_cache)

            # Render convergence chart (log scale evaluations, plotting HC and SA) and stack it below
            chart_bgr = render_convergence_chart(results, width=route_img.shape[1], height=280)
            combined_bgr = np.vstack([route_img, chart_bgr])

            combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)
            self._map_display.texture = numpy_to_texture(combined_rgb)

            # Build convergence summary text
            conv_lines = []
            for k, r in results.items():
                if r.score_history and len(r.score_history) > 1:
                    pct = ((r.score_history[0] - r.score_history[-1]) / r.score_history[0] * 100
                           if r.score_history[0] > 0 else 0.0)
                    conv_lines.append(
                        f"{r.algorithm}: {r.score_history[0]:.0f} -> {r.score_history[-1]:.0f}  "
                        f"({pct:.1f}% decr over {len(r.score_history)} evals)"
                    )

            conv_text = ""
            if conv_lines:
                conv_text = "\nImprovement: " + " | ".join(conv_lines)

            # Best Heuristic Gap: computed against Exact Global Optimum
            gap_text = ""
            if "exact" in results and results["exact"].feasible and best_heur is not None:
                exact_score = results["exact"].score
                if exact_score > 0:
                    gap = ((best_heur.score - exact_score) / exact_score) * 100.0
                    gap_text = f"\nExact Optimal: {exact_score:.1f}  |  Best Heuristic Gap: {gap:.2f}% ({best_heur.algorithm})"

            self._lbl_route.text = (
                f"Best: {best.algorithm}  |  "
                f"Road Dist: {best.total_distance:.1f}px  |  "
                f"Drive: {best.travel_time_min:.1f}m  |  "
                f"Wait: {best.waiting_time_min:.1f}m  |  "
                f"Elapsed: {best.total_elapsed_min:.1f}m\n"
                f"Route: {' -> '.join(best.route_labels)} -> {best.route_labels[0]}"
                + gap_text
                + conv_text
            )

        self._metrics.populate(results)
        self._lbl_status.text = "Optimisation complete (A* Road Routing) [OK]"
        self._progress.value  = 100

    # ── Error popup ───────────────────────────────────────────────────────────

    def _show_error(self, msg: str):
        content = BoxLayout(orientation="vertical", padding=dp(10))
        content.add_widget(Label(text=msg, color=(1, 0.3, 0.3, 1),
                                 font_size=dp(13)))
        btn = StyledButton(text="OK", bg_color=C_WARN,
                           size_hint_y=None, height=dp(40))
        popup = Popup(title="Error", content=content,
                      size_hint=(0.6, 0.35))
        btn.bind(on_press=popup.dismiss)
        content.add_widget(btn)
        popup.open()
        self._lbl_status.text = f"Error: {msg}"
        self._progress.value  = 0


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

class DeliveryOptimizerApp(App):
    title = "Last-Mile Delivery Route Optimiser"

    def build(self):
        Window.clearcolor = C_DARK
        Window.size = (1100, 700)   # desktop preview size; ignored on Android
        return DeliveryOptimizerLayout()

    def on_start(self):
        # Create a synthetic test map on first launch if none exists
        sample_path = os.path.join(os.path.dirname(__file__), "sample_map.png")
        if not os.path.exists(sample_path):
            self._create_sample_map(sample_path)
        if os.path.exists(sample_path) and hasattr(self, "root") and self.root:
            Clock.schedule_once(lambda dt: self.root._load_image(sample_path), 0.2)

    @staticmethod
    def _create_sample_map(path: str):
        create_city_map(out_path=path, n_houses=8)
        print(f"[app] Procedural city map created → {path}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DeliveryOptimizerApp().run()

