"""
app.py — Interactive Streamlit Application
Last-Mile Delivery Route Optimiser (Road-Aware TSP-TW)
-------------------------------------------------------
Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import time
from typing import Optional, Any
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from generate_sample_map import create_city_map
from pathfinding import (
    RoadGrid,
    build_road_distance_matrix,
    draw_road_following_route,
    extract_road_mask,
)
from solver import (
    BenchmarkResult,
    DeliveryGraph,
    NodeInfo,
    TSPSolver,
    benchmark_multi_seed,
    render_convergence_chart,
)
from vision import DetectedNode, MapVisionProcessor

# ──────────────────────────────────────────────────────────────────────────────
# Page Configuration & Styling
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Last-Mile Route Optimiser (Road-Aware TSP-TW)",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #f7f9fa;
        border-radius: 8px;
        padding: 12px 18px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 10px;
    }
    .badge-on-time {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-late {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────────────────────────────────────

if "map_image_bgr" not in st.session_state:
    img, mask, depot_pos, house_pts = create_city_map(out_path=None, n_houses=8, seed=42)
    st.session_state["map_image_bgr"] = img
    st.session_state["road_mask"] = mask
    st.session_state["map_source"] = "procedural"
    st.session_state["processor"] = None
    st.session_state["detected_nodes"] = None
    st.session_state["solve_results"] = None
    st.session_state["path_cache"] = None
    st.session_state["dist_matrix"] = None
    st.session_state["custom_stops"] = []


# ──────────────────────────────────────────────────────────────────────────────
# Helper: Vision Node Extraction with Session Caching
# ──────────────────────────────────────────────────────────────────────────────

def run_vision_detection(image: np.ndarray):
    """Run MapVisionProcessor with color format normalization."""
    processor = MapVisionProcessor()
    nodes = processor.process_image(image)
    return processor, nodes


def format_minutes_to_time(minutes: float) -> str:
    """Format minutes from midnight to HH:MM."""
    mins = int(round(minutes)) % 1440
    hr = mins // 60
    mn = mins % 60
    return f"{hr:02d}:{mn:02d}"


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar Controls
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Route Settings")
    st.markdown("---")

    # 1. Map Source Selection
    st.subheader("1. 🗺️ Map Input")
    map_mode = st.radio(
        "Choose Map Source:",
        ["🎲 Procedural City Map", "📁 Upload Custom Map"],
        index=0 if st.session_state.get("map_source") == "procedural" else 1,
    )

    if map_mode == "🎲 Procedural City Map":
        n_houses = st.slider("Number of Delivery Houses", min_value=3, max_value=12, value=8, step=1)
        seed_col1, seed_col2 = st.columns([2, 1])
        with seed_col1:
            custom_seed = st.number_input("Random Seed", value=42, step=1)
        with seed_col2:
            st.write("")
            st.write("")
            randomize = st.button("🎲 New")

        if randomize:
            new_seed = int(time.time() * 1000) % 100000
            img, mask, _, _ = create_city_map(out_path=None, n_houses=n_houses, seed=new_seed)
            st.session_state["map_image_bgr"] = img
            st.session_state["road_mask"] = mask
            st.session_state["map_source"] = "procedural"
            st.session_state["processor"] = None
            st.session_state["detected_nodes"] = None
            st.session_state["solve_results"] = None
            st.session_state["custom_stops"] = []
            st.rerun()

        gen_btn = st.button("🔄 Regenerate Map", width="stretch")
        if gen_btn:
            img, mask, _, _ = create_city_map(out_path=None, n_houses=n_houses, seed=int(custom_seed))
            st.session_state["map_image_bgr"] = img
            st.session_state["road_mask"] = mask
            st.session_state["map_source"] = "procedural"
            st.session_state["processor"] = None
            st.session_state["detected_nodes"] = None
            st.session_state["solve_results"] = None
            st.session_state["custom_stops"] = []
            st.rerun()

    else:
        uploaded_file = st.file_uploader(
            "Upload 2D Map (PNG / JPG / BMP)",
            type=["png", "jpg", "jpeg", "bmp"],
        )
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                st.session_state["map_image_bgr"] = img
                st.session_state["map_source"] = "upload"
                st.session_state["processor"] = None
                st.session_state["detected_nodes"] = None
                st.session_state["solve_results"] = None
                st.session_state["custom_stops"] = []

        road_color_type = st.selectbox(
            "Road Surface Contrast:",
            ["dark", "light"],
            format_func=lambda x: "Dark Roads on Light Map" if x == "dark" else "Light Roads on Dark Map",
        )
        road_thresh = st.slider("Road Detection Grayscale Threshold", min_value=30, max_value=240, value=130, step=5)

    st.markdown("---")
    st.subheader("2. 🔍 Detection Verification")

    current_map = st.session_state["map_image_bgr"]
    if st.session_state["processor"] is None or st.session_state["detected_nodes"] is None:
        processor, nodes = run_vision_detection(current_map)
        st.session_state["processor"] = processor
        st.session_state["detected_nodes"] = nodes
    else:
        processor = st.session_state["processor"]
        nodes = st.session_state["detected_nodes"]

    depots_detected = processor.all_detected_depots
    houses_detected = processor.houses

    if len(depots_detected) == 1:
        st.success(f"🟢 **Verified Red Depot**: 1 Hub at `({depots_detected[0].cx}, {depots_detected[0].cy})`")
    elif len(depots_detected) > 1:
        st.warning(
            f"⚠️ **Multiple Red Blobs**: {len(depots_detected)} found.\n"
            f"Primary Hub selected by largest area (`area={depots_detected[0].area:.0f}px²` at `({depots_detected[0].cx}, {depots_detected[0].cy})`)."
        )
    else:
        st.error("❌ **No Red Depot Detected!** Upload an image with a red depot or click to place one.")

    st.info(f"🔵 **Delivery Houses Detected**: {len(houses_detected)} stops")

    st.markdown("---")
    st.subheader("3. 🛣️ Schedule & Algorithms")

    # Unified Time & Physics Settings (D3, D4, FR-6)
    start_time_val = st.number_input(
        "Depot Departure Time (Minutes from 00:00, e.g. 540 = 09:00 AM):",
        min_value=0, max_value=1440, value=540, step=15,
        help="Unified departure time used identically by solver and turn-by-turn itinerary."
    )
    depot_close_val = st.number_input(
        "Depot Closing Time (Minutes from 00:00, e.g. 1080 = 18:00 PM):",
        min_value=0, max_value=1440, value=1080, step=15,
        help="Returns to depot after this time are penalized."
    )
    speed_val = st.slider("Vehicle Travel Speed (px/min):", min_value=20, max_value=120, value=60, step=5)
    hard_tw_mode = st.checkbox("Hard Time Windows (Reject late arrivals)", value=False)
    tw_weight = st.slider("Time Window Penalty Weight (λ):", min_value=0.5, max_value=5.0, value=1.5, step=0.5)

    grid_scale = st.select_slider(
        "A* Search Grid Resolution:",
        options=[1, 2, 3],
        value=2,
        format_func=lambda s: {1: "1x (Full Pixel Precision)", 2: "2x (Fast & Accurate)", 3: "3x (Ultra Fast)"}[s],
    )
    show_mask_preview = st.checkbox("👁️ Preview Extracted Road Mask", value=False)

    algo_options = {
        "nn": "Nearest Neighbour",
        "greedy": "Cheapest Insertion",
        "2opt": "2-opt Local Search",
        "hill": "Hill Climbing",
        "sa": "Simulated Annealing",
        "exact": "Exact Brute-Force / Held-Karp (≤ 12 stops)",
    }

    # Default to all algorithms (D9)
    selected_algos = st.multiselect(
        "Algorithms to Benchmark:",
        options=list(algo_options.keys()),
        default=list(algo_options.keys()),
        format_func=lambda k: algo_options[k],
    )

    opt_seed = st.number_input("Optimization Random Seed (for reproducibility):", value=42, step=1)


# ──────────────────────────────────────────────────────────────────────────────
# Main Application Content
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🚚 Last-Mile Delivery Route Optimiser</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Single-Depot Road-Aware TSP with Time Windows (TSP-TW) & Computer Vision &nbsp;|&nbsp; Start: <b>{format_minutes_to_time(start_time_val)}</b> &nbsp;|&nbsp; Depot Close: <b>{format_minutes_to_time(depot_close_val)}</b></div>',
    unsafe_allow_html=True,
)

depot_nodes = [n for n in nodes if n.is_depot]
house_nodes = [n for n in nodes if not n.is_depot]

# Metrics Row
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    depot_str = f"Hub at ({depot_nodes[0].cx}, {depot_nodes[0].cy})" if depot_nodes else "None"
    st.metric("Verified Red Depot", depot_str)
with col_m2:
    st.metric("Total Delivery Stops", f"{len(house_nodes)} Stops")
with col_m3:
    st.metric("Road Routing Engine", f"A* 8-Way (Scale: {grid_scale}x)")
with col_m4:
    opt_status = "Optimized ✓" if st.session_state.get("solve_results") else "Ready to Optimize"
    st.metric("Optimization Status", opt_status)

st.markdown("---")


# ──────────────────────────────────────────────────────────────────────────────
# Stop & Address Management (FR-1, FR-2, FR-3, User Prompt Constraint)
# ──────────────────────────────────────────────────────────────────────────────

with st.expander("📍 **Manage Stops, Addresses & Time Windows** (Input Custom Addresses / CSV Import)", expanded=False):
    tab_input, tab_table, tab_csv = st.tabs(["➕ Add Custom Stop", "📝 Edit Time Windows & Service Times", "📁 Import Stops CSV"])

    with tab_input:
        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            new_id = st.text_input("Stop ID / Label:", value=f"H{len(house_nodes)+1}")
            new_address = st.text_input("Street Address:", value=f"{len(house_nodes)*12 + 10} Market St")
        with col_in2:
            new_x = st.number_input("Pixel X Coordinate:", min_value=10, max_value=1190, value=300, step=10)
            new_y = st.number_input("Pixel Y Coordinate:", min_value=10, max_value=1190, value=300, step=10)
        with col_in3:
            new_early = st.number_input("Earliest Window (min from 00:00):", min_value=0, max_value=1440, value=570, step=15)
            new_late = st.number_input("Latest Window (min from 00:00):", min_value=0, max_value=1440, value=660, step=15)
            new_service = st.number_input("Service Duration (min):", min_value=1, max_value=60, value=5, step=1)

        if st.button("➕ Add Stop to Delivery Network"):
            new_node = DetectedNode(
                node_id=new_id,
                cx=int(new_x),
                cy=int(new_y),
                is_depot=False,
                area=500.0,
                time_window=(int(new_early), int(new_late)),
                service_time=int(new_service),
                address=new_address,
            )
            # Add to nodes
            st.session_state["detected_nodes"].append(new_node)
            st.session_state["solve_results"] = None
            st.success(f"Added stop '{new_id}' at ({new_x}, {new_y}) with window {format_minutes_to_time(new_early)} - {format_minutes_to_time(new_late)}")
            st.rerun()

    with tab_table:
        st.caption("Edit delivery time windows and service times directly. Validates earliest <= latest.")
        # Build dataframe for data_editor
        table_rows = []
        for n in house_nodes:
            table_rows.append({
                "Stop ID": n.node_id,
                "Address": getattr(n, "address", "") or f"Delivery Location {n.node_id}",
                "X (px)": n.cx,
                "Y (px)": n.cy,
                "Earliest (min)": n.time_window[0],
                "Latest (min)": n.time_window[1],
                "Service Time (min)": n.service_time,
            })
        df_editor = pd.DataFrame(table_rows)
        edited_df = st.data_editor(df_editor, width="stretch", key="stops_editor")

        if st.button("💾 Apply Updated Time Windows"):
            # Update in-memory nodes
            for _, row in edited_df.iterrows():
                target_id = row["Stop ID"]
                for n in house_nodes:
                    if n.node_id == target_id:
                        e_min = int(row["Earliest (min)"])
                        l_min = int(row["Latest (min)"])
                        if e_min > l_min:
                            st.warning(f"Warning: {target_id} earliest ({e_min}) > latest ({l_min}). Swap applied.")
                            e_min, l_min = l_min, e_min
                        n.time_window = (e_min, l_min)
                        n.service_time = int(row["Service Time (min)"])
                        n.address = str(row["Address"])
            st.session_state["solve_results"] = None
            st.success("Time windows and service times updated successfully!")
            st.rerun()

    with tab_csv:
        st.caption("Upload a CSV file with columns: `id`, `x`, `y`, `earliest`, `latest`, `service_time`, `address`.")
        csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv_import")
        if csv_file is not None:
            try:
                uploaded_csv = pd.read_csv(csv_file)
                st.dataframe(uploaded_csv.head(5), width="stretch")
                if st.button("📥 Import Stops from CSV"):
                    for _, r in uploaded_csv.iterrows():
                        sid = str(r.get("id", f"H{len(house_nodes)+1}"))
                        sx = int(r.get("x", 200))
                        sy = int(r.get("y", 200))
                        e = int(r.get("earliest", 540))
                        l = int(r.get("latest", 720))
                        srv = int(r.get("service_time", 5))
                        addr = str(r.get("address", sid))
                        st.session_state["detected_nodes"].append(
                            DetectedNode(
                                node_id=sid, cx=sx, cy=sy, is_depot=False,
                                time_window=(e, l), service_time=srv, address=addr
                            )
                        )
                    st.session_state["solve_results"] = None
                    st.success(f"Imported {len(uploaded_csv)} stops from CSV!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error reading CSV: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Interactive House Selection
# ──────────────────────────────────────────────────────────────────────────────

st.subheader("📍 Select Delivery Destinations")
house_ids = [h.node_id for h in house_nodes]

col_check1, col_check2 = st.columns([4, 1])
with col_check2:
    st.write("")
    btn_all = st.button("Select All", width="stretch")
    btn_none = st.button("Clear All", width="stretch")

with col_check1:
    default_selected = house_ids if not btn_none else []
    if btn_all:
        default_selected = house_ids

    selected_house_ids = st.multiselect(
        "Included Delivery Stops:",
        options=house_ids,
        default=default_selected if not btn_none else [],
        format_func=lambda hid: f"{hid} ({getattr(next((h for h in house_nodes if h.node_id == hid), None), 'address', '') or 'Stop'} | Window: {format_minutes_to_time(next((h for h in house_nodes if h.node_id == hid), None).time_window[0])} - {format_minutes_to_time(next((h for h in house_nodes if h.node_id == hid), None).time_window[1])})",
    )

# Form the run nodes: Verified Red Depot strictly at Index 0, followed by chosen houses
selected_run_nodes = (depot_nodes[:1] if depot_nodes else []) + [h for h in house_nodes if h.node_id in selected_house_ids]

# Optimize Button
col_btn1, col_btn2 = st.columns([1, 3])
with col_btn1:
    run_opt = st.button("🚀 Optimize Delivery Route", type="primary", width="stretch")

if run_opt:
    if not depot_nodes:
        st.error("No verified Red Depot found! Cannot optimize without a starting hub.")
    elif len(selected_run_nodes) < 2:
        st.error("Please select at least 1 delivery house to form a route.")
    elif not selected_algos:
        st.error("Please select at least 1 algorithm in the sidebar.")
    else:
        with st.spinner("Extracting road network & computing A* shortest paths strictly from Red Depot..."):
            t_start = time.perf_counter()

            # Road mask extraction
            node_coords = [(n.cx, n.cy) for n in selected_run_nodes]
            if st.session_state.get("map_source") == "procedural" and "road_mask" in st.session_state:
                road_mask = st.session_state["road_mask"].copy()
                for cx, cy in node_coords:
                    cv2.circle(road_mask, (int(cx), int(cy)), 18, 255, -1)
            else:
                road_type_val = road_color_type if "road_color_type" in locals() else "dark"
                thresh_val = road_thresh if "road_thresh" in locals() else 130
                road_mask = extract_road_mask(
                    current_map,
                    road_type=road_type_val,
                    threshold=thresh_val,
                    node_coords=node_coords,
                )

            st.session_state["current_road_mask"] = road_mask

            # Pairwise A* road distances (Index 0 is guaranteed to be the primary Red Depot)
            dist_matrix, path_cache = build_road_distance_matrix(
                selected_run_nodes,
                road_mask,
                grid_scale=grid_scale,
            )
            st.session_state["dist_matrix"] = dist_matrix
            st.session_state["path_cache"] = path_cache

            # Graph & Solver with unified start time, depot closing, and speed
            graph = DeliveryGraph(
                [
                    NodeInfo(
                        node_id=n.node_id,
                        x=float(n.cx),
                        y=float(n.cy),
                        time_window=n.time_window,
                        service_time=getattr(n, "service_time", 5),
                        address=getattr(n, "address", ""),
                    )
                    for n in selected_run_nodes
                ],
                dist_matrix=dist_matrix,
                start_time=start_time_val,
                depot_close_time=depot_close_val,
                speed_px_per_min=speed_val,
                hard_windows=hard_tw_mode,
            )
            solver = TSPSolver(graph, tw_weight=tw_weight, seed=int(opt_seed))
            results = solver.run_all(selected_algos)

            st.session_state["solve_results"] = results
            st.session_state["last_run_nodes"] = selected_run_nodes
            st.session_state["last_graph"] = graph

        st.success(f"Optimisation complete in {(time.perf_counter() - t_start):.2f}s!")
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Visual Display & Benchmarking Tabs
# ──────────────────────────────────────────────────────────────────────────────

tab_map, tab_metrics, tab_conv, tab_multi, tab_itinerary = st.tabs([
    "🗺️ Route Map",
    "📊 Benchmark Comparison",
    "📈 Convergence Curves",
    "🔬 Multi-Seed Study",
    "📋 Turn-by-Turn Manifest",
])

# ── Tab 1: Route Map ──────────────────────────────────────────────────────────
with tab_map:
    proc = st.session_state.get("processor") or processor

    if st.session_state.get("solve_results") and st.session_state.get("path_cache"):
        results = st.session_state["solve_results"]
        path_cache = st.session_state["path_cache"]
        last_nodes = st.session_state["last_run_nodes"]

        # Dropdown to choose which algorithm's route to display
        col_sel_algo, _ = st.columns([2, 3])
        with col_sel_algo:
            view_algo_key = st.selectbox(
                "Display Route for Algorithm:",
                options=[k for k in results.keys() if results[k].score < float("inf")],
                index=0,
                format_func=lambda k: f"{results[k].algorithm} (Road Dist: {results[k].total_distance:.1f}px | Score: {results[k].score:.1f})",
            )

        chosen_result = results[view_algo_key]
        route_display_bgr = proc.draw_route(
            chosen_result.route,
            last_nodes,
            color=(0, 235, 110),
            thickness=4,
            path_cache=path_cache,
        )
        route_display_rgb = cv2.cvtColor(route_display_bgr, cv2.COLOR_BGR2RGB)

        st.image(
            route_display_rgb,
            caption=f"Optimized Road Route: {chosen_result.algorithm} (Starts & Ends at Verified Red Depot via A*)",
            width="stretch",
        )

        is_success, buffer = cv2.imencode(".png", route_display_bgr)
        if is_success:
            st.download_button(
                label="📥 Download Route Map (PNG)",
                data=buffer.tobytes(),
                file_name="optimized_road_delivery_route.png",
                mime="image/png",
            )

    else:
        labeled_rgb = proc.get_labeled_image_rgb()
        if labeled_rgb is not None:
            st.image(
                labeled_rgb,
                caption="Detected Primary Red Depot (Index 0) and Blue Delivery Houses with Time Windows",
                width="stretch",
            )

    if show_mask_preview and "current_road_mask" in st.session_state:
        st.subheader("Walkable Road Mask (White = Road, Black = Buildings)")
        st.image(st.session_state["current_road_mask"], caption="A* Walkable Road Grid", width="stretch")


# ── Tab 2: Performance Benchmark ──────────────────────────────────────────────
with tab_metrics:
    st.subheader("⚡ Performance Benchmark (Single Run)")

    if st.session_state.get("solve_results"):
        results = st.session_state["solve_results"]

        rows = []
        for k, r in results.items():
            gap_str = f"{r.optimality_gap_pct:.2f}%" if r.optimality_gap_pct is not None else ("0.00% (Ref)" if k == "exact" and r.score < float("inf") else "N/A")
            feas_str = "✅ Yes" if r.feasible else f"❌ No ({len(r.late_stops)} late)"
            late_str = ", ".join(r.late_stops) if r.late_stops else "None"

            rows.append({
                "Algorithm": r.algorithm,
                "Road Distance (px)": f"{r.total_distance:.1f}",
                "Drive Time (min)": f"{r.travel_time_min:.1f}",
                "Wait Time (min)": f"{r.waiting_time_min:.1f}",
                "Lateness (min)": f"{r.tw_penalty:.1f}",
                "Total Elapsed Time (min)": f"{r.total_elapsed_min:.1f}",
                "Combined Score": f"{r.score:.1f}" if r.score < float("inf") else "Refused",
                "Feasible?": feas_str,
                "Late Stops": late_str,
                "Max Lateness (min)": f"{r.max_lateness_min:.1f}",
                "CPU Time (ms)": f"{r.cpu_time_ms:.3f}",
                "Evaluations": r.evaluations,
                "Optimality Gap vs Exact": gap_str,
            })

        df_bench = pd.DataFrame(rows)
        st.dataframe(df_bench, width="stretch")

        # CSV Download for benchmark
        csv_buffer = df_bench.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Benchmark Results as CSV",
            data=csv_buffer,
            file_name="benchmark_results.csv",
            mime="text/csv",
        )

        # Compute Best HEURISTIC Solution (strictly excluding the exact solver)
        heur_results = {k: r for k, r in results.items() if k != "exact" and r.score < float("inf")}
        if heur_results:
            best_algo_key = min(heur_results.keys(), key=lambda k: heur_results[k].score)
            best_r = heur_results[best_algo_key]

            gap_info = ""
            if "exact" in results and results["exact"].feasible and results["exact"].score < float("inf"):
                ex_score = results["exact"].score
                if ex_score > 0:
                    gap_pct = ((best_r.score - ex_score) / ex_score) * 100.0
                    gap_info = f"&nbsp;|&nbsp; <b>Optimality Gap vs Exact:</b> <span style='color:#00C853; font-weight:700;'>{gap_pct:.2f}%</span>"

            st.markdown(
                f"""
                <div class="metric-card">
                    <b>🏆 Best Heuristic Solution:</b> <span style="color:#1E88E5; font-size:1.1rem; font-weight:700;">{best_r.algorithm}</span><br>
                    <b>Starting Hub:</b> Verified Red Depot (Index 0) &nbsp;|&nbsp;
                    <b>Road Distance:</b> {best_r.total_distance:.1f} px ({best_r.travel_time_min:.1f} min drive) &nbsp;|&nbsp;
                    <b>Waiting Time:</b> {best_r.waiting_time_min:.1f} min &nbsp;|&nbsp;
                    <b>Total Mission Duration:</b> {best_r.total_elapsed_min:.1f} min &nbsp;|&nbsp;
                    <b>Lateness:</b> {best_r.tw_penalty:.1f} min &nbsp;|&nbsp;
                    <b>Feasible:</b> {'✅ Yes' if best_r.feasible else '❌ No'} &nbsp;|&nbsp;
                    <b>Solve Runtime:</b> {best_r.cpu_time_ms:.2f} ms
                    {gap_info}
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        st.info("Click **'🚀 Optimize Delivery Route'** above to run and view benchmark metrics across all algorithms.")


# ── Tab 3: Evaluation-Based Convergence Curves (D6, FR-13) ───────────────────
with tab_conv:
    st.subheader("📈 Solution Improvement vs. Candidate Route Evaluations")
    st.caption(
        "Shows solution score progression on a shared horizontal axis of route evaluations. "
        "Lower is better. Includes exact global optimum as a benchmark reference line."
    )

    if st.session_state.get("solve_results"):
        results = st.session_state["solve_results"]

        # Render OpenCV convergence chart
        chart_img = render_convergence_chart(results, width=850, height=420)
        st.image(chart_img, caption="Evaluation-Based Convergence Comparison (Shared Evaluations Axis)", width="stretch")

        # Summary improvement table
        conv_summary = []
        for k, r in results.items():
            if r.score_history and len(r.score_history) > 1:
                initial = r.score_history[0]
                final = r.score_history[-1]
                impr_pct = ((initial - final) / initial * 100) if initial > 0 else 0.0
                conv_summary.append({
                    "Algorithm": r.algorithm,
                    "Starting Score": f"{initial:.1f}",
                    "Final Best Score": f"{final:.1f}",
                    "Total Improvement": f"{initial - final:.1f} pts ({impr_pct:.1f}%)",
                    "Total Route Evaluations": r.evaluations,
                })
        if conv_summary:
            st.dataframe(pd.DataFrame(conv_summary), width="stretch")
    else:
        st.info("Run optimization to generate convergence tracking curves.")


# ── Tab 4: Multi-Seed Statistical Validation (FR-11, FR-12) ───────────────────
with tab_multi:
    st.subheader("🔬 Multi-Seed Validation Study (Statistical Comparison)")
    st.caption("Executes stochastic methods over multiple random seeds to measure mean, std dev, best, worst, and mean gap to exact.")

    if st.session_state.get("last_graph"):
        graph = st.session_state["last_graph"]
        n_seeds_choice = st.slider("Number of Random Seeds to Evaluate:", min_value=5, max_value=30, value=15, step=5)

        if st.button("🚀 Run Multi-Seed Validation Benchmark"):
            with st.spinner(f"Benchmarking across {n_seeds_choice} random seeds..."):
                multi_summary = benchmark_multi_seed(graph, n_seeds=n_seeds_choice, tw_weight=tw_weight)

                summary_rows = []
                for k, s in multi_summary.items():
                    mgap_str = f"{s['mean_gap_pct']:.2f}%" if s['mean_gap_pct'] is not None else ("0.00% (Ref)" if k == "exact" else "N/A")
                    summary_rows.append({
                        "Algorithm": s["algorithm"],
                        "Runs": s["runs"],
                        "Mean Score": f"{s['mean_score']:.1f}",
                        "Std Dev": f"±{s['std_score']:.1f}",
                        "Best Score": f"{s['best_score']:.1f}",
                        "Worst Score": f"{s['worst_score']:.1f}",
                        "Feasibility %": f"{s['feasibility_pct']:.1f}%",
                        "Mean Road Dist (px)": f"{s['mean_distance']:.1f}",
                        "Mean Lateness (min)": f"{s['mean_lateness']:.1f}",
                        "Mean CPU (ms)": f"{s['mean_cpu_ms']:.2f}",
                        "Mean Gap vs Exact": mgap_str,
                    })

                df_multi = pd.DataFrame(summary_rows)
                st.dataframe(df_multi, width="stretch")

                csv_multi = df_multi.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Multi-Seed Validation Study (CSV)",
                    data=csv_multi,
                    file_name="multiseed_validation_study.csv",
                    mime="text/csv",
                )
    else:
        st.info("Run optimization on a delivery map first to initialize the graph for multi-seed benchmarking.")


# ── Tab 5: Turn-by-Turn Delivery Manifest (D4) ─────────────────────────────────
with tab_itinerary:
    st.subheader("📋 Turn-by-Turn Delivery Schedule & Manifest")

    if st.session_state.get("solve_results"):
        results = st.session_state["solve_results"]
        last_nodes = st.session_state["last_run_nodes"]
        dist_matrix = st.session_state["dist_matrix"]

        valid_keys = [k for k in results.keys() if results[k].score < float("inf")]
        best_algo_key = min(valid_keys, key=lambda k: results[k].score)
        best_r = results[best_algo_key]
        route = best_r.route

        itinerary_data = []
        current_time_min = float(start_time_val)

        for i in range(len(route)):
            curr_idx = route[i]
            next_idx = route[(i + 1) % len(route)]

            curr_node = last_nodes[curr_idx]
            next_node = last_nodes[next_idx]

            leg_dist = dist_matrix[curr_idx][next_idx]
            travel_time = leg_dist / speed_val

            arrival_time_min = current_time_min + travel_time
            earliest, latest = next_node.time_window
            is_depot_return = next_node.is_depot and i == len(route) - 1

            if is_depot_return:
                if arrival_time_min <= depot_close_val:
                    status_badge = f'<span class="badge-on-time">Depot Return ✓ (Before {format_minutes_to_time(depot_close_val)})</span>'
                else:
                    over = int(arrival_time_min - depot_close_val)
                    status_badge = f'<span class="badge-late">Depot Late by {over}m ⚠️</span>'
            elif arrival_time_min < earliest:
                wait_m = int(earliest - arrival_time_min)
                status_badge = f'<span class="badge-on-time">Early (Wait {wait_m}m)</span>'
                current_time_min = float(earliest)
            elif arrival_time_min > latest:
                lateness = int(arrival_time_min - latest)
                status_badge = f'<span class="badge-late">Late by {lateness}m ⚠️</span>'
                current_time_min = arrival_time_min
            else:
                status_badge = '<span class="badge-on-time">On-Time ✓</span>'
                current_time_min = arrival_time_min

            dest_addr = getattr(next_node, "address", "") or next_node.node_id
            window_str = f"Depot Close ({format_minutes_to_time(depot_close_val)})" if is_depot_return else f"{format_minutes_to_time(earliest)} - {format_minutes_to_time(latest)}"

            itinerary_data.append({
                "Leg": f"{i + 1}",
                "From → To": f"{curr_node.node_id} → {next_node.node_id}",
                "Destination Address": dest_addr,
                "Road Distance (px)": f"{leg_dist:.1f}",
                "Travel Time (min)": f"{travel_time:.1f}",
                "Est. Arrival": format_minutes_to_time(arrival_time_min),
                "Customer Window": window_str,
                "Delivery Status": status_badge,
            })
            current_time_min += getattr(next_node, "service_time", 5)

        st.write(f"**Optimal Route Tour ({best_r.algorithm}):** `{' → '.join(best_r.route_labels)} → {best_r.route_labels[0]}`")

        # HTML Table
        table_html = """
        <table style="width:100%; border-collapse: collapse; margin-top: 10px;">
            <thead>
                <tr style="border-bottom: 2px solid #ddd; text-align: left;">
                    <th style="padding: 8px;">Leg</th>
                    <th style="padding: 8px;">From → To</th>
                    <th style="padding: 8px;">Destination Address</th>
                    <th style="padding: 8px;">Road Dist</th>
                    <th style="padding: 8px;">Travel Time</th>
                    <th style="padding: 8px;">Est. Arrival</th>
                    <th style="padding: 8px;">Customer Window</th>
                    <th style="padding: 8px;">Status</th>
                </tr>
            </thead>
            <tbody>
        """ + "".join(
            f"""
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 8px;">{d['Leg']}</td>
                <td style="padding: 8px; font-weight:600;">{d['From → To']}</td>
                <td style="padding: 8px;">{d['Destination Address']}</td>
                <td style="padding: 8px;">{d['Road Distance (px)']} px</td>
                <td style="padding: 8px;">{d['Travel Time (min)']} min</td>
                <td style="padding: 8px;">{d['Est. Arrival']}</td>
                <td style="padding: 8px;">{d['Customer Window']}</td>
                <td style="padding: 8px;">{d['Delivery Status']}</td>
            </tr>
            """
            for d in itinerary_data
        ) + "</tbody></table>"

        st.markdown(table_html, unsafe_allow_html=True)

        # CSV Download for manifest
        df_manifest = pd.DataFrame(itinerary_data)
        # Strip HTML from status for clean CSV
        df_manifest["Delivery Status"] = df_manifest["Delivery Status"].str.replace(r'<[^>]*>', '', regex=True)
        csv_manifest = df_manifest.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Turn-by-Turn Manifest as CSV",
            data=csv_manifest,
            file_name="delivery_manifest.csv",
            mime="text/csv",
        )

    else:
        st.info("Run optimization to generate turn-by-turn delivery schedule.")
