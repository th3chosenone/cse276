"""
Last-Mile Delivery Route Optimiser – FastAPI backend.
Purely stateless serverless API for Vercel deployment.
Implements the exact computational and visualization pipeline of app.py.
"""

from __future__ import annotations

import base64
import os
import random
import sys
import time
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from generate_sample_map import create_city_map  # noqa: E402
from pathfinding import (  # noqa: E402
    build_road_distance_matrix,
    extract_road_mask,
)
from solver import (  # noqa: E402
    BenchmarkResult,
    DeliveryGraph,
    NodeInfo,
    TSPSolver,
    benchmark_multi_seed,
    render_convergence_chart,
)
from vision import DetectedNode, MapVisionProcessor  # noqa: E402

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------
app = FastAPI(title="Last-Mile Delivery Route Optimiser API", version="2.0.0")
router = APIRouter()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _decode_b64_image(b64str: str) -> np.ndarray:
    """Decode base64 string (with or without data-uri prefix) to BGR image."""
    try:
        if "," in b64str:
            b64str = b64str.split(",", 1)[1]
        raw = base64.b64decode(b64str)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image bytes")
        return img
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {exc}") from exc


def _encode_bgr_to_b64_png(img: np.ndarray) -> str:
    """Encode BGR numpy array to base64 PNG string."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image to PNG")
    return base64.b64encode(buf.tobytes()).decode()


def _node_to_dict(n: DetectedNode) -> dict:
    return {
        "id": n.node_id,
        "cx": int(n.cx),
        "cy": int(n.cy),
        "is_depot": bool(n.is_depot),
        "time_window": [int(n.time_window[0]), int(n.time_window[1])],
        "service_time": int(n.service_time),
        "address": str(n.address or ""),
        "area": float(n.area) if hasattr(n, "area") else 0.0,
    }


def _dict_to_node(d: dict) -> DetectedNode:
    tw = d.get("time_window", [540, 1080])
    return DetectedNode(
        node_id=str(d["id"]),
        cx=int(d["cx"]),
        cy=int(d["cy"]),
        is_depot=bool(d.get("is_depot", False)),
        area=float(d.get("area", 0.0)),
        time_window=(int(tw[0]), int(tw[1])),
        service_time=int(d.get("service_time", 5)),
        address=str(d.get("address", "")),
    )


def _format_minutes_to_time(minutes: float) -> str:
    mins = int(round(minutes)) % 1440
    hr = mins // 60
    mn = mins % 60
    return f"{hr:02d}:{mn:02d}"


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class GenerateMapRequest(BaseModel):
    n_houses: int = Field(default=8, ge=3, le=12)
    seed: Optional[int] = None


class OptimizeRequest(BaseModel):
    map_image: str  # base64
    seed: Optional[int] = None
    nodes: list[dict]
    selected_ids: list[str]
    algorithms: list[str] = Field(
        default=["nn", "greedy", "2opt", "hill", "sa", "exact"]
    )
    start_time: float = 540.0
    depot_close: float = 1080.0
    speed: float = 60.0
    hard_windows: bool = False
    tw_weight: float = 1.5
    grid_scale: int = 2
    opt_seed: int = 42
    road_type: str = "dark"
    road_thresh: int = 130


class MultiSeedRequest(BaseModel):
    map_image: str  # base64
    seed: Optional[int] = None
    nodes: list[dict]
    selected_ids: list[str]
    n_seeds: int = Field(default=15, ge=5, le=30)
    start_time: float = 540.0
    depot_close: float = 1080.0
    speed: float = 60.0
    hard_windows: bool = False
    tw_weight: float = 1.5
    grid_scale: int = 2
    road_type: str = "dark"
    road_thresh: int = 130


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/generate-map")
def generate_map(req: GenerateMapRequest):
    """
    Generate a procedural city map in memory.
    If seed is not provided, pick a random seed like app.py does.
    """
    chosen_seed = req.seed if req.seed is not None else int(time.time() * 1000) % 100000

    # Procedural map generation in RAM (no disk writing)
    canvas, road_mask, depot_pos, house_pts = create_city_map(
        out_path=None,
        n_houses=req.n_houses,
        seed=chosen_seed,
    )

    processor = MapVisionProcessor()
    detected_nodes = processor.process_image(canvas)

    depot = next((n for n in detected_nodes if n.is_depot), None)
    houses = [n for n in detected_nodes if not n.is_depot]

    labeled_bgr = processor.get_labeled_image()
    if labeled_bgr is None:
        labeled_bgr = canvas

    return {
        "seed": chosen_seed,
        "n_houses": len(houses),
        "map_image": _encode_bgr_to_b64_png(canvas),
        "labeled_image": _encode_bgr_to_b64_png(labeled_bgr),
        "road_mask": _encode_bgr_to_b64_png(road_mask),
        "depot": {"cx": depot.cx, "cy": depot.cy} if depot else None,
        "all_depots_count": len(processor.all_detected_depots),
        "houses": [_node_to_dict(h) for h in houses],
    }


@router.post("/upload-map")
async def upload_map(
    file: UploadFile = File(...),
    road_type: str = Form("dark"),
    road_thresh: int = Form(130),
):
    """Upload custom map and detect nodes."""
    file_bytes = await file.read()
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image file.")

    processor = MapVisionProcessor()
    detected_nodes = processor.process_image(img)

    depot = next((n for n in detected_nodes if n.is_depot), None)
    houses = [n for n in detected_nodes if not n.is_depot]

    # Extract road mask
    node_coords = [(n.cx, n.cy) for n in detected_nodes]
    road_mask = extract_road_mask(
        img,
        road_type=road_type,
        threshold=road_thresh,
        node_coords=node_coords,
    )

    labeled_bgr = processor.get_labeled_image()
    if labeled_bgr is None:
        labeled_bgr = img

    return {
        "map_image": _encode_bgr_to_b64_png(img),
        "labeled_image": _encode_bgr_to_b64_png(labeled_bgr),
        "road_mask": _encode_bgr_to_b64_png(road_mask),
        "depot": {"cx": depot.cx, "cy": depot.cy} if depot else None,
        "all_depots_count": len(processor.all_detected_depots),
        "houses": [_node_to_dict(h) for h in houses],
    }


@router.post("/optimize")
def optimize_route(req: OptimizeRequest):
    """
    Run full route optimization pipeline mirroring app.py.
    Returns metrics, route overlays per algorithm, convergence chart, and turn-by-turn itinerary.
    """
    base_img = _decode_b64_image(req.map_image)
    all_nodes = [_dict_to_node(d) for d in req.nodes]

    depot = next((n for n in all_nodes if n.is_depot), None)
    if depot is None:
        raise HTTPException(status_code=400, detail="No verified Red Depot found! Cannot optimize without a starting hub.")

    selected_houses = [n for n in all_nodes if not n.is_depot and n.node_id in req.selected_ids]
    if len(selected_houses) < 1:
        raise HTTPException(status_code=400, detail="Please select at least 1 delivery house to form a route.")

    run_nodes = [depot] + selected_houses
    node_coords = [(n.cx, n.cy) for n in run_nodes]

    # Obtain road mask
    if req.seed is not None:
        # Recreate exact procedural road mask
        _, proc_mask, _, _ = create_city_map(out_path=None, n_houses=len(selected_houses), seed=req.seed)
        road_mask = proc_mask.copy()
        for cx, cy in node_coords:
            cv2.circle(road_mask, (int(cx), int(cy)), 18, 255, -1)
    else:
        road_mask = extract_road_mask(
            base_img,
            road_type=req.road_type,
            threshold=req.road_thresh,
            node_coords=node_coords,
        )

    # Compute pairwise A* road distance matrix
    dist_matrix, path_cache = build_road_distance_matrix(
        run_nodes,
        road_mask,
        grid_scale=req.grid_scale,
    )

    # Build DeliveryGraph with unified schedule parameters
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
            for n in run_nodes
        ],
        dist_matrix=dist_matrix,
        start_time=req.start_time,
        depot_close_time=req.depot_close,
        speed_px_per_min=req.speed,
        hard_windows=req.hard_windows,
    )

    solver = TSPSolver(graph, tw_weight=req.tw_weight, seed=int(req.opt_seed))
    results = solver.run_all(req.algorithms)

    # Vision processor for drawing route overlays
    processor = MapVisionProcessor()
    processor._original_bgr = base_img.copy()
    processor._labeled_bgr = base_img.copy()

    # Determine best heuristic (excluding exact solver)
    heur_results = {k: r for k, r in results.items() if k != "exact" and r.score < float("inf")}
    best_algo_key = min(heur_results.keys(), key=lambda k: heur_results[k].score) if heur_results else None

    # Render route overlay image for each algorithm
    route_images = {}
    for k, r in results.items():
        if r.score < float("inf") and r.route and len(r.route) >= 2:
            route_bgr = processor.draw_route(
                r.route,
                run_nodes,
                color=(0, 235, 110),
                thickness=4,
                path_cache=path_cache,
            )
            route_images[k] = _encode_bgr_to_b64_png(route_bgr)

    # Render convergence chart image
    chart_bgr = render_convergence_chart(results, width=850, height=420)
    convergence_chart_b64 = _encode_bgr_to_b64_png(chart_bgr)

    # Build Turn-by-Turn Manifest for best heuristic route (identical to app.py Tab 5)
    itinerary_data = []
    if best_algo_key and best_algo_key in results:
        best_r = results[best_algo_key]
        route = best_r.route
        current_time_min = float(req.start_time)

        for i in range(len(route)):
            curr_idx = route[i]
            next_idx = route[(i + 1) % len(route)]

            curr_node = run_nodes[curr_idx]
            next_node = run_nodes[next_idx]

            leg_dist = dist_matrix[curr_idx][next_idx]
            travel_time = leg_dist / req.speed
            arrival_time_min = current_time_min + travel_time
            earliest, latest = next_node.time_window
            is_depot_return = next_node.is_depot and i == len(route) - 1

            if is_depot_return:
                if arrival_time_min <= req.depot_close:
                    status_badge = f'<span class="badge-on-time">Depot Return ✓ (Before {_format_minutes_to_time(req.depot_close)})</span>'
                    status_plain = f"Depot Return ✓ (Before {_format_minutes_to_time(req.depot_close)})"
                else:
                    over = int(arrival_time_min - req.depot_close)
                    status_badge = f'<span class="badge-late">Depot Late by {over}m ⚠️</span>'
                    status_plain = f"Depot Late by {over}m ⚠️"
            elif arrival_time_min < earliest:
                wait_m = int(earliest - arrival_time_min)
                status_badge = f'<span class="badge-on-time">Early (Wait {wait_m}m)</span>'
                status_plain = f"Early (Wait {wait_m}m)"
                current_time_min = float(earliest)
            elif arrival_time_min > latest:
                lateness = int(arrival_time_min - latest)
                status_badge = f'<span class="badge-late">Late by {lateness}m ⚠️</span>'
                status_plain = f"Late by {lateness}m ⚠️"
                current_time_min = arrival_time_min
            else:
                status_badge = '<span class="badge-on-time">On-Time ✓</span>'
                status_plain = "On-Time ✓"
                current_time_min = arrival_time_min

            dest_addr = getattr(next_node, "address", "") or next_node.node_id
            window_str = (
                f"Depot Close ({_format_minutes_to_time(req.depot_close)})"
                if is_depot_return
                else f"{_format_minutes_to_time(earliest)} - {_format_minutes_to_time(latest)}"
            )

            itinerary_data.append({
                "leg": i + 1,
                "from_to": f"{curr_node.node_id} → {next_node.node_id}",
                "destination_address": dest_addr,
                "road_distance_px": round(leg_dist, 1),
                "travel_time_min": round(travel_time, 1),
                "est_arrival": _format_minutes_to_time(arrival_time_min),
                "customer_window": window_str,
                "status_badge": status_badge,
                "status_plain": status_plain,
            })
            current_time_min += getattr(next_node, "service_time", 5)

    # Format Benchmark comparison rows
    benchmark_table = []
    for k, r in results.items():
        gap_str = (
            f"{r.optimality_gap_pct:.2f}%"
            if r.optimality_gap_pct is not None
            else ("0.00% (Ref)" if k == "exact" and r.score < float("inf") else "N/A")
        )
        feas_str = "✅ Yes" if r.feasible else f"❌ No ({len(r.late_stops)} late)"
        late_str = ", ".join(r.late_stops) if r.late_stops else "None"

        benchmark_table.append({
            "key": k,
            "algorithm": r.algorithm,
            "road_distance_px": round(r.total_distance, 1),
            "drive_time_min": round(r.travel_time_min, 1),
            "wait_time_min": round(r.waiting_time_min, 1),
            "lateness_min": round(r.tw_penalty, 1),
            "total_elapsed_min": round(r.total_elapsed_min, 1),
            "score": round(r.score, 1) if r.score < float("inf") else "Refused (>12 stops)",
            "feasible": r.feasible,
            "feasibility_str": feas_str,
            "late_stops": late_str,
            "max_lateness_min": round(r.max_lateness_min, 1),
            "cpu_time_ms": round(r.cpu_time_ms, 3),
            "evaluations": r.evaluations,
            "gap_vs_exact": gap_str,
            "is_best_heuristic": k == best_algo_key,
            "is_exact": k == "exact",
        })

    # Convergence summary table
    conv_summary = []
    for k, r in results.items():
        if r.score_history and len(r.score_history) > 1:
            initial = r.score_history[0]
            final = r.score_history[-1]
            impr_pct = ((initial - final) / initial * 100) if initial > 0 else 0.0
            conv_summary.append({
                "algorithm": r.algorithm,
                "starting_score": round(initial, 1),
                "final_best_score": round(final, 1),
                "improvement_pts": round(initial - final, 1),
                "improvement_pct": round(impr_pct, 1),
                "evaluations": r.evaluations,
            })

    # Exact score for reference card
    exact_score = results["exact"].score if "exact" in results and results["exact"].score < float("inf") else None

    return {
        "best_algo_key": best_algo_key,
        "exact_score": exact_score,
        "benchmark_table": benchmark_table,
        "route_images": route_images,
        "convergence_chart": convergence_chart_b64,
        "convergence_summary": conv_summary,
        "itinerary": itinerary_data,
        "road_mask": _encode_bgr_to_b64_png(road_mask),
    }


@router.post("/multi-seed")
def multi_seed_study(req: MultiSeedRequest):
    """Run multi-seed statistical study across stochastic algorithms."""
    base_img = _decode_b64_image(req.map_image)
    all_nodes = [_dict_to_node(d) for d in req.nodes]

    depot = next((n for n in all_nodes if n.is_depot), None)
    if depot is None:
        raise HTTPException(status_code=400, detail="No depot found")

    selected_houses = [n for n in all_nodes if not n.is_depot and n.node_id in req.selected_ids]
    if len(selected_houses) < 1:
        raise HTTPException(status_code=400, detail="Select at least 1 house")

    run_nodes = [depot] + selected_houses
    node_coords = [(n.cx, n.cy) for n in run_nodes]

    if req.seed is not None:
        _, proc_mask, _, _ = create_city_map(out_path=None, n_houses=len(selected_houses), seed=req.seed)
        road_mask = proc_mask.copy()
        for cx, cy in node_coords:
            cv2.circle(road_mask, (int(cx), int(cy)), 18, 255, -1)
    else:
        road_mask = extract_road_mask(
            base_img,
            road_type=req.road_type,
            threshold=req.road_thresh,
            node_coords=node_coords,
        )

    dist_matrix, _ = build_road_distance_matrix(run_nodes, road_mask, grid_scale=req.grid_scale)

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
            for n in run_nodes
        ],
        dist_matrix=dist_matrix,
        start_time=req.start_time,
        depot_close_time=req.depot_close,
        speed_px_per_min=req.speed,
        hard_windows=req.hard_windows,
    )

    multi_summary = benchmark_multi_seed(graph, n_seeds=req.n_seeds, tw_weight=req.tw_weight)

    rows = []
    for k, s in multi_summary.items():
        mgap_str = f"{s['mean_gap_pct']:.2f}%" if s["mean_gap_pct"] is not None else ("0.00% (Ref)" if k == "exact" else "N/A")
        rows.append({
            "algorithm": s["algorithm"],
            "runs": s["runs"],
            "mean_score": round(s["mean_score"], 1),
            "std_score": round(s["std_score"], 1),
            "best_score": round(s["best_score"], 1),
            "worst_score": round(s["worst_score"], 1),
            "feasibility_pct": round(s["feasibility_pct"], 1),
            "mean_distance": round(s["mean_distance"], 1),
            "mean_lateness": round(s["mean_lateness"], 1),
            "mean_cpu_ms": round(s["mean_cpu_ms"], 2),
            "mean_gap_pct": mgap_str,
        })

    return {"summary_rows": rows}


# ---------------------------------------------------------------------------
# Compatibility Endpoints (Sample map / Health)
# ---------------------------------------------------------------------------

@router.get("/sample")
def get_sample():
    """Return default sample map."""
    sample_path = os.path.join(_ROOT, "sample_map.png")
    if os.path.exists(sample_path):
        img = cv2.imread(sample_path)
    else:
        img, _, _, _ = create_city_map(out_path=None, n_houses=8, seed=42)

    proc = MapVisionProcessor()
    nodes = proc.process_image(img)
    depot = next((n for n in nodes if n.is_depot), None)
    houses = [n for n in nodes if not n.is_depot]

    labeled = proc.get_labeled_image()
    if labeled is None:
        labeled = img

    return {
        "seed": 42,
        "n_houses": len(houses),
        "map_image": _encode_bgr_to_b64_png(img),
        "labeled_image": _encode_bgr_to_b64_png(labeled),
        "depot": {"cx": depot.cx, "cy": depot.cy} if depot else None,
        "all_depots_count": len(proc.all_detected_depots),
        "houses": [_node_to_dict(h) for h in houses],
    }


# ---------------------------------------------------------------------------
# Include routers with and without /api prefix
# ---------------------------------------------------------------------------
app.include_router(router, prefix="/api")
app.include_router(router)

# Mount static files for local uvicorn preview (Vercel edge handles /public automatically)
_public_dir = os.path.join(_ROOT, "public")
if os.path.isdir(_public_dir):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=_public_dir, html=True), name="static")

