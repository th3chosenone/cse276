"""
Last-Mile Delivery Route Optimiser – FastAPI serverless backend for Vercel.
Stateless: the client sends the map image (base64) with every request.
"""

from __future__ import annotations

import base64
import os
import sys
import tempfile
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so we can import the core modules.
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
    TSPSolver,
    benchmark_multi_seed,
    build_graph_from_detected_nodes,
)
from vision import DetectedNode, MapVisionProcessor  # noqa: E402

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Last-Mile Delivery Route Optimiser API", version="1.0.0")
router = APIRouter()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_MAP_PATH = os.path.join(_ROOT, "sample_map.png")
_PROCESSOR = MapVisionProcessor()


def _decode_b64_image(b64str: str) -> np.ndarray:
    """Decode a base64-encoded PNG/JPEG string to a BGR numpy array."""
    try:
        # Strip optional data-URL prefix
        if "," in b64str:
            b64str = b64str.split(",", 1)[1]
        raw = base64.b64decode(b64str)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        return img
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unreadable image: {exc}") from exc


def _encode_bgr_image(img: np.ndarray) -> str:
    """Encode a BGR numpy array to a base64 PNG string."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode()


def _node_to_dict(n: DetectedNode) -> dict:
    return {
        "id": n.node_id,
        "x": n.cx,
        "y": n.cy,
        "is_depot": n.is_depot,
        "time_window": list(n.time_window),
        "service_time": n.service_time,
        "address": n.address,
    }


def _dict_to_node(d: dict) -> DetectedNode:
    node = DetectedNode(
        node_id=d["id"],
        cx=int(d["x"]),
        cy=int(d["y"]),
        is_depot=bool(d.get("is_depot", False)),
        area=float(d.get("area", 0.0)),
        time_window=tuple(d.get("time_window", [0, 1440])),
        service_time=int(d.get("service_time", 5)),
        address=d.get("address", ""),
    )
    return node


def _result_to_dict(r: BenchmarkResult) -> dict:
    return {
        "algorithm": r.algorithm,
        "route": r.route,
        "route_labels": r.route_labels or [],
        "distance": round(r.total_distance, 1),
        "late_min": round(r.tw_penalty, 1),
        "drive_min": round(r.travel_time_min, 1),
        "wait_min": round(r.waiting_time_min, 1),
        "elapsed_min": round(r.total_elapsed_min, 1),
        "score": round(r.score, 1),
        "cpu_ms": round(r.cpu_time_ms, 2),
        "evaluations": r.evaluations,
        "gap_pct": round(r.optimality_gap_pct, 2) if r.optimality_gap_pct is not None else None,
        "feasible": r.feasible,
        "late_stops": r.late_stops,
        "score_history": r.score_history or [],
        "evaluation_history": r.evaluation_history or [],
    }


def _run_full_pipeline(
    img: np.ndarray,
    node_dicts: list[dict],
    selected_ids: list[str],
    algorithms: list[str],
    seed: int = 42,
) -> tuple[dict[str, BenchmarkResult], dict, list[DetectedNode]]:
    """
    Run the full optimisation pipeline.
    Returns (results, path_cache, run_nodes).
    """
    all_nodes = [_dict_to_node(d) for d in node_dicts]

    depot = next((n for n in all_nodes if n.is_depot), None)
    if depot is None:
        raise HTTPException(status_code=400, detail="No depot found in node list")

    houses = [n for n in all_nodes if not n.is_depot and n.node_id in selected_ids]
    if len(houses) < 1:
        raise HTTPException(status_code=400, detail="Select at least 1 house")

    run_nodes = [depot] + houses

    node_coords = [(n.cx, n.cy) for n in run_nodes]
    road_mask = extract_road_mask(img, road_type="dark", threshold=130, node_coords=node_coords)
    dist_matrix, path_cache = build_road_distance_matrix(run_nodes, road_mask, grid_scale=2)
    graph = build_graph_from_detected_nodes(run_nodes, dist_matrix=dist_matrix)

    solver = TSPSolver(graph, seed=seed)
    results = solver.run_all(algorithms)

    return results, path_cache, run_nodes


def _extract_route_polylines(
    best_result: BenchmarkResult,
    run_nodes: list[DetectedNode],
    path_cache: dict,
) -> list[list[list[int]]]:
    """
    Convert path_cache entries for the best-heuristic route into a list of
    polylines (one per leg), each a list of [x, y] pixel coords.
    """
    route_indices = best_result.route  # list of indices into run_nodes
    polylines = []
    n = len(route_indices)
    for i in range(n):  # include the return leg back to the depot
        a = route_indices[i]
        b = route_indices[(i + 1) % n]
        path = path_cache.get((a, b)) or path_cache.get((b, a), [])
        if path and path_cache.get((b, a)) and not path_cache.get((a, b)):
            path = list(reversed(path))
        polylines.append([[p[0], p[1]] for p in path])
    return polylines


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RandomMapRequest(BaseModel):
    seed: Optional[int] = None
    n_houses: int = Field(default=8, ge=2, le=15)


class DetectRequest(BaseModel):
    image: str  # base64


class OptimiseRequest(BaseModel):
    image: str  # base64
    nodes: list[dict]
    selected_ids: list[str]
    algorithms: list[str] = Field(
        default=["nn", "greedy", "2opt", "hill", "sa", "exact"]
    )
    seed: int = 42


class BenchmarkRequest(BaseModel):
    image: str  # base64
    nodes: list[dict]
    selected_ids: list[str]
    n_seeds: int = Field(default=10, ge=1, le=30)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/sample")
def get_sample():
    """Return the sample map image and its detected nodes."""
    img = cv2.imread(SAMPLE_MAP_PATH)
    if img is None:
        raise HTTPException(status_code=500, detail="sample_map.png not found on server")

    nodes = _PROCESSOR.process(SAMPLE_MAP_PATH)
    return {
        "image": _encode_bgr_image(img),
        "nodes": [_node_to_dict(n) for n in nodes],
    }


@router.post("/random-map")
def random_map(req: RandomMapRequest):
    """Generate a random city map and return image + nodes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "random_map.png")
        img_arr, _, _, _ = create_city_map(
            out_path=out_path,
            n_houses=req.n_houses,
            seed=req.seed,
        )
        # Process the saved file so vision can find nodes
        nodes = _PROCESSOR.process(out_path)
        img = cv2.imread(out_path)

    return {
        "image": _encode_bgr_image(img),
        "nodes": [_node_to_dict(n) for n in nodes],
    }


@router.post("/detect")
def detect(req: DetectRequest):
    """Detect nodes in an uploaded map image."""
    img = _decode_b64_image(req.image)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    try:
        cv2.imwrite(tmp_path, img)
        nodes = _PROCESSOR.process(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not any(n.is_depot for n in nodes):
        raise HTTPException(status_code=400, detail="No depot found. Ensure the map has a red depot marker.")

    return {"nodes": [_node_to_dict(n) for n in nodes]}


@router.post("/optimise")
def optimise(req: OptimiseRequest):
    """Run route optimisation and return per-algorithm results."""
    img = _decode_b64_image(req.image)

    # Validate inputs
    if not req.selected_ids:
        raise HTTPException(status_code=400, detail="Select at least 1 house")

    results, path_cache, run_nodes = _run_full_pipeline(
        img, req.nodes, req.selected_ids, req.algorithms, req.seed
    )

    # Determine best heuristic (exclude exact)
    heuristic_keys = [k for k in results if k != "exact" and results[k].score < float("inf")]
    best_key = min(heuristic_keys, key=lambda k: results[k].score) if heuristic_keys else None

    # Extract road polylines for best heuristic route
    polylines = []
    if best_key:
        try:
            polylines = _extract_route_polylines(results[best_key], run_nodes, path_cache)
        except Exception:
            polylines = []

    # Exact reference score
    exact_score = results["exact"].score if "exact" in results else None

    # Serialise results
    out_results = {}
    for k, r in results.items():
        d = _result_to_dict(r)
        d["is_best_heuristic"] = k == best_key
        d["is_exact"] = k == "exact"
        out_results[k] = d

    # Node info for drawing on the canvas
    nodes_out = [_node_to_dict(n) for n in run_nodes]

    return {
        "results": out_results,
        "best_heuristic_key": best_key,
        "polylines": polylines,  # list of legs, each a list of [x,y]
        "exact_reference_score": round(exact_score, 1) if exact_score and exact_score < float("inf") else None,
        "nodes": nodes_out,
    }


@router.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    """Run multi-seed benchmark and return summary statistics."""
    img = _decode_b64_image(req.image)

    if not req.selected_ids:
        raise HTTPException(status_code=400, detail="Select at least 1 house")

    all_nodes = [_dict_to_node(d) for d in req.nodes]
    depot = next((n for n in all_nodes if n.is_depot), None)
    if depot is None:
        raise HTTPException(status_code=400, detail="No depot found")

    houses = [n for n in all_nodes if not n.is_depot and n.node_id in req.selected_ids]
    if len(houses) < 1:
        raise HTTPException(status_code=400, detail="Select at least 1 house")

    run_nodes = [depot] + houses
    node_coords = [(n.cx, n.cy) for n in run_nodes]
    road_mask = extract_road_mask(img, road_type="dark", threshold=130, node_coords=node_coords)
    dist_matrix, _ = build_road_distance_matrix(run_nodes, road_mask, grid_scale=2)
    graph = build_graph_from_detected_nodes(run_nodes, dist_matrix=dist_matrix)

    summary = benchmark_multi_seed(graph, n_seeds=req.n_seeds)

    # summary is dict[algo_name, dict[metric, value]]
    return {"summary": summary}


# ---------------------------------------------------------------------------
# Register the routes under every path form Vercel might hand to the function:
#   /api/sample  (original path), /sample, and /api/index/sample.
# ---------------------------------------------------------------------------
app.include_router(router, prefix="/api")
app.include_router(router)
app.include_router(router, prefix="/api/index")


@app.api_route("/{full_path:path}", methods=["GET", "POST"], include_in_schema=False)
def not_found_debug(full_path: str, request: Request):
    """Shows which path the function actually received (helps debug 404s)."""
    raise HTTPException(
        status_code=404,
        detail=f"No API route for path '/{full_path}' (method {request.method}).",
    )
