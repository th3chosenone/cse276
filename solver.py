"""
solver.py — TSP with Time-Windows (TSPTW) Algorithmic Core
Last-Mile Delivery Route Optimiser
-----------------------------------------------------------
Provides:
  • DeliveryGraph: Graph construction, A* distance integration, unified time units.
  • Unified time accounting: start_time, depot_close_time, travel speed, and service times.
  • Six routing algorithms:
      1. Nearest Neighbour (NN)
      2. Cheapest Insertion (true cheapest insertion heuristic; never falls back to NN)
      3. 2-opt Local Search
      4. Hill Climbing (multi-restart with seed control)
      5. Simulated Annealing (SA with cooling schedule, seed control, and evaluation budget)
      6. Exact Solver: Exact Branch-and-Bound / Held-Karp for N <= 12; explicitly refuses larger instances
  • Feasibility analysis: feasible flag, late stops tracking, and max lateness calculation.
  • Reproducibility: optional random seed on all stochastic methods.
  • Fair comparison: evaluation-based convergence logging for all iterative methods.
  • Multi-seed benchmarking: statistical evaluation across multiple random seeds.
  • BenchmarkResult dataclass for detailed reporting and export.
"""

from __future__ import annotations

import math
import time
import random
import itertools
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Config & Result types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NodeInfo:
    """Coordinate + time-window + address struct used by the solver."""
    node_id: str
    x: float
    y: float
    time_window: tuple[int, int] = (540, 1080)   # (earliest, latest) in minutes from midnight (09:00 - 18:00)
    service_time: int = 5                         # minutes spent at stop
    address: str = ""                             # street address or human-readable description


@dataclass
class BenchmarkResult:
    """Comprehensive result metrics for a TSPTW solution."""
    algorithm: str
    route: list[int]                    # node indices in visit order (starts at 0)
    total_distance: float               # sum of road/Euclidean distances in pixels
    tw_penalty: float                   # penalty for time-window violations (minutes of lateness)
    score: float                        # unified routing cost (lower = better)
    estimated_time_min: float           # total elapsed mission duration (travel + service + waiting) in minutes
    cpu_time_ms: float                  # CPU solve runtime in milliseconds
    iterations: int                     # iteration count (where applicable)
    improvements: int                   # accepted improvements count
    waiting_time_min: float = 0.0       # time spent waiting for customer time windows to open (minutes)
    travel_time_min: float = 0.0        # pure driving travel time (minutes)
    service_time_min: float = 0.0       # total service duration across all visited stops (minutes)
    total_elapsed_min: float = 0.0      # total elapsed mission time: travel + service + waiting (minutes)
    route_labels: list[str] = field(default_factory=list)
    score_history: list[float] = field(default_factory=list)          # best score at logged evaluation milestones
    evaluation_history: list[int] = field(default_factory=list)       # candidate evaluation counts corresponding to score_history
    evaluations: int = 0                # total candidate route evaluations performed
    feasible: bool = True               # True if all stops visited within their time windows
    late_stops: list[str] = field(default_factory=list)               # IDs of late stops
    max_lateness_min: float = 0.0       # maximum lateness at any stop in minutes
    optimality_gap_pct: Optional[float] = None                        # % gap vs exact optimum (when exact is available)


# ──────────────────────────────────────────────────────────────────────────────
# Graph builder & Time Accounting
# ──────────────────────────────────────────────────────────────────────────────

class DeliveryGraph:
    """
    Builds and owns the adjacency/distance matrix and schedule simulation for NodeInfo stops.

    Nodes are ordered: nodes[0] = DEPOT, nodes[1..n] = delivery houses.
    Index 0 is always the start of the delivery tour.
    """

    PX_PER_MINUTE = 60.0   # Default calibration: 60 px = 1 minute travel (approx. 30 km/h at 500m/grid)

    def __init__(
        self,
        nodes: list[NodeInfo],
        dist_matrix: Optional[list[list[float]]] = None,
        return_to_depot: bool = True,
        start_time: float = 540.0,            # 09:00 AM (minutes from midnight)
        depot_close_time: float = 1080.0,      # 18:00 PM (minutes from midnight)
        speed_px_per_min: float = 60.0,
        hard_windows: bool = False,
    ):
        if len(nodes) < 2:
            raise ValueError("Need at least 2 nodes (depot + 1 house).")
        self.nodes = nodes
        self.n = len(nodes)
        self.return_to_depot = return_to_depot
        self.start_time = float(start_time)
        self.depot_close_time = float(depot_close_time)
        self.speed_px_per_min = float(speed_px_per_min) if speed_px_per_min > 0 else 60.0
        self.hard_windows = hard_windows

        if dist_matrix is not None:
            self.dist = dist_matrix
        else:
            self.dist = self._build_matrix()

    # ── Public Methods ────────────────────────────────────────────────────────

    def route_distance(self, route: list[int]) -> float:
        """Total distance along the route. If return_to_depot is True, includes return to depot."""
        total = 0.0
        n = len(route)
        end = n if self.return_to_depot else n - 1
        for i in range(end):
            a, b = route[i], route[(i + 1) % n]
            total += self.dist[a][b]
        return total

    def travel_time(self, route: list[int]) -> float:
        """Total driving travel time in minutes."""
        return self.route_distance(route) / self.speed_px_per_min

    def route_score(self, route: list[int], tw_weight: float = 1.5) -> tuple[float, float]:
        """
        Calculates unified score and lateness penalty.

        Physics / Units:
          travel_dist_px: total distance in pixels
          equivalent_delay_px: lateness_minutes * speed_px_per_min
          score = travel_dist_px + tw_weight * equivalent_delay_px
        This ensures distance and lateness penalty are evaluated in unified physical units.
        If hard_windows is True and there is lateness, an infeasibility barrier is added.
        """
        dist = self.route_distance(route)
        penalty_min, _, _, _, _ = self._simulate_schedule(route)

        # Unified penalty in equivalent distance units (60 px/min * minutes = px)
        penalty_equiv_px = penalty_min * self.speed_px_per_min
        base_score = dist + (tw_weight * penalty_equiv_px)

        if self.hard_windows and penalty_min > 0:
            base_score += 1e7 * penalty_min   # Infeasibility barrier

        return base_score, penalty_min

    def evaluate_route(self, route: list[int], tw_weight: float = 1.5) -> tuple[float, float, bool, list[str], float]:
        """
        Full schedule evaluation returning:
        (score, penalty_min, feasible, late_stops, max_lateness_min)
        """
        score, penalty = self.route_score(route, tw_weight)
        pen_min, late_stops, max_late, _, _ = self._simulate_schedule(route)
        feasible = (pen_min == 0.0)
        return score, pen_min, feasible, late_stops, max_late

    def estimated_time(self, route: list[int]) -> float:
        """
        Total mission elapsed duration from depot departure to final return (minutes).
        Includes driving travel time, service duration at customer stops, and customer waiting time.
        """
        _, _, _, _, total_elapsed = self._simulate_schedule(route)
        return total_elapsed

    def node_labels(self, route: list[int]) -> list[str]:
        return [self.nodes[i].node_id for i in route]

    # ── Schedule Simulation ───────────────────────────────────────────────────

    def _simulate_schedule(self, route: list[int]) -> tuple[float, list[str], float, float, float]:
        """
        Simulate vehicle departure from depot at start_time and visit each stop in sequence.
        Returns: (total_penalty_min, late_stops_list, max_lateness_min, total_wait_min, total_elapsed_min)
        """
        penalty = 0.0
        total_wait = 0.0
        t = self.start_time
        prev = route[0]
        late_stops = []
        max_late = 0.0

        for idx in route[1:]:
            travel = self.dist[prev][idx] / self.speed_px_per_min
            t += travel
            node = self.nodes[idx]
            earliest, latest = node.time_window

            if t < earliest:
                wait_duration = float(earliest) - t
                total_wait += wait_duration
                t = float(earliest)  # Driver arrives early and waits for customer window to open
            elif t > latest:
                lateness = t - latest
                penalty += lateness
                late_stops.append(node.node_id)
                if lateness > max_late:
                    max_late = lateness

            t += node.service_time
            prev = idx

        # Check return to depot against depot closing hours
        if self.return_to_depot and len(route) > 1:
            ret_travel = self.dist[prev][route[0]] / self.speed_px_per_min
            t += ret_travel
            if t > self.depot_close_time:
                depot_lateness = t - self.depot_close_time
                penalty += depot_lateness
                late_stops.append(self.nodes[route[0]].node_id)
                if depot_lateness > max_late:
                    max_late = depot_lateness

        total_elapsed = t - self.start_time
        return penalty, late_stops, max_late, total_wait, total_elapsed

    def _build_matrix(self) -> list[list[float]]:
        n = self.n
        d = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dx = self.nodes[i].x - self.nodes[j].x
                dy = self.nodes[i].y - self.nodes[j].y
                dist = math.hypot(dx, dy)
                d[i][j] = d[j][i] = dist
        return d


# ──────────────────────────────────────────────────────────────────────────────
# TSP-TW Solvers
# ──────────────────────────────────────────────────────────────────────────────

class TSPSolver:
    """
    Solves TSP with Time Windows on a DeliveryGraph using heuristics and exact algorithms.
    Supports reproducible random seeding and fair evaluation-count convergence tracking.
    """

    SA_T_INIT = 1000.0
    SA_T_MIN = 1e-3
    SA_ALPHA = 0.995

    def __init__(self, graph: DeliveryGraph, tw_weight: float = 1.5, seed: Optional[int] = None):
        self.g = graph
        self.tw_weight = tw_weight
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.eval_count = 0

    # ── Entry point: run all / selected ──────────────────────────────────────

    def run_all(self, algorithms: Optional[list[str]] = None, seed: Optional[int] = None) -> dict[str, BenchmarkResult]:
        """
        Run selected algorithms and return dictionary of BenchmarkResult objects.
        Automatically computes optimality gap % for every algorithm against Exact if available.
        """
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(seed)

        all_algos = {
            "nn": self.solve_nearest_neighbour,
            "greedy": self.solve_cheapest_insertion,
            "2opt": self.solve_2opt,
            "hill": self.solve_hill_climbing,
            "sa": self.solve_simulated_annealing,
            "exact": self.solve_exact,
        }
        if algorithms is None:
            algorithms = list(all_algos.keys())

        results: dict[str, BenchmarkResult] = {}
        for name in algorithms:
            if name in all_algos:
                try:
                    results[name] = all_algos[name]()
                except Exception as e:
                    print(f"[solver] {name} failed: {e}")

        # Compute optimality gap for EVERY algorithm if Exact found a valid optimum
        if "exact" in results and results["exact"].feasible and results["exact"].score < float("inf"):
            exact_score = results["exact"].score
            if exact_score > 0:
                for k, r in results.items():
                    if r.score < float("inf"):
                        r.optimality_gap_pct = ((r.score - exact_score) / exact_score) * 100.0

        return results

    # ── Timing Helper ────────────────────────────────────────────────────────

    @staticmethod
    def _timer_start() -> tuple[float, float]:
        """Returns (cpu_time_start, wall_time_start)."""
        return time.process_time(), time.perf_counter()

    @staticmethod
    def _timer_elapsed_ms(t_cpu_start: float, t_wall_start: float) -> float:
        """
        Computes elapsed CPU time in milliseconds.
        Falls back to wall-clock time if CPU time is below system tick resolution on Windows.
        """
        cpu_ms = (time.process_time() - t_cpu_start) * 1000.0
        wall_ms = (time.perf_counter() - t_wall_start) * 1000.0
        return cpu_ms if cpu_ms > 0.0 else max(wall_ms, 0.001)

    # ── 1. Nearest Neighbour ─────────────────────────────────────────────────

    def solve_nearest_neighbour(self) -> BenchmarkResult:
        t_cpu0, t_wall0 = self._timer_start()
        n = self.g.n
        visited = [False] * n
        route = [0]
        visited[0] = True

        for _ in range(n - 1):
            current = route[-1]
            nearest = None
            best_d = float("inf")
            for j in range(n):
                if not visited[j] and self.g.dist[current][j] < best_d:
                    best_d = self.g.dist[current][j]
                    nearest = j
            route.append(nearest)
            visited[nearest] = True

        cpu = self._timer_elapsed_ms(t_cpu0, t_wall0)
        return self._make_result("Nearest Neighbour", route, cpu, 0, 0, evaluations=1)

    # ── 2. Cheapest Insertion (True Heuristic - Never Falls Back to NN) ────────

    def solve_cheapest_insertion(self) -> BenchmarkResult:
        """
        Textbook Cheapest Insertion Heuristic:
        Starts with depot (0) and the closest node to depot.
        Iteratively inserts the unvisited stop into the position that minimizes added tour cost.
        Guaranteed to produce a valid tour without ever falling back to Nearest Neighbour.
        """
        t_cpu0, t_wall0 = self._timer_start()
        n = self.g.n
        evals = 0

        if n <= 2:
            route = list(range(n))
            cpu = self._timer_elapsed_ms(t_cpu0, t_wall0)
            return self._make_result("Cheapest Insertion", route, cpu, 0, 0, evaluations=1)

        # 1. Start with depot (0) and the closest stop
        best_first = min(range(1, n), key=lambda j: self.g.dist[0][j])
        tour = [0, best_first]
        unvisited = set(range(1, n)) - {best_first}

        # 2. Repeatedly insert the unvisited stop that causes the cheapest increase
        while unvisited:
            best_u = None
            best_pos = None
            best_cost = float("inf")

            for u in unvisited:
                for pos in range(1, len(tour) + 1):
                    cand = tour[:pos] + [u] + tour[pos:]
                    cost, _ = self.g.route_score(cand, self.tw_weight)
                    evals += 1
                    if cost < best_cost:
                        best_cost = cost
                        best_u = u
                        best_pos = pos

            tour.insert(best_pos, best_u)
            unvisited.remove(best_u)

        # Ensure depot is at index 0
        if tour[0] != 0:
            idx0 = tour.index(0)
            tour = tour[idx0:] + tour[:idx0]

        # Check reverse orientation and keep the one with lower time-window score
        if len(tour) >= 3:
            rev_tour = [0] + tour[1:][::-1]
            s_fwd, _ = self.g.route_score(tour, self.tw_weight)
            s_rev, _ = self.g.route_score(rev_tour, self.tw_weight)
            evals += 2
            if s_rev < s_fwd:
                tour = rev_tour

        cpu = self._timer_elapsed_ms(t_cpu0, t_wall0)
        return self._make_result("Cheapest Insertion", tour, cpu, n - 2, n - 2, evaluations=evals)

    # Alias for backward compatibility
    solve_greedy_insertion = solve_cheapest_insertion

    # ── 3. 2-opt Local Search ─────────────────────────────────────────────────

    def solve_2opt(self, initial_route: Optional[list[int]] = None) -> BenchmarkResult:
        t_cpu0, t_wall0 = self._timer_start()
        route = list(initial_route or self.solve_nearest_neighbour().route)
        n = len(route)
        improved = True
        iterations = 0
        improvements = 0
        evals = 0

        score_history: list[float] = []
        eval_history: list[int] = []

        s_cur, _ = self.g.route_score(route, self.tw_weight)
        evals += 1
        score_history.append(s_cur)
        eval_history.append(evals)

        while improved:
            improved = False
            iterations += 1
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    s_new, _ = self.g.route_score(new_route, self.tw_weight)
                    evals += 1

                    if s_new < s_cur - 1e-9:
                        route = new_route
                        s_cur = s_new
                        improved = True
                        improvements += 1
                        score_history.append(s_cur)
                        eval_history.append(evals)

            # Record milestone at end of pass
            if not score_history or score_history[-1] != s_cur:
                score_history.append(s_cur)
                eval_history.append(evals)

        cpu = self._timer_elapsed_ms(t_cpu0, t_wall0)
        return self._make_result(
            "2-opt", route, cpu, iterations, improvements,
            score_history=score_history, evaluation_history=eval_history, evaluations=evals
        )

    # ── 4. Hill Climbing (Multi-Restart with Seed Control) ───────────────────

    def solve_hill_climbing(self, restarts: int = 10, seed: Optional[int] = None) -> BenchmarkResult:
        t_cpu0, t_wall0 = self._timer_start()
        n = self.g.n
        rng = random.Random(seed if seed is not None else self.seed)

        best_route = None
        best_score = float("inf")
        total_iters = 0
        total_impr = 0
        evals = 0

        score_history: list[float] = []
        eval_history: list[int] = []

        for _ in range(restarts):
            perm = list(range(1, n))
            rng.shuffle(perm)
            route = [0] + perm

            s_cur, _ = self.g.route_score(route, self.tw_weight)
            evals += 1

            if s_cur < best_score:
                best_score = s_cur
                best_route = list(route)
            score_history.append(best_score)
            eval_history.append(evals)

            improved = True
            while improved:
                improved = False
                total_iters += 1
                for i in range(1, n - 1):
                    for j in range(i + 1, n):
                        route[i], route[j] = route[j], route[i]
                        s_new, _ = self.g.route_score(route, self.tw_weight)
                        evals += 1

                        if s_new < s_cur - 1e-9:
                            s_cur = s_new
                            improved = True
                            total_impr += 1
                            if s_new < best_score:
                                best_score = s_new
                                best_route = list(route)
                                score_history.append(best_score)
                                eval_history.append(evals)
                        else:
                            route[i], route[j] = route[j], route[i]

        cpu = self._timer_elapsed_ms(t_cpu0, t_wall0)
        return self._make_result(
            "Hill Climbing", best_route or [0] + list(range(1, n)), cpu, total_iters, total_impr,
            score_history=score_history, evaluation_history=eval_history, evaluations=evals
        )

    # ── 5. Simulated Annealing (Evaluation-Budgeted & Scaled Geometric Cooling) ─

    def solve_simulated_annealing(
        self,
        seed: Optional[int] = None,
        max_evaluations: int = 3000,
    ) -> BenchmarkResult:
        """
        Simulated Annealing for TSPTW:
        - Scales initial temperature T0 to the typical worsening score delta (80% initial acceptance).
        - Cools systematically per evaluation across the evaluation budget down to T_min = 1e-3.
        - Supports reproducible random seeding.
        """
        t_cpu0, t_wall0 = self._timer_start()
        n = self.g.n
        rng = random.Random(seed if seed is not None else self.seed)

        # Initial solution via Nearest Neighbour
        current = list(self.solve_nearest_neighbour().route)
        best = list(current)
        s_cur, _ = self.g.route_score(current, self.tw_weight)
        s_best = s_cur

        evals = 1
        score_history: list[float] = [s_best]
        eval_history: list[int] = [evals]

        # Handle small n edge cases
        if n <= 3:
            alt = [0] + current[1:][::-1]
            s_alt, _ = self.g.route_score(alt, self.tw_weight)
            evals += 1
            if s_alt < s_best:
                best = alt
                s_best = s_alt
            score_history.append(s_best)
            eval_history.append(evals)
            cpu = self._timer_elapsed_ms(t_cpu0, t_wall0)
            return self._make_result(
                "Simulated Annealing", best, cpu, 1, 1 if s_alt < s_cur else 0,
                score_history=score_history, evaluation_history=eval_history, evaluations=evals
            )

        # 1. Scale initial temperature (T0) to typical score delta
        # Sample random 2-opt perturbations to estimate typical worsening cost delta
        deltas: list[float] = []
        for _ in range(20):
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            cand = current[:i] + current[i:j + 1][::-1] + current[j + 1:]
            s_cand, _ = self.g.route_score(cand, self.tw_weight)
            evals += 1
            d = s_cand - s_cur
            if d > 0:
                deltas.append(d)

        delta_avg = float(np.mean(deltas)) if deltas else max(s_cur * 0.05, 10.0)

        # Target initial acceptance probability P0 = 0.80 for worsening moves:
        # P0 = exp(-delta_avg / T0) => T0 = -delta_avg / ln(0.80)
        t_init = -delta_avg / math.log(0.80)
        t_min = 1e-3

        # Continuous geometric cooling per evaluation:
        # T_final = T_init * (alpha ** remaining_budget) = T_min
        remaining_budget = max(max_evaluations - evals, 100)
        alpha = (t_min / t_init) ** (1.0 / remaining_budget)

        T = t_init
        iterations = 0
        improvements = 0
        log_interval = max(20, max_evaluations // 60)

        while evals < max_evaluations and T > t_min:
            i = rng.randint(1, n - 2)
            j = rng.randint(i + 1, n - 1)
            candidate = current[:i] + current[i:j + 1][::-1] + current[j + 1:]
            s_new, _ = self.g.route_score(candidate, self.tw_weight)
            evals += 1

            delta = s_new - s_cur
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-6)):
                current = candidate
                s_cur = s_new
                if s_new < s_best:
                    best = list(current)
                    s_best = s_new
                    improvements += 1

            iterations += 1
            # Cool temperature per evaluation so algorithm thoroughly explores early and refines late
            T *= alpha

            if evals % log_interval == 0:
                score_history.append(s_best)
                eval_history.append(evals)

        if not score_history or score_history[-1] != s_best:
            score_history.append(s_best)
            eval_history.append(evals)

        cpu = self._timer_elapsed_ms(t_cpu0, t_wall0)
        return self._make_result(
            "Simulated Annealing", best, cpu, iterations, improvements,
            score_history=score_history, evaluation_history=eval_history, evaluations=evals
        )

    # ── 6. Exact Solver (Branch-and-Bound / Held-Karp with Clean Refusal) ─────

    def solve_exact(self, max_nodes: int = 12) -> BenchmarkResult:
        """
        Exact TSP-TW Solver using Depth-First Branch and Bound with lower-bound pruning.
        Solves globally optimal routes for N <= 12 in tens of milliseconds.
        Refuses explicitly when N > max_nodes, never substituting heuristics under the Exact label.
        """
        t_cpu0, t_wall0 = self._timer_start()
        n = self.g.n

        # Explicit refusal above safe exponential threshold
        if n > max_nodes:
            cpu = self._timer_elapsed_ms(t_cpu0, t_wall0)
            return BenchmarkResult(
                algorithm=f"Exact (Refused: N={n} > {max_nodes})",
                route=[],
                total_distance=0.0,
                tw_penalty=0.0,
                score=float("inf"),
                estimated_time_min=0.0,
                cpu_time_ms=cpu,
                iterations=0,
                improvements=0,
                route_labels=[],
                score_history=[],
                evaluation_history=[],
                evaluations=0,
                feasible=False,
                late_stops=[],
                max_lateness_min=0.0,
                optimality_gap_pct=None,
            )

        dist = self.g.dist
        time_windows = [node.time_window for node in self.g.nodes]
        service_times = [node.service_time for node in self.g.nodes]
        speed = self.g.speed_px_per_min
        tw_weight = self.tw_weight
        start_time = self.g.start_time
        depot_close = self.g.depot_close_time
        return_to_depot = self.g.return_to_depot

        # Precompute minimum outgoing edge per node for lower-bound pruning
        min_outgoing = [min(dist[i][j] for j in range(n) if j != i) for i in range(n)]

        best_score = float("inf")
        best_route = None
        evals = 0

        # Initial upper bound from Nearest Neighbour
        nn_init = self.solve_nearest_neighbour()
        best_score = nn_init.score
        best_route = list(nn_init.route)

        def dfs(u: int, visited_mask: int, current_time: float, current_dist: float, current_penalty: float, path: list[int]):
            nonlocal best_score, best_route, evals
            evals += 1

            # Lower bound pruning: current cost + minimum remaining distance
            rem_dist_lb = sum(min_outgoing[i] for i in range(n) if not (visited_mask & (1 << i)))
            current_cost_lb = current_dist + rem_dist_lb + (tw_weight * speed * current_penalty)
            if current_cost_lb >= best_score - 1e-9:
                return

            if visited_mask == (1 << n) - 1:
                # All stops visited
                if return_to_depot:
                    ret_dist = dist[u][0]
                    tot_dist = current_dist + ret_dist
                    t_ret = current_time + (ret_dist / speed)
                    tot_penalty = current_penalty
                    if t_ret > depot_close:
                        tot_penalty += (t_ret - depot_close)
                else:
                    tot_dist = current_dist
                    tot_penalty = current_penalty

                tot_score = tot_dist + (tw_weight * speed * tot_penalty)
                if tot_score < best_score:
                    best_score = tot_score
                    best_route = list(path)
                return

            # Branch to unvisited stops
            for v in range(1, n):
                if not (visited_mask & (1 << v)):
                    d_uv = dist[u][v]
                    arr_time = current_time + (d_uv / speed)
                    earliest, latest = time_windows[v]
                    arr_actual = max(arr_time, float(earliest))
                    lateness = max(0.0, arr_time - latest)
                    depart_time = arr_actual + service_times[v]

                    dfs(v, visited_mask | (1 << v), depart_time, current_dist + d_uv, current_penalty + lateness, path + [v])

        dfs(0, 1, start_time, 0.0, 0.0, [0])

        cpu = self._timer_elapsed_ms(t_cpu0, t_wall0)
        final_route = best_route if best_route is not None else list(range(n))
        return self._make_result("Exact (Brute-Force)", final_route, cpu, evals, 0, evaluations=evals)

    # ── Result Struct Factory ────────────────────────────────────────────────

    def _make_result(
        self,
        name: str,
        route: list[int],
        cpu_ms: float,
        iterations: int,
        improvements: int,
        score_history: Optional[list[float]] = None,
        evaluation_history: Optional[list[int]] = None,
        evaluations: int = 0,
    ) -> BenchmarkResult:
        # Guarantee route starts at index 0 (Depot)
        if route and route[0] != 0 and 0 in route:
            idx0 = route.index(0)
            route = route[idx0:] + route[:idx0]

        score, _ = self.g.route_score(route, self.tw_weight)
        pen_min, late_stops, max_late, wait_min, elapsed_min = self.g._simulate_schedule(route)
        dist = self.g.route_distance(route)
        travel_min = self.g.travel_time(route)
        service_min = sum(self.g.nodes[i].service_time for i in route[1:])

        return BenchmarkResult(
            algorithm=name,
            route=route,
            total_distance=dist,
            tw_penalty=pen_min,
            score=score,
            estimated_time_min=elapsed_min,
            cpu_time_ms=cpu_ms,
            iterations=iterations,
            improvements=improvements,
            waiting_time_min=wait_min,
            travel_time_min=travel_min,
            service_time_min=service_min,
            total_elapsed_min=elapsed_min,
            route_labels=self.g.node_labels(route),
            score_history=score_history if score_history is not None else [score],
            evaluation_history=evaluation_history if evaluation_history is not None else [evaluations],
            evaluations=evaluations,
            feasible=(pen_min == 0.0),
            late_stops=late_stops,
            max_lateness_min=max_late,
            optimality_gap_pct=None,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Seed Benchmarking (FR-11, FR-12)
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_multi_seed(
    graph: DeliveryGraph,
    algorithms: Optional[list[str]] = None,
    n_seeds: int = 20,
    tw_weight: float = 1.5,
    exact_max_nodes: int = 12,
) -> dict[str, dict[str, Any]]:
    """
    Executes all methods across multiple random seeds and returns aggregate metrics:
    Mean, Std Dev, Best, Worst, Feasibility %, Mean Runtime, and Mean Optimality Gap vs Exact.
    """
    if algorithms is None:
        algorithms = ["nn", "greedy", "2opt", "hill", "sa"]
        if graph.n <= exact_max_nodes:
            algorithms.append("exact")

    # Run Exact once to establish ground-truth reference if applicable
    exact_score = None
    if graph.n <= exact_max_nodes:
        ref_solver = TSPSolver(graph, tw_weight=tw_weight)
        exact_res = ref_solver.solve_exact(max_nodes=exact_max_nodes)
        if exact_res.feasible and exact_res.score < float("inf"):
            exact_score = exact_res.score

    raw_results: dict[str, list[BenchmarkResult]] = {algo: [] for algo in algorithms}

    # Stochastic algorithms run across distinct seeds; deterministic algorithms run once
    stochastic_algos = {"hill", "sa"}

    for s_idx in range(n_seeds):
        seed_val = 1000 + s_idx * 7
        solver = TSPSolver(graph, tw_weight=tw_weight, seed=seed_val)

        for algo in algorithms:
            if algo in stochastic_algos or s_idx == 0:
                if algo == "hill":
                    res = solver.solve_hill_climbing(seed=seed_val)
                elif algo == "sa":
                    res = solver.solve_simulated_annealing(seed=seed_val)
                elif algo == "nn":
                    res = solver.solve_nearest_neighbour()
                elif algo == "greedy":
                    res = solver.solve_cheapest_insertion()
                elif algo == "2opt":
                    res = solver.solve_2opt()
                elif algo == "exact":
                    res = solver.solve_exact(max_nodes=exact_max_nodes)
                else:
                    continue

                if exact_score is not None and res.score < float("inf"):
                    res.optimality_gap_pct = ((res.score - exact_score) / exact_score) * 100.0

                raw_results[algo].append(res)

    summary: dict[str, dict[str, Any]] = {}
    for algo, res_list in raw_results.items():
        if not res_list:
            continue
        scores = [r.score for r in res_list if r.score < float("inf")]
        distances = [r.total_distance for r in res_list]
        latenesses = [r.tw_penalty for r in res_list]
        runtimes = [r.cpu_time_ms for r in res_list]
        feasibles = [1 if r.feasible else 0 for r in res_list]
        gaps = [r.optimality_gap_pct for r in res_list if r.optimality_gap_pct is not None]

        algo_display = res_list[0].algorithm

        summary[algo] = {
            "algorithm": algo_display,
            "runs": len(res_list),
            "mean_score": float(np.mean(scores)) if scores else float("inf"),
            "std_score": float(np.std(scores)) if len(scores) > 1 else 0.0,
            "best_score": float(np.min(scores)) if scores else float("inf"),
            "worst_score": float(np.max(scores)) if scores else float("inf"),
            "mean_distance": float(np.mean(distances)),
            "mean_lateness": float(np.mean(latenesses)),
            "feasibility_pct": float(np.mean(feasibles) * 100.0),
            "mean_cpu_ms": float(np.mean(runtimes)),
            "mean_gap_pct": float(np.mean(gaps)) if gaps else None,
        }

    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Visual Convergence Chart Generator (Evaluation-Based Shared Axis)
# ──────────────────────────────────────────────────────────────────────────────

def render_convergence_chart(
    results: dict[str, BenchmarkResult],
    width: int = 700,
    height: int = 400,
) -> np.ndarray:
    """
    Renders an evaluation-based convergence chart to an RGB image using OpenCV.
    Plots Hill Climbing and Simulated Annealing on a logarithmic evaluations axis,
    preventing fast-converging methods like Hill Climbing from being crushed on the left edge.
    Includes the Exact Global Optimum as a clean horizontal reference line.
    """
    import cv2

    canvas = np.full((height, width, 3), 26, dtype=np.uint8)

    # Filter methods with evaluation histories: Plot only HC and SA per project spec
    allowed_keys = ["hill", "sa"]
    histories = {}
    for k in allowed_keys:
        if k in results:
            r = results[k]
            if getattr(r, "score_history", None) and getattr(r, "evaluation_history", None):
                if len(r.score_history) >= 1 and len(r.evaluation_history) >= 1:
                    histories[k] = (r.evaluation_history, r.score_history)

    # Fallback to any available histories if neither HC nor SA was executed
    if not histories:
        for k, r in results.items():
            if getattr(r, "score_history", None) and getattr(r, "evaluation_history", None):
                if len(r.score_history) >= 1 and len(r.evaluation_history) >= 1:
                    histories[k] = (r.evaluation_history, r.score_history)

    if not histories:
        cv2.putText(
            canvas, "No convergence history to display.",
            (width // 4, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA,
        )
        return canvas

    pad_left, pad_right, pad_top, pad_bottom = 80, 35, 60, 60
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    # Determine evaluation (X) and score (Y) bounds
    max_evals = max(max(evals_list) for evals_list, _ in histories.values())
    max_evals = max(max_evals, 10)

    # Log10 evaluation axis: from 1 to max_evals
    log_min = 0.0  # log10(1)
    log_max = math.log10(max_evals)
    if log_max <= log_min + 1e-4:
        log_max = log_min + 1.0

    all_scores = [s for _, scores_list in histories.values() for s in scores_list]
    if "exact" in results and results["exact"].feasible and results["exact"].score < float("inf"):
        all_scores.append(results["exact"].score)

    min_s = min(all_scores) * 0.96
    max_s = max(all_scores) * 1.04
    if max_s <= min_s + 1e-4:
        max_s = min_s + 10.0

    # Draw chart background & border
    cv2.rectangle(canvas, (pad_left, pad_top), (pad_left + plot_w, pad_top + plot_h), (40, 40, 48), -1)
    cv2.rectangle(canvas, (pad_left, pad_top), (pad_left + plot_w, pad_top + plot_h), (80, 80, 90), 1)

    # Title & Subtitle
    cv2.putText(
        canvas, "Convergence: Solution Score vs. Route Evaluations (Log Scale)",
        (pad_left, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, "Hill Climbing vs. Simulated Annealing (Evaluation-Budgeted)",
        (pad_left, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 180), 1, cv2.LINE_AA,
    )

    # Gridlines and Y ticks (Score)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y_val = min_s + frac * (max_s - min_s)
        py = int(pad_top + plot_h - frac * plot_h)
        cv2.line(canvas, (pad_left, py), (pad_left + plot_w, py), (55, 55, 65), 1)
        cv2.putText(canvas, f"{y_val:.0f}", (14, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1)

    # Gridlines and X ticks on Log Scale: 1, 10, 100, 1000, max_evals
    tick_candidates = [1, 10, 100, 1000, max_evals]
    drawn_ticks = set()
    for tick in tick_candidates:
        if tick > max_evals or tick in drawn_ticks:
            continue
        drawn_ticks.add(tick)
        log_val = math.log10(max(1, tick))
        frac_x = (log_val - log_min) / (log_max - log_min)
        px = int(pad_left + frac_x * plot_w)
        cv2.line(canvas, (px, pad_top), (px, pad_top + plot_h), (55, 55, 65), 1)
        label = "1k" if tick == 1000 else str(tick)
        cv2.putText(canvas, label, (px - 10, pad_top + plot_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (170, 170, 170), 1)

    colors = {
        "hill": (30, 160, 255),    # Amber / Orange (BGR)
        "sa":   (70, 230, 90),     # Emerald Green (BGR)
        "2opt": (240, 200, 0),     # Cyan / Yellow
        "greedy": (220, 100, 240), # Violet
    }

    legend_y = pad_top + 22
    for algo_key, (eval_pts, score_pts) in histories.items():
        color = colors.get(algo_key, (200, 200, 200))
        pts = []
        for e_val, s_val in zip(eval_pts, score_pts):
            log_e = math.log10(max(1, e_val))
            frac_x = (log_e - log_min) / (log_max - log_min)
            px = int(pad_left + frac_x * plot_w)
            frac_y = (s_val - min_s) / (max_s - min_s)
            py = int(pad_top + plot_h - frac_y * plot_h)
            pts.append((px, py))

        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], color, 2, cv2.LINE_AA)

        name = results[algo_key].algorithm
        pct = ((score_pts[0] - score_pts[-1]) / score_pts[0] * 100) if score_pts[0] > 0 else 0
        leg_str = f"{name}: {score_pts[0]:.0f} -> {score_pts[-1]:.0f} (-{pct:.1f}%)"
        cv2.circle(canvas, (pad_left + 15, legend_y), 5, color, -1)
        cv2.putText(canvas, leg_str, (pad_left + 26, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
        legend_y += 20

    # Draw exact optimal baseline if available
    if "exact" in results and results["exact"].feasible and results["exact"].score < float("inf"):
        exact_s = results["exact"].score
        frac = (exact_s - min_s) / (max_s - min_s)
        py = int(pad_top + plot_h - frac * plot_h)
        for x_step in range(pad_left, pad_left + plot_w, 14):
            cv2.line(canvas, (x_step, py), (min(x_step + 7, pad_left + plot_w), py), (0, 0, 240), 2)
        cv2.putText(
            canvas, f"Exact Global Optimum: {exact_s:.1f}",
            (pad_left + plot_w - 230, py - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 240), 1, cv2.LINE_AA,
        )

    # X axis title
    cv2.putText(
        canvas, "Route Evaluations (Logarithmic Scale: log10(evaluations))",
        (pad_left + plot_w // 5, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA,
    )

    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────────────────────

def build_graph_from_detected_nodes(
    detected_nodes,
    dist_matrix: Optional[list[list[float]]] = None,
    return_to_depot: bool = True,
    start_time: float = 540.0,
    depot_close_time: float = 1080.0,
    speed_px_per_min: float = 60.0,
    hard_windows: bool = False,
) -> DeliveryGraph:
    """
    Convert vision.DetectedNode objects to NodeInfo and build a DeliveryGraph.
    Guarantees the verified primary Red Depot is always at Index 0.
    """
    nodes_copy = list(detected_nodes)
    matrix_copy = [list(row) for row in dist_matrix] if dist_matrix is not None else None

    # Defensive check: ensure the node marked as depot is strictly at Index 0
    depot_indices = [
        i for i, n in enumerate(nodes_copy)
        if getattr(n, "is_depot", False) or getattr(n, "node_id", "") == "DEPOT"
    ]
    if depot_indices and depot_indices[0] != 0:
        d_idx = depot_indices[0]
        depot_node = nodes_copy.pop(d_idx)
        nodes_copy.insert(0, depot_node)

        if matrix_copy is not None:
            order = [d_idx] + [i for i in range(len(nodes_copy)) if i != d_idx]
            n_len = len(order)
            reordered = [[0.0] * n_len for _ in range(n_len)]
            for r_new, r_old in enumerate(order):
                for c_new, c_old in enumerate(order):
                    reordered[r_new][c_new] = matrix_copy[r_old][c_old]
            matrix_copy = reordered

    node_infos = [
        NodeInfo(
            node_id=dn.node_id,
            x=float(dn.cx),
            y=float(dn.cy),
            time_window=getattr(dn, "time_window", (540, 1080)),
            service_time=getattr(dn, "service_time", 5),
            address=getattr(dn, "address", ""),
        )
        for dn in nodes_copy
    ]
    return DeliveryGraph(
        node_infos,
        dist_matrix=matrix_copy,
        return_to_depot=return_to_depot,
        start_time=start_time,
        depot_close_time=depot_close_time,
        speed_px_per_min=speed_px_per_min,
        hard_windows=hard_windows,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI Smoke Test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    test_nodes = [
        NodeInfo("DEPOT", 100, 300, (540, 1080), address="Central Hub"),
        NodeInfo("H1", 200, 150, (540, 600), address="14 Oak Lane"),
        NodeInfo("H2", 350, 400, (600, 660), address="88 Elm St"),
        NodeInfo("H3", 500, 200, (660, 720), address="204 Maple Ave"),
        NodeInfo("H4", 600, 350, (720, 780), address="55 Pine Rd"),
        NodeInfo("H5", 250, 500, (780, 840), address="12 Birch Blvd"),
        NodeInfo("H6", 450, 100, (840, 900), address="90 Cedar Way"),
    ]

    graph = DeliveryGraph(test_nodes)
    solver = TSPSolver(graph, seed=42)

    print("\n" + "=" * 80)
    print("  Last-Mile TSP-TW Benchmark (Single Run, Seed=42)")
    print("=" * 80)
    print(f"{'Algorithm':<22} {'Distance':>9} {'TW Pen':>8} {'Score':>9} {'Feasible':>9} {'Time(ms)':>9} {'Gap%':>7}")
    print("-" * 80)

    res = solver.run_all()
    for name, r in res.items():
        gap_str = f"{r.optimality_gap_pct:.2f}%" if r.optimality_gap_pct is not None else "N/A"
        feas_str = "YES" if r.feasible else f"NO({len(r.late_stops)})"
        print(f"{r.algorithm:<22} {r.total_distance:>9.1f} {r.tw_penalty:>8.1f} {r.score:>9.1f} {feas_str:>9} {r.cpu_time_ms:>9.2f} {gap_str:>7}")

    print("\n" + "=" * 80)
    print("  Multi-Seed Validation Study (10 Seeds)")
    print("=" * 80)
    summary = benchmark_multi_seed(graph, n_seeds=10)
    print(f"{'Algorithm':<22} {'Mean Score':>11} {'Std Dev':>8} {'Best':>8} {'Feas%':>7} {'CPU(ms)':>9} {'Mean Gap%':>10}")
    print("-" * 80)
    for k, s in summary.items():
        mgap = f"{s['mean_gap_pct']:.2f}%" if s['mean_gap_pct'] is not None else "N/A"
        print(f"{s['algorithm']:<22} {s['mean_score']:>11.1f} {s['std_score']:>8.1f} {s['best_score']:>8.1f} {s['feasibility_pct']:>6.0f}% {s['mean_cpu_ms']:>9.2f} {mgap:>10}")
