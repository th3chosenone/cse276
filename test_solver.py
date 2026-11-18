"""
test_solver.py — Unit tests for the TSP solver module
Run with: python -m pytest test_solver.py -v
"""

import math
import pytest
from solver import NodeInfo, DeliveryGraph, TSPSolver, build_graph_from_detected_nodes


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_nodes():
    """6-node test instance (depot + 5 houses on a circle)."""
    nodes = [NodeInfo("DEPOT", 0.0, 0.0, (0, 1440))]
    for i in range(5):
        angle = math.tau * i / 5
        x = 200 * math.cos(angle)
        y = 200 * math.sin(angle)
        tw_start = 540 + i * 60
        nodes.append(NodeInfo(f"H{i+1}", x, y, (tw_start, tw_start + 60)))
    return nodes


@pytest.fixture
def tiny_nodes():
    """3-node instance for exact solver verification."""
    return [
        NodeInfo("DEPOT", 0.0,   0.0,   (0, 1440)),
        NodeInfo("H1",    100.0, 0.0,   (540, 600)),
        NodeInfo("H2",    50.0,  86.6,  (600, 660)),
    ]


@pytest.fixture
def small_graph(small_nodes):
    return DeliveryGraph(small_nodes)


@pytest.fixture
def tiny_graph(tiny_nodes):
    return DeliveryGraph(tiny_nodes)


# ──────────────────────────────────────────────────────────────────────────────
# DeliveryGraph tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDeliveryGraph:
    def test_matrix_symmetry(self, small_graph):
        n = small_graph.n
        for i in range(n):
            for j in range(n):
                assert math.isclose(small_graph.dist[i][j], small_graph.dist[j][i], rel_tol=1e-9)

    def test_zero_diagonal(self, small_graph):
        for i in range(small_graph.n):
            assert small_graph.dist[i][i] == 0.0

    def test_triangle_inequality(self, small_graph):
        """dist(i,k) <= dist(i,j) + dist(j,k) for all triples."""
        d = small_graph.dist
        n = small_graph.n
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    assert d[i][k] <= d[i][j] + d[j][k] + 1e-9

    def test_route_distance_positive(self, small_graph):
        route = list(range(small_graph.n))
        assert small_graph.route_distance(route) > 0

    def test_route_distance_full_cycle(self, tiny_graph):
        route = [0, 1, 2]
        d = tiny_graph.route_distance(route)
        # Manual: 0→1 + 1→2 + 2→0
        d01 = math.hypot(100, 0)
        d12 = math.hypot(50, 86.6)
        d20 = math.hypot(50, 86.6)
        expected = d01 + d12 + d20
        assert math.isclose(d, expected, rel_tol=1e-4)

    def test_minimum_nodes_error(self):
        with pytest.raises(ValueError):
            DeliveryGraph([NodeInfo("DEPOT", 0, 0)])

    def test_estimated_time_positive(self, small_graph):
        route = list(range(small_graph.n))
        assert small_graph.estimated_time(route) > 0


# ──────────────────────────────────────────────────────────────────────────────
# TSPSolver algorithm tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTSPSolver:
    def _check_result(self, result, graph):
        """Common assertions for any algorithm result."""
        assert len(result.route) == graph.n
        assert result.route[0] == 0            # starts at depot
        assert set(result.route) == set(range(graph.n))   # visits all
        assert result.total_distance > 0
        assert result.cpu_time_ms >= 0
        assert result.score >= result.total_distance

    def test_nn_visits_all(self, small_graph):
        solver = TSPSolver(small_graph)
        r = solver.solve_nearest_neighbour()
        self._check_result(r, small_graph)

    def test_greedy_insertion_visits_all(self, small_graph):
        solver = TSPSolver(small_graph)
        r = solver.solve_greedy_insertion()
        self._check_result(r, small_graph)

    def test_2opt_improves_or_equals_nn(self, small_graph):
        solver = TSPSolver(small_graph)
        nn_r   = solver.solve_nearest_neighbour()
        opt_r  = solver.solve_2opt()
        self._check_result(opt_r, small_graph)
        # 2-opt should not be worse than its seed (NN)
        assert opt_r.score <= nn_r.score + 1e-6

    def test_hill_climbing_visits_all(self, small_graph):
        solver = TSPSolver(small_graph)
        r = solver.solve_hill_climbing(restarts=3)
        self._check_result(r, small_graph)

    def test_sa_visits_all(self, small_graph):
        solver = TSPSolver(small_graph)
        r = solver.solve_simulated_annealing()
        self._check_result(r, small_graph)

    def test_exact_tiny(self, tiny_graph):
        solver = TSPSolver(tiny_graph)
        r = solver.solve_exact()
        self._check_result(r, tiny_graph)
        assert r.algorithm == "Exact (Brute-Force)"

    def test_exact_fallback_large(self, small_graph):
        """With > 10 nodes, exact solver cleanly refuses instead of substituting a heuristic."""
        # Build an 11-node graph
        nodes = [NodeInfo(f"N{i}", float(i*50), 0.0) for i in range(11)]
        graph  = DeliveryGraph(nodes)
        solver = TSPSolver(graph)
        r = solver.solve_exact(max_nodes=10)
        assert "refused" in r.algorithm.lower() or "skipped" in r.algorithm.lower() or "fallback" in r.algorithm.lower()
        assert r.feasible is False

    def test_cheapest_insertion_does_not_fallback(self):
        """Verify cheapest insertion constructs a valid tour independently without NN fallback."""
        # Collinear layout where NN produces a suboptimal zigzag
        nodes = [
            NodeInfo("DEPOT", 0.0, 0.0),
            NodeInfo("H1", 10.0, 0.0),
            NodeInfo("H2", 20.0, 0.0),
            NodeInfo("H3", -15.0, 0.0),
            NodeInfo("H4", 30.0, 0.0),
        ]
        graph = DeliveryGraph(nodes)
        solver = TSPSolver(graph)
        r_ci = solver.solve_cheapest_insertion()
        assert r_ci.algorithm == "Cheapest Insertion"
        assert len(r_ci.route) == 5
        assert set(r_ci.route) == {0, 1, 2, 3, 4}
        assert r_ci.route[0] == 0

    def test_seed_reproducibility(self, small_graph):
        """Stochastic algorithms must yield identical results when given the same seed."""
        solver1 = TSPSolver(small_graph, seed=42)
        r1 = solver1.solve_simulated_annealing()

        solver2 = TSPSolver(small_graph, seed=42)
        r2 = solver2.solve_simulated_annealing()

        assert r1.route == r2.route
        assert math.isclose(r1.score, r2.score, rel_tol=1e-7)

    def test_feasibility_flags(self):
        """Test feasible vs infeasible reporting."""
        # Feasible instance with wide windows
        nodes_feas = [
            NodeInfo("DEPOT", 0.0, 0.0, (540, 1440)),
            NodeInfo("H1", 60.0, 0.0, (540, 700)),
        ]
        g_feas = DeliveryGraph(nodes_feas, start_time=540.0)
        s_feas = TSPSolver(g_feas)
        res_f = s_feas.solve_nearest_neighbour()
        assert res_f.feasible is True
        assert len(res_f.late_stops) == 0

        # Infeasible instance with impossible window (requires 10 min travel, window closes in 2 min)
        nodes_infeas = [
            NodeInfo("DEPOT", 0.0, 0.0, (540, 1440)),
            NodeInfo("H1", 600.0, 0.0, (540, 542)), # 600 px at 60 px/min = 10 min, deadline 542 (2 min)
        ]
        g_infeas = DeliveryGraph(nodes_infeas, start_time=540.0)
        s_infeas = TSPSolver(g_infeas)
        res_inf = s_infeas.solve_nearest_neighbour()
        assert res_inf.feasible is False
        assert "H1" in res_inf.late_stops
        assert res_inf.max_lateness_min > 0

    def test_multi_seed_benchmarking(self, small_graph):
        """Verify multi-seed benchmark returns summary stats."""
        from solver import benchmark_multi_seed
        summary = benchmark_multi_seed(small_graph, algorithms=["nn", "greedy", "2opt", "sa"], n_seeds=5)
        assert "sa" in summary
        assert summary["sa"]["runs"] == 5
        assert summary["sa"]["mean_score"] > 0
        assert summary["sa"]["best_score"] <= summary["sa"]["worst_score"]

    def test_run_all_returns_dict(self, small_graph):
        solver  = TSPSolver(small_graph)
        results = solver.run_all(["nn", "2opt", "sa"])
        assert set(results.keys()) == {"nn", "2opt", "sa"}
        for r in results.values():
            self._check_result(r, small_graph)

    def test_benchmark_labels(self, small_graph):
        solver = TSPSolver(small_graph)
        r = solver.solve_nearest_neighbour()
        assert r.route_labels[0] == "DEPOT"
        assert len(r.route_labels) == small_graph.n


# ──────────────────────────────────────────────────────────────────────────────
# Integration: vision → solver pipeline (mock detected nodes)
# ──────────────────────────────────────────────────────────────────────────────

class TestVisionSolverIntegration:
    def test_build_graph_from_detected_nodes(self):
        """Simulate what the UI does after vision detection."""
        # Minimal mock of vision.DetectedNode
        class MockNode:
            def __init__(self, nid, cx, cy, tw, is_depot=False):
                self.node_id = nid
                self.cx = cx; self.cy = cy
                self.time_window = tw
                self.is_depot = is_depot

        detected = [
            MockNode("DEPOT", 100, 300, (0, 1440), is_depot=True),
            MockNode("H1",    200, 150, (540, 600)),
            MockNode("H2",    350, 400, (600, 660)),
            MockNode("H3",    500, 200, (660, 720)),
        ]

        graph  = build_graph_from_detected_nodes(detected)
        assert graph.n == 4
        solver = TSPSolver(graph)
        result = solver.solve_nearest_neighbour()
        assert len(result.route) == 4
        assert result.route[0] == 0

    def test_build_graph_with_custom_dist_matrix(self):
        """Verify precomputed A* distance matrix override."""
        class MockNode:
            def __init__(self, nid, cx, cy):
                self.node_id = nid
                self.cx = cx
                self.cy = cy
                self.time_window = (0, 1440)
                self.service_time = 5

        detected = [MockNode("DEPOT", 0, 0), MockNode("H1", 10, 0), MockNode("H2", 20, 0)]
        custom_matrix = [
            [0.0, 50.0, 100.0],
            [50.0, 0.0, 50.0],
            [100.0, 50.0, 0.0],
        ]
        graph = build_graph_from_detected_nodes(detected, dist_matrix=custom_matrix)
        assert graph.dist == custom_matrix
        solver = TSPSolver(graph)
        r = solver.solve_nearest_neighbour()
        assert r.total_distance == 200.0

    def test_time_accounting_breakdown(self):
        """Verify travel, service, waiting time, and total elapsed duration."""
        # Depot starts at 540 (09:00).
        # H1 is 60px away (1 min travel at 60px/min). Window starts at 600 (driver arrives at 541 -> waits 59 min).
        # H1 service time = 5 min (departs at 605). Return to depot is 60px (1 min -> arrives depot at 606).
        nodes = [
            NodeInfo("DEPOT", 0.0, 0.0, (540, 1080), service_time=0),
            NodeInfo("H1", 60.0, 0.0, (600, 720), service_time=5),
        ]
        graph = DeliveryGraph(nodes, start_time=540.0, speed_px_per_min=60.0)
        solver = TSPSolver(graph)
        r = solver.solve_nearest_neighbour()

        assert math.isclose(r.travel_time_min, 2.0, rel_tol=1e-4)      # 1 min out + 1 min back
        assert math.isclose(r.service_time_min, 5.0, rel_tol=1e-4)     # 5 min service
        assert math.isclose(r.waiting_time_min, 59.0, rel_tol=1e-4)    # 541 to 600
        assert math.isclose(r.total_elapsed_min, 66.0, rel_tol=1e-4)   # 2 + 5 + 59 = 66
        assert math.isclose(r.estimated_time_min, 66.0, rel_tol=1e-4)

    def test_sa_exploration_and_cooling(self, small_graph):
        """Verify Simulated Annealing logs evaluation history and performs multi-step cooling."""
        solver = TSPSolver(small_graph, seed=42)
        r = solver.solve_simulated_annealing(max_evaluations=500)
        assert len(r.score_history) > 1
        assert len(r.evaluation_history) > 1
        assert r.evaluations <= 500
        assert r.cpu_time_ms >= 0

    def test_best_heuristic_gap_excludes_exact(self, tiny_graph):
        """Verify Best Heuristic Gap evaluates heuristics against Exact without exact setting its own gap."""
        solver = TSPSolver(tiny_graph, seed=42)
        results = solver.run_all(["nn", "exact"])
        assert "exact" in results
        assert "nn" in results

        exact_res = results["exact"]
        nn_res = results["nn"]
        assert exact_res.optimality_gap_pct == 0.0

        # NN gap must be >= 0% against exact
        if nn_res.optimality_gap_pct is not None:
            assert nn_res.optimality_gap_pct >= -1e-6

