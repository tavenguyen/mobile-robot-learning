"""Đường ngắn nhất trên lưới và bám waypoint P an toàn."""
import math
import unittest
from types import SimpleNamespace

import numpy as np
from matplotlib.patches import Circle, Rectangle

from src.Obstacle import Obstacle
from src.global_planner import (
    GlobalPlannerController, _CollisionMap, _goal_heuristics,
    _octile_distance, plan_path,
)


class OctileHeuristicTests(unittest.TestCase):
    def test_horizontal_diagonal_and_mixed_distances_in_meters(self):
        for resolution in (0.5, 1.0):
            with self.subTest(resolution=resolution):
                origin = np.array([2, -3])*resolution
                goals = origin+np.array([[0, 0], [3, 0], [2, 2], [-4, 2]])*resolution
                expected = np.array([0, 3, 2*math.sqrt(2), 2+2*math.sqrt(2)])*resolution
                np.testing.assert_allclose(_octile_distance(origin, goals), expected)
                np.testing.assert_allclose(_octile_distance(goals, origin), expected)

    def test_offgrid_goal_keeps_euclidean_connector_cost(self):
        coordinates = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
        goal = np.array([0.25, 0.1])
        costs = np.linalg.norm(coordinates-goal, axis=1)
        heuristics = _goal_heuristics(coordinates, dict(enumerate(costs)))
        # Octile trực tiếp tới goal lệch lưới có thể đánh giá quá cao.
        self.assertAlmostEqual(heuristics[0], math.hypot(0.25, 0.1))
        self.assertLess(heuristics[0], _octile_distance(coordinates[0], goal))
        np.testing.assert_allclose(heuristics, costs)

    def test_connector_heuristic_is_consistent_and_admissible(self):
        for resolution in (0.5, 1.0):
            with self.subTest(resolution=resolution):
                coordinates = np.array([(x, y) for x in range(4) for y in range(4)])*resolution
                goal = np.array([2.3, 1.2])*resolution
                goal_nodes = [9, 10, 13, 14]
                connectors = {node: float(np.linalg.norm(coordinates[node]-goal))
                              for node in goal_nodes}
                heuristics = _goal_heuristics(coordinates, connectors)
                for node, cost in connectors.items():
                    self.assertLessEqual(heuristics[node], cost+1e-12)
                for node, position in enumerate(coordinates):
                    for neighbor, target in enumerate(coordinates):
                        delta = np.abs(position-target)
                        if np.max(delta) <= resolution+1e-12:
                            self.assertLessEqual(heuristics[node],
                                                 math.dist(position, target)+heuristics[neighbor]+1e-12)
                    # Dijkstra là tham chiếu độc lập cho chi phí tới goal thật.
                    result = plan_path(Obstacle(), position, goal, algorithm='Dijkstra',
                                       resolution=resolution,
                                       bounds=(0, 3*resolution, 0, 3*resolution), clearance=0)
                    self.assertEqual(result.status, 'OK')
                    self.assertLessEqual(heuristics[node], result.cost+1e-10)

    def test_offgrid_endpoints_and_obstacles_keep_dijkstra_cost(self):
        obstacles = Obstacle()
        obstacles.add(Circle((0, 0), 0.6))
        obstacles.add(Rectangle((1, -2), 0.3, 2.5, angle=20))
        for resolution in (0.5, 1.0):
            with self.subTest(resolution=resolution):
                results = [plan_path(obstacles, (-2.3, -0.2), (2.2, 1.3),
                                     algorithm=algorithm, resolution=resolution,
                                     bounds=(-4, 4, -4, 4))
                           for algorithm in ('Dijkstra', 'A*')]
                self.assertEqual(results[0].status, 'OK')
                self.assertEqual(results[1].status, 'OK')
                self.assertAlmostEqual(results[0].cost, results[1].cost)
                self.assertLessEqual(results[1].expanded, results[0].expanded)


class GlobalPlannerTests(unittest.TestCase):
    def setUp(self):
        self.obstacles = Obstacle()

    def test_empty_map_algorithms_same_cost_exact_endpoints(self):
        start, goal = (-1.2, -0.3), (4.1, 2.2)
        results = [plan_path(self.obstacles, start, goal, algorithm=algorithm)
                   for algorithm in ('Dijkstra', 'A*')]
        for result in results:
            self.assertEqual(result.status, 'OK', result.message)
            self.assertEqual(result.path[0], start)
            self.assertEqual(result.path[-1], goal)
            measured = sum(math.dist(a, b) for a, b in zip(result.path, result.path[1:]))
            self.assertAlmostEqual(result.cost, measured)
        self.assertAlmostEqual(results[0].cost, results[1].cost)
        self.assertLess(results[1].expanded, results[0].expanded)

    def test_circle_and_rotated_rectangle_safe_segments(self):
        self.obstacles.add(Circle((0, 0), 1))
        self.obstacles.add(Rectangle((2, -2), 0.2, 3, angle=23))
        clearance = 0.3
        for algorithm in ('Dijkstra', 'A*'):
            result = plan_path(self.obstacles, (-3, 0), (5, 0),
                               algorithm=algorithm, clearance=clearance)
            self.assertEqual(result.status, 'OK', result.message)
            collision = _CollisionMap(self.obstacles, clearance)
            for a, b in zip(result.path, result.path[1:]):
                self.assertTrue(collision.free(a, b))

    def test_no_diagonal_corner_cut(self):
        # Hai ô kề bị chặn nhưng đoạn chéo hình học vẫn không chạm circle.
        self.obstacles.add(Circle((1, 0), 0.1))
        self.obstacles.add(Circle((0, 1), 0.1))
        result = plan_path(self.obstacles, (0, 0), (1, 1), resolution=1,
                           bounds=(-1, 2, -1, 2), clearance=0)
        self.assertEqual(result.status, 'OK')
        self.assertGreater(result.cost, math.sqrt(2))

    def test_unreachable_and_invalid_endpoints(self):
        self.obstacles.add(Rectangle((-0.2, -3), 0.4, 6))
        result = plan_path(self.obstacles, (-1, 0), (1, 0), bounds=(-2, 2, -2, 2))
        self.assertEqual(result.status, 'FAILED')
        self.assertEqual(result.path, [])
        self.assertTrue(math.isinf(result.cost))
        for start, goal in [((0, 0), (1, 0)), ((-1, 0), (0, 0)),
                            ((-21, 0), (1, 0))]:
            self.assertEqual(plan_path(self.obstacles, start, goal).status, 'FAILED')

    def test_same_start_goal(self):
        result = plan_path(self.obstacles, (1, 2), (1, 2))
        self.assertEqual(result.path, [(1, 2)])
        self.assertEqual(result.cost, 0)

    def test_invalid_configuration(self):
        for values in ({'resolution': 0}, {'clearance': -1},
                       {'bounds': (1, -1, -1, 1)}, {'bounds': (1, 2)},
                       {'algorithm': 'BFS'}, {'resolution': math.nan}):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    plan_path(self.obstacles, (0, 0), (1, 1), **values)

    def test_thin_rectangle_blocks_segment_and_tangent_circle_is_collision(self):
        self.obstacles.add(Rectangle((0.123, -1), 0.001, 2))
        collision = _CollisionMap(self.obstacles, 0)
        self.assertFalse(collision.free((-1, 0), (1, 0)))
        self.obstacles.clear()
        self.obstacles.add(Circle((0, 0), 1))
        collision = _CollisionMap(self.obstacles, 0.2)
        self.assertFalse(collision.free((-2, 1.2), (2, 1.2)))
        self.assertTrue(collision.free((-2, 1.201), (2, 1.201)))

    def test_rotated_rectangle_clearance_matches_nearest_point(self):
        self.obstacles.add(Rectangle((1, 2), 2, 1, angle=37, rotation_point='center'))
        collision = _CollisionMap(self.obstacles, 0.3)
        points = np.random.default_rng(42).uniform(-2, 5, (200, 2))
        expected = [self.obstacles.nearest(p)[2] > 0.3+1e-10 for p in points]
        np.testing.assert_array_equal(collision.free(points, points), expected)

    def test_u_trap_path_exits_open_side(self):
        self.obstacles.add(Rectangle((-3, -3), 7, 0.5))
        self.obstacles.add(Rectangle((-3, 2.5), 7, 0.5))
        self.obstacles.add(Rectangle((3, -3), 0.5, 6))
        results = [plan_path(self.obstacles, (0, 0), (8, 0), algorithm=algorithm,
                             clearance=1.27) for algorithm in ('Dijkstra', 'A*')]
        for result in results:
            self.assertEqual(result.status, 'OK', result.message)
            self.assertLess(min(p[0] for p in result.path), -3)
            self.assertGreater(result.cost, 8)
        self.assertAlmostEqual(results[0].cost, results[1].cost)


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.obstacles = Obstacle()
        self.controller = GlobalPlannerController(self.obstacles)
        self.robot = SimpleNamespace(x=0., y=0., theta=0., L=0.5,
                                     robot_length=0.2, robot_width=0.2)

    def test_turn_in_place_then_advance_and_arrive(self):
        first = self.controller.step(self.robot, (0, 2), 0.1)
        self.assertEqual(first.phase, 'TURN')
        self.assertAlmostEqual(first.v_l+first.v_r, 0)
        self.assertLessEqual(abs((first.v_r-first.v_l)/self.robot.L), 1.5)
        self.robot.theta = math.pi/2
        moving = self.controller.step(self.robot, (0, 2), 0.1)
        self.assertEqual(moving.phase, 'TRACK')
        self.assertGreater(moving.v_l, 0)
        self.assertEqual(moving.v_l, moving.v_r)
        self.assertLessEqual(moving.v_l, 0.8)
        self.robot.y = 1.99
        arrived = self.controller.step(self.robot, (0, 2), 0.1)
        self.assertTrue(arrived.arrived)
        self.assertEqual((arrived.v_l, arrived.v_r), (0, 0))

    def test_collision_stops_until_reset_and_changed_goal_replans(self):
        self.controller.reset(self.robot, (5, 0))
        self.controller.on_collision(self.robot)
        self.assertTrue(self.controller.step(self.robot, (5, 0), 0.1).failed)
        self.controller.reset(self.robot, (5, 0))
        self.assertFalse(self.controller.step(self.robot, (5, 0), 0.1).failed)
        self.controller.step(self.robot, (4, 1), 0.1)
        self.assertEqual(self.controller.path[-1], (4, 1))
        self.assertIsNotNone(self.controller.debug['target'])

    def test_failed_plan_returns_zero_and_invalid_dt_raises(self):
        self.obstacles.add(Circle((0, 0), 1))
        result = self.controller.step(self.robot, (5, 0), 0.1)
        self.assertTrue(result.failed)
        self.assertEqual((result.v_l, result.v_r), (0, 0))
        for dt in (0, -1, math.nan):
            with self.assertRaises(ValueError):
                self.controller.step(self.robot, (5, 0), dt)


if __name__ == '__main__':
    unittest.main()
