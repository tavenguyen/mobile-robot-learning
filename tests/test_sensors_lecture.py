"""Kiểm tra range sensor, biên offset và hình học phục vụ BTH2.1."""
import math
import unittest

import numpy as np
from matplotlib.patches import Circle, Rectangle

from src.Obstacle import Obstacle, point_polygon_distance
from src.boundary import offset_boundary, offset_polygon
from src.sensors import RangeSensor


class RangeSensorTests(unittest.TestCase):
    def setUp(self):
        self.obstacles = Obstacle()

    def test_ray_hits_rectangle(self):
        self.obstacles.add(Rectangle((2, -1), 2, 2))
        sensor = RangeSensor(self.obstacles, n_rays=72, max_range=20)
        distance, point, hit, index = sensor.cast((0, 0), 0.0)
        self.assertTrue(hit)
        self.assertEqual(index, 0)
        self.assertAlmostEqual(distance, 2.0)
        self.assertAlmostEqual(point[0], 2.0)
        self.assertAlmostEqual(point[1], 0.0)

    def test_ray_hits_circle(self):
        self.obstacles.add(Circle((5, 0), 1.0))
        sensor = RangeSensor(self.obstacles, max_range=20)
        distance, _, hit, index = sensor.cast((0, 0), 0.0)
        self.assertTrue(hit)
        self.assertEqual(index, 0)
        self.assertAlmostEqual(distance, 4.0)

    def test_ray_miss_returns_max_range(self):
        self.obstacles.add(Rectangle((2, 5), 1, 1))
        sensor = RangeSensor(self.obstacles, max_range=10)
        distance, point, hit, index = sensor.cast((0, 0), 0.0)
        self.assertFalse(hit)
        self.assertEqual(index, -1)
        self.assertAlmostEqual(distance, 10.0)
        self.assertAlmostEqual(point[0], 10.0)

    def test_rotated_rectangle_is_detected(self):
        self.obstacles.add(Rectangle((2, -1), 2, 2, angle=45))
        sensor = RangeSensor(self.obstacles, max_range=20)
        distance, _, hit, index = sensor.cast((0, 0), 0.0)
        self.assertTrue(hit)
        self.assertEqual(index, 0)
        self.assertGreater(distance, 1.0)

    def test_scan_shape_and_hits(self):
        self.obstacles.add(Rectangle((2, -1), 2, 2))
        sensor = RangeSensor(self.obstacles, n_rays=36, max_range=20)
        scan = sensor.scan(0, 0, 0)
        self.assertEqual(len(scan.angles), 36)
        self.assertEqual(scan.points.shape, (36, 2))
        self.assertTrue(scan.hit.any())
        self.assertTrue((scan.ranges >= 0).all())

    def test_clear_path_and_first_blocker(self):
        self.obstacles.add(Rectangle((2, -1), 2, 2))
        sensor = RangeSensor(self.obstacles, max_range=20)
        self.assertFalse(sensor.clear_path((0, 0), (5, 0), 0.5))
        self.assertTrue(sensor.clear_path((0, 3), (5, 3), 0.5))
        index, point = sensor.first_blocker((0, 0), (5, 0), 0.5)
        self.assertEqual(index, 0)
        self.assertIsNotNone(point)

    def test_intervals_and_goal_interval(self):
        self.obstacles.add(Rectangle((2, -1), 2, 2))
        sensor = RangeSensor(self.obstacles, n_rays=72, max_range=20)
        scan = sensor.scan(0, 0, 0)
        intervals = sensor.intervals(scan)
        self.assertTrue(intervals)
        interval = sensor.interval_for_obstacle(scan, 0, 0.0)
        self.assertIsNotNone(interval)
        self.assertTrue(sensor.interval_contains(scan, interval, 0.0))

    def test_tangent_endpoints_have_h_and_clearance(self):
        self.obstacles.add(Rectangle((2, -1), 2, 2))
        sensor = RangeSensor(self.obstacles, n_rays=72, max_range=20)
        origin = np.array([0.0, 0.0])
        goal = np.array([6.0, 0.0])
        scan = sensor.scan(0, 0, 0)
        interval = sensor.interval_for_obstacle(scan, 0, 0.0)
        endpoints = sensor.endpoints_for_interval(
            scan, interval, origin, goal, 1.0)
        self.assertEqual(len(endpoints), 2)
        for endpoint in endpoints:
            self.assertIn(endpoint.side, ('left', 'right'))
            expected = float(np.linalg.norm(endpoint.point - origin)) + float(
                np.linalg.norm(goal - endpoint.point))
            self.assertAlmostEqual(endpoint.h, expected)
            safe_distance = float(np.linalg.norm(endpoint.safe_point - origin))
            self.assertAlmostEqual(safe_distance, endpoint.distance - 1.0)

    def test_invalid_arguments(self):
        sensor = RangeSensor(self.obstacles, n_rays=72)
        with self.assertRaises(ValueError):
            RangeSensor(self.obstacles, n_rays=4)
        with self.assertRaises(ValueError):
            RangeSensor(self.obstacles, max_range=0)
        with self.assertRaises(ValueError):
            sensor.cast((0,), 0.0)
        with self.assertRaises(ValueError):
            sensor.clear_path((0, 0), (1,))


class BoundaryTests(unittest.TestCase):
    def test_offset_polygon_keeps_clearance(self):
        square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        boundary = offset_polygon(square, 0.5)
        self.assertGreater(boundary.perimeter, 8.0)
        for point in boundary.points:
            self.assertAlmostEqual(
                point_polygon_distance(point, square), 0.5, places=6)

    def test_boundary_project_point_at_tangent(self):
        obstacle = Rectangle((2, -1), 2, 2)
        boundary = offset_boundary(obstacle, 1.0)
        s, point, distance = boundary.project((0.0, 0.0))
        self.assertAlmostEqual(distance, np.linalg.norm(point))
        self.assertAlmostEqual(np.linalg.norm(point - (0.0, 0.0)), distance)
        self.assertAlmostEqual(
            np.linalg.norm(boundary.point_at(s) - point), 0.0, places=6)
        tangent = boundary.tangent_at(s)
        self.assertAlmostEqual(float(np.linalg.norm(tangent)), 1.0)
        self.assertAlmostEqual(
            boundary.forward_distance(s, s + 2.0), 2.0, places=6)

    def test_offset_circle(self):
        obstacle = Circle((3, 0), 1.5)
        boundary = offset_boundary(obstacle, 0.75)
        distances = np.linalg.norm(
            boundary.points - np.array([3.0, 0.0]), axis=1)
        self.assertTrue(np.allclose(distances, 2.25, atol=1e-9))
        self.assertAlmostEqual(
            boundary.perimeter, 2 * math.pi * 2.25, delta=0.01)

    def test_invalid_clearance(self):
        with self.assertRaises(ValueError):
            offset_boundary(Rectangle((0, 0), 1, 1), 0)
        with self.assertRaises(ValueError):
            offset_polygon(np.array([[0, 0], [1, 0], [1, 1]]), -1)


if __name__ == '__main__':
    unittest.main()
