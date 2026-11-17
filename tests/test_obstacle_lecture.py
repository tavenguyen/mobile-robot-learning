"""Kiểm tra quản lý vật cản và phát hiện va chạm (BTH1)."""
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

from src.Obstacle import Obstacle
from src.Robot import Differential_Robot


def make_robot():
    return Differential_Robot(
        L=1.0, robot_length=2.0, robot_width=1.0,
        wheel_length=0.5, wheel_width=0.5,
    )


class ObstacleCollectionTests(unittest.TestCase):
    def test_add_remove_clear(self):
        obstacles = Obstacle()
        rect = Rectangle((1, 1), 2, 2)
        circle = Circle((5, 5), 1)
        obstacles.add(rect)
        obstacles.add(circle)
        self.assertEqual(obstacles.obs, [rect, circle])
        obstacles.remove(rect)
        self.assertEqual(obstacles.obs, [circle])
        obstacles.remove(Rectangle((0, 0), 1, 1))  # không có: bỏ qua
        obstacles.clear()
        self.assertEqual(obstacles.obs, [])

    def test_rejects_unsupported_patch(self):
        obstacles = Obstacle()
        with self.assertRaises(TypeError):
            obstacles.add(Polygon([[0, 0], [1, 0], [1, 1]]))

    def test_labels(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((-5, -5), 3, 3))
        obstacles.add(Circle((10, 10), 2))
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        obstacles.label_obstacles(ax)
        self.assertEqual([text.get_text() for text in ax.texts],
                         ['Obs 0', 'Obs 1'])

    def test_random_obstacles_reproducible(self):
        first = Obstacle()
        second = Obstacle()
        kwargs = dict(count=4, area=(-8, 8), size=(1, 2), seed=7)
        first.generate_random_obstacles(**kwargs)
        second.generate_random_obstacles(**kwargs)
        self.assertEqual(len(first.obs), 4)
        for a, b in zip(first.obs, second.obs):
            np.testing.assert_allclose(a.get_xy(), b.get_xy())
            self.assertAlmostEqual(a.get_width(), b.get_width())
            self.assertAlmostEqual(a.get_height(), b.get_height())

    def test_random_obstacles_are_spread_out(self):
        obstacles = Obstacle()
        obstacles.generate_random_obstacles(
            count=5, area=(-8, 8), size=(1, 2), seed=3, start_clearance=1.0,
        )
        for obs in obstacles.obs:
            x, y = obs.get_xy()
            self.assertGreaterEqual(x, -8)
            self.assertLessEqual(x, 8)
            self.assertGreaterEqual(y, -8)
            self.assertLessEqual(y, 8)

            nearest_x = min(max(0.0, x), x + obs.get_width())
            nearest_y = min(max(0.0, y), y + obs.get_height())
            self.assertGreaterEqual(
                np.hypot(nearest_x, nearest_y), 1.0 - 1e-12)

        for i, first in enumerate(obstacles.obs):
            ax0, ay0 = first.get_xy()
            ax1 = ax0 + first.get_width()
            ay1 = ay0 + first.get_height()
            for second in obstacles.obs[i + 1:]:
                bx0, by0 = second.get_xy()
                bx1 = bx0 + second.get_width()
                by1 = by0 + second.get_height()
                separated = (ax1 <= bx0 or bx1 <= ax0
                             or ay1 <= by0 or by1 <= ay0)
                self.assertTrue(separated)

    def test_random_count_zero(self):
        obstacles = Obstacle()
        self.assertEqual(obstacles.generate_random_obstacles(count=0), [])

    def test_random_invalid_arguments(self):
        obstacles = Obstacle()
        for kwargs in [dict(count=-1), dict(area=(1, 1)),
                       dict(size=(0, 2)), dict(area=(0, np.inf)),
                       dict(start_clearance=-1)]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    obstacles.generate_random_obstacles(**kwargs)


class RectangleCollisionTests(unittest.TestCase):
    def setUp(self):
        self.robot = make_robot()

    def test_overlap(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((0, 0), 0.5, 0.5))
        self.assertTrue(obstacles.collides_pose(self.robot, 0, 0, 0))

    def test_touching_edge_counts(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((1, -0.25), 0.5, 0.5))
        self.assertTrue(obstacles.collides_pose(self.robot, 0, 0, 0))

    def test_small_gap_misses(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((1.01, 0), 1, 1))
        self.assertFalse(obstacles.collides_pose(self.robot, 0, 0, 0))

    def test_far_obstacle_misses(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((5, 5), 1, 1))
        self.assertFalse(obstacles.collides_pose(self.robot, 0, 0, 0))

    def test_rotation_changes_result(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((0.0, 0.9), 0.5, 0.3))
        self.assertFalse(obstacles.collides_pose(self.robot, 0, 0, 0))
        self.assertTrue(
            obstacles.collides_pose(self.robot, 0, 0, np.deg2rad(45)))

    def test_rotated_obstacle_patch(self):
        obstacles = Obstacle()
        flat = Rectangle((0.9, 0.6), 0.4, 0.1)
        tilted = Rectangle((0.9, 0.6), 0.4, 0.1, angle=-45)
        obstacles.add(flat)
        self.assertFalse(obstacles.collides_pose(self.robot, 0, 0, 0))
        obstacles.clear()
        obstacles.add(tilted)
        self.assertTrue(obstacles.collides_pose(self.robot, 0, 0, 0))


class CircleCollisionTests(unittest.TestCase):
    def setUp(self):
        self.robot = make_robot()

    def test_center_inside_robot(self):
        obstacles = Obstacle()
        obstacles.add(Circle((0, 0), 0.1))
        self.assertTrue(obstacles.collides_pose(self.robot, 0, 0, 0))

    def test_corner_inside_circle(self):
        obstacles = Obstacle()
        obstacles.add(Circle((1.2, 0.5), 0.3))
        self.assertTrue(obstacles.collides_pose(self.robot, 0, 0, 0))

    def test_edge_cut_without_corners_inside(self):
        obstacles = Obstacle()
        obstacles.add(Circle((1.5, 0), 0.6))
        corners = self.robot.get_robot_corners(0, 0, 0)
        distances = np.linalg.norm(corners - [1.5, 0], axis=1)
        self.assertTrue(np.all(distances > 0.6))
        self.assertTrue(obstacles.collides_pose(self.robot, 0, 0, 0))

    def test_circle_encloses_robot(self):
        obstacles = Obstacle()
        obstacles.add(Circle((0, 0), 5))
        self.assertTrue(obstacles.collides_pose(self.robot, 0, 0, 0))

    def test_touching_circle_counts(self):
        obstacles = Obstacle()
        obstacles.add(Circle((2, 0), 1))
        self.assertTrue(obstacles.collides_pose(self.robot, 0, 0, 0))

    def test_miss(self):
        obstacles = Obstacle()
        obstacles.add(Circle((2.5, 0), 1))
        self.assertFalse(obstacles.collides_pose(self.robot, 0, 0, 0))

    def test_collides_validates_corners(self):
        obstacles = Obstacle()
        with self.assertRaises(ValueError):
            obstacles.collides(np.zeros((2, 2)))
        with self.assertRaises(ValueError):
            obstacles.collides(np.array([[0, 0], [1, np.nan], [0, 1]]))


if __name__ == '__main__':
    unittest.main()
