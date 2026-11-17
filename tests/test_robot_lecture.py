"""Kiểm tra động học và hình học của Differential_Robot (BTH1)."""
import unittest

import matplotlib
matplotlib.use('Agg')
import numpy as np

from src.Robot import Differential_Robot


def make_robot():
    return Differential_Robot(
        L=1.0, robot_length=2.0, robot_width=1.0,
        wheel_length=0.5, wheel_width=0.5,
    )


class RobotKinematicsTests(unittest.TestCase):
    def test_straight_line(self):
        robot = make_robot()
        robot.set_wheel_velocities(1.5, 1.5)
        for _ in range(10):
            robot.update_pose(0.1)
        self.assertAlmostEqual(robot.x, 1.5, places=12)
        self.assertAlmostEqual(robot.y, 0.0, places=12)
        self.assertAlmostEqual(robot.theta, 0.0, places=12)
        self.assertAlmostEqual(robot.v, 1.5, places=12)
        self.assertAlmostEqual(robot.omega, 0.0, places=12)
        self.assertEqual(len(robot.x_data), 10)

    def test_spin_in_place(self):
        robot = make_robot()
        robot.set_wheel_velocities(-1.0, 1.0)
        robot.update_pose(0.1)
        self.assertAlmostEqual(robot.theta, 0.2, places=12)
        self.assertAlmostEqual(robot.x, 0.0, places=12)
        self.assertAlmostEqual(robot.y, 0.0, places=12)
        self.assertAlmostEqual(robot.omega, 2.0, places=12)

    def test_arc_step(self):
        robot = make_robot()
        robot.set_wheel_velocities(0.5, 1.5)
        robot.update_pose(0.1)
        self.assertAlmostEqual(robot.v, 1.0, places=12)
        self.assertAlmostEqual(robot.omega, 1.0, places=12)
        self.assertAlmostEqual(robot.theta, 0.1, places=12)
        self.assertAlmostEqual(robot.x, 0.1, places=12)
        self.assertAlmostEqual(robot.y, 0.0, places=12)

    def test_normalize_angle(self):
        for angle, expected in [(0.0, 0.0), (np.pi, -np.pi),
                                (3 * np.pi, -np.pi), (-3 * np.pi, -np.pi),
                                (np.pi / 2, np.pi / 2)]:
            self.assertAlmostEqual(
                Differential_Robot.normalize_angle(angle), expected, places=12)

    def test_predict_pose_does_not_mutate(self):
        robot = make_robot()
        robot.set_wheel_velocities(1.0, 1.0)
        predicted = robot.predict_pose(0.1)
        np.testing.assert_allclose(predicted, [0.1, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            [robot.x, robot.y, robot.theta], [0.0, 0.0, 0.0], atol=1e-12)
        self.assertEqual(robot.x_data, [])

    def test_reset(self):
        robot = make_robot()
        robot.set_wheel_velocities(1.0, 1.0)
        robot.update_pose(0.1)
        robot.reset()
        np.testing.assert_allclose(
            [robot.x, robot.y, robot.theta], [0.0, 0.0, 0.0], atol=1e-12)
        self.assertEqual(
            (robot.v_l, robot.v_r, robot.v, robot.omega), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(robot.x_data, [])
        self.assertEqual(robot.y_data, [])
        self.assertEqual(robot.theta_data, [])


class RobotGeometryTests(unittest.TestCase):
    def test_corners_at_pose(self):
        robot = make_robot()
        corners = robot.get_robot_corners(2.0, 3.0, 0.0)
        np.testing.assert_allclose(corners, [
            [1.0, 2.5], [3.0, 2.5], [3.0, 3.5], [1.0, 3.5],
        ], atol=1e-12)
        np.testing.assert_allclose(corners.mean(axis=0), [2.0, 3.0],
                                   atol=1e-12)

    def test_corners_follow_rotation(self):
        robot = make_robot()
        theta = np.pi / 2
        corners = robot.get_robot_corners(2.0, 3.0, theta)
        local = np.array([
            [-1.0, -0.5], [1.0, -0.5], [1.0, 0.5], [-1.0, 0.5],
        ])
        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        np.testing.assert_allclose(
            corners, local @ rotation.T + [2.0, 3.0], atol=1e-12)

    def test_corners_are_counterclockwise(self):
        robot = make_robot()
        corners = robot.get_robot_corners(0.0, 0.0, 0.0)
        twice_area = 0.0
        for i, start in enumerate(corners):
            end = corners[(i + 1) % len(corners)]
            twice_area += start[0] * end[1] - end[0] * start[1]
        self.assertGreater(twice_area, 0.0)

    def test_corners_default_to_current_pose(self):
        robot = make_robot()
        robot.set_wheel_velocities(1.0, 1.0)
        robot.update_pose(0.1)
        np.testing.assert_allclose(
            robot.get_robot_corners(),
            robot.get_robot_corners(robot.x, robot.y, robot.theta),
            atol=1e-12,
        )
        with self.assertRaises(ValueError):
            robot.get_robot_corners(1.0, 2.0, None)

    def test_update_graphics_returns_artists(self):
        import matplotlib.pyplot as plt
        robot = make_robot()
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        artists = robot.update_graphics(ax)
        self.assertEqual(len(artists), 6)
        for artist in artists:
            self.assertIsNotNone(artist.get_transform())


class RobotValidationTests(unittest.TestCase):
    def test_invalid_dimensions(self):
        for values in [
            dict(L=0.0), dict(L=-1.0), dict(L=np.nan), dict(L=np.inf),
            dict(robot_length=0.0), dict(robot_width=-2.0),
            dict(wheel_length=np.nan), dict(wheel_width=np.inf),
        ]:
            kwargs = dict(
                L=1.0, robot_length=2.0, robot_width=1.0,
                wheel_length=0.5, wheel_width=0.5,
            )
            kwargs.update(values)
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    Differential_Robot(**kwargs)

    def test_invalid_dt(self):
        robot = make_robot()
        for dt in [0.0, -0.1, np.nan, np.inf, None, 'x']:
            with self.subTest(dt=dt):
                with self.assertRaises(ValueError):
                    robot.update_pose(dt)
                with self.assertRaises(ValueError):
                    robot.predict_pose(dt)


if __name__ == '__main__':
    unittest.main()
