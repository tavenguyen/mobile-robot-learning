"""Kiểm tra luật điều khiển auto-mode tới goal (BTH2.1)."""
import math
import unittest

import numpy as np

from src.Robot import Differential_Robot
from src.navigation import goal_command


def make_robot(L=1.0, theta=0.0):
    robot = Differential_Robot(
        L, robot_length=2.0, robot_width=1.0,
        wheel_length=0.5, wheel_width=0.5,
    )
    robot.x, robot.y, robot.theta = 0.0, 0.0, theta
    return robot


class GoalCommandTests(unittest.TestCase):
    def test_straight_ahead(self):
        v_l, v_r, distance, arrived = goal_command(make_robot(), (5.0, 0.0))
        self.assertFalse(arrived)
        self.assertAlmostEqual(distance, 5.0)
        self.assertAlmostEqual(v_l, 2.0)
        self.assertAlmostEqual(v_r, 2.0)

    def test_goal_behind_turns_in_place(self):
        v_l, v_r, _, arrived = goal_command(make_robot(), (-5.0, 0.0))
        self.assertFalse(arrived)
        self.assertAlmostEqual(v_l, 1.0)
        self.assertAlmostEqual(v_r, -1.0)

    def test_goal_at_45_degrees(self):
        v_l, v_r, distance, arrived = goal_command(make_robot(), (5.0, 5.0))
        self.assertFalse(arrived)
        self.assertAlmostEqual(distance, math.hypot(5.0, 5.0))
        self.assertAlmostEqual(v_l, 2.0 - 0.5 * math.pi / 2)
        self.assertAlmostEqual(v_r, 2.0 + 0.5 * math.pi / 2)

    def test_goal_on_the_left_turns_counterclockwise(self):
        v_l, v_r, _, _ = goal_command(make_robot(), (0.0, 5.0))
        self.assertLess(v_l, v_r)

    def test_arrival_tolerance(self):
        v_l, v_r, distance, arrived = goal_command(make_robot(), (0.2, 0.0))
        self.assertTrue(arrived)
        self.assertAlmostEqual(distance, 0.2)
        self.assertEqual((v_l, v_r), (0.0, 0.0))

    def test_tolerance_boundary_not_arrived(self):
        v_l, v_r, _, arrived = goal_command(make_robot(), (0.3, 0.0))
        self.assertFalse(arrived)
        self.assertAlmostEqual(v_l, 0.3)
        self.assertAlmostEqual(v_r, 0.3)

    def test_custom_tolerance(self):
        _, _, _, arrived = goal_command(
            make_robot(), (0.2, 0.0), tolerance=0.1)
        self.assertFalse(arrived)

    def test_half_track_scales_wheel_speeds(self):
        v_l, v_r, _, _ = goal_command(make_robot(L=2.0), (5.0, 5.0))
        self.assertAlmostEqual(v_l, 2.0 - 1.0 * math.pi / 2)
        self.assertAlmostEqual(v_r, 2.0 + 1.0 * math.pi / 2)

    def test_velocity_limits(self):
        v_l, v_r, _, _ = goal_command(make_robot(), (5.0, 0.0), max_v=1.0)
        self.assertAlmostEqual(v_l, 1.0)
        self.assertAlmostEqual(v_r, 1.0)

        v_l, v_r, _, _ = goal_command(
            make_robot(), (-5.0, 0.0), max_omega=1.0)
        self.assertAlmostEqual(v_l, 0.5)
        self.assertAlmostEqual(v_r, -0.5)

    def test_heading_correction_is_wrapped(self):
        robot = make_robot(theta=-0.1)
        v_l, v_r, _, _ = goal_command(robot, (-5.0, 0.0))
        self.assertAlmostEqual(v_l, -v_r)
        self.assertGreater(v_l, 0.0)

    def test_invalid_goal(self):
        robot = make_robot()
        for goal in [None, (np.nan, 0.0), (1.0,), ('a', 'b')]:
            with self.subTest(goal=goal):
                with self.assertRaises(ValueError):
                    goal_command(robot, goal)


if __name__ == '__main__':
    unittest.main()
