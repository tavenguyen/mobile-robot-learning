"""Kiểm tra cửa sổ động, điểm số, quỹ đạo và tính an toàn DWA."""
import math
import unittest
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from matplotlib.patches import Circle, Rectangle

from src.Obstacle import Obstacle
from src.dwa import DWAConfig, DWAController


def make_robot(v=0, w=0):
    return SimpleNamespace(x=0.0, y=0.0, theta=0.0, L=0.5,
                           v_l=v-w*0.25, v_r=v+w*0.25,
                           robot_length=0.2, robot_width=0.2)


class DWAConfigTests(unittest.TestCase):
    def test_defaults_and_immutable_live_weight_update(self):
        config = DWAConfig()
        self.assertEqual((config.alpha, config.beta, config.gamma), (0.6, 0.3, 0.1))
        with self.assertRaises(FrozenInstanceError):
            config.alpha = 1
        self.assertEqual(replace(config, alpha=1).alpha, 1)

    def test_invalid_configuration(self):
        for values in ({'alpha': -1}, {'beta': math.inf}, {'gamma': math.nan},
                       {'alpha': 0, 'beta': 0, 'gamma': 0},
                       {'v_min': -1}, {'v_max': 0}, {'w_max': -3},
                       {'a_max': 0}, {'angular_a_max': 0}, {'horizon': 0},
                       {'v_resolution': 0}, {'w_resolution': -1},
                       {'dist_cap': 0}, {'clearance_margin': -1}):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    DWAConfig(**values)


class DWATests(unittest.TestCase):
    def setUp(self):
        self.obstacles = Obstacle()
        self.controller = DWAController(self.obstacles)
        self.robot = make_robot()

    def test_dynamic_window_and_wheel_conversion(self):
        self.robot = make_robot(v=0.073, w=0.047)
        control = self.controller.step(self.robot, (5, 0), 0.1)
        candidates = self.controller.candidates
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertGreaterEqual(candidate.v, 0)
            self.assertLessEqual(abs(candidate.v - 0.073), 0.1 + 1e-10)
            self.assertLessEqual(abs(candidate.w - 0.047), 0.2 + 1e-10)
        self.assertTrue(any(abs(c.v-0.073) < 1e-10 for c in candidates))
        self.assertTrue(any(abs(c.w-0.047) < 1e-10 for c in candidates))
        best = self.controller.selected_candidate
        self.assertAlmostEqual((control.v_l + control.v_r)/2, best.v)
        self.assertAlmostEqual((control.v_r - control.v_l)/self.robot.L, best.w)
        self.assertIs(best, candidates[0])

    def test_euler_rollout_matches_robot_and_does_not_mutate_it(self):
        before = vars(self.robot).copy()
        self.controller.step(self.robot, (5, 0), 0.1)
        for candidate in self.controller.candidates:
            trajectory = candidate.trajectory
            np.testing.assert_allclose(trajectory[0], (0, 0, 0))
            h = self.controller.config.horizon / (len(trajectory)-1)
            np.testing.assert_allclose(np.diff(trajectory[:, 0]),
                                       candidate.v*np.cos(trajectory[:-1, 2])*h, atol=1e-12)
            np.testing.assert_allclose(np.diff(trajectory[:, 1]),
                                       candidate.v*np.sin(trajectory[:-1, 2])*h, atol=1e-12)
            np.testing.assert_allclose(np.diff(trajectory[:, 2]), candidate.w*h, atol=1e-12)
        self.assertEqual(vars(self.robot), before)

    def test_score_formula_and_sort_order(self):
        self.controller.step(self.robot, (5, 1), 0.1)
        for candidate in self.controller.candidates:
            end = candidate.trajectory[-1]
            angle = math.atan2(1-end[1], 5-end[0])
            error = (angle-end[2]+math.pi) % (2*math.pi)-math.pi
            self.assertAlmostEqual(candidate.heading, 1-abs(error)/math.pi)
            self.assertEqual(candidate.dist, 1)
            self.assertAlmostEqual(candidate.velocity, candidate.v/2)
            self.assertAlmostEqual(candidate.score, 0.6*candidate.heading+0.3+0.1*candidate.v/2)
        keys = [(c.score, c.heading, c.dist, c.v, -abs(c.w), -c.w)
                for c in self.controller.candidates]
        self.assertEqual(keys, sorted(keys, reverse=True))

    def test_clearance_term_with_circle(self):
        self.obstacles.add(Circle((2, 0), 0.2))
        self.controller.step(self.robot, (5, 1), 0.1)
        radius = 0.5*math.hypot(0.2, 0.2)+0.15
        for candidate in self.controller.candidates:
            clearance = (np.linalg.norm(candidate.trajectory[:, :2]-[2, 0], axis=1)-0.2-radius).min()
            self.assertAlmostEqual(candidate.dist, min(clearance/3, 1))

    def test_goal_arrival_and_invalid_goal(self):
        control = self.controller.step(self.robot, (0.1, 0), 0.1)
        self.assertTrue(control.arrived)
        self.assertEqual((control.v_l, control.v_r), (0, 0))
        self.assertIsNone(self.controller.selected_candidate)
        for goal in ((math.nan, 1), ('x', 1), (1,)):
            with self.assertRaises(ValueError):
                self.controller.step(self.robot, goal, 0.1)
        for dt in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                self.controller.step(self.robot, (5, 0), dt)

    def test_current_pose_and_goal_inflated_collisions_fail(self):
        for location in ((0, 0.25), (5, 0.25)):
            with self.subTest(location=location):
                obstacles = Obstacle()
                obstacles.add(Circle(location, 0.1))
                control = DWAController(obstacles).step(self.robot, (5, 0), 0.1)
                self.assertTrue(control.failed)
                self.assertEqual((control.v_l, control.v_r), (0, 0))

    def test_braking_rejects_clear_horizon_with_unsafe_stop(self):
        self.obstacles.add(Circle((1.2, 0), 0.2))
        controller = DWAController(self.obstacles, DWAConfig(horizon=0.2))
        control = controller.step(make_robot(v=1.2), (5, 0), 0.1)
        self.assertTrue(control.failed)
        self.assertEqual(controller.candidates, [])

    def test_rotated_thin_rectangle_candidates_are_safe(self):
        obstacle = Rectangle((0.7, -1), 0.02, 2, angle=10)
        self.obstacles.add(obstacle)
        self.controller.step(make_robot(v=0.5), (5, 0), 0.1)
        radius = math.hypot(0.2, 0.2)/2+0.15
        for candidate in self.controller.candidates:
            for x, y in candidate.trajectory[:, :2]:
                index, _, distance, _ = self.obstacles.nearest((x, y))
                self.assertIsNotNone(index)
                self.assertGreater(distance, radius)

    def test_reset_clears_output_and_collision_failure(self):
        self.controller.step(self.robot, (5, 0), 0.1)
        self.controller.on_collision(self.robot)
        self.assertTrue(self.controller.step(self.robot, (5, 0), 0.1).failed)
        self.assertEqual(self.controller.candidates, [])
        self.controller.reset(self.robot, (6, 1))
        self.assertEqual(self.controller.candidates, [])
        self.assertIsNone(self.controller.selected_candidate)
        self.assertFalse(self.controller.step(self.robot, (6, 1), 0.1).failed)

    def test_changed_goal_restarts_failed_controller(self):
        self.controller.reset(self.robot, (5, 0))
        self.controller.on_collision(self.robot)
        control = self.controller.step(self.robot, (5, 1), 0.1)
        self.assertFalse(control.failed)
        np.testing.assert_allclose(self.controller.goal, (5, 1))

    def test_weight_changes_apply_without_reset(self):
        self.controller.step(self.robot, (5, 1), 0.1)
        self.controller.config = replace(self.controller.config, alpha=0, beta=0, gamma=2)
        self.controller.step(self.robot, (5, 1), 0.1)
        for candidate in self.controller.candidates:
            self.assertAlmostEqual(candidate.score, 2*candidate.velocity)

    def test_terminal_speed_cap_respects_acceleration_and_horizon(self):
        goal = (0.6, 0)
        self.controller.step(self.robot, goal, 0.1)
        self.assertLessEqual(max(c.v for c in self.controller.candidates), 0.1)
        self.controller.step(make_robot(v=0.8), goal, 0.1)
        for candidate in self.controller.candidates:
            self.assertAlmostEqual(candidate.v, 0.7)

    def test_small_goal_bearing_has_exact_angular_sample(self):
        goal = (5, 0.2)
        self.controller.step(self.robot, goal, 0.1)
        desired_w = math.atan2(goal[1], goal[0])/3
        self.assertTrue(any(abs(c.w-desired_w) < 1e-12
                            for c in self.controller.candidates))

    def test_tracking_target_changes_heading_without_reset_or_arrival(self):
        self.controller.reset(self.robot, (5, 0))
        with patch.object(self.controller, 'reset', wraps=self.controller.reset) as reset:
            control = self.controller.step(self.robot, (5, 0), 0.1,
                                           tracking_target=(0, 0))
            self.assertFalse(control.arrived)
            self.controller.step(self.robot, (5, 0), 0.1, tracking_target=(0, 2))
            reset.assert_not_called()
        np.testing.assert_array_equal(self.controller.goal, (5, 0))
        for candidate in self.controller.candidates:
            end = candidate.trajectory[-1]
            error = (math.atan2(2-end[1], -end[0])-end[2]+math.pi) % (2*math.pi)-math.pi
            self.assertAlmostEqual(candidate.heading, 1-abs(error)/math.pi)

    def test_final_goal_arrival_ignores_distant_tracking_target(self):
        control = self.controller.step(self.robot, (0.1, 0), 0.1,
                                       tracking_target=(5, 0))
        self.assertTrue(control.arrived)

    def test_bounds_filter_braking_beyond_safe_horizon(self):
        robot = make_robot(v=1.2)
        robot.x = 19.5
        controller = DWAController(self.obstacles, DWAConfig(horizon=0.2))
        unrestricted = controller.step(robot, (25, 0), 0.1)
        self.assertFalse(unrestricted.failed)
        controller.reset(robot, (25, 0))
        restricted = controller.step(robot, (25, 0), 0.1,
                                     bounds=(-20, 20, -20, 20))
        self.assertTrue(restricted.failed)

    def test_bounds_filter_keeps_candidates_inside_map(self):
        self.robot.x = 19.8
        self.controller.step(self.robot, (19, 2), 0.1,
                             bounds=(-20, 20, -20, 20))
        self.assertTrue(self.controller.candidates)
        for candidate in self.controller.candidates:
            self.assertTrue(np.all(np.abs(candidate.trajectory[:, :2]) <= 20))

    def test_progress_admissibility_filters_stationary_without_changing_score(self):
        self.controller.step(self.robot, (5, 0), 0.1,
                             tracking_target=(1, 0), progress_direction=(1, 0))
        self.assertTrue(self.controller.candidates)
        for candidate in self.controller.candidates:
            self.assertGreaterEqual(candidate.trajectory[-1, 0], 0.01)
            self.assertAlmostEqual(candidate.score, 0.6*candidate.heading+
                                   0.3*candidate.dist+0.1*candidate.velocity)

    def test_progress_admissibility_retains_rotation_if_forward_impossible(self):
        self.controller.step(self.robot, (-5, 0), 0.1,
                             tracking_target=(-1, 0), progress_direction=(-1, 0))
        self.assertTrue(any(c.v == 0 and abs(c.w) > 0
                            for c in self.controller.candidates))
        self.assertFalse(any(-c.trajectory[-1, 0] >= 0.01
                             for c in self.controller.candidates))


if __name__ == '__main__':
    unittest.main()
