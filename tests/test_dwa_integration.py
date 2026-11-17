"""Kiểm tra DWA với robot và vật cản thật, không cần Tkinter."""
import math
import unittest

import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Circle, Rectangle

from src.Obstacle import Obstacle
from src.Robot import Differential_Robot
from src.dwa import DWAController


def make_robot():
    return Differential_Robot(1.0, 2.0, 1.0, 0.5, 0.5)


class DWAIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.robot = make_robot()
        self.obstacles = Obstacle()
        self.controller = DWAController(self.obstacles)

    def test_empty_map_reaches_front_and_rear_goals_without_planner_mutation(self):
        for goal in ((5.0, 0.0), (-4.0, 0.0)):
            with self.subTest(goal=goal):
                robot = make_robot()
                controller = DWAController(self.obstacles)
                controller.reset(robot, goal)
                for _ in range(400):
                    before = robot.pose
                    control = controller.step(robot, goal, 0.1)
                    self.assertEqual(robot.pose, before)
                    self.assertFalse(control.failed, control.message)
                    if control.arrived:
                        break
                    robot.set_wheel_velocities(control.v_l, control.v_r)
                    robot.update_pose(0.1)
                else:
                    self.fail('DWA không tới goal trên map trống')
                self.assertLessEqual(math.hypot(robot.x - goal[0], robot.y), 0.3)

    def test_front_obstacle_does_not_produce_collision_commands(self):
        self.obstacles.add(Rectangle((2.0, -1.0), 1.0, 2.0))
        goal = (7.0, 0.0)
        self.controller.reset(self.robot, goal)
        for _ in range(35):
            control = self.controller.step(self.robot, goal, 0.1)
            if control.failed or control.arrived:
                self.assertEqual((control.v_l, control.v_r), (0.0, 0.0))
                break
            self.robot.set_wheel_velocities(control.v_l, control.v_r)
            self.assertFalse(self.obstacles.collides_pose(
                self.robot, *self.robot.predict_pose(0.1)))
            self.robot.update_pose(0.1)

    def test_candidates_sorted_and_selected_actuation_is_deterministic(self):
        goal = (6.0, 2.0)
        self.controller.reset(self.robot, goal)
        first = self.controller.step(self.robot, goal, 0.1)
        candidates = self.controller.candidates
        self.assertTrue(candidates)
        self.assertEqual([c.score for c in candidates],
                         sorted((c.score for c in candidates), reverse=True))
        chosen = self.controller.selected_candidate
        self.assertIs(chosen, candidates[0])
        self.assertAlmostEqual((first.v_l + first.v_r) / 2, chosen.v)
        self.assertAlmostEqual((first.v_r - first.v_l) / self.robot.L, chosen.w)
        second_controller = DWAController(self.obstacles)
        second_controller.reset(self.robot, goal)
        second = second_controller.step(self.robot, goal, 0.1)
        self.assertEqual((first.v_l, first.v_r), (second.v_l, second.v_r))
        self.assertEqual([(c.v, c.w, c.score) for c in candidates],
                         [(c.v, c.w, c.score) for c in second_controller.candidates])

    def test_map_change_reset_clears_candidates_and_detects_blocked_goal(self):
        goal = (5.0, 0.0)
        self.controller.reset(self.robot, goal)
        self.controller.step(self.robot, goal, 0.1)
        self.assertTrue(self.controller.candidates)
        self.obstacles.add(Circle(goal, 1.0))
        self.controller.reset(self.robot, goal)
        self.assertFalse(self.controller.candidates)
        self.assertIsNone(self.controller.selected_candidate)
        control = self.controller.step(self.robot, goal, 0.1)
        self.assertTrue(control.failed)
        self.assertEqual((control.v_l, control.v_r), (0.0, 0.0))
        self.assertFalse(self.controller.candidates)

    def test_goal_behind_robot_selects_turn_within_acceleration_window(self):
        goal = (-4.0, 0.0)
        self.controller.reset(self.robot, goal)
        control = self.controller.step(self.robot, goal, 0.1)
        self.assertFalse(control.failed)
        selected = self.controller.selected_candidate
        self.assertGreater(abs(selected.w), 0.0)
        self.assertLessEqual(abs(selected.w), 0.2 + 1e-12)
        self.assertLessEqual(selected.v, 0.1 + 1e-12)

    def test_invalid_goal_rejected(self):
        with self.assertRaises(ValueError):
            self.controller.reset(self.robot, (float('nan'), 0.0))


if __name__ == '__main__':
    unittest.main()
