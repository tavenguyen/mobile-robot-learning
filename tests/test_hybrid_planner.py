"""Tích hợp A* và DWA, mục tiêu cục bộ, tiến độ và phục hồi có giới hạn."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from matplotlib.patches import Circle

from src.Obstacle import Obstacle
from src.avoidance import Control
from src.dwa import DWAConfig
from src.hybrid_planner import AStarDWAController


def make_robot():
    return SimpleNamespace(x=0., y=0., theta=0., L=0.5,
                           v_l=0., v_r=0., robot_length=0.2, robot_width=0.2)


class HybridPlannerTests(unittest.TestCase):
    def setUp(self):
        self.obstacles = Obstacle()
        self.controller = AStarDWAController(self.obstacles)
        self.robot = make_robot()

    def test_reset_plans_and_forwards_config_candidates(self):
        self.controller.reset(self.robot, (5, 0))
        self.assertEqual(self.controller.result.status, 'OK')
        self.assertEqual(self.controller.path[0], (0, 0))
        self.assertEqual(self.controller.path[-1], (5, 0))
        self.assertEqual(self.controller.local_target, (1.5, 0))
        updated = replace(DWAConfig(), alpha=1)
        self.controller.config = updated
        self.assertIs(self.controller.local.config, updated)
        control = self.controller.step(self.robot, (5, 0), 0.1)
        self.assertFalse(control.failed)
        self.assertIs(self.controller.selected_candidate, self.controller.candidates[0])
        self.assertIsNotNone(self.controller.debug['trajectory'])
        self.assertEqual(self.controller.debug['target'], self.controller.local_target)

    def test_progress_monotone_on_active_segment_without_branch_jump(self):
        self.controller.reset(self.robot, (5, 0))
        self.controller.path = [(0, 0), (4, 0), (4, 1), (0, 1)]
        self.robot.x, self.robot.y = 2, 0.95
        self.controller._update_target(self.robot)
        self.assertEqual(self.controller._segment, 0)
        self.assertAlmostEqual(self.controller._segment_progress, 2)
        np.testing.assert_allclose(self.controller.local_target, (3.5, 0))
        self.robot.x = 1
        self.controller._update_target(self.robot)
        self.assertAlmostEqual(self.controller._segment_progress, 2)
        self.assertEqual(self.controller._segment, 0)

    def test_corner_transition_requires_local_clear_line(self):
        self.controller.reset(self.robot, (5, 0))
        self.controller.path = [(0, 0), (2, 0), (2, 2)]
        self.robot.x = 1.8
        # Vật cản mới chặn đường nối sang đoạn sau.
        self.obstacles.add(Circle((1.9, 0.7), 0.2))
        self.controller._update_target(self.robot)
        self.assertEqual(self.controller._segment, 0)
        self.assertEqual(self.controller.local_target, (2, 0))
        self.obstacles.clear()
        self.controller._update_target(self.robot)
        self.assertEqual(self.controller._segment, 1)
        self.assertEqual(self.controller.local_target, (2, 1.5))

    def test_no_progress_stops_and_replans_twice_then_fails(self):
        self.controller.reset(self.robot, (5, 0))
        with patch.object(self.controller.local, 'step', return_value=Control(0, 0)):
            # Cần candidate cho dòng debug, nhưng không cập nhật robot: kẹt thật.
            self.controller.local.selected_candidate = SimpleNamespace(v=0, score=0, trajectory=None)
            for attempt in (1, 2):
                for _ in range(49):
                    self.controller.local.selected_candidate = SimpleNamespace(v=0, score=0, trajectory=None)
                    control = self.controller.step(self.robot, (5, 0), 0.1)
                    self.assertFalse(control.failed)
                control = self.controller.step(self.robot, (5, 0), 0.1)
                self.assertEqual(control.phase, 'REPLAN')
                self.assertEqual((control.v_l, control.v_r), (0, 0))
                self.assertEqual(self.controller._replan_attempts, attempt)
            for _ in range(50):
                self.controller.local.selected_candidate = SimpleNamespace(v=0, score=0, trajectory=None)
                control = self.controller.step(self.robot, (5, 0), 0.1)
            self.assertTrue(control.failed)

    def test_no_safe_command_stops_replans_and_eventually_fails(self):
        self.controller.reset(self.robot, (5, 0))
        with patch.object(self.controller.local, 'step', return_value=Control(0, 0, failed=True)):
            self.assertEqual(self.controller.step(self.robot, (5, 0), 0.1).phase, 'REPLAN')
            self.assertEqual(self.controller.step(self.robot, (5, 0), 0.1).phase, 'REPLAN')
            self.assertTrue(self.controller.step(self.robot, (5, 0), 0.1).failed)

    def test_progress_clears_replan_attempts(self):
        self.controller.reset(self.robot, (5, 0))
        self.controller._replan_attempts = 2
        self.robot.x = 0.15
        self.controller.step(self.robot, (5, 0), 0.1)
        self.assertEqual(self.controller._replan_attempts, 0)

    def test_local_target_is_not_final_arrival(self):
        self.controller.reset(self.robot, (5, 0))
        self.robot.x = 1.5
        control = self.controller.step(self.robot, (5, 0), 0.1)
        self.assertFalse(control.arrived)
        self.assertGreater(self.controller.local_target[0], 1.5)
        self.robot.x = 4.9
        self.assertTrue(self.controller.step(self.robot, (5, 0), 0.1).arrived)

    def test_collision_failed_until_reset_and_goal_change_replans(self):
        self.controller.reset(self.robot, (5, 0))
        self.controller.on_collision(self.robot)
        self.assertTrue(self.controller.step(self.robot, (5, 0), 0.1).failed)
        control = self.controller.step(self.robot, (4, 1), 0.1)
        self.assertFalse(control.failed)
        self.assertEqual(self.controller.path[-1], (4, 1))

    def test_failed_global_plan_and_outside_map_stop(self):
        self.obstacles.add(Circle((5, 0), 1))
        self.assertTrue(self.controller.step(self.robot, (5, 0), 0.1).failed)
        self.assertTrue(self.controller.step(self.robot, (21, 0), 0.1).failed)


if __name__ == '__main__':
    unittest.main()
