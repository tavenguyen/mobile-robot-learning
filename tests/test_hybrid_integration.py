"""Kiểm chứng A* + DWA với robot thật và các tình huống map."""
import math
import unittest

import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Circle, Rectangle

from src.Obstacle import Obstacle
from src.Robot import Differential_Robot
from src.hybrid_planner import AStarDWAController


def make_robot():
    return Differential_Robot(1.0, 2.0, 1.0, 0.5, 0.5)


class HybridIntegrationTests(unittest.TestCase):
    def drive(self, obstacles, goal, max_steps=2500):
        robot = make_robot()
        controller = AStarDWAController(obstacles)
        controller.reset(robot, goal)
        self.assertTrue(len(controller.path), controller.result.message)
        previous_result = controller.result
        previous_segment = controller.debug.get('segment', 0)
        for tick in range(max_steps):
            before = robot.pose
            command = controller.step(robot, goal, .1)
            segment = controller.debug.get('segment', 0)
            if controller.result is previous_result:
                self.assertGreaterEqual(segment, previous_segment)
                self.assertLessEqual(segment, previous_segment+1)
            previous_result, previous_segment = controller.result, segment
            self.assertEqual(robot.pose, before)
            self.assertFalse(command.failed,
                             f'{command.message}; tick={tick}; pose={robot.pose}; target={controller.local_target}')
            if command.arrived:
                self.assertLessEqual(math.dist(robot.pose[:2], goal), .3)
                self.assertEqual((command.v_l, command.v_r), (0., 0.))
                return controller
            robot.set_wheel_velocities(command.v_l, command.v_r)
            self.assertFalse(obstacles.collides_pose(robot, *robot.predict_pose(.1)),
                             f'Va chạm tại bước {tick}')
            robot.update_pose(.1)
        self.fail(f'Không tới goal: pose={robot.pose}, target={controller.local_target}')

    def test_empty_map_front_and_rear_goals(self):
        for goal in ((5., 0.), (-4., 0.)):
            with self.subTest(goal=goal):
                self.drive(Obstacle(), goal)

    def test_frontal_obstacle_detour_reaches_final_goal(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((2., -1.), 1., 2.))
        self.drive(obstacles, (7., 0.))

    def test_u_trap_escapes_without_jumping_to_other_branch(self):
        obstacles = Obstacle()
        for xy, width, height in [((-3., -3.5), 7., .5),
                                  ((-3., 3.), 7., .5),
                                  ((3.5, -3.5), .5, 7.)]:
            obstacles.add(Rectangle(xy, width, height))
        controller = self.drive(obstacles, (8., 0.))
        self.assertLess(min(p[0] for p in controller.path), -3.)

    def test_scores_and_selected_wheel_command_match(self):
        robot = make_robot()
        controller = AStarDWAController(Obstacle())
        controller.reset(robot, (5., 1.))
        command = controller.step(robot, (5., 1.), .1)
        self.assertTrue(controller.candidates)
        cfg = controller.config
        for candidate in controller.candidates:
            self.assertAlmostEqual(candidate.score,
                cfg.alpha*candidate.heading + cfg.beta*candidate.dist + cfg.gamma*candidate.velocity)
        self.assertEqual([c.score for c in controller.candidates],
                         sorted((c.score for c in controller.candidates), reverse=True))
        best = controller.selected_candidate
        self.assertIs(best, controller.candidates[0])
        self.assertAlmostEqual((command.v_l+command.v_r)/2, best.v)
        self.assertAlmostEqual((command.v_r-command.v_l)/robot.L, best.w)

    def test_map_edit_reset_discards_stale_results_and_rejects_blocked_goal(self):
        robot = make_robot()
        obstacles = Obstacle()
        controller = AStarDWAController(obstacles)
        goal = (6., 0.)
        controller.reset(robot, goal)
        controller.step(robot, goal, .1)
        self.assertTrue(controller.candidates)
        obstacles.add(Circle(goal, 1.))
        controller.reset(robot, goal)
        self.assertFalse(controller.candidates)
        self.assertIsNone(controller.selected_candidate)
        self.assertFalse(len(controller.path))
        command = controller.step(robot, goal, .1)
        self.assertTrue(command.failed)
        self.assertEqual((command.v_l, command.v_r), (0., 0.))
        obstacles.clear()
        controller.reset(robot, goal)
        self.assertTrue(len(controller.path))
        self.assertFalse(controller.step(robot, goal, .1).failed)

    def test_stationary_robot_recovery_is_bounded(self):
        # Không áp dụng lệnh: mô phỏng robot bị kẹt dù map vẫn có đường.
        robot = make_robot()
        controller = AStarDWAController(Obstacle())
        controller.reset(robot, (5., 0.))
        replans = 0
        for _ in range(180):
            command = controller.step(robot, (5., 0.), .1)
            if command.phase == 'REPLAN':
                replans += 1
                self.assertEqual((command.v_l, command.v_r), (0., 0.))
            if command.failed:
                self.assertEqual(replans, 2)
                self.assertEqual((command.v_l, command.v_r), (0., 0.))
                break
        else:
            self.fail('Không dừng sau hai lần phục hồi khi robot không tiến')

    def test_unreachable_wall_stops(self):
        robot = make_robot()
        obstacles = Obstacle()
        obstacles.add(Rectangle((2., -20.), 1., 40.))
        controller = AStarDWAController(obstacles)
        controller.reset(robot, (7., 0.))
        command = controller.step(robot, (7., 0.), .1)
        self.assertTrue(command.failed)
        self.assertEqual((command.v_l, command.v_r), (0., 0.))


if __name__ == '__main__':
    unittest.main()
