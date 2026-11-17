"""Kiểm chứng tìm đường và bám waypoint với robot/vật cản thật."""
import math
import unittest

import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Rectangle

from src.Obstacle import Obstacle
from src.Robot import Differential_Robot
from src.global_planner import GlobalPlannerController, plan_path


def robot():
    return Differential_Robot(1.0, 2.0, 1.0, 0.5, 0.5)


class GlobalPlannerIntegrationTests(unittest.TestCase):
    def drive_to_goal(self, obstacles, goal, algorithm, max_steps=2200):
        vehicle = robot()
        controller = GlobalPlannerController(obstacles, algorithm=algorithm)
        controller.reset(vehicle, goal)
        self.assertTrue(len(controller.path), controller.result.message)
        for tick in range(max_steps):
            before = vehicle.pose
            command = controller.step(vehicle, goal, 0.1)
            self.assertEqual(vehicle.pose, before)
            self.assertFalse(command.failed, command.message)
            if command.arrived:
                self.assertLess(math.hypot(vehicle.x-goal[0], vehicle.y-goal[1]), 0.31)
                return controller
            vehicle.set_wheel_velocities(command.v_l, command.v_r)
            self.assertFalse(obstacles.collides_pose(vehicle, *vehicle.predict_pose(0.1)),
                             f'{algorithm}: va chạm tại bước {tick}')
            vehicle.update_pose(0.1)
        self.fail(f'{algorithm}: không tới goal sau {max_steps} bước; pose={vehicle.pose}')

    def test_empty_map_front_and_rear_arrival(self):
        for algorithm in ('A*', 'Dijkstra'):
            for goal in ((5., 0.), (-4., 0.)):
                with self.subTest(algorithm=algorithm, goal=goal):
                    self.drive_to_goal(Obstacle(), goal, algorithm)

    def test_front_obstacle_detour_arrives_without_collision(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((2., -1.), 1., 2.))
        for algorithm in ('A*', 'Dijkstra'):
            with self.subTest(algorithm=algorithm):
                controller = self.drive_to_goal(obstacles, (7., 0.), algorithm)
                self.assertGreater(max(abs(point[1]) for point in controller.path), 1.)

    def test_u_shaped_trap_escapes_and_reaches_goal(self):
        obstacles = Obstacle()
        for xy, width, height in [((-3., -3.5), 7., .5),
                                  ((-3., 3.), 7., .5),
                                  ((3.5, -3.5), .5, 7.)]:
            obstacles.add(Rectangle(xy, width, height))
        for algorithm in ('A*', 'Dijkstra'):
            with self.subTest(algorithm=algorithm):
                controller = self.drive_to_goal(obstacles, (8., 0.), algorithm)
                self.assertLess(min(point[0] for point in controller.path), -3.)

    def test_equal_grid_cost_with_fewer_astar_expansions(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((2., -1.), 1., 2.))
        results = [plan_path(obstacles, (0., 0.), (7., 0.), algorithm=algorithm,
                             clearance=1.3) for algorithm in ('A*', 'Dijkstra')]
        self.assertTrue(all(len(result.path) for result in results))
        self.assertAlmostEqual(results[0].cost, results[1].cost)
        self.assertLessEqual(results[0].expanded, results[1].expanded)

    def test_unreachable_wall_then_map_edit_reset_replans(self):
        obstacles = Obstacle()
        wall = Rectangle((2., -20.), 1., 40.)
        obstacles.add(wall)
        for algorithm in ('A*', 'Dijkstra'):
            with self.subTest(algorithm=algorithm):
                vehicle = robot()
                controller = GlobalPlannerController(obstacles, algorithm=algorithm)
                controller.reset(vehicle, (7., 0.))
                command = controller.step(vehicle, (7., 0.), .1)
                self.assertTrue(command.failed)
                self.assertEqual((command.v_l, command.v_r), (0., 0.))
                self.assertFalse(len(controller.path))
                obstacles.remove(wall)
                controller.reset(vehicle, (7., 0.))
                self.assertTrue(len(controller.path))
                self.assertFalse(controller.step(vehicle, (7., 0.), .1).failed)
                obstacles.add(wall)


if __name__ == '__main__':
    unittest.main()
