"""Kiểm tra ba thuật toán né tránh Bug1, Bug2 và Tangent Bug (BTH2.1)."""
import unittest

import numpy as np
from matplotlib.patches import Circle, Rectangle

from src.Obstacle import Obstacle
from src.Robot import Differential_Robot
from src.avoidance import (
    Bug1Controller, Bug2Controller, TangentBugController, robot_clearance,
)
from src.boundary import offset_boundary

DT = 0.1


def make_robot():
    robot = Differential_Robot(
        1.0, robot_length=2.0, robot_width=1.0,
        wheel_length=0.5, wheel_width=0.5,
    )
    robot.x, robot.y, robot.theta = 0.0, 0.0, 0.0
    return robot


def make_obstacles(kind='box'):
    obstacles = Obstacle()
    if kind == 'box':
        obstacles.add(Rectangle((2, -1), 2, 2, color='gray'))
    elif kind == 'circle':
        obstacles.add(Circle((3, 0), 1.5, color='gray'))
    elif kind == 'wall':
        obstacles.add(Rectangle((2, -3), 1, 6, color='gray'))
    return obstacles


def drive(controller, obstacles, robot, goal, max_steps=4000):
    """Mô phỏng headless; trả (phase, số bước, số va chạm)."""
    controller.reset(robot, goal)
    collisions = 0
    for step in range(max_steps):
        control = controller.step(robot, goal, DT)
        if control.arrived or control.failed:
            return control.phase, step, collisions
        robot.set_wheel_velocities(control.v_l, control.v_r)
        x_next, y_next, theta_next = robot.predict_pose(DT)
        if obstacles.collides_pose(robot, x_next, y_next, theta_next):
            collisions += 1
            robot.set_wheel_velocities(0.0, 0.0)
            controller.on_collision(robot)
        else:
            robot.update_pose(DT)
    return 'TIMEOUT', max_steps, collisions


class ControllerBasicsTests(unittest.TestCase):
    def test_clearance_covers_robot_body(self):
        robot = make_robot()
        clearance = robot_clearance(robot)
        body_radius = 0.5 * np.hypot(
            robot.robot_length, robot.robot_width)
        self.assertGreater(clearance, body_radius)

    def test_goal_inside_obstacle_fails(self):
        obstacles = make_obstacles('circle')
        robot = make_robot()
        for controller in (
                Bug1Controller(obstacles),
                Bug2Controller(obstacles),
                TangentBugController(obstacles)):
            controller.reset(robot, (3.0, 0.0))
            control = controller.step(robot, (3.0, 0.0), DT)
            self.assertTrue(control.failed)
            self.assertIn('vật cản', control.message)

    def test_goal_too_close_to_obstacle_fails(self):
        obstacles = make_obstacles('circle')
        robot = make_robot()
        # Cách tâm (3, 0) 1.6 m: ngoài hình tròn nhưng sát hơn bán kính robot.
        for controller in (
                Bug1Controller(obstacles),
                Bug2Controller(obstacles),
                TangentBugController(obstacles)):
            controller.reset(robot, (4.6, 0.0))
            control = controller.step(robot, (4.6, 0.0), DT)
            self.assertTrue(control.failed)


class Bug1Tests(unittest.TestCase):
    def test_box_records_path_and_leave_point(self):
        obstacles = make_obstacles('box')
        robot = make_robot()
        controller = Bug1Controller(obstacles)
        phase, _, collisions = drive(controller, obstacles, robot, (6.0, 0.0))
        self.assertEqual(phase, 'ARRIVED')
        self.assertGreaterEqual(collisions, 1)
        self.assertIsNotNone(controller.debug['hit'])
        self.assertIsNotNone(controller.debug['leave'])
        self.assertIsNotNone(controller.debug['boundary'])

        boundary = offset_boundary(
            obstacles.obs[0], robot_clearance(robot))
        _, _, distance = boundary.project(controller.leave_point)
        self.assertLess(distance, 1e-6)

    def test_leave_point_only_after_full_loop(self):
        obstacles = make_obstacles('box')
        robot = make_robot()
        controller = Bug1Controller(obstacles)
        goal = (6.0, 0.0)
        controller.reset(robot, goal)

        checked_early = False
        saw_leave_point = False
        for _ in range(4000):
            control = controller.step(robot, goal, DT)
            if (control.phase == 'FOLLOW' and controller.boundary is not None
                    and controller._accumulated
                    < 0.5 * controller.boundary.perimeter):
                checked_early = True
                self.assertIsNone(controller._l_point)
                self.assertIsNone(controller.debug.get('target_leave'))
            if controller._l_point is not None:
                saw_leave_point = True
                self.assertGreaterEqual(
                    controller._accumulated,
                    controller.boundary.perimeter - 0.25,
                )
            if control.arrived or control.failed:
                break
            robot.set_wheel_velocities(control.v_l, control.v_r)
            x_next, y_next, theta_next = robot.predict_pose(DT)
            if obstacles.collides_pose(robot, x_next, y_next, theta_next):
                robot.set_wheel_velocities(0.0, 0.0)
                controller.on_collision(robot)
            else:
                robot.update_pose(DT)

        self.assertTrue(checked_early)
        self.assertTrue(saw_leave_point)
        self.assertEqual(control.phase, 'ARRIVED')
        self.assertIsNotNone(controller.debug['leave'])

    def test_circle_arrives(self):
        obstacles = make_obstacles('circle')
        robot = make_robot()
        phase, _, _ = drive(
            Bug1Controller(obstacles), obstacles, robot, (6.0, 0.0))
        self.assertEqual(phase, 'ARRIVED')

    def test_wall_arrives(self):
        obstacles = make_obstacles('wall')
        robot = make_robot()
        phase, _, _ = drive(
            Bug1Controller(obstacles), obstacles, robot, (6.0, 0.0))
        self.assertEqual(phase, 'ARRIVED')


class Bug2Tests(unittest.TestCase):
    def test_m_line_and_leave_point(self):
        obstacles = make_obstacles('box')
        robot = make_robot()
        controller = Bug2Controller(obstacles)
        goal = np.array([6.0, 0.0])
        phase, _, collisions = drive(controller, obstacles, robot, goal)
        self.assertEqual(phase, 'ARRIVED')
        self.assertGreaterEqual(collisions, 1)
        self.assertIsNotNone(controller.debug['m_line'])
        start, end = controller.debug['m_line']
        self.assertTrue(np.allclose(start, (0.0, 0.0)))
        self.assertTrue(np.allclose(end, goal))
        self.assertIsNotNone(controller.debug['leave'])

        leave = np.asarray(controller.debug['leave'], dtype=float)
        direction = goal - start
        projection = start + float(
            (leave - start) @ direction / (direction @ direction)
        ) * direction
        self.assertAlmostEqual(float(np.linalg.norm(leave - projection)), 0.0)
        hit = np.asarray(controller.debug['hit'], dtype=float)
        self.assertLess(
            float(np.linalg.norm(leave - goal)),
            float(np.linalg.norm(hit - goal)),
        )

    def test_wall_arrives(self):
        obstacles = make_obstacles('wall')
        robot = make_robot()
        phase, _, _ = drive(
            Bug2Controller(obstacles), obstacles, robot, (6.0, 0.0))
        self.assertEqual(phase, 'ARRIVED')


class TangentBugTests(unittest.TestCase):
    def test_box_uses_range_sensor_without_collision(self):
        obstacles = make_obstacles('box')
        robot = make_robot()
        controller = TangentBugController(obstacles)
        goal = (6.0, 0.0)
        controller.reset(robot, goal)

        followed = False
        saw_tangent = False
        saw_boundary = False
        collisions = 0
        for _ in range(4000):
            control = controller.step(robot, goal, DT)
            if control.phase == 'FOLLOW':
                followed = True
                saw_tangent = saw_tangent or (
                    controller.debug.get('tangent') is not None)
                saw_boundary = saw_boundary or (
                    controller.debug.get('boundary') is not None)
            if control.arrived or control.failed:
                break
            robot.set_wheel_velocities(control.v_l, control.v_r)
            x_next, y_next, theta_next = robot.predict_pose(DT)
            if obstacles.collides_pose(robot, x_next, y_next, theta_next):
                collisions += 1
                robot.set_wheel_velocities(0.0, 0.0)
                controller.on_collision(robot)
            else:
                robot.update_pose(DT)

        self.assertEqual(control.phase, 'ARRIVED')
        self.assertTrue(followed)
        self.assertTrue(saw_tangent)
        self.assertTrue(saw_boundary)
        self.assertIsNotNone(controller.last_scan)
        self.assertEqual(collisions, 0)
        self.assertIsNotNone(controller.debug['leave'])

    def test_wall_arrives_without_collision(self):
        obstacles = make_obstacles('wall')
        robot = make_robot()
        controller = TangentBugController(obstacles)
        phase, _, collisions = drive(
            controller, obstacles, robot, (6.0, 0.0))
        self.assertEqual(phase, 'ARRIVED')
        self.assertEqual(collisions, 0)

    def test_circle_arrives_without_collision(self):
        obstacles = make_obstacles('circle')
        robot = make_robot()
        controller = TangentBugController(obstacles)
        phase, _, collisions = drive(
            controller, obstacles, robot, (6.0, 0.0))
        self.assertEqual(phase, 'ARRIVED')
        self.assertEqual(collisions, 0)


if __name__ == '__main__':
    unittest.main()
