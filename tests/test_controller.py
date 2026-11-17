import unittest
import numpy as np
from src.controller import RobotController


class ControllerTests(unittest.TestCase):
    def test_drive_stop(self):
        robot = RobotController()
        robot.drive(1)
        for _ in range(50):
            robot.tick()
        np.testing.assert_allclose(robot.state, [.1, 0, 0], atol=1e-12)
        robot.drive(-1)
        for _ in range(50):
            robot.tick()
        np.testing.assert_allclose(robot.state, [0, 0, 0], atol=1e-12)
        robot.noise_std = (1, 1)
        robot.turn(1)
        robot.stop()
        before = robot.state.copy()
        robot.tick()
        np.testing.assert_array_equal(robot.state, before)

    def test_turn_commands(self):
        for direction in [-1, 1]:
            for drive in [-1, 0, 1]:
                robot = RobotController()
                robot.drive(drive)
                robot.turn(direction)
                for _ in range(30):
                    robot.tick()
                self.assertAlmostEqual(robot.state[2], direction*np.deg2rad(15))
                self.assertEqual(robot.velocity[1], 0)
                self.assertEqual(robot.speed, drive*.1)
        robot = RobotController()
        robot.turn(1)
        robot.turn(1)
        robot.turn(-1)
        for _ in range(30):
            robot.tick()
        self.assertAlmostEqual(robot.state[2], np.deg2rad(15))
        robot.turn(1)
        robot.drive(1)
        self.assertEqual(robot.turn_remaining, 0)
