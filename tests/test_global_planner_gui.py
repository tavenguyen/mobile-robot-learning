"""Tích hợp GUI planner toàn cục không cần Tkinter/display."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from src.global_planner import GlobalPlannerController
from tests.test_obstacle_gui_layout import gui_class


class GlobalPlannerGUITests(unittest.TestCase):
    def setUp(self):
        cls, _ = gui_class()
        cls.__init__.__globals__.update(
            np=np, GlobalPlannerController=GlobalPlannerController,
            AVOIDANCE_MODES=('Tắt né tránh', 'Bug1', 'Bug2', 'Tangent Bug',
                             'DWA', 'Dijkstra', 'A*'))
        self.gui = cls.__new__(cls)
        self.gui.avoidance_mode = 'A*'
        self.gui.obstacles = Mock()
        self.gui.robot = Mock()
        self.gui.destination = (8, 8)
        self.gui.auto_mode = False
        self.gui.avoidance_controller = Mock(spec=GlobalPlannerController)
        self.gui.avoidance_controller.path = [(0, 0), (1, 1), (8, 8)]
        self.gui.avoidance_controller.debug = {
            'target': (1, 1), 'message': 'A* | cost=12 | expanded=20 | waypoint=1/3'}
        self.gui.global_path_line = Mock()
        self.gui.waypoint_marker = Mock()
        self.gui.nav_label = Mock()
        self.gui.canvas = Mock()
        self.gui.ax = Mock()

    def test_factory_supports_both_standalone_global_algorithms(self):
        for algorithm in ('Dijkstra', 'A*'):
            self.gui.avoidance_mode = algorithm
            controller = self.gui._make_controller()
            self.assertIsInstance(controller, GlobalPlannerController)
            self.assertEqual(controller.algorithm, algorithm)
            self.assertIs(controller.obstacles, self.gui.obstacles)

    def test_switch_stops_before_build_and_plans_while_paused(self):
        self.gui.avoidance_mode = 'Bug1'
        self.gui.avoidance_combobox = Mock()
        self.gui.avoidance_combobox.get.return_value = 'Dijkstra'
        calls = Mock()
        self.gui.stop = calls.stop
        self.gui._make_controller = calls.factory
        self.gui._sync_dwa_visibility = Mock()
        self.gui._clear_avoidance_overlays = Mock()
        self.gui.avoidance_label = Mock()
        self.gui.on_avoidance_change()
        self.assertEqual([call[0] for call in calls.mock_calls[:2]],
                         ['stop', 'factory'])
        self.gui.avoidance_controller.reset.assert_called_once_with(
            self.gui.robot, self.gui.destination)
        self.assertFalse(self.gui.auto_mode)

    def test_display_global_path_waypoint_and_status_outside_local_modes(self):
        self.gui._update_avoidance_overlays()
        data = self.gui.global_path_line.set_data.call_args.args
        np.testing.assert_equal(data[0], [0, 1, 8])
        np.testing.assert_equal(data[1], [0, 1, 8])
        self.gui.waypoint_marker.set_data.assert_called_once_with([1.0], [1.0])
        self.gui.nav_label.config.assert_called_once_with(
            text=self.gui.avoidance_controller.debug['message'])
        self.gui.avoidance_controller.path = []
        self.gui.avoidance_controller.debug['target'] = None
        self.gui._update_global_display()
        self.gui.global_path_line.set_data.assert_called_with([], [])
        self.gui.waypoint_marker.set_data.assert_called_with([], [])

    def test_map_change_replans_even_while_paused(self):
        self.gui.follow_x = [1]
        self.gui.follow_y = [2]
        self.gui.ax = Mock()
        self.gui._clear_avoidance_overlays = Mock()
        self.gui._update_obstacle_controls = Mock()
        self.gui._sync_obstacle_dimensions = Mock()
        self.gui._on_obstacles_changed('resize')
        self.gui.avoidance_controller.reset.assert_called_once_with(
            self.gui.robot, self.gui.destination)
        self.gui._clear_avoidance_overlays.assert_called_once_with()
        self.assertFalse(self.gui.auto_mode)

    def test_collision_stops_global_without_bug_recovery(self):
        self.gui.auto_mode = True
        self.gui.dt = .1
        self.gui.last_collision_state = False
        self.gui.follow_x = []
        self.gui.follow_y = []
        self.gui.robot.predict_pose.return_value = (1, 2, 0)
        self.gui.obstacles.collides_pose.return_value = True
        self.gui._auto_control = Mock(return_value=SimpleNamespace(
            arrived=False, failed=False))
        self.gui.stop = Mock()
        self.gui._update_auto_button = Mock()
        self.gui._update_avoidance_overlays = Mock()
        self.gui.update_velocity_labels = Mock()
        for name in ('path_line', 'follow_line', 'robot_dot', 'collision_label'):
            setattr(self.gui, name, Mock())
        self.gui.update()
        self.gui.avoidance_controller.on_collision.assert_called_once_with(
            self.gui.robot)
        self.gui.stop.assert_called_once_with()
        self.assertFalse(self.gui.auto_mode)
        self.gui._update_auto_button.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
