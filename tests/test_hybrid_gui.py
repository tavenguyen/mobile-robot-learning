"""Kiểm tra chế độ A* + DWA độc lập với Tkinter/display."""
from dataclasses import replace
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from src.dwa import DWAConfig
from src.hybrid_planner import AStarDWAController
from tests import test_global_planner_gui as global_gui
from tests.test_obstacle_gui_layout import Variable


class HybridGUITests(unittest.TestCase):
    def setUp(self):
        global_gui.GlobalPlannerGUITests.setUp(self)
        self.gui.__class__.__init__.__globals__.update(
            AStarDWAController=AStarDWAController, replace=replace, math=math)
        self.gui.avoidance_mode = 'A* + DWA'
        self.gui.dwa_config = DWAConfig()
        self.gui.avoidance_controller.name = 'A* + DWA'

    def test_factory_uses_hybrid_and_shared_weight_config(self):
        controller = self.gui._make_controller()
        self.assertIsInstance(controller, AStarDWAController)
        self.assertIs(controller.config, self.gui.dwa_config)

    def test_weights_update_without_replanning_or_stopping(self):
        self.gui.avoidance_controller = self.gui._make_controller()
        controller = self.gui.avoidance_controller
        controller.reset = Mock()
        self.gui.stop = Mock()
        self.gui.dwa_weight_vars = {}
        self.gui.dwa_status_label = Mock()
        for key, value in (('alpha', '1'), ('beta', '.2'), ('gamma', '.3')):
            variable = Variable()
            variable.set(value)
            self.gui.dwa_weight_vars[key] = variable
        self.gui._apply_dwa_weights()
        self.assertEqual(controller.config.alpha, 1)
        self.assertEqual(controller.config.beta, .2)
        controller.reset.assert_not_called()
        self.gui.stop.assert_not_called()

    def test_hybrid_displays_global_and_local_overlays(self):
        self.gui._update_global_display = Mock()
        self.gui._update_dwa_display = Mock()
        self.gui._update_avoidance_overlays()
        self.gui._update_global_display.assert_called_once_with()
        self.gui._update_dwa_display.assert_called_once_with()
        self.gui.dwa_frame = Mock()
        self.gui._sync_dwa_visibility()
        self.gui.dwa_frame.grid.assert_called_once_with()

    def test_paused_map_edit_stops_before_replan(self):
        timeline = Mock()
        self.gui.stop = timeline.stop
        self.gui.avoidance_controller.reset = timeline.reset
        self.gui.follow_x = []
        self.gui.follow_y = []
        self.gui._clear_avoidance_overlays = Mock()
        self.gui._update_obstacle_controls = Mock()
        self.gui._on_obstacles_changed('move')
        self.assertEqual([call[0] for call in timeline.mock_calls],
                         ['stop', 'reset'])
        timeline.reset.assert_called_once_with(self.gui.robot,
                                               self.gui.destination)
        self.assertFalse(self.gui.auto_mode)

    def test_new_goal_stops_before_replan(self):
        timeline = Mock()
        self.gui.stop = timeline.stop
        self.gui.avoidance_controller.reset = timeline.reset
        self.gui.toolbar = None
        self.gui._update_auto_button = Mock()
        self.gui._clear_collision_message = Mock()
        self.gui._clear_avoidance_overlays = Mock()
        self.gui.destination_marker = None
        self.gui.ax.plot.return_value = (Mock(),)
        event = SimpleNamespace(inaxes=self.gui.ax, xdata=4, ydata=5, button=1)
        self.gui.on_click(event)
        self.assertEqual([call[0] for call in timeline.mock_calls],
                         ['stop', 'reset'])
        timeline.reset.assert_called_once_with(self.gui.robot, (4., 5.))
        self.assertTrue(self.gui.auto_mode)

    def test_switching_into_hybrid_stops_before_controller_creation(self):
        self.gui.avoidance_mode = 'DWA'
        self.gui.avoidance_combobox = Mock()
        self.gui.avoidance_combobox.get.return_value = 'A* + DWA'
        timeline = Mock()
        self.gui.stop = timeline.stop
        self.gui._make_controller = timeline.factory
        self.gui._sync_dwa_visibility = Mock()
        self.gui._clear_avoidance_overlays = Mock()
        self.gui.avoidance_label = Mock()
        self.gui.on_avoidance_change()
        self.assertEqual([call[0] for call in timeline.mock_calls[:2]],
                         ['stop', 'factory'])
        self.gui.avoidance_controller.reset.assert_called_once_with(
            self.gui.robot, self.gui.destination)

    def test_collision_stops_hybrid_without_bug_recovery(self):
        global_gui.GlobalPlannerGUITests.test_collision_stops_global_without_bug_recovery(self)

    def test_reset_clears_global_and_local_diagnostics(self):
        self.gui.avoidance_controller = self.gui._make_controller()
        self.gui.avoidance_controller.path = [(0, 0), (1, 1)]
        self.gui.avoidance_controller.local.candidates = [Mock()]
        self.gui.dwa_rollout = Mock()
        self.gui.dwa_table = Mock()
        self.gui.dwa_table.get_children.return_value = ('stale',)
        for name in ('scan_collection', 'goal_line', 'tangent_line',
                     'boundary_line', 'm_line', 'leave_line', 'hit_marker',
                     'target_leave_marker', 'leave_marker', 'path_line',
                     'follow_line', 'robot_dot', 'collision_label'):
            setattr(self.gui, name, Mock())
        self.gui.obstacle_editor = None
        self.gui.destination_marker = None
        self.gui._update_auto_button = Mock()
        self.gui.update_velocity_labels = Mock()
        self.gui._update_obstacle_controls = Mock()
        self.gui.reset()
        self.assertEqual(self.gui.avoidance_controller.path, [])
        self.assertEqual(self.gui.avoidance_controller.candidates, [])
        self.gui.global_path_line.set_data.assert_called_with([], [])
        self.gui.waypoint_marker.set_data.assert_called_with([], [])
        self.gui.dwa_rollout.set_data.assert_called_with([], [])
        self.gui.dwa_table.delete.assert_called_once_with('stale')


if __name__ == '__main__':
    unittest.main()
