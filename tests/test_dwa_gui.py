"""Kiểm tra DWA GUI headless với controller thật và widget giả."""
from dataclasses import replace
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from src.dwa import DWAConfig, DWAController
from tests.test_obstacle_gui_layout import gui_class, Variable


class DWAGUITests(unittest.TestCase):
    def setUp(self):
        cls, _ = gui_class()
        self.clock = Mock(return_value=10.0)
        cls.__init__.__globals__.update(
            DWAConfig=DWAConfig, DWAController=DWAController,
            replace=replace, math=math, np=np,
            time=SimpleNamespace(monotonic=self.clock),
            AVOIDANCE_MODES=('Tắt né tránh', 'Bug1', 'Bug2', 'Tangent Bug', 'DWA'))
        self.gui = cls.__new__(cls)
        self.gui.dwa_config = DWAConfig()
        self.gui.dwa_weight_vars = {}
        for key, value in (('alpha', '.6'), ('beta', '.3'), ('gamma', '.1')):
            variable = Variable()
            variable.set(value)
            self.gui.dwa_weight_vars[key] = variable
        self.gui.dwa_status_label = Mock()
        self.gui.obstacles = Mock()
        self.gui.avoidance_controller = DWAController(self.gui.obstacles)
        self.gui.avoidance_mode = 'DWA'
        self.gui.auto_mode = True
        self.gui.dwa_rollout = Mock()
        self.gui.dwa_table = Mock()
        self.gui.dwa_table.get_children.return_value = ('old',)
        self.gui.dwa_table.yview.return_value = (.25, .5)
        self.gui._dwa_last_refresh = float('-inf')

    def test_apply_weights_updates_controller_without_reset_or_stop(self):
        controller = self.gui.avoidance_controller
        controller.reset = Mock()
        self.gui.stop = Mock()
        self.gui.dwa_weight_vars['alpha'].set('2')
        self.assertEqual(self.gui._apply_dwa_weights(), 'break')
        self.assertEqual(controller.config.alpha, 2)
        self.assertIs(controller.config, self.gui.dwa_config)
        controller.reset.assert_not_called()
        self.gui.stop.assert_not_called()

    def test_invalid_weights_are_atomic(self):
        original = self.gui.dwa_config
        for values in (('nan', '1', '1'), ('inf', '1', '1'),
                       ('-1', '1', '1'), ('0', '0', '0'), ('bad', '1', '1')):
            for key, value in zip(('alpha', 'beta', 'gamma'), values):
                self.gui.dwa_weight_vars[key].set(value)
            self.gui._apply_dwa_weights()
            self.assertIs(self.gui.dwa_config, original)
            self.assertIsNotNone(self.gui.dwa_status_label.config.call_args)

    def test_factory_reuses_applied_config(self):
        controller = self.gui._make_controller()
        self.assertIsInstance(controller, DWAController)
        self.assertIs(controller.config, self.gui.dwa_config)

    def test_switch_into_dwa_stops_before_controller_creation(self):
        self.gui.avoidance_mode = 'Bug1'
        self.gui.avoidance_combobox = Mock()
        self.gui.avoidance_combobox.get.return_value = 'DWA'
        calls = Mock()
        self.gui.stop = calls.stop
        self.gui._make_controller = calls.factory
        self.gui._sync_dwa_visibility = Mock()
        self.gui._clear_avoidance_overlays = Mock()
        self.gui.destination = (3, 4)
        self.gui.robot = Mock()
        self.gui.avoidance_label = Mock()
        self.gui.nav_label = Mock()
        self.gui.canvas = Mock()
        self.gui.on_avoidance_change()
        self.assertEqual([call[0] for call in calls.mock_calls[:2]],
                         ['stop', 'factory'])
        self.gui.avoidance_controller.reset.assert_called_once_with(
            self.gui.robot, self.gui.destination)

    def test_table_shows_all_candidates_selected_and_throttles(self):
        def candidate(score):
            return SimpleNamespace(v=.2, w=.1, heading=.5, dist=.3,
                                   velocity=.1, score=score,
                                   trajectory=np.array([[0, 0, 0], [1, 2, .1]]))
        candidates = [candidate(1 - index / 100) for index in range(40)]
        controller = self.gui.avoidance_controller
        controller.candidates = candidates
        controller.selected_candidate = candidates[0]
        self.gui._update_dwa_display()
        self.assertEqual(self.gui.dwa_table.insert.call_count, 40)
        first = self.gui.dwa_table.insert.call_args_list[0].kwargs
        self.assertEqual(first['tags'], ('selected',))
        self.assertEqual(first['values'][0], '✓')
        self.gui.dwa_table.yview_moveto.assert_called_with(.25)
        plotted = self.gui.dwa_rollout.set_data.call_args.args
        np.testing.assert_equal(plotted[0], [0, 1])
        np.testing.assert_equal(plotted[1], [0, 2])
        self.clock.return_value = 10.1
        self.gui._update_dwa_display()
        self.assertEqual(self.gui.dwa_table.insert.call_count, 40)
        self.clock.return_value = 10.21
        self.gui._update_dwa_display()
        self.assertEqual(self.gui.dwa_table.insert.call_count, 80)

    def test_clear_removes_rollout_table_and_allows_immediate_refresh(self):
        self.gui._clear_dwa_display()
        self.gui.dwa_rollout.set_data.assert_called_once_with([], [])
        self.gui.dwa_table.delete.assert_called_once_with('old')
        self.assertEqual(self.gui._dwa_last_refresh, float('-inf'))

    def test_controls_hide_outside_dwa(self):
        self.gui.dwa_frame = Mock()
        self.gui.avoidance_mode = 'Bug2'
        self.gui._sync_dwa_visibility()
        self.gui.dwa_frame.grid_remove.assert_called_once_with()
        self.gui.avoidance_mode = 'DWA'
        self.gui._sync_dwa_visibility()
        self.gui.dwa_frame.grid.assert_called_once_with()

    def test_map_edit_and_reset_clear_dwa_diagnostics(self):
        for name in ('scan_collection', 'goal_line', 'tangent_line',
                     'boundary_line', 'm_line', 'leave_line', 'hit_marker',
                     'target_leave_marker', 'leave_marker', 'path_line',
                     'follow_line', 'robot_dot', 'ax', 'canvas', 'robot',
                     'nav_label', 'collision_label'):
            setattr(self.gui, name, Mock())
        self.gui._update_obstacle_controls = Mock()
        self.gui._update_auto_button = Mock()
        self.gui.update_velocity_labels = Mock()
        self.gui.follow_x = [1]
        self.gui.follow_y = [2]
        self.gui.destination = (3, 4)
        self.gui.avoidance_controller.reset = Mock()
        self.gui.auto_mode = False
        self.gui._on_obstacles_changed('move')
        self.gui.avoidance_controller.reset.assert_called_once_with(
            self.gui.robot, self.gui.destination)
        self.gui.dwa_table.delete.assert_called_once_with('old')
        self.gui.dwa_rollout.set_data.assert_called_with([], [])
        self.gui.dwa_table.delete.reset_mock()
        self.gui.destination_marker = None
        self.gui.obstacle_editor = None
        self.gui.reset()
        self.gui.dwa_table.delete.assert_called_once_with('old')
        self.gui.dwa_rollout.set_data.assert_called_with([], [])
        self.assertIsInstance(self.gui.avoidance_controller, DWAController)
        self.assertFalse(self.gui.auto_mode)


if __name__ == '__main__':
    unittest.main()
