"""Kiểm tra GUI bằng Agg và widget giả, không cần Tkinter/display."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from src.Obstacle import Obstacle
from src.obstacle_editor import MODE_RECT, ObstacleEditor


def gui_class():
    source = Path(__file__).resolve().parents[1] / 'src' / 'GUI.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    gui = next(node for node in tree.body
               if isinstance(node, ast.ClassDef)
               and node.name == 'RobotSimulatorGUI')
    namespace = {'plt': plt, 'Circle': Circle, 'Rectangle': Rectangle}
    exec(compile(ast.Module(body=[gui], type_ignores=[]), str(source), 'exec'),
         namespace)
    return namespace['RobotSimulatorGUI'], gui


class Variable:
    def __init__(self):
        self.value = ''

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class ObstacleGUITests(unittest.TestCase):
    def setUp(self):
        cls, self.tree = gui_class()
        self.gui = cls.__new__(cls)
        self.gui.obstacle_editor = SimpleNamespace(selected=None,
                                                   resize_selected=Mock())
        self.gui.obstacle_dimension_vars = {
            key: Variable() for key in ('radius', 'length', 'width')}
        self.gui.obstacle_dimension_entries = {
            key: Mock() for key in self.gui.obstacle_dimension_vars}
        self.gui.apply_obstacle_button = Mock()
        self.gui._dimension_selection = None
        self.gui.obstacle_status_label = Mock()

    def test_single_square_map_uses_figure(self):
        setup = next(node for node in self.tree.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == 'setup_gui')
        statements = []
        for statement in setup.body:
            if statements and isinstance(statement, ast.Assign):
                break
            statements.append(statement)
        exec(compile(ast.Module(body=statements, type_ignores=[]),
                     '<gui axes>', 'exec'), {'plt': plt, 'self': self.gui})
        self.addCleanup(plt.close, self.gui.fig)
        self.gui.fig.canvas.draw()
        self.assertEqual(len(self.gui.fig.axes), 1)
        self.assertAlmostEqual(self.gui.ax.bbox.width,
                               self.gui.ax.bbox.height)
        self.assertGreater(self.gui.ax.bbox.width, 400)

    def test_selection_populates_only_applicable_fields(self):
        self.gui.obstacle_editor.selected = Circle((1, 2), 3)
        self.gui._sync_obstacle_dimensions()
        self.assertEqual(self.gui.obstacle_dimension_vars['radius'].get(), '3.0')
        self.gui.obstacle_dimension_entries['radius'].config.assert_called_with(
            state='normal')
        self.gui.obstacle_dimension_entries['length'].config.assert_called_with(
            state='disabled')
        self.gui.obstacle_editor.selected = Rectangle((0, 0), 4, 5)
        self.gui._sync_obstacle_dimensions()
        self.assertEqual(self.gui.obstacle_dimension_vars['length'].get(), '4.0')
        self.assertEqual(self.gui.obstacle_dimension_vars['width'].get(), '5.0')
        self.assertEqual(self.gui.obstacle_dimension_vars['radius'].get(), '')
        self.gui.obstacle_editor.selected = None
        self.gui._sync_obstacle_dimensions()
        self.gui.apply_obstacle_button.config.assert_called_with(state='disabled')
        self.assertTrue(all(v.get() == '' for v in
                            self.gui.obstacle_dimension_vars.values()))

    def test_refresh_preserves_typing_and_apply_reports_invalid_input(self):
        self.gui.obstacle_editor.selected = Circle((0, 0), 1)
        self.gui._sync_obstacle_dimensions()
        self.gui.obstacle_dimension_vars['radius'].set('bad')
        self.gui._sync_obstacle_dimensions()
        self.assertEqual(self.gui.obstacle_dimension_vars['radius'].get(), 'bad')
        self.gui.obstacle_editor.resize_selected.side_effect = ValueError('Số sai')
        self.assertEqual(self.gui._apply_obstacle_dimensions(), 'break')
        self.gui.obstacle_editor.resize_selected.assert_called_once_with(radius='bad')
        self.gui.obstacle_status_label.config.assert_called_with(text='Số sai')
        self.assertEqual(self.gui.obstacle_dimension_vars['radius'].get(), 'bad')

    def test_apply_rectangle_passes_both_dimensions(self):
        self.gui.obstacle_editor.selected = Rectangle((0, 0), 2, 3)
        self.gui._sync_obstacle_dimensions()
        self.gui.obstacle_dimension_vars['length'].set('4')
        self.gui.obstacle_dimension_vars['width'].set('5')
        self.gui._apply_obstacle_dimensions()
        self.gui.obstacle_editor.resize_selected.assert_called_once_with(
            length='4', width='5')

    def test_shortcuts_do_not_run_during_text_entry(self):
        self.gui.root = Mock()
        self.gui._bind_keys()
        handlers = {call.args[0]: call.args[1]
                    for call in self.gui.root.bind.call_args_list}
        for widget_class in ('TEntry', 'Entry', 'Treeview'):
            event = SimpleNamespace(widget=SimpleNamespace(
                winfo_class=lambda: widget_class))
            for handler in handlers.values():
                handler(event)
        action = Mock()
        self.gui._run_shortcut(SimpleNamespace(widget=SimpleNamespace(
            winfo_class=lambda: 'Canvas')), action)
        action.assert_called_once_with()

    def test_resize_refreshes_labels_and_resets_active_avoidance(self):
        self.gui.obstacles = Mock()
        self.gui.ax = Mock()
        self.gui.canvas = Mock()
        self.gui.robot = Mock()
        self.gui.auto_mode = True
        self.gui.destination = (7, 8)
        self.gui.avoidance_controller = Mock()
        self.gui.follow_x = [1]
        self.gui.follow_y = [2]
        self.gui._clear_avoidance_overlays = Mock()
        self.gui._update_obstacle_controls = Mock()
        self.gui._on_obstacles_changed('resize')
        self.gui.obstacles.label_obstacles.assert_called_once_with(self.gui.ax)
        self.gui._clear_avoidance_overlays.assert_called_once_with()
        self.gui.avoidance_controller.reset.assert_called_once_with(
            self.gui.robot, self.gui.destination)
        self.assertEqual(self.gui.follow_x, [])
        self.assertEqual(self.gui.follow_y, [])


class CanvasRoutingTests(unittest.TestCase):
    """Editor phải nuốt click vật cản/palette; click nền vẫn đặt goal."""

    def setUp(self):
        cls, _ = gui_class()
        self.gui = cls.__new__(cls)
        self.fig, self.ax = plt.subplots()
        self.addCleanup(plt.close, self.fig)
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)

        self.obstacles = Obstacle()
        self.obstacles.add(Rectangle((0, 0), 2, 2, color='gray'))
        self.ax.add_patch(self.obstacles.obs[0])

        self.gui.ax = self.ax
        self.gui.obstacles = self.obstacles
        self.gui.obstacle_editor = ObstacleEditor(self.ax, self.obstacles)
        self.gui.toolbar = SimpleNamespace(mode='')
        self.gui.canvas = Mock()
        self.gui._update_obstacle_controls = Mock()
        self.gui.on_click = Mock()

    def press(self, x, y):
        return SimpleNamespace(
            inaxes=self.ax, xdata=x, ydata=y, button=1)

    def test_click_obstacle_is_consumed(self):
        self.gui.on_canvas_press(self.press(1, 1))
        self.gui.on_click.assert_not_called()
        self.assertIsNotNone(self.gui.obstacle_editor.selected)

    def test_click_empty_map_still_sets_goal(self):
        self.gui.on_canvas_press(self.press(-5, -5))
        self.gui.on_click.assert_called_once()

    def test_draw_mode_consumes_press(self):
        self.gui.obstacle_editor.set_mode(MODE_RECT)
        self.gui.on_canvas_press(self.press(-5, -5))
        self.gui.on_click.assert_not_called()
        self.assertEqual(self.gui.obstacle_editor._drag_kind, 'create_rect')

    def test_toolbar_mode_blocks_editor_and_goal(self):
        self.gui.toolbar.mode = 'pan/zoom'
        self.gui.on_canvas_press(self.press(-5, -5))
        self.gui.on_click.assert_not_called()
        self.assertIsNone(self.gui.obstacle_editor._drag_kind)


if __name__ == '__main__':
    unittest.main()
