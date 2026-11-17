"""Kiểm tra tích hợp ObstacleEditor với collection Obstacle thật."""
import unittest
import numpy as np
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from src.Obstacle import Obstacle
from src.obstacle_editor import ObstacleEditor


def event(ax, x=None, y=None, button=1):
    return SimpleNamespace(inaxes=ax, xdata=x, ydata=y, button=button)


class ObstacleEditorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fig = plt.figure(figsize=(6, 4))
        self.ax = self.fig.add_axes([0.05, 0.05, 0.70, 0.90])
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.obstacles = Obstacle()
        self.changes = []
        self.editor = ObstacleEditor(
            self.ax, self.obstacles,
            on_change=self._on_change,
        )
        self.addCleanup(plt.close, self.fig)

    def _on_change(self, reason):
        self.changes.append(reason)
        if reason in ('create', 'move', 'delete', 'clear', 'resize'):
            self.obstacles.label_obstacles(self.ax)

    def add_obstacle(self, patch):
        self.obstacles.add(patch)
        self.ax.add_patch(patch)

    def test_move_and_delete_real_obstacle_updates_collection_and_labels(self):
        rect = Rectangle((0, 0), 2, 2, color='gray')
        self.add_obstacle(rect)
        self.obstacles.label_obstacles(self.ax)

        self.editor.set_mode(ObstacleEditor.MODE_SELECT)
        self.assertTrue(self.editor.on_press(event(self.ax, 1, 1)))
        self.assertTrue(self.editor.on_motion(event(self.ax, 4, 5)))
        self.assertTrue(self.editor.on_release(event(self.ax, 4, 5)))
        self.assertAlmostEqual(rect.get_x(), 3.0)
        self.assertAlmostEqual(rect.get_y(), 4.0)
        self.assertEqual(self.changes, ['move'])
        self.assertEqual(self.ax.texts[0].get_position(), (3.0, 4.0))

        self.assertTrue(self.editor.delete_selected())
        self.assertEqual(self.obstacles.obs, [])
        self.assertNotIn(rect, self.ax.patches)
        self.assertEqual(list(self.ax.texts), [])
        self.assertEqual(self.changes, ['move', 'delete'])

    def test_select_topmost_real_obstacle(self):
        bottom = Rectangle((0, 0), 3, 3, color='gray')
        top = Circle((2, 2), 1, color='gray')
        self.add_obstacle(bottom)
        self.add_obstacle(top)

        self.editor.set_mode(ObstacleEditor.MODE_SELECT)
        self.assertTrue(self.editor.on_press(event(self.ax, 2, 2)))
        self.assertIs(self.editor.selected, top)

    def test_rotated_rectangle_move_preserves_angle(self):
        rect = Rectangle((0, 0), 2, 1, angle=30, color='gray')
        self.add_obstacle(rect)
        center = rect.get_patch_transform().transform([[0.5, 0.5]])[0]

        self.editor.set_mode(ObstacleEditor.MODE_SELECT)
        self.editor.on_press(event(self.ax, center[0], center[1]))
        self.editor.on_motion(event(self.ax, center[0] + 1, center[1] + 2))
        self.editor.on_release(event(self.ax, center[0] + 1, center[1] + 2))

        self.assertAlmostEqual(rect.get_angle(), 30.0)
        self.assertAlmostEqual(rect.get_x(), 1.0)
        self.assertAlmostEqual(rect.get_y(), 2.0)
        self.assertAlmostEqual(self.editor._highlight.get_angle(), 30.0)

    def test_clear_all_removes_patches_and_labels(self):
        self.add_obstacle(Rectangle((0, 0), 2, 2, color='gray'))
        self.add_obstacle(Circle((5, 5), 1, color='gray'))
        self.obstacles.label_obstacles(self.ax)

        self.assertTrue(self.editor.clear_all())
        self.assertEqual(self.obstacles.obs, [])
        self.assertEqual(list(self.ax.patches), [])
        self.assertEqual(list(self.ax.texts), [])
        self.assertEqual(self.changes, ['clear'])


class ResizeIntegrationTests(unittest.TestCase):
    setUp = ObstacleEditorIntegrationTests.setUp
    _on_change = ObstacleEditorIntegrationTests._on_change
    add_obstacle = ObstacleEditorIntegrationTests.add_obstacle

    def test_resize_without_selection_is_noop(self):
        self.assertFalse(self.editor.resize_selected(radius=2))
        self.assertEqual(self.changes, [])

    def test_resize_selected_circle_updates_highlight_and_callback(self):
        circle = Circle((2, 3), 1)
        self.add_obstacle(circle)
        self.editor.on_press(event(self.ax, 2, 3))
        self.editor.on_release(event(self.ax, 2, 3))
        self.changes.clear()
        self.assertTrue(self.editor.resize_selected(radius=3))
        self.assertEqual(circle.center, (2, 3))
        self.assertEqual(self.editor._highlight.radius, 3)
        self.assertEqual(self.changes, ['resize'])
        self.assertFalse(self.editor.resize_selected(radius=3))
        self.assertEqual(self.changes, ['resize'])

    def test_resize_selected_rectangle_and_invalid_input(self):
        rect = Rectangle((0, 0), 2, 1, angle=30)
        self.add_obstacle(rect)
        center = rect.get_patch_transform().transform([[0.5, 0.5]])[0]
        self.editor.on_press(event(self.ax, *center))
        self.editor.on_release(event(self.ax, *center))
        self.changes.clear()
        with self.assertRaises(ValueError):
            self.editor.resize_selected(length=4, width=0)
        self.assertEqual(self.changes, [])
        self.assertTrue(self.editor.resize_selected(length=4, width=3))
        self.assertEqual(self.changes, ['resize'])
        highlight = self.editor._highlight
        self.assertEqual((highlight.get_width(), highlight.get_height()), (4, 3))
        self.assertEqual(highlight.get_angle(), 30)
        np.testing.assert_allclose(
            highlight.get_patch_transform().transform([[0.5, 0.5]])[0], center)


if __name__ == '__main__':
    unittest.main()
