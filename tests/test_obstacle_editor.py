"""Kiểm tra trình soạn thảo vật cản Tk/Matplotlib (ObstacleEditor).

Collection thật (``src.Obstacle.Obstacle``) đang được bổ sung API
``pick/index_of/translate/remove_at`` song song, nên test dùng một fake object
cùng hợp đồng. Editor trong production vẫn gọi đúng API thật.
"""
import math
import unittest
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Rectangle
from matplotlib.path import Path

from src.Obstacle import Obstacle
from src.obstacle_editor import (
    MODE_CIRCLE, MODE_RECT, MODE_SELECT, ObstacleEditor,
)


class FakeObstacleCollection:
    """Collection tối thiểu, giữ đúng API mà ObstacleEditor yêu cầu."""

    def __init__(self):
        self.obs = []

    def add(self, patch):
        self.obs.append(patch)

    def pick(self, point):
        """Trả index vật cản trên cùng chứa point, hoặc None."""
        for index in range(len(self.obs) - 1, -1, -1):
            if self._contains(self.obs[index], point):
                return index
        return None

    def index_of(self, patch):
        try:
            return self.obs.index(patch)
        except ValueError:
            return None

    def move_to(self, patch, x, y):
        if isinstance(patch, Rectangle):
            old_x, old_y = patch.get_xy()
            self.translate(patch, x - old_x, y - old_y)
        else:
            old_x, old_y = patch.center
            self.translate(patch, x - old_x, y - old_y)

    def translate(self, patch, dx, dy):
        if isinstance(patch, Rectangle):
            x, y = patch.get_xy()
            patch.set_xy((x + dx, y + dy))
        else:
            x, y = patch.center
            patch.center = (x + dx, y + dy)

    def remove_at(self, index):
        return self.obs.pop(index)

    def clear(self):
        self.obs.clear()

    def label_obstacles(self, ax):
        return None

    @staticmethod
    def _contains(patch, point):
        if isinstance(patch, Rectangle):
            corners = patch.get_patch_transform().transform(
                patch.get_path().vertices[:4, :2])
            return Path(corners).contains_point(point, radius=1e-9)
        return math.hypot(
            point[0] - patch.center[0],
            point[1] - patch.center[1],
        ) <= patch.radius + 1e-12


def make_event(ax, x=None, y=None, button=1):
    """Tạo MouseEvent giả giống các trường editor sử dụng."""
    return SimpleNamespace(inaxes=ax, xdata=x, ydata=y, button=button)


class ObstacleEditorTestBase(unittest.TestCase):
    def setUp(self):
        self.fig = plt.figure(figsize=(6, 4))
        self.ax = self.fig.add_axes([0.05, 0.05, 0.70, 0.90])
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_aspect('equal')

        self.obstacles = FakeObstacleCollection()
        self.changes = []
        self.statuses = []
        self.editor = ObstacleEditor(
            self.ax,
            self.obstacles,
            on_change=self.changes.append,
            on_status=self.statuses.append,
        )
        self.addCleanup(plt.close, self.fig)

    # ------------------------------------------------------------------
    # Tiện ích test
    # ------------------------------------------------------------------
    def add_obstacle(self, patch):
        """Mô phỏng GUI: thêm vào collection và hiển thị trên axes."""
        self.obstacles.add(patch)
        self.ax.add_patch(patch)

    def event(self, x, y, button=1):
        return make_event(self.ax, x, y, button)

    def outside_event(self, button=1):
        return make_event(None, None, None, button)

    def draw(self, mode, start, end):
        self.editor.set_mode(mode)
        self.assertTrue(self.editor.on_press(self.event(*start)))
        self.assertTrue(self.editor.on_motion(self.event(*end)))
        self.assertTrue(self.editor.on_release(self.event(*end)))

    def click(self, x, y):
        """Click chọn (press + release cùng điểm)."""
        self.editor.set_mode(ObstacleEditor.MODE_SELECT)
        consumed = self.editor.on_press(self.event(x, y))
        self.editor.on_release(self.event(x, y))
        return consumed

    def assert_no_preview(self):
        self.assertIsNone(self.editor._preview)
        for patch in self.ax.patches:
            self.assertNotEqual(
                patch.get_edgecolor(), to_rgba('#1E90FF'))

    def assert_no_highlight(self):
        self.assertIsNone(self.editor._highlight)
        self.assertIsNone(self.editor.selected)


class DrawingTests(ObstacleEditorTestBase):
    def test_draw_rect_forward(self):
        self.draw(ObstacleEditor.MODE_RECT, (0, 0), (2, 1))

        self.assertEqual(len(self.obstacles.obs), 1)
        rect = self.obstacles.obs[0]
        self.assertIsInstance(rect, Rectangle)
        self.assertAlmostEqual(rect.get_x(), 0.0)
        self.assertAlmostEqual(rect.get_y(), 0.0)
        self.assertAlmostEqual(rect.get_width(), 2.0)
        self.assertAlmostEqual(rect.get_height(), 1.0)
        self.assertEqual(self.changes, ['create'])
        self.assertIsNone(self.editor._preview)

    def test_draw_rect_backwards(self):
        self.draw(ObstacleEditor.MODE_RECT, (2, 1), (0, 0))

        rect = self.obstacles.obs[0]
        self.assertAlmostEqual(rect.get_x(), 0.0)
        self.assertAlmostEqual(rect.get_y(), 0.0)
        self.assertAlmostEqual(rect.get_width(), 2.0)
        self.assertAlmostEqual(rect.get_height(), 1.0)

    def test_draw_circle_center_and_radius(self):
        self.draw(ObstacleEditor.MODE_CIRCLE, (1, 1), (2, 1))

        circle = self.obstacles.obs[0]
        self.assertIsInstance(circle, Circle)
        self.assertAlmostEqual(circle.center[0], 1.0)
        self.assertAlmostEqual(circle.center[1], 1.0)
        self.assertAlmostEqual(circle.radius, 1.0)
        self.assertEqual(self.changes, ['create'])
        self.assertIsNone(self.editor._preview)

    def test_draw_circle_too_small_is_cancelled(self):
        self.draw(ObstacleEditor.MODE_CIRCLE, (0, 0), (0.05, 0))

        self.assertEqual(self.obstacles.obs, [])
        self.assertEqual(self.changes, [])
        self.assertIn('nhỏ', self.statuses[-1])
        self.assertIsNone(self.editor._preview)

    def test_draw_rect_too_thin_is_cancelled(self):
        self.draw(ObstacleEditor.MODE_RECT, (0, 0), (2, 0.1))

        self.assertEqual(self.obstacles.obs, [])
        self.assertEqual(self.changes, [])
        self.assertIn('nhỏ', self.statuses[-1])
        self.assertIsNone(self.editor._preview)

    def test_preview_exists_during_drag_and_is_removed(self):
        self.editor.set_mode(ObstacleEditor.MODE_RECT)
        self.editor.on_press(self.event(0, 0))
        self.editor.on_motion(self.event(2, 2))
        self.assertIsNotNone(self.editor._preview)
        self.assertEqual(self.editor._preview.get_linestyle(), '--')
        self.assertIn(self.editor._preview, self.ax.patches)

        self.editor.on_release(self.event(2, 2))
        self.assertIsNone(self.editor._preview)

    def test_release_without_motion_uses_release_point(self):
        self.editor.set_mode(ObstacleEditor.MODE_RECT)
        self.editor.on_press(self.event(0, 0))
        self.editor.on_release(self.event(2, 1))

        rect = self.obstacles.obs[0]
        self.assertAlmostEqual(rect.get_width(), 2.0)
        self.assertAlmostEqual(rect.get_height(), 1.0)

    def test_ignores_non_left_button(self):
        self.editor.set_mode(ObstacleEditor.MODE_RECT)
        self.assertFalse(self.editor.on_press(self.event(0, 0, button=3)))
        self.assertIsNone(self.editor._preview)
        self.assertFalse(self.editor.on_release(self.event(0, 0, button=3)))
        self.assertEqual(self.obstacles.obs, [])


class SelectionTests(ObstacleEditorTestBase):
    def setUp(self):
        super().setUp()
        self.bottom = Rectangle((0, 0), 2, 2, color='gray')
        self.top = Rectangle((1, 1), 2, 2, color='gray')
        self.add_obstacle(self.bottom)
        self.add_obstacle(self.top)

    def test_select_topmost_and_highlight(self):
        self.assertTrue(self.click(1.5, 1.5))
        self.assertIs(self.editor.selected, self.top)
        self.assertIsNotNone(self.editor._highlight)
        self.assertIn(self.editor._highlight, self.ax.patches)
        self.assertEqual(
            self.editor._highlight.get_edgecolor(),
            to_rgba('#FF8C00'),
        )
        self.assertEqual(self.editor._highlight.get_linestyle(), '--')
        self.assertEqual(self.changes, ['select'])

    def test_highlight_does_not_change_original_style(self):
        original_edge = self.top.get_edgecolor()
        original_line = self.top.get_linestyle()
        self.click(1.5, 1.5)

        self.assertEqual(self.top.get_edgecolor(), original_edge)
        self.assertEqual(self.top.get_linestyle(), original_line)

    def test_click_background_returns_false_and_deselects(self):
        self.click(1.5, 1.5)
        self.assertIsNotNone(self.editor.selected)

        consumed = self.editor.on_press(self.event(-5, -5))
        self.assertFalse(consumed)
        self.assert_no_highlight()
        self.assertEqual(self.changes, ['select', 'select'])

    def test_click_background_without_selection_returns_false(self):
        self.editor.set_mode(ObstacleEditor.MODE_SELECT)
        self.assertFalse(self.editor.on_press(self.event(-5, -5)))
        self.assertEqual(self.changes, [])

    def test_click_on_patch_returns_true(self):
        self.assertTrue(self.click(0.5, 0.5))
        self.assertIs(self.editor.selected, self.bottom)


class MoveTests(ObstacleEditorTestBase):
    def test_move_rect_and_notify_only_on_release(self):
        rect = Rectangle((0, 0), 2, 2, color='gray')
        self.add_obstacle(rect)
        self.editor.set_mode(ObstacleEditor.MODE_SELECT)

        self.assertTrue(self.editor.on_press(self.event(1, 1)))
        self.assertEqual(self.changes, [])
        self.assertTrue(self.editor.on_motion(self.event(2, 3)))
        self.assertEqual(self.changes, [])
        self.assertAlmostEqual(rect.get_x(), 1.0)
        self.assertAlmostEqual(rect.get_y(), 2.0)

        self.assertTrue(self.editor.on_release(self.event(2, 3)))
        self.assertEqual(self.changes, ['move'])
        self.assertAlmostEqual(rect.get_x(), 1.0)
        self.assertAlmostEqual(rect.get_y(), 2.0)
        self.assertAlmostEqual(self.editor._highlight.get_x(), 1.0)
        self.assertAlmostEqual(self.editor._highlight.get_y(), 2.0)

    def test_move_circle(self):
        circle = Circle((0, 0), 1, color='gray')
        self.add_obstacle(circle)
        self.editor.set_mode(ObstacleEditor.MODE_SELECT)

        self.editor.on_press(self.event(0, 0))
        self.editor.on_motion(self.event(3, 4))
        self.editor.on_release(self.event(3, 4))

        self.assertAlmostEqual(circle.center[0], 3.0)
        self.assertAlmostEqual(circle.center[1], 4.0)
        self.assertEqual(self.changes, ['move'])

    def test_move_rotated_rect_preserves_angle(self):
        rect = Rectangle((0, 0), 2, 1, angle=30, color='gray')
        self.add_obstacle(rect)
        center = rect.get_patch_transform().transform([[0.5, 0.5]])[0]

        self.editor.set_mode(ObstacleEditor.MODE_SELECT)
        self.editor.on_press(self.event(center[0], center[1]))
        self.editor.on_motion(self.event(center[0] + 1, center[1] + 2))
        self.editor.on_release(self.event(center[0] + 1, center[1] + 2))

        self.assertAlmostEqual(rect.get_angle(), 30.0)
        self.assertAlmostEqual(rect.get_x(), 1.0)
        self.assertAlmostEqual(rect.get_y(), 2.0)
        self.assertIsInstance(self.editor._highlight, Rectangle)
        self.assertAlmostEqual(self.editor._highlight.get_angle(), 30.0)

    def test_simple_click_notifies_select_not_move(self):
        rect = Rectangle((0, 0), 2, 2, color='gray')
        self.add_obstacle(rect)
        self.click(1, 1)

        self.assertEqual(self.changes, ['select'])
        self.assertAlmostEqual(rect.get_x(), 0.0)

    def test_reclick_selected_obstacle_updates_status_without_notify(self):
        rect = Rectangle((0, 0), 2, 2, color='gray')
        self.add_obstacle(rect)
        self.click(1, 1)
        self.statuses.clear()

        self.click(1, 1)

        self.assertEqual(self.changes, ['select'])
        self.assertIn('Đã chọn', self.statuses[-1])


class DeleteClearTests(ObstacleEditorTestBase):
    def test_delete_selected(self):
        rect = Rectangle((0, 0), 2, 2, color='gray')
        circle = Circle((5, 5), 1, color='gray')
        self.add_obstacle(rect)
        self.add_obstacle(circle)
        self.click(1, 1)

        self.assertTrue(self.editor.delete_selected())
        self.assertEqual(self.obstacles.obs, [circle])
        self.assert_no_highlight()
        self.assertEqual(self.changes, ['select', 'delete'])
        self.assertEqual(len(self.ax.patches), 1)

    def test_delete_without_selection_returns_false(self):
        self.assertFalse(self.editor.delete_selected())
        self.assertEqual(self.changes, [])

    def test_clear_all(self):
        self.add_obstacle(Rectangle((0, 0), 2, 2, color='gray'))
        self.add_obstacle(Circle((5, 5), 1, color='gray'))
        self.click(1, 1)

        self.assertTrue(self.editor.clear_all())
        self.assertEqual(self.obstacles.obs, [])
        self.assert_no_highlight()
        self.assertEqual(self.changes, ['select', 'clear'])
        self.assertEqual(len(self.ax.patches), 0)

    def test_clear_all_empty_returns_false(self):
        self.assertFalse(self.editor.clear_all())
        self.assertEqual(self.changes, [])


class CancelRefreshTests(ObstacleEditorTestBase):
    def test_cancel_move_restores_rectangle_and_rotation(self):
        rect = Rectangle((2, 3), 2, 1, angle=30)
        self.add_obstacle(rect)
        center = rect.get_patch_transform().transform([[0.5, 0.5]])[0]
        self.editor.on_press(self.event(*center))
        self.editor.on_motion(self.event(center[0] + 3, center[1] + 4))

        self.editor.cancel()

        self.assertEqual(rect.get_xy(), (2, 3))
        self.assertEqual(rect.get_angle(), 30)
        self.assertEqual(self.changes, [])
        self.assert_no_highlight()
        self.assertFalse(self.editor.on_release(self.event(*center)))

    def test_cancel_move_restores_circle(self):
        circle = Circle((2, 3), 1)
        self.add_obstacle(circle)
        self.editor.on_press(self.event(2, 3))
        self.editor.on_motion(self.event(6, 7))

        self.editor.cancel()

        self.assertEqual(circle.center, (2, 3))
        self.assertEqual(circle.radius, 1)
        self.assertEqual(self.changes, [])

    def test_mode_switch_restores_move_and_highlight(self):
        rect = Rectangle((0, 0), 2, 2)
        self.add_obstacle(rect)
        self.editor.on_press(self.event(1, 1))
        self.editor.on_motion(self.event(4, 5))

        self.editor.set_mode(ObstacleEditor.MODE_CIRCLE)

        self.assertEqual(rect.get_xy(), (0, 0))
        self.assertEqual(self.editor._highlight.get_xy(), (0, 0))
        self.assertEqual(self.changes, [])
        self.assertFalse(self.editor.on_release(self.event(4, 5)))

    def test_cancel_after_completed_move_keeps_committed_position(self):
        rect = Rectangle((0, 0), 2, 2)
        self.add_obstacle(rect)
        self.editor.on_press(self.event(1, 1))
        self.editor.on_motion(self.event(4, 5))
        self.editor.on_release(self.event(4, 5))

        self.editor.cancel()

        self.assertEqual(rect.get_xy(), (3, 4))
        self.assertEqual(self.changes, ['move'])

    def test_cancel_during_create_removes_preview(self):
        self.editor.set_mode(ObstacleEditor.MODE_RECT)
        self.editor.on_press(self.event(0, 0))
        self.editor.on_motion(self.event(2, 2))
        self.assertIsNotNone(self.editor._preview)

        self.editor.cancel()
        self.assert_no_preview()
        self.assert_no_highlight()
        self.assertEqual(self.obstacles.obs, [])
        self.assertFalse(self.editor.on_release(self.event(2, 2)))
        self.assertEqual(self.changes, [])

    def test_cancel_clears_selection(self):
        rect = Rectangle((0, 0), 2, 2, color='gray')
        self.add_obstacle(rect)
        self.click(1, 1)
        self.assertIsNotNone(self.editor.selected)

        self.editor.cancel()
        self.assert_no_highlight()
        self.assertEqual(len(self.ax.patches), 1)

    def test_mode_switch_cancels_drag(self):
        self.editor.set_mode(ObstacleEditor.MODE_RECT)
        self.editor.on_press(self.event(0, 0))
        self.editor.on_motion(self.event(2, 2))
        self.assertIsNotNone(self.editor._preview)

        self.editor.set_mode(ObstacleEditor.MODE_CIRCLE)
        self.assert_no_preview()
        self.assertIsNone(self.editor._drag_kind)
        self.assertFalse(self.editor.on_release(self.event(2, 2)))
        self.assertEqual(self.obstacles.obs, [])

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            self.editor.set_mode('không-hợp-lệ')

    def test_refresh_clears_selection_when_patch_removed_externally(self):
        rect = Rectangle((0, 0), 2, 2, color='gray')
        self.add_obstacle(rect)
        self.click(1, 1)
        self.assertIsNotNone(self.editor.selected)

        self.obstacles.clear()
        self.editor.refresh()
        self.assert_no_highlight()
        self.assertIn('không còn tồn tại', self.statuses[-1])

    def test_external_removal_during_move_aborts_cleanly(self):
        rect = Rectangle((0, 0), 2, 2)
        self.add_obstacle(rect)
        self.editor.on_press(self.event(1, 1))
        self.editor.on_motion(self.event(3, 3))

        self.obstacles.obs.remove(rect)
        rect.remove()

        self.assertTrue(self.editor.on_motion(self.event(4, 4)))
        self.assertIsNone(self.editor._drag_kind)
        self.assert_no_highlight()
        self.assertFalse(self.editor.on_release(self.event(4, 4)))

    def test_refresh_updates_highlight_after_external_move(self):
        rect = Rectangle((0, 0), 2, 2, color='gray')
        self.add_obstacle(rect)
        self.click(1, 1)

        self.obstacles.move_to(rect, 5, 6)
        self.editor.refresh()
        self.assertAlmostEqual(self.editor._highlight.get_x(), 5.0)
        self.assertAlmostEqual(self.editor._highlight.get_y(), 6.0)


class ModuleApiTests(unittest.TestCase):
    def test_mode_constants_are_exported(self):
        """GUI import trực tiếp các hằng số chế độ từ module."""
        self.assertEqual(MODE_SELECT, ObstacleEditor.MODE_SELECT)
        self.assertEqual(MODE_RECT, ObstacleEditor.MODE_RECT)
        self.assertEqual(MODE_CIRCLE, ObstacleEditor.MODE_CIRCLE)


class RealObstacleIntegrationTests(unittest.TestCase):
    """Kiểm tra nhanh với collection thật (API SA1 đã hoàn thiện)."""

    def setUp(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.addCleanup(plt.close, self.fig)
        self.obstacles = Obstacle()
        self.changes = []
        self.editor = ObstacleEditor(
            self.ax, self.obstacles, on_change=self.changes.append)

    def test_create_select_move_delete_with_real_collection(self):
        self.editor.set_mode(ObstacleEditor.MODE_RECT)
        self.editor.on_press(make_event(self.ax, 0, 0))
        self.editor.on_motion(make_event(self.ax, 2, 2))
        self.editor.on_release(make_event(self.ax, 2, 2))

        self.assertEqual(len(self.obstacles.obs), 1)
        rect = self.obstacles.obs[0]
        self.assertIn(rect, self.ax.patches)
        self.assertIs(self.editor.selected, rect)

        self.editor.set_mode(ObstacleEditor.MODE_SELECT)
        self.editor.on_press(make_event(self.ax, 1, 1))
        self.editor.on_motion(make_event(self.ax, 2, 3))
        self.editor.on_release(make_event(self.ax, 2, 3))

        self.assertEqual(self.changes, ['create', 'move'])
        self.assertAlmostEqual(rect.get_x(), 1.0)
        self.assertAlmostEqual(rect.get_y(), 2.0)

        self.assertTrue(self.editor.delete_selected())
        self.assertEqual(self.obstacles.obs, [])
        self.assertNotIn(rect, self.ax.patches)


if __name__ == '__main__':
    unittest.main()
