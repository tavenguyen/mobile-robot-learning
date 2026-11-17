"""Kiểm tra API thao tác vật cản theo từng đối tượng (BTH1)."""
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from src.Obstacle import Obstacle


def make_axes(testcase):
    fig, ax = plt.subplots()
    testcase.addCleanup(plt.close, fig)
    return fig, ax


class PickTests(unittest.TestCase):
    def test_pick_empty_collection(self):
        self.assertIsNone(Obstacle().pick((0, 0)))

    def test_pick_returns_topmost_when_overlapping(self):
        obstacles = Obstacle()
        bottom = Rectangle((0, 0), 3, 3)
        top = Rectangle((1, 1), 3, 3)
        obstacles.add(bottom)
        obstacles.add(top)
        self.assertEqual(obstacles.pick((1.5, 1.5)), 1)

    def test_pick_topmost_circle_over_rectangle(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((0, 0), 4, 4))
        obstacles.add(Circle((2, 2), 1))
        self.assertEqual(obstacles.pick((2, 2)), 1)

    def test_pick_boundary_and_tolerance(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((0, 0), 2, 2))
        self.assertEqual(obstacles.pick((2, 1)), 0)  # đúng trên biên
        self.assertIsNone(obstacles.pick((2.5, 1)))
        self.assertIsNone(obstacles.pick((2.5, 1), tolerance=0.49))
        self.assertEqual(obstacles.pick((2.5, 1), tolerance=0.5), 0)
        self.assertIsNone(obstacles.pick((10, 10), tolerance=1.0))

    def test_pick_circle_tolerance(self):
        obstacles = Obstacle()
        obstacles.add(Circle((0, 0), 1))
        self.assertEqual(obstacles.pick((0, 0)), 0)
        self.assertIsNone(obstacles.pick((1.5, 0), tolerance=0.49))
        self.assertEqual(obstacles.pick((1.5, 0), tolerance=0.5), 0)

    def test_pick_rotated_rectangle(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((0, 0), 2, 0.2, angle=90))
        # Hình chữ nhật xoay nằm trong dải x thuộc [-0.2, 0], y thuộc [0, 2].
        self.assertEqual(obstacles.pick((-0.1, 1.5)), 0)
        self.assertIsNone(obstacles.pick((0.1, 1.5)))

    def test_pick_degenerate_rectangle_does_not_match_every_point(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((0, 0), 0, 0))
        self.assertIsNone(obstacles.pick((100, 100)))
        self.assertEqual(obstacles.pick((0, 0)), 0)

    def test_pick_degenerate_rectangle_measures_edges(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((0, 0), 0, 3))
        self.assertEqual(obstacles.pick((0, 1.5)), 0)
        self.assertIsNone(obstacles.pick((1, 1.5)))
        self.assertEqual(obstacles.pick((0.25, 1.5), tolerance=0.25), 0)

    def test_pick_validates_point(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((0, 0), 1, 1))
        for point in [None, (1,), (1, 2, 3), (np.nan, 0), ('a', 'b'),
                      (1, np.inf)]:
            with self.subTest(point=point):
                with self.assertRaises(ValueError):
                    obstacles.pick(point)

    def test_pick_validates_tolerance(self):
        obstacles = Obstacle()
        for tolerance in [-0.1, np.nan, np.inf, 'x', None]:
            with self.subTest(tolerance=tolerance):
                with self.assertRaises(ValueError):
                    obstacles.pick((0, 0), tolerance=tolerance)


class IndexAndMoveTests(unittest.TestCase):
    def test_index_of_uses_identity(self):
        obstacles = Obstacle()
        rect = Rectangle((0, 0), 1, 1)
        circle = Circle((5, 5), 1)
        obstacles.add(rect)
        obstacles.add(circle)
        self.assertEqual(obstacles.index_of(rect), 0)
        self.assertEqual(obstacles.index_of(circle), 1)
        self.assertIsNone(obstacles.index_of(Rectangle((0, 0), 1, 1)))
        self.assertIsNone(obstacles.index_of(None))

    def test_move_to_rectangle_keeps_angle(self):
        obstacles = Obstacle()
        rect = Rectangle((1, 2), 3, 1, angle=30)
        obstacles.add(rect)
        obstacles.move_to(rect, -2, 4)
        np.testing.assert_allclose(rect.get_xy(), (-2, 4))
        self.assertAlmostEqual(rect.angle, 30)
        self.assertAlmostEqual(rect.get_width(), 3)
        self.assertAlmostEqual(rect.get_height(), 1)

    def test_move_to_circle_changes_center(self):
        obstacles = Obstacle()
        circle = Circle((1, 2), 1.5)
        obstacles.add(circle)
        obstacles.move_to(circle, -3, 7)
        np.testing.assert_allclose(circle.center, (-3, 7))
        self.assertAlmostEqual(circle.radius, 1.5)

    def test_translate_rectangle_and_circle(self):
        obstacles = Obstacle()
        rect = Rectangle((1, 2), 2, 1, angle=45)
        circle = Circle((0, 0), 1)
        obstacles.add(rect)
        obstacles.add(circle)

        obstacles.translate(rect, -0.5, 3)
        np.testing.assert_allclose(rect.get_xy(), (0.5, 5))
        self.assertAlmostEqual(rect.angle, 45)

        obstacles.translate(circle, 4, -2)
        np.testing.assert_allclose(circle.center, (4, -2))

    def test_move_and_translate_validate_arguments(self):
        obstacles = Obstacle()
        rect = Rectangle((0, 0), 1, 1)
        obstacles.add(rect)

        for x, y in [(np.nan, 0), (0, np.inf), ('x', 0)]:
            with self.subTest(x=x, y=y):
                with self.assertRaises(ValueError):
                    obstacles.move_to(rect, x, y)

        for dx, dy in [(np.nan, 0), (0, np.inf), (None, 0)]:
            with self.subTest(dx=dx, dy=dy):
                with self.assertRaises(ValueError):
                    obstacles.translate(rect, dx, dy)

        with self.assertRaises(ValueError):
            obstacles.move_to(Rectangle((0, 0), 1, 1), 1, 1)
        with self.assertRaises(TypeError):
            obstacles.move_to(object(), 1, 1)
        with self.assertRaises(ValueError):
            obstacles.translate(Rectangle((0, 0), 1, 1), 1, 1)
        with self.assertRaises(TypeError):
            obstacles.translate(object(), 1, 1)


class RemoveTests(unittest.TestCase):
    def test_remove_at_returns_patch_and_keeps_order(self):
        obstacles = Obstacle()
        first = Rectangle((0, 0), 1, 1)
        second = Circle((2, 2), 1)
        third = Rectangle((4, 4), 1, 1)
        for patch in (first, second, third):
            obstacles.add(patch)

        removed = obstacles.remove_at(1)
        self.assertIs(removed, second)
        self.assertEqual(obstacles.obs, [first, third])
        self.assertIs(obstacles.remove_at(1), third)
        self.assertEqual(obstacles.obs, [first])

    def test_remove_at_validates_index(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((0, 0), 1, 1))
        obstacles.add(Circle((2, 2), 1))

        for index in [-1, 2, 100]:
            with self.subTest(index=index):
                with self.assertRaises(IndexError):
                    obstacles.remove_at(index)
        for index in ['0', 1.0, None, True]:
            with self.subTest(index=index):
                with self.assertRaises(TypeError):
                    obstacles.remove_at(index)

    def test_remove_clears_labels(self):
        obstacles = Obstacle()
        first = Rectangle((0, 0), 1, 1)
        second = Circle((2, 2), 1)
        obstacles.add(first)
        obstacles.add(second)
        fig, ax = make_axes(self)
        obstacles.label_obstacles(ax)

        obstacles.remove(first)
        self.assertEqual(obstacles.obs, [second])
        self.assertEqual(list(ax.texts), [])
        self.assertEqual(obstacles._labels, [])

    def test_remove_at_clears_labels(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((0, 0), 1, 1))
        fig, ax = make_axes(self)
        obstacles.label_obstacles(ax)
        self.assertEqual(len(ax.texts), 1)

        obstacles.remove_at(0)
        self.assertEqual(list(ax.texts), [])
        self.assertEqual(obstacles._labels, [])


class LabelTests(unittest.TestCase):
    def test_label_obstacles_is_idempotent(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((-5, -5), 3, 3))
        obstacles.add(Circle((10, 10), 2))
        fig, ax = make_axes(self)

        obstacles.label_obstacles(ax)
        obstacles.label_obstacles(ax)
        self.assertEqual([text.get_text() for text in ax.texts],
                         ['Obs 0', 'Obs 1'])

    def test_labels_follow_moved_obstacles(self):
        obstacles = Obstacle()
        rect = Rectangle((1, 2), 1, 1)
        circle = Circle((3, 4), 1)
        obstacles.add(rect)
        obstacles.add(circle)
        fig, ax = make_axes(self)

        obstacles.label_obstacles(ax)
        obstacles.move_to(rect, 6, 7)
        obstacles.move_to(circle, -2, -3)
        obstacles.label_obstacles(ax)

        self.assertEqual(len(ax.texts), 2)
        np.testing.assert_allclose(ax.texts[0].get_position(), (6, 7))
        np.testing.assert_allclose(ax.texts[1].get_position(), (-2, -3))

    def test_clear_labels_only_removes_obstacle_labels(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((0, 0), 1, 1))
        fig, ax = make_axes(self)
        other = ax.text(99, 99, 'khác')

        obstacles.label_obstacles(ax)
        self.assertEqual(len(ax.texts), 2)
        obstacles.clear_labels()
        self.assertEqual(list(ax.texts), [other])
        self.assertEqual(obstacles._labels, [])

    def test_clear_removes_obstacles_and_labels(self):
        obstacles = Obstacle()
        obstacles.add(Rectangle((0, 0), 1, 1))
        obstacles.add(Circle((2, 2), 1))
        fig, ax = make_axes(self)
        obstacles.label_obstacles(ax)

        obstacles.clear()
        self.assertEqual(obstacles.obs, [])
        self.assertEqual(obstacles._labels, [])
        self.assertEqual(list(ax.texts), [])


class ResizeTests(unittest.TestCase):
    def test_rectangle_resize_keeps_center_and_rotation(self):
        for rotation_point in ('xy', 'center', (5, -2)):
            with self.subTest(rotation_point=rotation_point):
                obstacles = Obstacle()
                rect = Rectangle((2, 3), 4, 2, angle=37,
                                 rotation_point=rotation_point)
                obstacles.add(rect)
                center = rect.get_patch_transform().transform([[0.5, 0.5]])[0]
                self.assertIs(obstacles.resize(rect, length=6, width=3), rect)
                np.testing.assert_allclose(
                    rect.get_patch_transform().transform([[0.5, 0.5]])[0], center)
                self.assertEqual((rect.get_width(), rect.get_height()), (6, 3))
                self.assertEqual(rect.get_angle(), 37)
                self.assertEqual(rect.rotation_point, rotation_point)

    def test_circle_resize_keeps_center_and_accepts_minimum(self):
        obstacles = Obstacle()
        circle = Circle((2, 3), 1)
        obstacles.add(circle)
        obstacles.resize(circle, radius=0.1)
        self.assertEqual(circle.center, (2, 3))
        self.assertEqual(circle.radius, 0.1)

    def test_rectangle_partial_resize_and_minimum(self):
        obstacles = Obstacle()
        rect = Rectangle((0, 0), 2, 1)
        obstacles.add(rect)
        obstacles.resize(rect, length=0.2)
        self.assertEqual((rect.get_width(), rect.get_height()), (0.2, 1))

    def test_invalid_sizes_leave_geometry_unchanged(self):
        obstacles = Obstacle()
        rect = Rectangle((2, 3), 4, 2, angle=30)
        circle = Circle((2, 3), 1)
        obstacles.add(rect)
        obstacles.add(circle)
        for value in (-1, 0, 0.09, float('nan'), float('inf'), 'abc'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    obstacles.resize(circle, radius=value)
                with self.assertRaises(ValueError):
                    obstacles.resize(rect, length=8, width=value)
        with self.assertRaises(ValueError):
            obstacles.resize(rect, radius=2)
        with self.assertRaises(ValueError):
            obstacles.resize(circle, length=2)
        with self.assertRaises(ValueError):
            obstacles.resize(Circle((0, 0), 1), radius=2)
        self.assertEqual((rect.get_width(), rect.get_height()), (4, 2))
        self.assertEqual(rect.get_xy(), (2, 3))
        self.assertEqual(circle.radius, 1)


if __name__ == '__main__':
    unittest.main()
