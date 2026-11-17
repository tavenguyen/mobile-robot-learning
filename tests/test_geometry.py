import unittest
import numpy as np
from src import get_robot_corners
from src.geometry import robot_polygons, transform_points


class GeometryTests(unittest.TestCase):
    def test_footprint(self):
        for theta in [0, np.pi/2, .7]:
            state = np.array([2., 3., theta])
            for wheels in [False, True]:
                corners = get_robot_corners(state, 1.2, .6, include_wheels=wheels)
                self.assertEqual(corners.shape, (4, 2))
                np.testing.assert_allclose(corners.mean(axis=0), state[:2])
                local = transform_points(corners-state[:2], (0, 0, -theta))
                half_width = .372 if wheels else .3
                expected = [[-.6, -half_width], [.6, -half_width],
                            [.6, half_width], [-.6, half_width]]
                np.testing.assert_allclose(local, expected, atol=1e-12)
                for polygon in robot_polygons(1.2, .6)[:5 if wheels else 1]:
                    self.assertTrue(np.all(polygon >= local.min(axis=0)-1e-12))
                    self.assertTrue(np.all(polygon <= local.max(axis=0)+1e-12))
            np.testing.assert_array_equal(state, [2, 3, theta])

    def test_invalid(self):
        for state in [(0, 0), (0, 0, np.nan)]:
            with self.assertRaises(ValueError):
                get_robot_corners(state)
        for options in [dict(length=0), dict(width=np.inf), dict(include_wheels=1)]:
            with self.assertRaises(ValueError):
                get_robot_corners((0, 0, 0), **options)
