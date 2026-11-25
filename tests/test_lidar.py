"""Kiểm tra LiDAR mô phỏng: raycast, hai khung toạ độ, nhiễu và vẽ."""
import math
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from src.Obstacle import Obstacle, rectangle_corners
from src.lidar import Lidar, LidarScan, LocalObstacle, RayHit


def _beam_index(lidar, angle):
    """Chỉ số tia gần nhất với góc tương đối cho trước."""
    return int(np.argmin(np.abs(lidar.angles - angle)))


class LidarRaycastTests(unittest.TestCase):
    def setUp(self):
        self.obstacles = Obstacle()

    def test_default_has_72_beams(self):
        lidar = Lidar(self.obstacles)
        self.assertEqual(lidar.num_beams, 72)
        self.assertEqual(len(lidar.angles), 72)

    def test_accepts_iterable_or_single_patch(self):
        patches = [Rectangle((2, -1), 2, 2), Circle((5, 3), 1.0)]
        lidar = Lidar(patches, max_range=20)
        self.assertEqual(len(lidar.obstacles.obs), 2)
        scan = lidar.scan(0, 0, 0)
        self.assertTrue(scan.hit.any())

        single = Lidar(Rectangle((2, -1), 2, 2), max_range=20)
        self.assertEqual(len(single.obstacles.obs), 1)
        self.assertTrue(single.scan(0, 0, 0).hit.any())

    def test_empty_map_has_only_misses(self):
        lidar = Lidar(self.obstacles, max_range=10)
        scan = lidar.scan(0, 0, 0)
        self.assertFalse(scan.hit.any())
        np.testing.assert_allclose(scan.ranges, 10.0)
        self.assertTrue((scan.obstacle_ids == -1).all())

    def test_circle_hit_in_world_and_local_frames(self):
        self.obstacles.add(Circle((5, 0), 1.0))
        lidar = Lidar(self.obstacles, max_range=20)
        scan = lidar.scan(0, 0, 0)
        index = _beam_index(lidar, 0.0)

        self.assertAlmostEqual(scan.angles[index], 0.0)
        self.assertTrue(scan.hit[index])
        self.assertEqual(scan.obstacle_ids[index], 0)
        self.assertAlmostEqual(scan.ranges[index], 4.0)
        np.testing.assert_allclose(
            scan.world_points[index], [4.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(
            scan.local_points[index], [4.0, 0.0], atol=1e-9)

    def test_rectangle_hit(self):
        self.obstacles.add(Rectangle((2, -1), 2, 2))
        lidar = Lidar(self.obstacles, max_range=20)
        scan = lidar.scan(0, 0, 0)
        index = _beam_index(lidar, 0.0)
        self.assertTrue(scan.hit[index])
        self.assertAlmostEqual(scan.ranges[index], 2.0)

    def test_rotated_rectangle_hit(self):
        # Hình vuông cạnh 2, tâm (5, 0), quay 45 độ: đỉnh trái nằm trên
        # tia +x tại x = 5 - sqrt(2).
        self.obstacles.add(Rectangle(
            (4, -1), 2, 2, angle=45, rotation_point='center'))
        lidar = Lidar(self.obstacles, max_range=20)
        scan = lidar.scan(0, 0, 0)
        index = _beam_index(lidar, 0.0)
        expected = 5.0 - math.sqrt(2)
        self.assertTrue(scan.hit[index])
        self.assertAlmostEqual(scan.ranges[index], expected, places=6)

    def test_miss_returns_max_range(self):
        self.obstacles.add(Rectangle((2, 5), 1, 1))
        lidar = Lidar(self.obstacles, max_range=10)
        scan = lidar.scan(0, 0, 0)
        index = _beam_index(lidar, 0.0)
        self.assertFalse(scan.hit[index])
        self.assertEqual(scan.obstacle_ids[index], -1)
        self.assertAlmostEqual(scan.ranges[index], 10.0)
        np.testing.assert_allclose(
            scan.world_points[index], [10.0, 0.0], atol=1e-9)

    def test_cast_returns_ray_hit(self):
        self.obstacles.add(Circle((5, 0), 1.0))
        lidar = Lidar(self.obstacles, max_range=20)
        hit = lidar.cast(0, 0, 0, 0.0)
        self.assertIsInstance(hit, RayHit)
        self.assertTrue(hit.hit)
        self.assertAlmostEqual(hit.distance, 4.0)
        np.testing.assert_allclose(hit.world_point, [4.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(hit.local_point, [4.0, 0.0], atol=1e-9)

    def test_origin_inside_obstacle_returns_zero(self):
        self.obstacles.add(Circle((0, 0), 1.0))
        lidar = Lidar(self.obstacles, max_range=10)
        scan = lidar.scan(0, 0, 0)
        self.assertTrue(scan.hit.all())
        np.testing.assert_allclose(scan.ranges, 0.0)
        np.testing.assert_allclose(scan.world_points, 0.0)
        np.testing.assert_allclose(scan.local_points, 0.0)
        self.assertTrue((scan.obstacle_ids == 0).all())


class LidarFrameTests(unittest.TestCase):
    def setUp(self):
        self.obstacles = Obstacle()

    def test_frame_roundtrip(self):
        lidar = Lidar(self.obstacles)
        rng = np.random.default_rng(0)
        points = rng.normal(size=(20, 2))
        pose = (1.5, -2.0, 0.7)
        world = lidar.local_to_world(points, pose)
        back = lidar.world_to_local(world, pose)
        np.testing.assert_allclose(back, points, atol=1e-12)

    def test_frame_single_point_shape(self):
        lidar = Lidar(self.obstacles)
        pose = (1.0, 2.0, -0.3)
        world = lidar.local_to_world(np.array([1.0, 2.0]), pose)
        self.assertEqual(world.shape, (2,))
        back = lidar.world_to_local(world, pose)
        self.assertEqual(back.shape, (2,))
        np.testing.assert_allclose(back, [1.0, 2.0], atol=1e-12)

    def test_local_points_follow_robot_heading(self):
        self.obstacles.add(Circle((2, 6), 0.5))
        lidar = Lidar(self.obstacles, max_range=20)
        scan = lidar.scan(2, 3, math.pi / 2)
        index = _beam_index(lidar, 0.0)
        self.assertTrue(scan.hit[index])
        self.assertAlmostEqual(scan.ranges[index], 2.5)
        np.testing.assert_allclose(
            scan.world_points[index], [2.0, 5.5], atol=1e-9)
        np.testing.assert_allclose(
            scan.local_points[index], [2.5, 0.0], atol=1e-9)

    def test_rotation_invariance_of_local_frame(self):
        phi = 0.7
        self.obstacles.add(Circle((5, 0), 1.0))
        lidar = Lidar(self.obstacles, max_range=20)
        base = lidar.scan(0, 0, 0)

        rotated = Obstacle()
        rotated.add(Circle((5 * math.cos(phi), 5 * math.sin(phi)), 1.0))
        lidar_rotated = Lidar(rotated, max_range=20)
        turned = lidar_rotated.scan(0, 0, phi)

        np.testing.assert_allclose(
            turned.local_points, base.local_points, atol=1e-9)
        np.testing.assert_allclose(turned.ranges, base.ranges, atol=1e-9)
        np.testing.assert_array_equal(turned.hit, base.hit)

    def test_mount_offset(self):
        self.obstacles.add(Circle((5, 0), 1.0))
        lidar = Lidar(
            self.obstacles, max_range=20, mount=(1.0, 0.0))
        scan = lidar.scan(0, 0, 0)
        index = _beam_index(lidar, 0.0)
        self.assertAlmostEqual(scan.ranges[index], 3.0)
        np.testing.assert_allclose(
            scan.world_points[index], [4.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(
            scan.local_points[index], [4.0, 0.0], atol=1e-9)

    def test_local_obstacles(self):
        rectangle = Rectangle((2, -1), 2, 2)
        circle = Circle((5, 3), 1.5)
        self.obstacles.add(rectangle)
        self.obstacles.add(circle)
        lidar = Lidar(self.obstacles)
        pose = (0.5, -1.0, 0.4)
        local = lidar.local_obstacles(pose)

        self.assertEqual(len(local), 2)
        self.assertIsInstance(local[0], LocalObstacle)
        self.assertEqual(local[0].kind, 'rectangle')
        np.testing.assert_allclose(
            local[0].corners,
            Lidar.world_to_local(rectangle_corners(rectangle), pose),
            atol=1e-12)
        self.assertEqual(local[1].kind, 'circle')
        np.testing.assert_allclose(
            local[1].center,
            Lidar.world_to_local(np.array([5.0, 3.0]), pose),
            atol=1e-12)
        self.assertAlmostEqual(local[1].radius, 1.5)

    def test_scan_does_not_mutate_obstacles(self):
        rectangle = Rectangle((2, -1), 2, 2, angle=15)
        circle = Circle((5, 3), 1.5)
        self.obstacles.add(rectangle)
        self.obstacles.add(circle)
        lidar = Lidar(self.obstacles, max_range=20)
        before_rect = rectangle_corners(rectangle).copy()
        before_center = np.asarray(circle.center, dtype=float).copy()
        lidar.scan(0, 0, 0)
        lidar.scan(1, 2, 0.5)
        np.testing.assert_allclose(
            rectangle_corners(rectangle), before_rect, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(circle.center, dtype=float), before_center,
            atol=1e-12)


class LidarAngleAndNoiseTests(unittest.TestCase):
    def setUp(self):
        self.obstacles = Obstacle()

    def test_full_circle_angles(self):
        lidar = Lidar(self.obstacles, num_beams=72)
        scan = lidar.scan(1, 2, 0.3)
        step = 2 * math.pi / 72
        self.assertAlmostEqual(scan.angles[0], -math.pi)
        self.assertAlmostEqual(scan.angles[-1], math.pi - step)
        np.testing.assert_allclose(np.diff(scan.angles), step, atol=1e-12)
        np.testing.assert_allclose(
            scan.world_angles, 0.3 + scan.angles, atol=1e-12)

    def test_partial_field_of_view(self):
        lidar = Lidar(
            self.obstacles, num_beams=5,
            angle_min=-0.5, angle_max=0.5)
        np.testing.assert_allclose(
            lidar.angles, [-0.5, -0.3, -0.1, 0.1, 0.3], atol=1e-12)

    def test_zero_noise_is_exact(self):
        self.obstacles.add(Rectangle((2, -1), 2, 2))
        lidar = Lidar(
            self.obstacles, max_range=20, noise_std=0.0)
        scan = lidar.scan(0, 0, 0)
        index = _beam_index(lidar, 0.0)
        self.assertAlmostEqual(scan.ranges[index], 2.0, places=12)

    def test_noise_is_reproducible_with_seed(self):
        self.obstacles.add(Rectangle((2, -1), 2, 2))
        first = Lidar(
            self.obstacles, max_range=20, noise_std=0.1, seed=7)
        second = Lidar(
            self.obstacles, max_range=20, noise_std=0.1, seed=7)
        scan_first = first.scan(0, 0, 0)
        scan_second = second.scan(0, 0, 0)
        np.testing.assert_allclose(scan_first.ranges, scan_second.ranges)
        np.testing.assert_allclose(
            scan_first.world_points, scan_second.world_points)

    def test_noise_only_on_hits_and_within_range(self):
        self.obstacles.add(Rectangle((2, -1), 2, 2))
        lidar = Lidar(
            self.obstacles, max_range=20, noise_std=0.2, seed=3)
        scan = lidar.scan(0, 0, 0)
        self.assertTrue(np.all(scan.ranges >= 0.0))
        self.assertTrue(np.all(scan.ranges <= 20.0))
        np.testing.assert_allclose(scan.ranges[~scan.hit], 20.0)
        self.assertFalse(np.allclose(
            scan.ranges[scan.hit], scan.ranges[scan.hit][0]))


class LidarPlotTests(unittest.TestCase):
    def setUp(self):
        self.obstacles = Obstacle()
        self.obstacles.add(Rectangle((2, -1), 2, 2))
        self.obstacles.add(Circle((5, 3), 1.0))
        self.lidar = Lidar(self.obstacles, max_range=12)

    def test_plot_frames_creates_two_panels(self):
        scan = self.lidar.scan(0, 0, 0)
        fig, (ax_global, ax_local) = self.lidar.plot_frames(scan)
        self.assertEqual(len(fig.axes), 2)
        self.assertGreater(len(ax_global.patches), 0)
        self.assertGreater(len(ax_local.patches), 0)
        self.assertGreater(len(ax_global.collections), 0)
        self.assertGreater(len(ax_local.collections), 0)
        self.assertIn('Global', ax_global.get_title())
        self.assertIn('local', ax_local.get_title().lower())
        plt.close(fig)

    def test_plot_frames_accepts_existing_axes(self):
        scan = self.lidar.scan(0, 0, 0)
        fig, axes = plt.subplots(1, 2)
        returned_fig, returned_axes = self.lidar.plot_frames(
            scan, ax_global=axes[0], ax_local=axes[1])
        self.assertIs(returned_fig, fig)
        self.assertIs(returned_axes[0], axes[0])
        self.assertIs(returned_axes[1], axes[1])
        plt.close(fig)

    def test_plot_frames_rejects_bad_scan(self):
        with self.assertRaises(TypeError):
            self.lidar.plot_frames('not a scan')


class LidarValidationTests(unittest.TestCase):
    def setUp(self):
        self.obstacles = Obstacle()

    def test_invalid_constructor_arguments(self):
        with self.assertRaises(ValueError):
            Lidar(self.obstacles, num_beams=0)
        with self.assertRaises(ValueError):
            Lidar(self.obstacles, num_beams=1.5)
        with self.assertRaises(ValueError):
            Lidar(self.obstacles, num_beams=True)
        with self.assertRaises(ValueError):
            Lidar(self.obstacles, max_range=0)
        with self.assertRaises(ValueError):
            Lidar(self.obstacles, angle_min=1, angle_max=1)
        with self.assertRaises(ValueError):
            Lidar(self.obstacles, angle_min=-4, angle_max=4)
        with self.assertRaises(ValueError):
            Lidar(self.obstacles, noise_std=-1)
        with self.assertRaises(ValueError):
            Lidar(self.obstacles, mount=(0,))
        with self.assertRaises(ValueError):
            Lidar(self.obstacles, seed=-1)
        with self.assertRaises(ValueError):
            Lidar(42)

    def test_invalid_scan_pose(self):
        lidar = Lidar(self.obstacles)
        with self.assertRaises(ValueError):
            lidar.scan(np.nan, 0, 0)
        with self.assertRaises(ValueError):
            lidar.scan(0, 0, None)
        with self.assertRaises(ValueError):
            lidar.cast(0, 0, 0, np.inf)

    def test_invalid_transform_arguments(self):
        with self.assertRaises(ValueError):
            Lidar.world_to_local([(0, 0, 0)], (0, 0, 0))
        with self.assertRaises(ValueError):
            Lidar.local_to_world((0, 0), (0, 0))
        with self.assertRaises(ValueError):
            Lidar.world_to_local(np.array([[0.0, np.nan]]), (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
