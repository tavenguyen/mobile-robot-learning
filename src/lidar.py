"""LiDAR 2D mô phỏng: khung toạ độ toàn cục và khung cục bộ của robot.

Vật cản được lưu ở khung toàn cục (world frame). Mỗi lần quét trả về cả
điểm chạm trong khung toàn cục và trong khung cục bộ có tâm robot là (0, 0),
trục +x hướng trước robot, trục +y hướng sang trái.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle

if __package__:
    from .Obstacle import Obstacle, rectangle_corners
    from .sensors import RangeSensor
else:
    from Obstacle import Obstacle, rectangle_corners
    from sensors import RangeSensor

_TOL = 1e-9


@dataclass
class RayHit:
    """Kết quả của một tia trong cả hai khung toạ độ."""

    angle: float
    world_angle: float
    distance: float
    local_point: np.ndarray
    world_point: np.ndarray
    hit: bool
    obstacle_id: int


@dataclass
class LidarScan:
    """Kết quả một lần quét; các mảng có cùng độ dài ``num_beams``."""

    pose: np.ndarray
    angles: np.ndarray
    world_angles: np.ndarray
    ranges: np.ndarray
    local_points: np.ndarray
    world_points: np.ndarray
    hit: np.ndarray
    obstacle_ids: np.ndarray


@dataclass
class LocalObstacle:
    """Hình học vật cản trong khung cục bộ của robot."""

    kind: str
    corners: np.ndarray | None = None
    center: np.ndarray | None = None
    radius: float | None = None


def _as_finite(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} phải là số hữu hạn') from exc
    if not np.isfinite(value):
        raise ValueError(f'{name} phải là số hữu hạn')
    return value


def _as_point(value, name='point'):
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} phải là cặp (x, y)') from exc
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f'{name} phải là cặp (x, y) hữu hạn')
    return point


def _as_points(points, name='points'):
    """Chuyển về mảng (N, 2) và cờ cho biết đầu vào là một điểm (2,)."""
    try:
        array = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} phải có dạng (N, 2) hoặc (2,)') from exc
    single = array.shape == (2,)
    if single:
        array = array.reshape(1, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f'{name} phải có dạng (N, 2) hoặc (2,)')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} phải là số hữu hạn')
    return array, single


def _as_pose(pose):
    try:
        pose = np.asarray(pose, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('pose phải gồm ba số hữu hạn (x, y, theta)') from exc
    if pose.shape != (3,) or not np.all(np.isfinite(pose)):
        raise ValueError('pose phải gồm ba số hữu hạn (x, y, theta)')
    return pose


def _as_pose_xyz(x, y, theta):
    return _as_pose([x, y, theta])


def _as_obstacles(obstacles):
    if isinstance(obstacles, Obstacle):
        return obstacles
    if isinstance(obstacles, (Rectangle, Circle)):
        patches = [obstacles]
    else:
        try:
            patches = list(obstacles)
        except TypeError as exc:
            raise ValueError(
                'obstacles phải là Obstacle hoặc iterable Rectangle/Circle'
            ) from exc
    collection = Obstacle()
    for patch in patches:
        collection.add(patch)
    return collection


class Lidar:
    """LiDAR 2D quét vật cản tĩnh trong mặt phẳng."""

    def __init__(self, obstacles, *, num_beams=72, max_range=10.0,
                 angle_min=-math.pi, angle_max=math.pi,
                 mount=(0.0, 0.0), mount_theta=0.0,
                 noise_std=0.0, seed=None):
        self.obstacles = _as_obstacles(obstacles)

        if isinstance(num_beams, (bool, np.bool_)) \
                or not isinstance(num_beams, (int, np.integer)):
            raise ValueError('num_beams phải là số nguyên dương')
        self.num_beams = int(num_beams)
        if self.num_beams < 1:
            raise ValueError('num_beams phải là số nguyên dương')

        self.max_range = _as_finite(max_range, 'max_range')
        if self.max_range <= 0:
            raise ValueError('max_range phải là số dương hữu hạn')

        self.angle_min = _as_finite(angle_min, 'angle_min')
        self.angle_max = _as_finite(angle_max, 'angle_max')
        span = self.angle_max - self.angle_min
        if span <= _TOL or span > 2 * math.pi + _TOL:
            raise ValueError(
                'angle_max - angle_min phải nằm trong (0, 2*pi]')

        self.mount = _as_point(mount, 'mount')
        self.mount_theta = _as_finite(mount_theta, 'mount_theta')
        self.noise_std = _as_finite(noise_std, 'noise_std')
        if self.noise_std < 0:
            raise ValueError('noise_std phải là số không âm hữu hạn')

        if seed is not None and (
                isinstance(seed, (bool, np.bool_))
                or not isinstance(seed, (int, np.integer))
                or seed < 0):
            raise ValueError('seed phải là số nguyên không âm hoặc None')
        self.seed = None if seed is None else int(seed)
        self._rng = np.random.default_rng(self.seed)

        # Tái sử dụng raycast analytic đã được kiểm thử của RangeSensor.
        self._sensor = RangeSensor(
            self.obstacles, n_rays=max(8, self.num_beams),
            max_range=self.max_range,
        )
        step = span / self.num_beams
        self.angles = self.angle_min + np.arange(self.num_beams) * step

    # ------------------------------------------------------------------
    # Chuyển đổi khung toạ độ
    # ------------------------------------------------------------------
    @staticmethod
    def local_to_world(points, pose):
        """Chuyển điểm trong khung robot sang khung toàn cục."""
        pose = _as_pose(pose)
        array, single = _as_points(points)
        x, y, theta = pose
        cosine, sine = math.cos(theta), math.sin(theta)
        local_x = array[:, 0]
        local_y = array[:, 1]
        world = np.column_stack((
            cosine * local_x - sine * local_y + x,
            sine * local_x + cosine * local_y + y,
        ))
        return world[0] if single else world

    @staticmethod
    def world_to_local(points, pose):
        """Chuyển điểm trong khung toàn cục sang khung robot (tâm 0, 0)."""
        pose = _as_pose(pose)
        array, single = _as_points(points)
        x, y, theta = pose
        cosine, sine = math.cos(theta), math.sin(theta)
        delta_x = array[:, 0] - x
        delta_y = array[:, 1] - y
        local = np.column_stack((
            cosine * delta_x + sine * delta_y,
            -sine * delta_x + cosine * delta_y,
        ))
        return local[0] if single else local

    # ------------------------------------------------------------------
    # Quét
    # ------------------------------------------------------------------
    def _sensor_origin(self, pose):
        """Vị trí cảm biến trong khung toàn cục, kể cả mount."""
        x, y, theta = pose
        cosine, sine = math.cos(theta), math.sin(theta)
        offset_x = cosine * self.mount[0] - sine * self.mount[1]
        offset_y = sine * self.mount[0] + cosine * self.mount[1]
        return np.array([x + offset_x, y + offset_y])

    def _build_hit(self, pose, origin, angle, inside, noise):
        world_angle = float(pose[2] + self.mount_theta + angle)
        if inside is not None:
            local_point = self.world_to_local(origin, pose)
            return RayHit(
                angle=float(angle),
                world_angle=world_angle,
                distance=0.0,
                local_point=local_point,
                world_point=origin.copy(),
                hit=True,
                obstacle_id=int(inside),
            )

        distance, point, hit, obstacle_id = self._sensor.cast(
            origin, world_angle)
        if hit and noise:
            distance = float(np.clip(
                distance + noise, 0.0, self.max_range))
            direction = np.array(
                [math.cos(world_angle), math.sin(world_angle)])
            point = origin + distance * direction

        return RayHit(
            angle=float(angle),
            world_angle=world_angle,
            distance=float(distance),
            local_point=self.world_to_local(point, pose),
            world_point=np.asarray(point, dtype=float),
            hit=bool(hit),
            obstacle_id=int(obstacle_id),
        )

    def cast(self, x, y, theta, angle):
        """Quét một tia tại góc tương đối ``angle``; trả :class:`RayHit`."""
        pose = _as_pose_xyz(x, y, theta)
        angle = _as_finite(angle, 'angle')
        origin = self._sensor_origin(pose)
        inside = self.obstacles.contains(origin)
        noise = 0.0
        if self.noise_std > 0:
            noise = float(self._rng.normal(0.0, self.noise_std))
        return self._build_hit(pose, origin, angle, inside, noise)

    def scan(self, x, y, theta):
        """Quét toàn bộ ``num_beams`` tia; trả :class:`LidarScan`."""
        pose = _as_pose_xyz(x, y, theta)
        origin = self._sensor_origin(pose)
        inside = self.obstacles.contains(origin)
        world_angles = pose[2] + self.mount_theta + self.angles

        ranges = np.empty(self.num_beams, dtype=float)
        world_points = np.empty((self.num_beams, 2), dtype=float)
        hit = np.zeros(self.num_beams, dtype=bool)
        obstacle_ids = np.full(self.num_beams, -1, dtype=int)
        noise = None
        if self.noise_std > 0:
            noise = self._rng.normal(
                0.0, self.noise_std, size=self.num_beams)

        for index, world_angle in enumerate(world_angles):
            if inside is not None:
                ranges[index] = 0.0
                world_points[index] = origin
                hit[index] = True
                obstacle_ids[index] = inside
                continue

            distance, point, ray_hit, obstacle_id = self._sensor.cast(
                origin, float(world_angle))
            if ray_hit and noise is not None:
                distance = float(np.clip(
                    distance + noise[index], 0.0, self.max_range))
                direction = np.array([
                    math.cos(world_angle), math.sin(world_angle)])
                point = origin + distance * direction

            ranges[index] = distance
            world_points[index] = point
            hit[index] = ray_hit
            obstacle_ids[index] = obstacle_id

        local_points = self.world_to_local(world_points, pose)
        return LidarScan(
            pose=pose.copy(),
            angles=self.angles.copy(),
            world_angles=np.asarray(world_angles, dtype=float),
            ranges=ranges,
            local_points=local_points,
            world_points=world_points,
            hit=hit,
            obstacle_ids=obstacle_ids,
        )

    def local_obstacles(self, pose):
        """Vật cản tĩnh được biểu diễn trong khung cục bộ của robot."""
        pose = _as_pose(pose)
        result = []
        for patch in self.obstacles.obs:
            if isinstance(patch, Rectangle):
                result.append(LocalObstacle(
                    kind='rectangle',
                    corners=self.world_to_local(
                        rectangle_corners(patch), pose),
                ))
            elif isinstance(patch, Circle):
                result.append(LocalObstacle(
                    kind='circle',
                    center=self.world_to_local(
                        np.asarray(patch.center, dtype=float), pose),
                    radius=float(patch.radius),
                ))
        return result

    # ------------------------------------------------------------------
    # Trực quan hoá
    # ------------------------------------------------------------------
    def plot_frames(self, scan, ax_global=None, ax_local=None):
        """Vẽ hai khung global/local; trả ``(fig, (ax_global, ax_local))``."""
        if not isinstance(scan, LidarScan):
            raise TypeError('scan phải là LidarScan')
        if (ax_global is None) != (ax_local is None):
            raise ValueError('phải truyền cả hai axes hoặc không truyền')

        import matplotlib.pyplot as plt

        if ax_global is None:
            fig, (ax_global, ax_local) = plt.subplots(
                1, 2, figsize=(12, 5.5))
            created = True
        else:
            fig = ax_global.figure
            if ax_local.figure is not fig:
                raise ValueError('ax_global và ax_local phải cùng một figure')
            created = False

        # --- Khung toàn cục -------------------------------------------------
        for patch in self.obstacles.obs:
            if isinstance(patch, Rectangle):
                ax_global.add_patch(Polygon(
                    rectangle_corners(patch), closed=True, fill=False,
                    edgecolor='0.35', linewidth=1.4, zorder=2))
            elif isinstance(patch, Circle):
                ax_global.add_patch(Circle(
                    patch.center, patch.radius, fill=False,
                    edgecolor='0.35', linewidth=1.4, zorder=2))

        origin = self._sensor_origin(scan.pose)
        segments = np.stack((
            np.broadcast_to(origin, scan.world_points.shape),
            scan.world_points,
        ), axis=1)
        beam_colors = np.where(scan.hit, '#d62728', '0.82')
        ax_global.add_collection(LineCollection(
            segments, colors=beam_colors, linewidths=0.6,
            alpha=0.85, zorder=1))

        if np.any(scan.hit):
            ax_global.scatter(
                scan.world_points[scan.hit, 0],
                scan.world_points[scan.hit, 1],
                s=14, color='#d62728', zorder=3)

        arrow = max(0.5, 0.08 * self.max_range)
        x, y, theta = scan.pose
        ax_global.plot(x, y, marker='o', color='#1f77b4',
                       markersize=5, zorder=4)
        ax_global.annotate(
            '', xy=(x + arrow * math.cos(theta),
                    y + arrow * math.sin(theta)),
            xytext=(x, y),
            arrowprops=dict(arrowstyle='->', color='#1f77b4',
                            linewidth=1.6),
            zorder=5)

        # --- Khung cục bộ ---------------------------------------------------
        for item in self.local_obstacles(scan.pose):
            if item.kind == 'rectangle':
                ax_local.add_patch(Polygon(
                    item.corners, closed=True, fill=False,
                    edgecolor='0.35', linewidth=1.4, zorder=2))
            else:
                ax_local.add_patch(Circle(
                    item.center, item.radius, fill=False,
                    edgecolor='0.35', linewidth=1.4, zorder=2))

        local_origin = self.mount
        local_segments = np.stack((
            np.broadcast_to(local_origin, scan.local_points.shape),
            scan.local_points,
        ), axis=1)
        ax_local.add_collection(LineCollection(
            local_segments, colors=beam_colors, linewidths=0.6,
            alpha=0.85, zorder=1))

        if np.any(scan.hit):
            ax_local.scatter(
                scan.local_points[scan.hit, 0],
                scan.local_points[scan.hit, 1],
                s=14, color='#d62728', zorder=3)

        ax_local.axhline(0.0, color='0.88', linewidth=0.8, zorder=0)
        ax_local.axvline(0.0, color='0.88', linewidth=0.8, zorder=0)
        ax_local.plot(0.0, 0.0, marker='o', color='#1f77b4',
                      markersize=5, zorder=4)
        ax_local.annotate(
            '', xy=(arrow, 0.0), xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle='->', color='#1f77b4',
                            linewidth=1.6),
            zorder=5)

        # --- Trang trí ------------------------------------------------------
        handles = [
            Line2D([0], [0], color='0.35', linewidth=1.4,
                   label='Obstacle'),
            Line2D([0], [0], color='#d62728', linewidth=1.0,
                   label='Beam hit'),
            Line2D([0], [0], color='0.82', linewidth=1.0,
                   label='Beam miss'),
            Line2D([0], [0], marker='o', linestyle='',
                   markerfacecolor='#d62728', markeredgecolor='#d62728',
                   label='Hit point'),
            Line2D([0], [0], marker='o', linestyle='',
                   markerfacecolor='#1f77b4', markeredgecolor='#1f77b4',
                   label='Robot'),
        ]
        for axis, title in (
                (ax_global, 'Global frame (world coordinates)'),
                (ax_local, 'Robot-local frame (robot at (0, 0), +x forward)')):
            axis.set_title(title)
            axis.set_xlabel('x (m)')
            axis.set_ylabel('y (m)')
            axis.set_aspect('equal', adjustable='box')
            axis.grid(True, alpha=0.3)
            axis.margins(0.08)
            axis.autoscale_view()
            axis.legend(handles=handles, loc='upper right', fontsize=8,
                        framealpha=0.9)

        if created:
            fig.tight_layout()
        return fig, (ax_global, ax_local)
