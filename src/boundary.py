"""Biên offset của vật cản và các phép toán theo độ dài cung (BTH2.1)."""
import math

import numpy as np
from matplotlib.patches import Circle

if __package__:
    from .Obstacle import rectangle_corners
else:
    from Obstacle import rectangle_corners

_TOL = 1e-9


def _as_points(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError('points phải có dạng (N, 2) với N >= 3')
    if not np.all(np.isfinite(points)):
        raise ValueError('points phải là số hữu hạn')
    if np.linalg.norm(points[0] - points[-1]) <= _TOL:
        points = points[:-1]
    return points


def _signed_area(points):
    following = np.roll(points, -1, axis=0)
    return 0.5 * float(np.sum(
        points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
    ))


def _as_ccw(points):
    points = _as_points(points)
    if _signed_area(points) < 0:
        points = points[::-1]
    return points


def offset_polygon(points, clearance, arc_samples=8):
    """Biên offset của đa giác lồi: cạnh dịch ra ngoài, góc bo tròn.

    Đây là biên Minkowski với hình tròn bán kính ``clearance`` nên mọi
    điểm trên biên cách vật cản gốc đúng ``clearance``.
    """
    clearance = float(clearance)
    if not math.isfinite(clearance) or clearance <= 0:
        raise ValueError('clearance phải là số dương hữu hạn')
    points = _as_ccw(points)
    count = len(points)

    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= _TOL):
        raise ValueError('đa giác có cạnh độ dài 0')
    normals = np.column_stack((edges[:, 1], -edges[:, 0])) / lengths[:, None]

    boundary = []
    for i, vertex in enumerate(points):
        normal_previous = normals[i - 1]
        normal_current = normals[i]
        angle_start = math.atan2(normal_previous[1], normal_previous[0])
        angle_end = math.atan2(normal_current[1], normal_current[0])
        sweep = (angle_end - angle_start) % (2 * math.pi)
        if sweep <= _TOL:
            boundary.append(vertex + clearance * normal_current)
            continue
        for step in range(arc_samples + 1):
            angle = angle_start + sweep * step / arc_samples
            boundary.append(
                vertex + clearance * np.array([math.cos(angle), math.sin(angle)])
            )

    boundary = np.asarray(boundary, dtype=float)
    # Loại điểm trùng nhau do các cung nối tiếp.
    keep = [0]
    for i in range(1, len(boundary)):
        if np.linalg.norm(boundary[i] - boundary[keep[-1]]) > 1e-7:
            keep.append(i)
    boundary = boundary[keep]
    if np.linalg.norm(boundary[0] - boundary[-1]) <= 1e-7:
        boundary = boundary[:-1]
    return BoundaryPath(boundary)


def offset_boundary(obstacle, clearance, circle_samples=96):
    """Sinh BoundaryPath cho một Rectangle hoặc Circle."""
    clearance = float(clearance)
    if not math.isfinite(clearance) or clearance <= 0:
        raise ValueError('clearance phải là số dương hữu hạn')
    if isinstance(obstacle, Circle):
        center = np.asarray(obstacle.center, dtype=float)
        radius = float(obstacle.radius) + clearance
        angles = np.linspace(0.0, 2 * math.pi, circle_samples, endpoint=False)
        points = center + radius * np.column_stack(
            (np.cos(angles), np.sin(angles))
        )
        return BoundaryPath(points)
    return offset_polygon(rectangle_corners(obstacle), clearance)


class BoundaryPath:
    """Đường biên kín với tham số hoá theo độ dài cung."""

    def __init__(self, points):
        points = _as_points(points)
        self.points = points
        following = np.roll(points, -1, axis=0)
        segments = following - points
        lengths = np.linalg.norm(segments, axis=1)
        if np.any(lengths <= _TOL):
            raise ValueError('biên có đoạn độ dài 0')
        self.segments = segments
        self.segment_lengths = lengths
        self.cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        self.perimeter = float(self.cumulative[-1])

    def project(self, point):
        """Trả (s, điểm gần nhất trên biên, khoảng cách)."""
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError('point phải là cặp (x, y) hữu hạn')
        best_s = 0.0
        best_point = None
        best_distance = float('inf')
        for i, start in enumerate(self.points):
            segment = self.segments[i]
            length_squared = float(segment @ segment)
            t = float(np.clip(
                (point - start) @ segment / length_squared, 0.0, 1.0))
            candidate = start + t * segment
            distance = float(np.linalg.norm(point - candidate))
            if distance < best_distance:
                best_distance = distance
                best_point = candidate
                best_s = float(self.cumulative[i] + t * self.segment_lengths[i])
        return best_s, best_point, best_distance

    def point_at(self, s):
        """Điểm trên biên tại độ dài cung s (tự động mod chu vi)."""
        s = float(s) % self.perimeter
        index = int(np.searchsorted(self.cumulative, s, side='right') - 1)
        index = min(max(index, 0), len(self.points) - 1)
        local = s - self.cumulative[index]
        if self.segment_lengths[index] <= _TOL:
            return self.points[index].copy()
        t = local / self.segment_lengths[index]
        return self.points[index] + t * self.segments[index]

    def tangent_at(self, s):
        """Vector đơn vị tiếp tuyến theo chiều tăng của s."""
        s = float(s) % self.perimeter
        index = int(np.searchsorted(self.cumulative, s, side='right') - 1)
        index = min(max(index, 0), len(self.points) - 1)
        return self.segments[index] / self.segment_lengths[index]

    def forward_distance(self, start_s, end_s):
        """Quãng đường đi tiếp (theo chiều tăng) từ start_s tới end_s."""
        return (float(end_s) - float(start_s)) % self.perimeter
