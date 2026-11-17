"""Cảm biến mô phỏng cho BTH2.1: contact sensor và range sensor 360°."""
import math
from dataclasses import dataclass

import numpy as np
from matplotlib.patches import Circle

if __package__:
    from .Obstacle import rectangle_corners
else:
    from Obstacle import rectangle_corners

_TOL = 1e-9


@dataclass
class Scan:
    """Kết quả quét: góc tuyệt đối, khoảng cách, điểm chạm, vật cản trúng."""

    angles: np.ndarray
    ranges: np.ndarray
    points: np.ndarray
    hit: np.ndarray
    obstacle_ids: np.ndarray


@dataclass
class TangentEndpoint:
    """Đầu mút của một khoảng trống nhìn thấy bởi range sensor."""

    angle: float
    point: np.ndarray
    safe_point: np.ndarray
    distance: float
    h: float
    side: str


def _ray_segment(origin, direction, start, end):
    """Giao tia origin + t*direction với đoạn start-end; trả t >= 0 hoặc None."""
    segment = end - start
    denominator = float(
        direction[0] * segment[1] - direction[1] * segment[0])
    if abs(denominator) <= _TOL:
        return None
    difference = start - origin
    t = float((difference[0] * segment[1]
               - difference[1] * segment[0]) / denominator)
    u = float((difference[0] * direction[1]
               - difference[1] * direction[0]) / denominator)
    if t >= -_TOL and -1e-7 <= u <= 1 + 1e-7:
        return max(t, 0.0)
    return None


def _ray_circle(origin, direction, center, radius):
    """Giao tia với đường tròn; trả t >= 0 nhỏ nhất hoặc None."""
    offset = origin - center
    b = 2.0 * float(offset @ direction)
    c = float(offset @ offset) - radius * radius
    discriminant = b * b - 4.0 * c
    if discriminant < 0.0:
        return None
    root = math.sqrt(max(discriminant, 0.0))
    candidates = [(-b - root) / 2.0, (-b + root) / 2.0]
    positive = [value for value in candidates if value >= -_TOL]
    if not positive:
        return None
    return max(min(positive), 0.0)


def _group_intervals(mask):
    """Nhóm các chỉ số True liên tiếp trên vòng tròn thành (start, end)."""
    mask = np.asarray(mask, dtype=bool)
    count = len(mask)
    if not mask.any():
        return []
    if mask.all():
        return [(0, count - 1)]

    starts = [
        index for index in range(count)
        if mask[index] and not mask[index - 1]
    ]
    intervals = []
    for start in starts:
        end = start
        while mask[(end + 1) % count] and (end + 1) % count != start:
            end = (end + 1) % count
        intervals.append((start, end))
    return intervals


def _interval_contains(interval, index, count):
    start, end = interval
    if start <= end:
        return start <= index <= end
    return index >= start or index <= end


def _interval_mid_angle(scan, interval):
    start, end = interval
    count = len(scan.angles)
    length = (end - start) % count + 1
    indices = [(start + k) % count for k in range(length)]
    sine = float(np.mean(np.sin(scan.angles[indices])))
    cosine = float(np.mean(np.cos(scan.angles[indices])))
    return math.atan2(sine, cosine)


def _angle_distance(first, second):
    return abs((first - second + math.pi) % (2 * math.pi) - math.pi)


class RangeSensor:
    """Range sensor 360° với n tia, tầm xa max_range (mét)."""

    def __init__(self, obstacles, n_rays=72, max_range=15.0):
        self.obstacles = obstacles
        self.n_rays = int(n_rays)
        if self.n_rays < 8:
            raise ValueError('n_rays phải >= 8')
        self.max_range = float(max_range)
        if not math.isfinite(self.max_range) or self.max_range <= 0:
            raise ValueError('max_range phải là số dương hữu hạn')
        self.step = 2 * math.pi / self.n_rays

    def cast(self, origin, angle):
        """Trả (distance, point, hit, obstacle_index) cho một tia."""
        origin = np.asarray(origin, dtype=float)
        if origin.shape != (2,) or not np.all(np.isfinite(origin)):
            raise ValueError('origin phải là cặp (x, y) hữu hạn')
        direction = np.array([math.cos(angle), math.sin(angle)])

        best_t = float('inf')
        best_index = -1
        for index, obstacle in enumerate(self.obstacles.obs):
            if isinstance(obstacle, Circle):
                center = np.asarray(obstacle.center, dtype=float)
                candidate = _ray_circle(
                    origin, direction, center, float(obstacle.radius))
                if candidate is not None and candidate < best_t:
                    best_t = candidate
                    best_index = index
            else:
                polygon = rectangle_corners(obstacle)
                for i, start in enumerate(polygon):
                    end = polygon[(i + 1) % len(polygon)]
                    candidate = _ray_segment(origin, direction, start, end)
                    if candidate is not None and candidate < best_t:
                        best_t = candidate
                        best_index = index

        if not math.isfinite(best_t) or best_t > self.max_range:
            point = origin + self.max_range * direction
            return self.max_range, point, False, -1
        point = origin + best_t * direction
        return float(best_t), point, True, best_index

    def scan(self, x, y, theta):
        """Quét 360° quanh pose (x, y, theta); trả Scan."""
        origin = np.asarray([x, y], dtype=float)
        if origin.shape != (2,) or not np.all(np.isfinite(origin)) \
                or not math.isfinite(float(theta)):
            raise ValueError('pose phải hữu hạn')
        angles = theta + np.arange(self.n_rays) * self.step
        ranges = np.empty(self.n_rays, dtype=float)
        points = np.empty((self.n_rays, 2), dtype=float)
        hit = np.zeros(self.n_rays, dtype=bool)
        obstacle_ids = np.full(self.n_rays, -1, dtype=int)
        for index, angle in enumerate(angles):
            distance, point, ray_hit, obstacle_index = self.cast(origin, angle)
            ranges[index] = distance
            points[index] = point
            hit[index] = ray_hit
            obstacle_ids[index] = obstacle_index
        return Scan(angles, ranges, points, hit, obstacle_ids)

    def clear_path(self, start, target, clearance=0.0):
        """Đoạn thẳng start-target có cách mọi vật cản >= clearance không."""
        blocker, _ = self.first_blocker(start, target, clearance)
        return blocker is None

    def first_blocker(self, start, target, clearance=0.0):
        """Trả (obstacle_index, điểm vi phạm) hoặc (None, None)."""
        try:
            start = np.asarray(start, dtype=float)
            target = np.asarray(target, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError('start/target phải là cặp (x, y)') from exc
        if start.shape != (2,) or target.shape != (2,):
            raise ValueError('start/target phải là cặp (x, y)')
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(target)):
            raise ValueError('start/target phải hữu hạn')

        length = float(np.linalg.norm(target - start))
        if length <= _TOL:
            index, _, distance, _ = self.obstacles.nearest(start)
            if index is not None and distance < clearance - 1e-7:
                return index, start.copy()
            return None, None

        direction = (target - start) / length
        step = 0.1 if clearance <= 0 else max(0.1, min(0.25, clearance / 4))
        count = max(2, int(math.ceil(length / step)) + 1)
        for k in range(1, count + 1):
            distance_along = length * k / count
            sample = start + direction * distance_along
            index, _, distance, _ = self.obstacles.nearest(sample)
            if index is not None and distance < clearance - 1e-7:
                return index, sample
        return None, None

    def intervals(self, scan):
        """Các khoảng góc bị chặn liên tiếp của scan."""
        return _group_intervals(scan.hit)

    def interval_for_obstacle(self, scan, obstacle_index, goal_angle):
        """Khoảng bị chặn của vật cản gần hướng goal nhất."""
        mask = (scan.obstacle_ids == obstacle_index) & scan.hit
        intervals = _group_intervals(mask)
        if not intervals:
            return None
        return min(
            intervals,
            key=lambda interval: _angle_distance(
                _interval_mid_angle(scan, interval), goal_angle),
        )

    def interval_contains(self, scan, interval, angle):
        """Góc tuyệt đối có nằm trong khoảng bị chặn không."""
        relative = (angle - scan.angles[0]) % (2 * math.pi)
        index = int(round(relative / self.step)) % self.n_rays
        return _interval_contains(interval, index, self.n_rays)

    def endpoints_for_interval(self, scan, interval, origin, goal, clearance):
        """Hai đầu mút của khoảng trống, kèm điểm an toàn và h(p)."""
        count = self.n_rays
        start, end = interval
        free_before = (start - 1) % count
        free_after = (end + 1) % count

        endpoints = []
        for side, free_index, blocked_index in (
                ('left', free_before, start),
                ('right', free_after, end)):
            angle = self._refine_tangent(
                scan, origin, free_index, blocked_index)
            distance, point, hit, _ = self.cast(origin, angle)
            if not hit:
                continue
            safe_distance = max(distance - clearance, 0.2)
            direction = np.array([math.cos(angle), math.sin(angle)])
            safe_point = origin + safe_distance * direction
            distance_robot = float(np.linalg.norm(point - origin))
            h = distance_robot + float(np.linalg.norm(goal - point))
            endpoints.append(TangentEndpoint(
                angle=float(angle),
                point=point,
                safe_point=safe_point,
                distance=distance_robot,
                h=h,
                side=side,
            ))
        return endpoints

    def _refine_tangent(self, scan, origin, free_index, blocked_index):
        """Bisection tìm góc chuyển giữa tia tự do và tia bị chặn."""
        angle_free = float(scan.angles[free_index])
        angle_blocked = float(scan.angles[blocked_index])
        while angle_blocked - angle_free > math.pi:
            angle_blocked -= 2 * math.pi
        while angle_free - angle_blocked > math.pi:
            angle_blocked += 2 * math.pi

        for _ in range(12):
            middle = 0.5 * (angle_free + angle_blocked)
            _, _, hit, _ = self.cast(origin, middle)
            if hit:
                angle_blocked = middle
            else:
                angle_free = middle
        angle = 0.5 * (angle_free + angle_blocked)
        _, _, hit, _ = self.cast(origin, angle)
        return angle if hit else angle_blocked
