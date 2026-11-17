"""Dynamic Window Approach: lấy mẫu vận tốc, dự đoán và kiểm tra phanh."""
import math
from dataclasses import dataclass

import numpy as np
from matplotlib.patches import Circle, Rectangle

if __package__:
    from .avoidance import Control, _validate_goal
    from .Obstacle import rectangle_corners
else:
    from avoidance import Control, _validate_goal
    from Obstacle import rectangle_corners


@dataclass(frozen=True)
class DWAConfig:
    alpha: float = 0.6
    beta: float = 0.3
    gamma: float = 0.1
    v_min: float = 0.0
    v_max: float = 2.0
    w_min: float = -2.0
    w_max: float = 2.0
    a_max: float = 1.0
    angular_a_max: float = 2.0
    v_resolution: float = 0.05
    w_resolution: float = 0.1
    horizon: float = 3.0
    clearance_margin: float = 0.15
    dist_cap: float = 3.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} phải là số hữu hạn')
        for name in ('a_max', 'angular_a_max', 'v_resolution', 'w_resolution',
                     'horizon', 'dist_cap'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} phải dương')
        if min(self.alpha, self.beta, self.gamma, self.clearance_margin) < 0:
            raise ValueError('trọng số và biên an toàn phải không âm')
        if self.alpha + self.beta + self.gamma <= 0:
            raise ValueError('ít nhất một trọng số phải dương')
        if self.v_min < 0 or self.v_max <= self.v_min or self.w_max <= self.w_min:
            raise ValueError('giới hạn vận tốc không hợp lệ')


@dataclass
class DWACandidate:
    v: float
    w: float
    heading: float
    dist: float
    velocity: float
    score: float
    trajectory: np.ndarray


def _samples(low, high, resolution, current):
    """Giữ hai đầu cửa sổ, vận tốc hiện tại và số 0 nếu thuộc khoảng."""
    if high < low:
        return np.empty(0)
    values = np.arange(low, high + resolution * 1e-8, resolution)
    values = np.append(values, high)
    if low <= 0 <= high:
        values = np.append(values, 0.0)
    if low <= current <= high:
        values = np.append(values, current)
    return np.unique(np.round(values, 12))


class DWAController:
    name = 'DWA'

    def __init__(self, obstacles, config=None):
        self.obstacles = obstacles
        self.config = config if config is not None else DWAConfig()
        self.goal = None
        self.phase = 'DWA'
        self.debug = {}
        self.candidates = []
        self.selected_candidate = None
        self._failed = False

    def reset(self, robot, goal):
        self.goal = np.asarray(_validate_goal(goal))
        self.phase = 'DWA'
        self.debug = {'state': self.phase, 'message': ''}
        self.candidates = []
        self.selected_candidate = None
        self._failed = False

    def _stop(self, phase, message):
        self.phase = phase
        self.debug.update(state=phase, message=message)
        self.candidates = []
        self.selected_candidate = None
        self._failed = phase == 'FAILED'
        return Control(0.0, 0.0, arrived=phase == 'ARRIVED',
                       failed=self._failed, phase=phase, message=message)

    def on_collision(self, robot):
        self._stop('FAILED', 'DWA | Va chạm — dừng robot.')

    def _clearances(self, points, radius):
        """Khoảng hở từ tâm đến vật cản trừ đường tròn bao robot."""
        result = np.full(points.shape[:-1], np.inf)
        for obstacle in self.obstacles.obs:
            if isinstance(obstacle, Circle):
                distance = np.linalg.norm(points - obstacle.center, axis=-1) - obstacle.radius
            elif isinstance(obstacle, Rectangle):
                polygon = rectangle_corners(obstacle)
                starts = polygon
                edges = np.roll(polygon, -1, axis=0) - polygon
                relative = points[..., None, :] - starts
                lengths = np.sum(edges * edges, axis=-1)
                projection = np.sum(relative * edges, axis=-1) / np.maximum(lengths, 1e-24)
                closest = relative - np.clip(projection, 0, 1)[..., None] * edges
                distance = np.linalg.norm(closest, axis=-1).min(axis=-1)
                crosses = edges[:, 0] * relative[..., 1] - edges[:, 1] * relative[..., 0]
                inside = np.all(crosses >= -1e-12, axis=-1) | np.all(crosses <= 1e-12, axis=-1)
                distance = np.where(inside, 0.0, distance)
            else:
                continue
            result = np.minimum(result, distance - radius)
        return result

    def step(self, robot, goal, dt, *, tracking_target=None, bounds=None,
             progress_direction=None):
        goal = np.asarray(_validate_goal(goal))
        target = (goal if tracking_target is None else
                  np.asarray(_validate_goal(tracking_target)))
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt phải dương hữu hạn')
        if self.goal is None or not np.array_equal(goal, self.goal):
            self.reset(robot, goal)
        if self._failed:
            return self._stop('FAILED', self.debug.get('message', 'DWA | Dừng.'))
        cfg = self.config
        radius = 0.5 * math.hypot(robot.robot_length, robot.robot_width) + cfg.clearance_margin
        if self._clearances(np.array([[robot.x, robot.y]]), radius)[0] <= 0:
            return self._stop('FAILED', 'DWA | Robot nằm trong vùng vật cản.')
        if self._clearances(goal[None, :], radius)[0] <= 0:
            return self._stop('FAILED', 'DWA | Goal nằm trong vùng vật cản.')
        goal_distance = float(np.linalg.norm(goal - [robot.x, robot.y]))
        if goal_distance < 0.3:
            return self._stop('ARRIVED', 'DWA | Destination reached!')
        target_distance = float(np.linalg.norm(target - [robot.x, robot.y]))
        # Dùng vận tốc bánh đang áp dụng, tránh trạng thái v/omega cũ sau stop.
        current_v = (robot.v_l + robot.v_r) / 2.0
        current_w = (robot.v_r - robot.v_l) / robot.L
        v_low = max(cfg.v_min, current_v - cfg.a_max * dt)
        # Không để horizon vượt quá khoảng cách goal; khi đang chạy nhanh,
        # vẫn giữ giới hạn giảm tốc của cửa sổ động để hãm dần.
        terminal_speed = max(v_low, target_distance / cfg.horizon)
        vs = _samples(v_low,
                      min(cfg.v_max, current_v + cfg.a_max * dt, terminal_speed),
                      cfg.v_resolution, current_v)
        ws = _samples(max(cfg.w_min, current_w - cfg.angular_a_max * dt),
                      min(cfg.w_max, current_w + cfg.angular_a_max * dt), cfg.w_resolution, current_w)
        # Bổ sung hướng chính xác tới goal để lưới omega không gây kẹt
        # tại chỗ khi sai lệch góc nhỏ hơn độ phân giải sau horizon.
        bearing = math.atan2(target[1] - robot.y, target[0] - robot.x)
        heading_error = (bearing - robot.theta + math.pi) % (2 * math.pi) - math.pi
        goal_w = heading_error / cfg.horizon
        if len(ws) and ws[0] <= goal_w <= ws[-1]:
            ws = np.unique(np.append(ws, goal_w))
        velocities, omegas = np.meshgrid(vs, ws, indexing='ij')
        velocities, omegas = velocities.ravel(), omegas.ravel()
        if not len(velocities):
            return self._stop('FAILED', 'DWA | Cửa sổ động rỗng.')
        # Bước thích nghi giới hạn cả dịch chuyển và quét góc của đường tròn bao.
        speed = np.max(np.abs(velocities) + radius * np.abs(omegas))
        h = min(float(dt), 0.1, 0.05 / max(speed, 1e-12))
        steps = max(1, math.ceil(cfg.horizon / h))
        h = cfg.horizon / steps
        poses = np.empty((len(velocities), steps + 1, 3))
        poses[:, 0, :] = (robot.x, robot.y, robot.theta)
        for index in range(steps):
            previous = poses[:, index, :]
            poses[:, index + 1, 0] = previous[:, 0] + velocities * np.cos(previous[:, 2]) * h
            poses[:, index + 1, 1] = previous[:, 1] + velocities * np.sin(previous[:, 2]) * h
            poses[:, index + 1, 2] = previous[:, 2] + omegas * h
        clearance = self._clearances(poses[..., :2], radius)
        # Mỗi điểm kiểm tra bao thêm nửa đoạn Euler: không xuyên vật cản mỏng.
        padding = np.abs(velocities) * h / 2.0
        safe = np.all(clearance > padding[:, None], axis=1)
        if bounds is not None:
            try:
                xmin, xmax, ymin, ymax = (float(value) for value in bounds)
            except (TypeError, ValueError) as exc:
                raise ValueError('bounds không hợp lệ') from exc
            if (not np.all(np.isfinite([xmin, xmax, ymin, ymax]))
                    or xmin >= xmax or ymin >= ymax):
                raise ValueError('bounds không hợp lệ')

            def in_bounds(points):
                return ((points[..., 0] >= xmin) & (points[..., 0] <= xmax)
                        & (points[..., 1] >= ymin) & (points[..., 1] <= ymax))

            safe &= np.all(in_bounds(poses[..., :2]), axis=1)
        min_clearance = clearance.min(axis=1)
        # Sau horizon, giảm cả v và w về 0; chỉ chấp nhận khi toàn đường phanh an toàn.
        braking_pose = poses[:, -1, :].copy()
        braking_v, braking_w = velocities.copy(), omegas.copy()
        brake_steps = math.ceil(max(np.max(np.abs(braking_v)) / cfg.a_max,
                                    np.max(np.abs(braking_w)) / cfg.angular_a_max) / h)
        for _ in range(brake_steps):
            braking_pose[:, 0] += braking_v * np.cos(braking_pose[:, 2]) * h
            braking_pose[:, 1] += braking_v * np.sin(braking_pose[:, 2]) * h
            braking_pose[:, 2] += braking_w * h
            safe &= self._clearances(braking_pose[:, :2], radius) > np.abs(braking_v) * h
            if bounds is not None:
                safe &= in_bounds(braking_pose[:, :2])
            braking_v = np.sign(braking_v) * np.maximum(0, np.abs(braking_v) - cfg.a_max * h)
            braking_w = np.sign(braking_w) * np.maximum(0, np.abs(braking_w) - cfg.angular_a_max * h)
        if progress_direction is not None:
            # Hybrid ưu tiên tập vận tốc thật sự tiến dọc đoạn đường hiện tại.
            # Nếu chưa có tiến an toàn, vẫn cho quay tại chỗ để căn hướng.
            direction = np.asarray(_validate_goal(progress_direction))
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                progress = (poses[:, -1, :2]-[robot.x, robot.y])@(direction/norm)
                advancing = safe & (progress >= 0.01)
                if advancing.any():
                    safe = advancing
        self.candidates = []
        for index in np.flatnonzero(safe):
            end = poses[index, -1]
            angle = math.atan2(target[1] - end[1], target[0] - end[0])
            error = (angle - end[2] + math.pi) % (2 * math.pi) - math.pi
            heading = 1.0 if np.linalg.norm(target - end[:2]) < 1e-9 else 1 - abs(error) / math.pi
            dist = min(float(min_clearance[index]) / cfg.dist_cap, 1.0)
            velocity = float(velocities[index]) / cfg.v_max
            score = cfg.alpha * heading + cfg.beta * dist + cfg.gamma * velocity
            self.candidates.append(DWACandidate(float(velocities[index]), float(omegas[index]),
                                               heading, dist, velocity, score, poses[index]))
        self.candidates.sort(key=lambda c: (c.score, c.heading, c.dist, c.v, -abs(c.w), -c.w), reverse=True)
        if not self.candidates:
            return self._stop('FAILED', 'DWA | Không có vận tốc an toàn để phanh.')
        self.selected_candidate = best = self.candidates[0]
        self.phase = 'DWA'
        message = f'DWA | v={best.v:.2f} | ω={best.w:.2f} | G={best.score:.3f}'
        self.debug = {'state': self.phase, 'message': message, 'trajectory': best.trajectory,
                      'dynamic_window': (float(vs[0]), float(vs[-1]), float(ws[0]), float(ws[-1]))}
        return Control(best.v - best.w * robot.L / 2,
                       best.v + best.w * robot.L / 2, phase=self.phase, message=message)
