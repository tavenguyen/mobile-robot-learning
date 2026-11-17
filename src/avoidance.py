"""Ba thuật toán né tránh vật cản cho robot vi sai: Bug1, Bug2, Tangent Bug.

Bug1/Bug2 dùng contact sensor: chỉ bắt đầu bám biên sau khi pose kế tiếp
sắp chạm vật cản. Tangent Bug dùng range sensor 360° và chọn tiếp tuyến
của khoảng trống có h(p) = d(robot, p) + d(p, goal) nhỏ nhất.
"""
import math
from dataclasses import dataclass

import numpy as np

if __package__:
    from .boundary import offset_boundary
    from .navigation import GOAL_TOLERANCE, goal_command, point_command
    from .sensors import RangeSensor
else:
    from boundary import offset_boundary
    from navigation import GOAL_TOLERANCE, goal_command, point_command
    from sensors import RangeSensor

CLEARANCE_MARGIN = 0.15
LOOKAHEAD = 0.7
FOLLOW_SPEED = 1.4
MAX_EPISODES = 12


@dataclass
class Control:
    v_l: float
    v_r: float
    arrived: bool = False
    failed: bool = False
    phase: str = 'GOAL'
    message: str = ''


def robot_clearance(robot, margin=CLEARANCE_MARGIN):
    """Bán kính bao ngoài robot cộng biên an toàn."""
    return 0.5 * math.hypot(robot.robot_length, robot.robot_width) + margin


def _validate_goal(goal):
    try:
        x, y = float(goal[0]), float(goal[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError('goal phải là cặp (x, y) hữu hạn') from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError('goal phải là cặp (x, y) hữu hạn')
    return x, y


class BaseAvoidanceController:
    """Khung state machine chung: goal, retreat, recover, follow, leave."""

    name = 'Avoidance'

    def __init__(self, obstacles):
        self.obstacles = obstacles
        self.goal = None
        self.clearance = 1.0
        self.phase = 'GOAL'
        self.boundary = None
        self.hit_point = None
        self.leave_point = None
        self.contact_index = None
        self.debug = {}
        self._arrived = False
        self._failed = False
        self._episodes = 0
        self._start_point = None
        self._retreat_target = None
        self._recover_target = None
        self._resume_phase = 'GOAL'
        self._s_previous = None
        self._accumulated = 0.0
        self._l_point = None
        self._s_l = None
        self._best_leave_distance = float('inf')
        self._best_s_l = None
        self._retreat_steps = 0
        self._retreat_start = None
        self._recover_steps = 0
        self._recover_start = None

    # ------------------------------------------------------------------
    # Vòng đời
    # ------------------------------------------------------------------
    def reset(self, robot, goal):
        gx, gy = _validate_goal(goal)
        self.goal = np.array([gx, gy], dtype=float)
        self.clearance = robot_clearance(robot)
        self.phase = 'GOAL'
        self.boundary = None
        self.hit_point = None
        self.leave_point = None
        self.contact_index = None
        self._arrived = False
        self._failed = False
        self._episodes = 0
        self._start_point = np.array([robot.x, robot.y], dtype=float)
        self._retreat_target = None
        self._recover_target = None
        self._resume_phase = 'GOAL'
        self._s_previous = None
        self._accumulated = 0.0
        self._l_point = None
        self._s_l = None
        self._best_leave_distance = float('inf')
        self._best_s_l = None
        self._retreat_steps = 0
        self._retreat_start = None
        self._recover_steps = 0
        self._recover_start = None
        self.debug = {
            'state': 'GOAL',
            'message': '',
            'hit': None,
            'leave': None,
            'target_leave': None,
            'boundary': None,
            'm_line': None,
            'scan': None,
            'tangent': None,
            'tangent_point': None,
            'leave_reason': None,
            'progress': 0.0,
            'h': None,
        }
        if self.obstacles.contains(self.goal) is not None:
            self._failed = True
            self.debug['state'] = 'FAILED'
            self.debug['message'] = (
                f'{self.name} | Goal nằm trong vật cản — không tới được.'
            )
        else:
            index, _, goal_distance, _ = self.obstacles.nearest(self.goal)
            body_radius = 0.5 * math.hypot(
                robot.robot_length, robot.robot_width)
            if index is not None and goal_distance < body_radius - 0.02:
                self._failed = True
                self.debug['state'] = 'FAILED'
                self.debug['message'] = (
                    f'{self.name} | Goal quá sát vật cản — robot không thể tới.'
                )
            else:
                self._reset_extra(robot)

    def _reset_extra(self, robot):
        """Hook cho lớp con."""

    def step(self, robot, goal, dt):
        gx, gy = _validate_goal(goal)
        if self.goal is None or not np.allclose(
                self.goal, (gx, gy), atol=1e-9):
            self.reset(robot, (gx, gy))

        if self._failed:
            return Control(0.0, 0.0, False, True, 'FAILED',
                           self.debug.get('message', 'Không tới được goal.'))
        if self._arrived:
            return Control(0.0, 0.0, True, False, 'ARRIVED',
                           self.debug.get('message', 'Destination reached!'))

        distance = float(np.linalg.norm(self.goal - np.array([robot.x, robot.y])))
        if distance < GOAL_TOLERANCE:
            self._arrived = True
            self.phase = 'ARRIVED'
            message = f'{self.name} | Destination reached!'
            self.debug.update(state='ARRIVED', message=message)
            return Control(0.0, 0.0, True, False, 'ARRIVED', message)

        control = self._step(robot, float(dt))
        self.debug['state'] = control.phase
        self.debug['message'] = control.message
        self.debug['progress'] = min(
            1.0,
            self._accumulated / self.boundary.perimeter
            if self.boundary is not None else 0.0,
        )
        return control

    def on_collision(self, robot):
        """Contact sensor: pose hiện tại đang quá sát vật cản."""
        index, _, distance, normal = self.obstacles.nearest((robot.x, robot.y))
        if index is None:
            return
        if normal is None:
            normal = np.array([1.0, 0.0])
        if self.phase in ('FOLLOW', 'RETURN_L'):
            self._begin_recover(robot, normal, resume=self.phase)
        elif self.phase == 'RECOVER':
            self._recover_target = (
                np.array([robot.x, robot.y])
                + normal * max(0.5, self.clearance)
            )
        elif self.phase == 'RETREAT':
            self._retreat_target = (
                np.array([robot.x, robot.y])
                + normal * max(0.25, self.clearance + 0.25)
            )
        else:
            self._start_contact(robot, index, distance, normal)

    # ------------------------------------------------------------------
    # Các pha dùng chung
    # ------------------------------------------------------------------
    def _start_contact(self, robot, index, distance, normal):
        self._episodes += 1
        if self._episodes > MAX_EPISODES:
            self._fail('quá nhiều lần chạm vật cản — dừng để tránh lặp vô hạn.')
            return
        self.contact_index = index
        self.hit_point = np.array([robot.x, robot.y], dtype=float)
        self.boundary = offset_boundary(
            self.obstacles.obs[index], self.clearance)
        retreat = max(0.3, self.clearance - float(distance) + 0.3)
        self._retreat_target = self.hit_point + normal * retreat
        self._s_previous = None
        self._accumulated = 0.0
        self._retreat_steps = 0
        self._retreat_start = self.hit_point.copy()
        self.phase = 'RETREAT'
        self._resume_phase = 'FOLLOW'
        self.debug.update(
            hit=self.hit_point.copy(),
            boundary=self.boundary.points.copy(),
            target_leave=None,
        )

    def _begin_recover(self, robot, normal, resume):
        self._resume_phase = resume
        self._recover_target = (
            np.array([robot.x, robot.y])
            + normal * max(0.5, self.clearance)
        )
        self._recover_steps = 0
        self._recover_start = np.array([robot.x, robot.y], dtype=float)
        self.phase = 'RECOVER'

    def _enter_follow(self, robot):
        self._s_previous, _, _ = self.boundary.project((robot.x, robot.y))
        self._accumulated = 0.0
        self.phase = self._resume_phase if self._resume_phase != 'RECOVER' \
            else 'FOLLOW'

    def _follow_target(self, robot):
        s, _, _ = self.boundary.project((robot.x, robot.y))
        if self._s_previous is None:
            self._s_previous = s
        delta = (s - self._s_previous) % self.boundary.perimeter
        if delta > self.boundary.perimeter / 2 or delta > 0.4:
            delta = 0.0
        self._accumulated += delta
        self._s_previous = s
        return s, self.boundary.point_at(s + LOOKAHEAD)

    def _return_target(self, robot, point, s_leave):
        s, _, _ = self.boundary.project((robot.x, robot.y))
        forward = (s_leave - s) % self.boundary.perimeter
        distance = float(np.linalg.norm(
            np.array([robot.x, robot.y]) - point))
        if forward <= LOOKAHEAD * 0.6 or distance < 0.35:
            return None
        return self.boundary.point_at(s + LOOKAHEAD)

    def _step_retreat(self, robot):
        index, _, obstacle_distance, normal = self.obstacles.nearest(
            (robot.x, robot.y))
        if normal is None:
            v_l, v_r = 0.0, 0.0
        else:
            heading = np.array([math.cos(robot.theta), math.sin(robot.theta)])
            sign = 1.0 if float(normal @ heading) >= 0.0 else -1.0
            v_l = v_r = sign * 0.9
        self._retreat_steps += 1
        moved = 0.0
        if self._retreat_start is not None:
            moved = float(np.linalg.norm(
                np.array([robot.x, robot.y]) - self._retreat_start))
        if (index is None or obstacle_distance >= self.clearance - 0.02
                or self._retreat_steps >= 10 or moved >= 1.2):
            self._enter_follow(robot)
        message = f'{self.name} | RETREAT — lùi tới khoảng cách an toàn.'
        return Control(v_l, v_r, False, False, 'RETREAT', message)

    def _step_recover(self, robot):
        index, _, obstacle_distance, normal = self.obstacles.nearest(
            (robot.x, robot.y))
        if normal is None:
            v_l, v_r = 0.0, 0.0
        else:
            heading = np.array([math.cos(robot.theta), math.sin(robot.theta)])
            sign = 1.0 if float(normal @ heading) >= 0.0 else -1.0
            v_l = v_r = sign * 0.9
        self._recover_steps += 1
        moved = 0.0
        if self._recover_start is not None:
            moved = float(np.linalg.norm(
                np.array([robot.x, robot.y]) - self._recover_start))
        if (index is None or obstacle_distance >= self.clearance - 0.02
                or self._recover_steps >= 10 or moved >= 0.9):
            self.phase = self._resume_phase
        message = f'{self.name} | RECOVER — lấy lại khoảng cách an toàn.'
        return Control(v_l, v_r, False, False, 'RECOVER', message)

    def _goal_control(self, robot, phase, message):
        v_l, v_r, _, arrived = goal_command(robot, self.goal)
        if arrived:
            self._arrived = True
            return Control(0.0, 0.0, True, False, 'ARRIVED',
                           f'{self.name} | Destination reached!')
        return Control(v_l, v_r, False, False, phase, message)

    def _steer(self, robot, target, phase, message, max_v=FOLLOW_SPEED):
        v_l, v_r, _, _ = point_command(
            robot, target, max_v=max_v, tolerance=0.05)
        return Control(v_l, v_r, False, False, phase, message)

    def _fail(self, message):
        self._failed = True
        self.phase = 'FAILED'
        self.debug.update(state='FAILED', message=f'{self.name} | {message}')

    def _debug_follow(self, s):
        self.debug['boundary'] = self.boundary.points.copy()
        self.debug['progress'] = min(1.0, self._accumulated / self.boundary.perimeter)
        self.debug['target_leave'] = (
            self._l_point.copy() if self._l_point is not None else None)


class Bug1Controller(BaseAvoidanceController):
    """Đi vòng đủ biên vật cản rồi rời tại L — điểm gần goal nhất."""

    name = 'Bug1'

    def _start_contact(self, robot, index, distance, normal):
        super()._start_contact(robot, index, distance, normal)
        if self._failed or self.boundary is None:
            return
        # Chưa biết L: chỉ ghi nhận dần trong lúc đi đủ một vòng.
        self._s_l = None
        self._l_point = None
        self._best_leave_distance = float('inf')
        self._best_s_l = None
        self.debug['target_leave'] = None

    def _step(self, robot, dt):
        if self.phase == 'RETREAT':
            return self._step_retreat(robot)
        if self.phase == 'RECOVER':
            return self._step_recover(robot)

        if self.phase == 'FOLLOW':
            s, target = self._follow_target(robot)
            self._debug_follow(s)

            # Ghi nhận điểm trên biên gần goal nhất trong suốt vòng đi.
            boundary_point = self.boundary.point_at(s)
            goal_distance = float(np.linalg.norm(boundary_point - self.goal))
            if goal_distance < self._best_leave_distance:
                self._best_leave_distance = goal_distance
                self._best_s_l = s

            if self._accumulated >= self.boundary.perimeter - 0.25:
                # Đã đủ một vòng: lúc này mới chốt L.
                self._s_l = self._best_s_l
                self._l_point = self.boundary.point_at(self._best_s_l)
                self.debug['target_leave'] = self._l_point.copy()
                self.phase = 'RETURN_L'
            else:
                progress = 100.0 * self._accumulated / self.boundary.perimeter
                message = (
                    f'Bug1 | FOLLOW vòng {progress:3.0f}% | '
                    f'đang ghi nhận điểm gần goal nhất'
                )
                return self._steer(robot, target, 'FOLLOW', message)

        if self.phase == 'RETURN_L':
            target = self._return_target(
                robot, self._l_point, self._s_l)
            if target is None:
                self.leave_point = self._l_point.copy()
                self.debug['leave'] = self.leave_point.copy()
                self.debug['leave_reason'] = 'L — điểm trên biên gần goal nhất'
                self.phase = 'LEAVE'
            else:
                return self._steer(
                    robot, target, 'RETURN_L',
                    'Bug1 | Đã đủ 1 vòng — RETURN tới L.')

        if self.phase == 'LEAVE':
            message = 'Bug1 | LEAVE @ L — đi thẳng tới goal.'
            return self._goal_control(robot, 'LEAVE', message)

        # Phòng trường hợp state lạ: quay lại goal.
        return self._goal_control(robot, 'GOAL', f'Bug1 | GOAL | d={0:.2f} m')


class Bug2Controller(BaseAvoidanceController):
    """Bám biên tới khi cắt lại m-line tại Q gần goal hơn H."""

    name = 'Bug2'

    def _reset_extra(self, robot):
        self._m_start = np.array([robot.x, robot.y], dtype=float)
        self._last_leave = None
        self._d_h = None
        self.debug['m_line'] = (self._m_start.copy(), self.goal.copy())

    def _m_line_projection(self, point):
        direction = self.goal - self._m_start
        length_squared = float(direction @ direction)
        if length_squared <= 1e-12:
            return self._m_start.copy(), 0.0, float(
                np.linalg.norm(point - self._m_start))
        t = float((point - self._m_start) @ direction / length_squared)
        t_clamped = min(max(t, 0.0), 1.0)
        projection = self._m_start + t_clamped * direction
        return projection, t, float(np.linalg.norm(point - projection))

    def _start_contact(self, robot, index, distance, normal):
        # Va chạm lại ngay sau khi rời Q: tiếp tục bám biên, không reset H.
        if (self._last_leave is not None
                and index == self.contact_index
                and float(np.linalg.norm(
                    np.array([robot.x, robot.y]) - self._last_leave)) < 0.6):
            self._begin_recover(robot, normal, resume='FOLLOW')
            return
        super()._start_contact(robot, index, distance, normal)
        if self._failed or self.boundary is None:
            return
        self._d_h = float(np.linalg.norm(
            np.array([robot.x, robot.y]) - self.goal))
        self.debug['m_line'] = (self._m_start.copy(), self.goal.copy())

    def _step(self, robot, dt):
        if self.phase == 'RETREAT':
            return self._step_retreat(robot)
        if self.phase == 'RECOVER':
            return self._step_recover(robot)

        if self.phase == 'FOLLOW':
            s, target = self._follow_target(robot)
            self._debug_follow(s)
            point = np.array([robot.x, robot.y])
            projection, t, line_distance = self._m_line_projection(point)
            closer = (
                self._d_h is not None
                and float(np.linalg.norm(projection - self.goal))
                < self._d_h - 0.2
            )
            if (0.0 <= t <= 1.0 and line_distance < 0.18
                    and closer and self._accumulated > 0.5):
                self.leave_point = projection.copy()
                self._last_leave = projection.copy()
                self.debug['leave'] = projection.copy()
                improvement = self._d_h - float(
                    np.linalg.norm(projection - self.goal))
                self.debug['leave_reason'] = (
                    f'Q trên m-line, gần goal hơn H {improvement:.2f} m')
                self.phase = 'LEAVE_Q'
            elif self._accumulated >= self.boundary.perimeter - 0.25:
                self._l_point = self.boundary.project(self.goal)[1]
                self._s_l = self.boundary.project(self.goal)[0]
                self.debug['target_leave'] = self._l_point.copy()
                self.phase = 'RETURN_L'
            else:
                message = (
                    f'Bug2 | FOLLOW | d(m-line)={line_distance:.2f} m | '
                    f'H cách goal {self._d_h:.2f} m'
                )
                return self._steer(robot, target, 'FOLLOW', message)

        if self.phase == 'RETURN_L':
            target = self._return_target(robot, self._l_point, self._s_l)
            if target is None:
                self.leave_point = self._l_point.copy()
                self._last_leave = self.leave_point.copy()
                self.debug['leave'] = self.leave_point.copy()
                self.debug['leave_reason'] = (
                    'fallback: m-line không cắt lại, rời tại L gần goal nhất')
                self.phase = 'LEAVE'
            else:
                return self._steer(
                    robot, target, 'RETURN_L',
                    'Bug2 | RETURN_L — m-line không cắt lại.')

        if self.phase == 'LEAVE_Q':
            v_l, v_r, distance, _ = point_command(
                robot, self.leave_point, max_v=FOLLOW_SPEED, tolerance=0.1)
            if distance < 0.2:
                self.phase = 'LEAVE'
            return Control(
                v_l, v_r, False, False, 'LEAVE_Q',
                'Bug2 | LEAVE_Q — quay lại m-line tại Q.')

        if self.phase == 'LEAVE':
            message = 'Bug2 | LEAVE @ Q — đi thẳng tới goal.'
            return self._goal_control(robot, 'LEAVE', message)

        return self._goal_control(robot, 'GOAL', 'Bug2 | GOAL — bám m-line.')


class TangentBugController(BaseAvoidanceController):
    """Range sensor + tiếp tuyến có h(p) nhỏ nhất; rời khi thấy goal."""

    name = 'Tangent Bug'

    def __init__(self, obstacles, n_rays=72, max_range=15.0):
        super().__init__(obstacles)
        self.sensor = RangeSensor(obstacles, n_rays=n_rays, max_range=max_range)
        self.last_scan = None
        self.last_endpoint = None
        self._boundaries = {}
        self._follow_direction = None
        self._last_blocker = None

    def _reset_extra(self, robot):
        self.last_scan = None
        self.last_endpoint = None
        self._boundaries = {}
        self._follow_direction = None
        self._last_blocker = None
        self.debug['scan'] = None
        self.debug['tangent'] = None

    def _boundary_for(self, index):
        if index not in self._boundaries:
            self._boundaries[index] = offset_boundary(
                self.obstacles.obs[index], self.clearance)
        return self._boundaries[index]

    def on_collision(self, robot):
        index, _, _, normal = self.obstacles.nearest((robot.x, robot.y))
        if index is None:
            return
        if normal is None:
            normal = np.array([1.0, 0.0])
        self._begin_recover(robot, normal, resume='GOAL')

    def _step(self, robot, dt):
        if self.phase == 'RECOVER':
            return self._step_recover(robot)

        scan = self.sensor.scan(robot.x, robot.y, robot.theta)
        self.last_scan = scan
        self.debug['scan'] = scan

        goal_array = self.goal
        point = np.array([robot.x, robot.y])
        blocker, _ = self.sensor.first_blocker(
            point, goal_array, self.clearance)

        if blocker is None:
            just_left = self.phase == 'FOLLOW'
            if just_left:
                self.leave_point = point.copy()
                self.debug['leave'] = point.copy()
                self.debug['leave_reason'] = 'LOS — đã thấy goal, rời tiếp tuyến'
            self._follow_direction = None
            self._last_blocker = None
            self.phase = 'GOAL'
            self.debug['tangent'] = None
            self.debug['tangent_point'] = None
            distance = float(np.linalg.norm(goal_array - point))
            if just_left:
                return self._goal_control(
                    robot, 'LEAVE',
                    f'Tangent Bug | LEAVE — đã thấy goal (LOS) | '
                    f'd={distance:.2f} m')
            message = f'Tangent Bug | GOAL | d={distance:.2f} m'
            return self._goal_control(robot, 'GOAL', message)

        goal_angle = math.atan2(
            goal_array[1] - robot.y, goal_array[0] - robot.x)
        interval = self.sensor.interval_for_obstacle(
            scan, blocker, goal_angle)
        if interval is None:
            # Vật cản chưa nằm trong tầm quét: tiến thăm dò về phía goal.
            message = 'Tangent Bug | GOAL — vật cản ngoài tầm quét, tiến thăm dò.'
            return self._goal_control(robot, 'GOAL', message)

        endpoints = self.sensor.endpoints_for_interval(
            scan, interval, point, goal_array, self.clearance)
        if not endpoints:
            # Quay tại chỗ để tìm khoảng trống.
            return Control(
                -0.8, 0.8, False, False, 'SEARCH',
                'Tangent Bug | SEARCH — quay tìm khoảng trống.')

        best = min(endpoints, key=lambda endpoint: endpoint.h)
        self.last_endpoint = best
        self.debug['tangent'] = best
        self.debug['tangent_point'] = best.point.copy()
        self.debug['h'] = best.h

        distance_goal = float(np.linalg.norm(goal_array - point))

        # Đi theo tiếp tuyến của biên offset. Giữ một chiều bám biên duy nhất
        # trong suốt lần né tránh để không quay đầu ở góc vật cản.
        boundary = self._boundary_for(blocker)
        s_robot, nearest_point, _ = boundary.project(point)
        tangent_unit = boundary.tangent_at(s_robot)
        if self.phase != 'FOLLOW' or self._follow_direction is None \
                or self._last_blocker != blocker:
            toward_endpoint = best.point - point
            self._follow_direction = (
                1.0 if float(tangent_unit @ toward_endpoint) >= 0.0 else -1.0)
            self._last_blocker = blocker
        tangent = tangent_unit * self._follow_direction

        index, closest, obstacle_distance, normal = self.obstacles.nearest(point)
        if obstacle_distance > self.clearance + 0.35:
            # Còn xa biên an toàn: đi theo tia tiếp tuyến tới đầu mút khoảng
            # trống, giống tia quét của range sensor.
            target = best.safe_point.copy()
        else:
            # Đã ở gần biên: bám đúng tiếp tuyến của biên offset.
            target = point + tangent * LOOKAHEAD
            if (index is not None and closest is not None
                    and obstacle_distance < self.clearance - 0.03):
                outward = point - closest
                norm = float(np.linalg.norm(outward))
                if norm > 1e-6:
                    outward = outward / norm
                elif normal is not None:
                    outward = np.asarray(normal, dtype=float)
                else:
                    outward = np.array([1.0, 0.0])
                target = target + outward * min(
                    1.0, self.clearance - obstacle_distance + 0.35)

        self.debug['boundary'] = boundary.points.copy()
        self.debug['progress'] = 0.0

        self.phase = 'FOLLOW'
        message = (
            f'Tangent Bug | FOLLOW p* ({best.side}) | '
            f'h={best.h:.2f} m | goal={distance_goal:.2f} m'
        )
        return self._steer(
            robot, target, 'FOLLOW', message,
            max_v=FOLLOW_SPEED)
