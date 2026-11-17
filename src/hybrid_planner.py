"""Đường toàn cục A* kết hợp DWA bám mục tiêu cục bộ theo thứ tự đoạn."""
import math

import numpy as np

if __package__:
    from .avoidance import Control, _validate_goal
    from .dwa import DWAController
    from .global_planner import DEFAULT_BOUNDS, _CollisionMap, plan_path
else:
    from avoidance import Control, _validate_goal
    from dwa import DWAController
    from global_planner import DEFAULT_BOUNDS, _CollisionMap, plan_path


class AStarDWAController:
    name = 'A* + DWA'

    def __init__(self, obstacles, config=None):
        self.obstacles = obstacles
        self.local = DWAController(obstacles, config)
        self.goal = None
        self.path = []
        self.result = None
        self.local_target = None
        self.phase = 'PLAN'
        self.debug = {}
        self._segment = 0
        self._segment_progress = 0.0
        self._progress_mark = 0.0
        self._stalled_time = 0.0
        self._replan_attempts = 0

    @property
    def config(self):
        return self.local.config

    @config.setter
    def config(self, value):
        self.local.config = value

    @property
    def candidates(self):
        return self.local.candidates

    @property
    def selected_candidate(self):
        return self.local.selected_candidate

    def _plan(self, robot):
        radius = math.hypot(robot.robot_length, robot.robot_width)/2
        self.result = plan_path(self.obstacles, (robot.x, robot.y), self.goal,
                                algorithm='A*', clearance=radius+0.25,
                                bounds=DEFAULT_BOUNDS)
        self.path = self.result.path
        self._segment = 0
        self._segment_progress = 0.0
        self._progress_mark = 0.0
        self._stalled_time = 0.0
        self.local.reset(robot, self.goal)
        self.local_target = None
        if self.result.status != 'OK':
            return False
        self._update_target(robot)
        return True

    def reset(self, robot, goal):
        self.goal = np.asarray(_validate_goal(goal))
        self._replan_attempts = 0
        if not self._plan(robot):
            self._stop('FAILED', f'Không tìm được đường: {self.result.message}')
        else:
            self.phase = 'TRACK'
            self._update_debug(f'Đã lập đường A* | cost={self.result.cost:.2f}'
                               f' | expanded={self.result.expanded}')

    def _update_debug(self, message):
        self.debug = {'state': self.phase, 'message': f'{self.name} | {message}',
                      'path': self.path, 'target': self.local_target,
                      'trajectory': (self.selected_candidate.trajectory
                                     if self.selected_candidate is not None else None),
                      'segment': self._segment, 'progress': self._arclength(),
                      'replans': self._replan_attempts}

    def _arclength(self):
        if len(self.path) < 2:
            return 0.0
        prefix = sum(math.dist(a, b) for a, b in
                     zip(self.path[:self._segment], self.path[1:self._segment+1]))
        return prefix+self._segment_progress

    def _target_on_segment(self, index, progress):
        start, end = np.asarray(self.path[index]), np.asarray(self.path[index+1])
        length = float(np.linalg.norm(end-start))
        return start+(end-start)*min(length, progress+1.5)/max(length, 1e-12)

    def _update_target(self, robot):
        if len(self.path) <= 1:
            self.local_target = tuple(self.goal)
            return
        point = np.array([robot.x, robot.y])
        start, end = np.asarray(self.path[self._segment]), np.asarray(self.path[self._segment+1])
        delta = end-start
        length = float(np.linalg.norm(delta))
        projection = float((point-start)@delta/max(length, 1e-12))
        self._segment_progress = max(self._segment_progress, min(length, max(0, projection)))
        if (self._segment < len(self.path)-2 and np.linalg.norm(point-end) <= 0.3):
            next_target = self._target_on_segment(self._segment+1, 0.0)
            radius = math.hypot(robot.robot_length, robot.robot_width)/2+self.config.clearance_margin
            if _CollisionMap(self.obstacles, radius).free(point, next_target):
                self._segment += 1
                self._segment_progress = 0.0
        self.local_target = tuple(self._target_on_segment(self._segment, self._segment_progress))

    def _stop(self, phase, message):
        self.phase = phase
        self.local.candidates = []
        self.local.selected_candidate = None
        self._update_debug(message)
        return Control(0, 0, arrived=phase == 'ARRIVED', failed=phase == 'FAILED',
                       phase=phase, message=self.debug['message'])

    def on_collision(self, robot):
        self._stop('FAILED', 'Va chạm — dừng robot.')

    def _replan(self, robot, reason):
        if self._replan_attempts >= 2:
            return self._stop('FAILED', f'{reason} Không phục hồi sau hai lần lập lại đường.')
        self._replan_attempts += 1
        if not self._plan(robot):
            return self._stop('FAILED', f'{reason} {self.result.message}')
        return self._stop('REPLAN', f'{reason} Lập lại đường lần {self._replan_attempts}/2.')

    def step(self, robot, goal, dt):
        goal = np.asarray(_validate_goal(goal))
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt phải dương hữu hạn')
        if self.goal is None or not np.array_equal(self.goal, goal):
            self.reset(robot, goal)
        if self.phase == 'FAILED':
            return self._stop('FAILED', self.debug['message'].removeprefix(f'{self.name} | '))
        if np.linalg.norm(goal-[robot.x, robot.y]) < 0.3:
            return self._stop('ARRIVED', 'Destination reached!')
        self._update_target(robot)
        progress = self._arclength()
        if progress-self._progress_mark >= 0.1:
            self._progress_mark = progress
            self._stalled_time = 0.0
            self._replan_attempts = 0
        else:
            self._stalled_time += dt
        if self._stalled_time >= 5.0-1e-10:
            return self._replan(robot, 'Không tiến thêm 0.1 m trong 5 s.')
        direction = (np.asarray(self.path[self._segment+1])-self.path[self._segment]
                     if len(self.path) > 1 else self.goal-[robot.x, robot.y])
        control = self.local.step(robot, self.goal, dt,
                                  tracking_target=self.local_target, bounds=DEFAULT_BOUNDS,
                                  progress_direction=direction)
        if control.failed:
            return self._replan(robot, 'DWA không có vận tốc an toàn.')
        if control.arrived:
            return self._stop('ARRIVED', 'Destination reached!')
        self.phase = 'TRACK'
        self._update_debug(f'Đoạn {self._segment+1}/{max(1, len(self.path)-1)}'
                           f' | cost={self.result.cost:.2f} | expanded={self.result.expanded}'
                           f' | v={self.selected_candidate.v:.2f} | G={self.selected_candidate.score:.3f}')
        control.phase, control.message = self.phase, self.debug['message']
        return control
