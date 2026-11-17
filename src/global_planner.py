"""Dijkstra/A* trên lưới và bộ bám waypoint P kiểu dừng–quay–đi."""
import heapq
import math
from dataclasses import dataclass, field

import numpy as np
from matplotlib.patches import Circle, Rectangle

if __package__:
    from .avoidance import Control, _validate_goal
    from .Obstacle import rectangle_corners
else:
    from avoidance import Control, _validate_goal
    from Obstacle import rectangle_corners

DEFAULT_BOUNDS = (-20.0, 20.0, -20.0, 20.0)


def _octile_distance(start, goal):
    """Khoảng cách lưới 8 hướng, nhận tọa độ mét và hỗ trợ broadcasting."""
    difference = np.abs(np.asarray(start, dtype=float)-np.asarray(goal, dtype=float))
    return difference.max(axis=-1)+(math.sqrt(2)-1)*difference.min(axis=-1)


def _goal_heuristics(coordinates, goals):
    """Cận dưới qua các nút nối goal thật; giữ đúng chi phí đoạn nối Euclid."""
    coordinates = np.asarray(coordinates, dtype=float)
    heuristics = np.full(len(coordinates), np.inf)
    for node, connector_cost in goals.items():
        heuristics = np.minimum(
            heuristics, _octile_distance(coordinates, coordinates[node])+connector_cost)
    return heuristics


@dataclass
class PlanResult:
    path: list = field(default_factory=list)
    status: str = 'FAILED'
    message: str = ''
    cost: float = math.inf
    expanded: int = 0


class _CollisionMap:
    """Kiểm tra đoạn thẳng chính xác, kể cả góc tròn của rectangle nở rộng."""

    def __init__(self, obstacles, clearance):
        self.clearance = clearance
        self.shapes = []
        for patch in obstacles.obs:
            if isinstance(patch, Circle):
                self.shapes.append(('circle', np.array(patch.center), patch.radius))
            elif isinstance(patch, Rectangle):
                angle = math.radians(patch.get_angle())
                basis = np.array([[math.cos(angle), -math.sin(angle)],
                                  [math.sin(angle), math.cos(angle)]])
                center = rectangle_corners(patch).mean(axis=0)
                half = np.array([abs(patch.get_width()), abs(patch.get_height())])/2
                self.shapes.append(('rect', center, (basis, half)))

    def free(self, starts, ends):
        starts, ends = np.asarray(starts, float), np.asarray(ends, float)
        scalar = starts.ndim == 1
        starts, ends = np.atleast_2d(starts), np.atleast_2d(ends)
        safe = np.ones(len(starts), dtype=bool)
        for kind, center, shape in self.shapes:
            a, b = starts-center, ends-center
            if kind == 'circle':
                delta = b-a
                t = np.clip(-np.sum(a*delta, axis=1) /
                            np.maximum(np.sum(delta*delta, axis=1), 1e-24), 0, 1)
                distance = np.linalg.norm(a+t[:, None]*delta, axis=1)-shape
            else:
                basis, half = shape
                a, b = a@basis, b@basis
                delta = b-a
                # Slab test của đoạn với rectangle chưa nở rộng.
                parallel = np.abs(delta) < 1e-14
                denominator = np.where(parallel, 1.0, delta)
                first, second = (-half-a)/denominator, (half-a)/denominator
                lower, upper = np.minimum(first, second), np.maximum(first, second)
                outside = parallel & (np.abs(a) > half)
                lower = np.where(parallel, -np.inf, lower)
                upper = np.where(parallel, np.inf, upper)
                intersects = ((np.maximum(lower.max(axis=1), 0) <=
                               np.minimum(upper.min(axis=1), 1)) & ~outside.any(axis=1))
                distance = np.minimum(np.linalg.norm(np.maximum(np.abs(a)-half, 0), axis=1),
                                      np.linalg.norm(np.maximum(np.abs(b)-half, 0), axis=1))
                corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])*half
                relative = corners[None, :, :]-a[:, None, :]
                lengths = np.maximum(np.sum(delta*delta, axis=1), 1e-24)
                t = np.clip(np.sum(relative*delta[:, None, :], axis=2)/lengths[:, None], 0, 1)
                corner_distance = np.linalg.norm(relative-t[..., None]*delta[:, None, :], axis=2).min(axis=1)
                distance = np.where(intersects, 0, np.minimum(distance, corner_distance))
            safe &= distance > self.clearance+1e-10
        return bool(safe[0]) if scalar else safe


def plan_path(obstacles, start, goal, *, algorithm='A*', clearance=0.15,
              resolution=0.5, bounds=DEFAULT_BOUNDS):
    """Tìm đường ngắn nhất trên lưới 8 hướng, trả cả tọa độ đầu/cuối thật."""
    if algorithm not in ('A*', 'Dijkstra'):
        raise ValueError('algorithm phải là A* hoặc Dijkstra')
    start, goal = np.array(_validate_goal(start)), np.array(_validate_goal(goal))
    try:
        resolution, clearance = float(resolution), float(clearance)
        xmin, xmax, ymin, ymax = (float(v) for v in bounds)
    except (TypeError, ValueError) as exc:
        raise ValueError('tham số lưới không hợp lệ') from exc
    if (not np.all(np.isfinite([resolution, clearance, xmin, xmax, ymin, ymax]))
            or resolution <= 0 or clearance < 0 or xmax <= xmin or ymax <= ymin):
        raise ValueError('tham số lưới không hợp lệ')
    for point in (start, goal):
        if not (xmin <= point[0] <= xmax and ymin <= point[1] <= ymax):
            return PlanResult(message='Điểm đầu hoặc goal nằm ngoài bản đồ.')
    collision = _CollisionMap(obstacles, clearance)
    if not collision.free(start, start) or not collision.free(goal, goal):
        return PlanResult(message='Điểm đầu hoặc goal nằm trong vùng vật cản.')
    if np.linalg.norm(goal-start) < 1e-12:
        return PlanResult([tuple(start)], 'OK', 'Đã ở goal.', 0.0, 0)
    xs = np.arange(xmin, xmax+resolution*1e-8, resolution)
    ys = np.arange(ymin, ymax+resolution*1e-8, resolution)
    nx, ny = len(xs), len(ys)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    coordinates = np.column_stack((xx.ravel(), yy.ravel()))
    free_nodes = collision.free(coordinates, coordinates).reshape(nx, ny)
    neighbors = [[] for _ in range(len(coordinates))]
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        indices = np.array([(i, j) for i in range(nx) for j in range(ny)
                            if 0 <= i+dx < nx and 0 <= j+dy < ny], dtype=int)
        if not len(indices):
            continue
        i, j = indices.T
        valid = free_nodes[i, j] & free_nodes[i+dx, j+dy]
        if dx and dy:
            valid &= free_nodes[i+dx, j] & free_nodes[i, j+dy]
        source, target = i*ny+j, (i+dx)*ny+j+dy
        valid &= collision.free(coordinates[source], coordinates[target])
        weight = resolution*math.hypot(dx, dy)
        for a, b in zip(source[valid], target[valid]):
            neighbors[a].append((int(b), weight))
            neighbors[b].append((int(a), weight))

    def connectors(point):
        scaled = (point-[xmin, ymin])/resolution
        choices = [{int(math.floor(value)), int(math.ceil(value))} for value in scaled]
        result = []
        for i in choices[0]:
            for j in choices[1]:
                if 0 <= i < nx and 0 <= j < ny and free_nodes[i, j]:
                    node = i*ny+j
                    if collision.free(point, coordinates[node]):
                        result.append((node, float(np.linalg.norm(point-coordinates[node]))))
        return result

    starts, goals = connectors(start), dict(connectors(goal))
    if not starts or not goals:
        return PlanResult(message='Không nối được điểm đầu/goal vào lưới an toàn.')
    heuristics = (_goal_heuristics(coordinates, goals) if algorithm == 'A*'
                  else np.zeros(len(coordinates)))
    costs, previous, queue = {}, {}, []
    for node, cost in starts:
        costs[node] = cost
        heapq.heappush(queue, (cost+heuristics[node], cost, node))
    expanded, visited = 0, set()
    best_goal, best_cost = None, math.inf
    while queue:
        priority, cost, node = heapq.heappop(queue)
        if node in visited or cost > costs[node]+1e-12:
            continue
        if priority >= best_cost-1e-12:
            break
        visited.add(node)
        expanded += 1
        if node in goals and cost+goals[node] < best_cost:
            best_goal, best_cost = node, cost+goals[node]
        for neighbor, weight in neighbors[node]:
            new_cost = cost+weight
            if new_cost < costs.get(neighbor, math.inf)-1e-12:
                costs[neighbor], previous[neighbor] = new_cost, node
                heapq.heappush(queue, (new_cost+heuristics[neighbor], new_cost, neighbor))
    if best_goal is None:
        return PlanResult(message='Không tìm được đường trên lưới.', expanded=expanded)
    nodes = [best_goal]
    while nodes[-1] in previous:
        nodes.append(previous[nodes[-1]])
    points = [tuple(start)]
    for node in reversed(nodes):
        point = tuple(coordinates[node])
        if np.linalg.norm(np.array(point)-points[-1]) > 1e-12:
            points.append(point)
    if np.linalg.norm(goal-points[-1]) > 1e-12:
        points.append(tuple(goal))
    # Gộp các đoạn thẳng cùng hướng, không làm tắt qua góc của đường lưới.
    simplified = []
    for point in points:
        while len(simplified) >= 2:
            a = np.array(simplified[-1])-simplified[-2]
            b = np.array(point)-simplified[-1]
            if abs(a[0]*b[1]-a[1]*b[0]) > 1e-10 or a@b <= 0:
                break
            simplified.pop()
        simplified.append(point)
    return PlanResult(simplified, 'OK', f'{algorithm} | Đã tìm đường.', best_cost, expanded)


class GlobalPlannerController:
    """Bám từng đoạn an toàn: dừng để quay rồi mới chạy thẳng."""

    def __init__(self, obstacles, algorithm='A*'):
        if algorithm not in ('A*', 'Dijkstra'):
            raise ValueError('algorithm phải là A* hoặc Dijkstra')
        self.obstacles, self.algorithm = obstacles, algorithm
        self.name = algorithm
        self.goal = None
        self.result = None
        self.path = []
        self.phase = 'PLAN'
        self.debug = {}
        self._waypoint = 0

    def reset(self, robot, goal):
        self.goal = np.array(_validate_goal(goal))
        clearance = math.hypot(robot.robot_length, robot.robot_width)/2+0.15
        self.result = plan_path(self.obstacles, (robot.x, robot.y), self.goal,
                                algorithm=self.algorithm, clearance=clearance)
        self.path = self.result.path
        self._waypoint = min(1, max(0, len(self.path)-1))
        self.phase = 'TRACK' if self.result.status == 'OK' else 'FAILED'
        message = (f'{self.algorithm} | {self.result.message} | cost={self.result.cost:.2f}'
                   f' | expanded={self.result.expanded} | waypoint {self._waypoint+1}/{len(self.path)}')
        self.debug = {'state': self.phase, 'message': message,
                      'path': self.path, 'target': self.path[self._waypoint] if self.path else None}

    def on_collision(self, robot):
        self.phase = 'FAILED'
        self.debug.update(state=self.phase, message=f'{self.algorithm} | Va chạm — dừng robot.')

    def _command(self, v, w, robot, phase):
        self.phase = phase
        message = (f'{self.algorithm} | {phase} | waypoint {self._waypoint+1}/{len(self.path)}'
                   f' | cost={self.result.cost:.2f} | expanded={self.result.expanded}')
        if phase == 'FAILED':
            message = self.debug.get('message', self.result.message)
        self.debug.update(state=phase, message=message,
                          target=self.path[self._waypoint] if self.path else None)
        return Control(v-w*robot.L/2, v+w*robot.L/2, arrived=phase == 'ARRIVED',
                       failed=phase == 'FAILED', phase=phase, message=message)

    def step(self, robot, goal, dt):
        goal = np.array(_validate_goal(goal))
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt phải dương hữu hạn')
        if self.goal is None or not np.array_equal(self.goal, goal):
            self.reset(robot, goal)
        if self.phase == 'FAILED':
            return self._command(0, 0, robot, 'FAILED')
        target = self.path[self._waypoint]
        distance = math.hypot(target[0]-robot.x, target[1]-robot.y)
        if distance <= 0.03:
            if self._waypoint == len(self.path)-1:
                return self._command(0, 0, robot, 'ARRIVED')
            self._waypoint += 1
            return self._command(0, 0, robot, 'TRACK')
        bearing = math.atan2(target[1]-robot.y, target[0]-robot.x)
        error = (bearing-robot.theta+math.pi) % (2*math.pi)-math.pi
        if abs(error) > 0.02:
            w = math.copysign(min(1.5, 2*abs(error), abs(error)/dt), error)
            return self._command(0, w, robot, 'TURN')
        v = min(0.8, 1.5*distance, distance/dt)
        return self._command(v, 0, robot, 'TRACK')
