"""Tập vật cản tĩnh và kiểm tra va chạm với robot (Bài thực hành 1)."""
import numpy as np
from matplotlib.patches import Circle, Rectangle

_TOL = 1e-12
_SUPPORTED = (Rectangle, Circle)


def _as_polygon(points, name='points'):
    """Chuyển về mảng (N, 2) hữu hạn, N >= 3."""
    try:
        points = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} phải là tọa độ số') from exc
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError(f'{name} phải có dạng (N, 2) với N >= 3')
    if not np.all(np.isfinite(points)):
        raise ValueError(f'{name} phải là số hữu hạn')
    return points


def _as_point(value, name='point'):
    """Chuyển về điểm (2,) hữu hạn."""
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} phải là cặp (x, y)') from exc
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f'{name} phải là cặp (x, y) hữu hạn')
    return point


def _as_finite(value, name='giá trị'):
    """Chuyển về số thực hữu hạn."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} phải là số hữu hạn') from exc
    if not np.isfinite(value):
        raise ValueError(f'{name} phải là số hữu hạn')
    return value


def _as_tolerance(value):
    """Chuyển về dung sai không âm hữu hạn."""
    tolerance = _as_finite(value, 'tolerance')
    if tolerance < 0:
        raise ValueError('tolerance phải là số không âm hữu hạn')
    return tolerance


def _anchor(patch):
    """Tọa độ neo của vật cản: góc dưới-trái (Rectangle) hoặc tâm (Circle)."""
    if isinstance(patch, Rectangle):
        return np.asarray(patch.get_xy(), dtype=float)
    if isinstance(patch, Circle):
        return np.asarray(patch.center, dtype=float)
    raise TypeError('chỉ hỗ trợ matplotlib.patches.Rectangle hoặc Circle')


def _rectangle_corners(patch):
    """Bốn góc của Rectangle trong hệ dữ liệu, kể cả khi patch bị xoay."""
    vertices = patch.get_path().vertices[:4, :2]
    return patch.get_patch_transform().transform(vertices)


def _edges(polygon):
    return np.roll(polygon, -1, axis=0) - polygon


def _axes(polygon):
    """Pháp tuyến đơn vị của các cạnh (trục tách cho SAT)."""
    edges = _edges(polygon)
    normals = np.column_stack((edges[:, 1], -edges[:, 0]))
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > _TOL
    return normals[keep] / lengths[keep, None]


def _polygons_intersect(a, b):
    """Hai đa giác lồi giao nhau (hoặc tiếp xúc) hay không, dùng SAT."""
    for axis in np.vstack((_axes(a), _axes(b))):
        pa = a @ axis
        pb = b @ axis
        if pa.max() < pb.min() - _TOL or pb.max() < pa.min() - _TOL:
            return False
    return True


def _signed_area2(polygon):
    """Hai lần diện tích có dấu của đa giác (dùng để phát hiện suy biến)."""
    twice_area = 0.0
    for i, current in enumerate(polygon):
        following = polygon[(i + 1) % len(polygon)]
        twice_area += current[0] * following[1] - following[0] * current[1]
    return twice_area


def _contains(polygon, point):
    """Điểm nằm trong (hoặc trên cạnh) đa giác lồi, không phụ thuộc chiều."""
    twice_area = _signed_area2(polygon)
    sign = 1.0 if twice_area >= 0 else -1.0

    for i, start in enumerate(polygon):
        end = polygon[(i + 1) % len(polygon)]
        cross = ((end[0] - start[0]) * (point[1] - start[1])
                 - (end[1] - start[1]) * (point[0] - start[0]))
        if sign * cross < -_TOL:
            return False
    return True


def _distance_to_segment(point, start, end):
    segment = end - start
    length_squared = float(segment @ segment)
    if length_squared <= _TOL:
        return float(np.linalg.norm(point - start))
    t = np.clip((point - start) @ segment / length_squared, 0.0, 1.0)
    return float(np.linalg.norm(point - (start + t * segment)))


def _circle_intersects_polygon(center, radius, polygon):
    """Hình tròn giao (hoặc tiếp xúc) đa giác, kể cả khi chỉ cắt cạnh."""
    if _contains(polygon, center):
        return True
    for i, start in enumerate(polygon):
        end = polygon[(i + 1) % len(polygon)]
        if _distance_to_segment(center, start, end) <= radius + _TOL:
            return True
    return False


def _point_polygon_distance(point, polygon):
    if _contains(polygon, point):
        return 0.0
    return min(
        _distance_to_segment(point, start, polygon[(i + 1) % len(polygon)])
        for i, start in enumerate(polygon)
    )


def _rectangle_point_distance(point, patch):
    """Khoảng cách từ điểm tới Rectangle, kể cả hình suy biến (0 cạnh)."""
    polygon = _rectangle_corners(patch)
    if abs(_signed_area2(polygon)) <= _TOL:
        # Đa giác diện tích 0: _contains luôn đúng nên phải đo theo các cạnh.
        return min(
            _distance_to_segment(point, polygon[i], polygon[(i + 1) % 4])
            for i in range(4)
        )
    return _point_polygon_distance(point, polygon)


def _closest_point_on_segment(point, start, end):
    """Hình chiếu gần nhất của point lên đoạn start-end."""
    segment = end - start
    length_squared = float(segment @ segment)
    if length_squared <= _TOL:
        return start.copy()
    t = float(np.clip((point - start) @ segment / length_squared, 0.0, 1.0))
    return start + t * segment


def rectangle_corners(patch):
    """Bốn góc của Rectangle trong hệ dữ liệu (kể cả patch bị xoay)."""
    return _rectangle_corners(patch)


def point_polygon_distance(point, polygon):
    """Khoảng cách từ điểm tới đa giác (0 nếu nằm trong)."""
    return _point_polygon_distance(point, polygon)


def _interval(value, name, positive=False):
    try:
        low, high = (float(value[0]), float(value[1]))
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f'{name} phải là cặp (min, max)') from exc
    if not np.all(np.isfinite([low, high])) or high <= low:
        raise ValueError(f'{name} phải là cặp (min, max) hữu hạn với min < max')
    if positive and low <= 0:
        raise ValueError(f'{name} phải dương')
    return low, high


class Obstacle:
    """Quản lý vật cản tĩnh (Rectangle/Circle) và va chạm với robot."""

    def __init__(self):
        self.obs = []
        self._labels = []

    def add(self, new_obs):
        if not isinstance(new_obs, _SUPPORTED):
            raise TypeError('chỉ hỗ trợ matplotlib.patches.Rectangle hoặc Circle')
        self.obs.append(new_obs)

    def clear(self):
        self.clear_labels()
        self.obs.clear()

    def remove(self, obs_to_remove):
        if obs_to_remove in self.obs:
            self.obs.remove(obs_to_remove)
            self.clear_labels()

    def index_of(self, patch):
        """Chỉ số của patch theo identity trong danh sách, hoặc None."""
        for index, obs in enumerate(self.obs):
            if obs is patch:
                return index
        return None

    def _require_patch(self, patch):
        """Trả chỉ số của patch thuộc collection, lỗi nếu không hợp lệ."""
        if not isinstance(patch, _SUPPORTED):
            raise TypeError('chỉ hỗ trợ matplotlib.patches.Rectangle hoặc Circle')
        index = self.index_of(patch)
        if index is None:
            raise ValueError('patch không thuộc danh sách vật cản')
        return index

    def move_to(self, patch, x, y):
        """Đặt vị trí neo của vật cản, giữ nguyên góc xoay của Rectangle."""
        x = _as_finite(x, 'x')
        y = _as_finite(y, 'y')
        self._require_patch(patch)
        if isinstance(patch, Rectangle):
            patch.set_xy((x, y))
        else:
            patch.center = (x, y)
        return patch

    def translate(self, patch, dx, dy):
        """Dịch chuyển vật cản theo delta (dx, dy)."""
        dx = _as_finite(dx, 'dx')
        dy = _as_finite(dy, 'dy')
        self._require_patch(patch)
        anchor = _anchor(patch)
        return self.move_to(patch, anchor[0] + dx, anchor[1] + dy)

    def remove_at(self, index):
        """Xoá vật cản theo chỉ số và trả patch đã xoá."""
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError('index phải là số nguyên')
        index = int(index)
        if index < 0 or index >= len(self.obs):
            raise IndexError('index vật cản ngoài phạm vi')
        patch = self.obs.pop(index)
        self.clear_labels()
        return patch

    def resize(self, patch, *, radius=None, length=None, width=None):
        """Đổi kích thước, giữ tâm và góc xoay; dài/rộng theo trục cục bộ."""
        self._require_patch(patch)
        if isinstance(patch, Circle):
            if length is not None or width is not None:
                raise ValueError('hình tròn chỉ nhận bán kính')
            value = _as_finite(patch.radius if radius is None else radius, 'bán kính')
            if value < 0.1:
                raise ValueError('bán kính phải >= 0.1')
            patch.set_radius(value)
            return patch

        if radius is not None:
            raise ValueError('hình chữ nhật chỉ nhận chiều dài và chiều rộng')
        new_length = _as_finite(patch.get_width() if length is None else length, 'chiều dài')
        new_width = _as_finite(patch.get_height() if width is None else width, 'chiều rộng')
        if new_length < 0.2 or new_width < 0.2:
            raise ValueError('chiều dài và chiều rộng phải >= 0.2')
        if (new_length, new_width) == (patch.get_width(), patch.get_height()):
            return patch
        center = _rectangle_corners(patch).mean(axis=0)
        patch.set_width(new_length)
        patch.set_height(new_width)
        delta = center - _rectangle_corners(patch).mean(axis=0)
        # Với tâm xoay cố định, dịch neo bị quay theo góc của Rectangle.
        if not isinstance(patch.rotation_point, str):
            angle = np.deg2rad(patch.get_angle())
            cosine, sine = np.cos(angle), np.sin(angle)
            delta = np.array([[cosine, sine], [-sine, cosine]]) @ delta
        patch.set_xy(np.asarray(patch.get_xy()) + delta)
        return patch

    def pick(self, point, tolerance=0.0):
        """Chỉ số vật cản trên cùng chứa point hoặc cách biên <= tolerance."""
        point = _as_point(point)
        tolerance = _as_tolerance(tolerance)
        for index in range(len(self.obs) - 1, -1, -1):
            obs = self.obs[index]
            if isinstance(obs, Rectangle):
                distance = _rectangle_point_distance(point, obs)
                if distance <= tolerance + _TOL:
                    return index
            elif isinstance(obs, Circle):
                center = np.asarray(obs.center, dtype=float)
                distance = float(np.linalg.norm(point - center))
                if distance <= float(obs.radius) + tolerance + _TOL:
                    return index
        return None

    def generate_random_obstacles(self, count=5, area=(-10, 10),
                                  size=(1.0, 3.0), seed=None,
                                  start_clearance=1.0):
        """Sinh vật cản không chồng nhau và không chặn vị trí xuất phát."""
        count = int(count)
        if count < 0:
            raise ValueError('count phải là số nguyên không âm')
        low, high = _interval(area, 'area')
        size_low, size_high = _interval(size, 'size', positive=True)
        start_clearance = float(start_clearance)
        if not np.isfinite(start_clearance) or start_clearance < 0:
            raise ValueError('start_clearance phải là số không âm hữu hạn')

        rng = np.random.default_rng(seed)
        for _ in range(count):
            for _attempt in range(200):
                x, y = rng.uniform(low, high, size=2)
                width, height = rng.uniform(size_low, size_high, size=2)
                candidate = Rectangle((x, y), width, height, color='gray')
                corners = _rectangle_corners(candidate)

                if start_clearance and _point_polygon_distance(
                        np.zeros(2), corners) < start_clearance:
                    continue
                if any(
                    isinstance(obs, Rectangle)
                    and _polygons_intersect(corners, _rectangle_corners(obs))
                    for obs in self.obs
                ):
                    continue
                if any(
                    isinstance(obs, Circle)
                    and _circle_intersects_polygon(
                        np.asarray(obs.center, dtype=float), obs.radius, corners)
                    for obs in self.obs
                ):
                    continue
                self.add(candidate)
                break
            else:
                raise RuntimeError(
                    'không đặt được vật cản không chồng lấn; giảm count, '
                    'mở rộng area hoặc giảm size')
        return self.obs

    def clear_labels(self):
        """Xoá các text label do label_obstacles tạo."""
        for text in self._labels:
            try:
                text.remove()
            except (ValueError, NotImplementedError):
                pass
        self._labels.clear()

    def label_obstacles(self, ax):
        """Gắn label cho từng vật cản; gọi lại sẽ thay label cũ."""
        self.clear_labels()
        for index, obs in enumerate(self.obs):
            if isinstance(obs, Rectangle):
                x, y = obs.get_xy()
            else:
                x, y = obs.center
            text = ax.text(x, y, f'Obs {index}', fontsize=8, color='red')
            self._labels.append(text)

    def collides(self, corners):
        """Kiểm tra đa giác robot (4, 2) có chạm vật cản nào không."""
        robot = _as_polygon(corners, 'corners')
        for obs in self.obs:
            if isinstance(obs, Rectangle):
                if _polygons_intersect(robot, _rectangle_corners(obs)):
                    return True
            elif isinstance(obs, Circle):
                if _circle_intersects_polygon(
                        np.asarray(obs.center, dtype=float), obs.radius, robot):
                    return True
        return False

    def collides_pose(self, robot, x, y, theta):
        """Kiểm tra robot tại pose (x, y, theta) có va chạm không."""
        return self.collides(robot.get_robot_corners(x, y, theta))

    def collides_index(self, corners):
        """Trả chỉ số vật cản bị chạm đầu tiên, hoặc None."""
        robot = _as_polygon(corners, 'corners')
        for index, obs in enumerate(self.obs):
            if isinstance(obs, Rectangle):
                if _polygons_intersect(robot, _rectangle_corners(obs)):
                    return index
            elif isinstance(obs, Circle):
                if _circle_intersects_polygon(
                        np.asarray(obs.center, dtype=float), obs.radius, robot):
                    return index
        return None

    def contact_at(self, robot, x=None, y=None, theta=None):
        """Contact sensor: vật cản mà robot tại pose sắp chạm, hoặc None."""
        corners = robot.get_robot_corners(x, y, theta)
        return self.collides_index(corners)

    def contains(self, point):
        """Chỉ số vật cản chứa điểm (kể cả trên biên), hoặc None."""
        try:
            point = np.asarray(point, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError('point phải là cặp (x, y)') from exc
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError('point phải là cặp (x, y) hữu hạn')
        for index, obs in enumerate(self.obs):
            if isinstance(obs, Rectangle):
                if _contains(_rectangle_corners(obs), point):
                    return index
            elif isinstance(obs, Circle):
                center = np.asarray(obs.center, dtype=float)
                if np.linalg.norm(point - center) <= obs.radius + _TOL:
                    return index
        return None

    def nearest(self, point):
        """Trả (index, điểm gần nhất trên biên, khoảng cách, pháp tuyến ra ngoài).

        Nếu chưa có vật cản, trả (None, None, inf, None). Khoảng cách bằng 0
        khi điểm nằm trong vật cản.
        """
        try:
            point = np.asarray(point, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError('point phải là cặp (x, y)') from exc
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError('point phải là cặp (x, y) hữu hạn')

        best = (None, None, float('inf'), None)
        for index, obs in enumerate(self.obs):
            if isinstance(obs, Rectangle):
                polygon = _rectangle_corners(obs)
                centroid = polygon.mean(axis=0)
                inside = _contains(polygon, point)
                closest = None
                distance = float('inf')
                for i, start in enumerate(polygon):
                    end = polygon[(i + 1) % len(polygon)]
                    candidate = _closest_point_on_segment(point, start, end)
                    candidate_distance = float(np.linalg.norm(point - candidate))
                    if candidate_distance < distance:
                        closest = candidate
                        distance = candidate_distance
                if inside:
                    distance = 0.0
                    closest = point.copy()
                normal = closest - centroid
                norm = float(np.linalg.norm(normal))
                normal = normal / norm if norm > _TOL else np.array([1.0, 0.0])
            else:
                center = np.asarray(obs.center, dtype=float)
                radius = float(obs.radius)
                delta = point - center
                norm = float(np.linalg.norm(delta))
                if norm <= _TOL:
                    normal = np.array([1.0, 0.0])
                else:
                    normal = delta / norm
                closest = center + radius * normal
                distance = max(0.0, norm - radius)

            if distance < best[2]:
                best = (index, closest, float(distance), normal)
        return best
