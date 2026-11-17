"""Robot vi sai cho Bài thực hành 1; đơn vị mét và radian."""
import numpy as np
import matplotlib.transforms as transforms
from matplotlib.patches import FancyArrowPatch, Rectangle

ARROW_SCALE = 1.0
HEADING_SCALE = 0.6


def _positive(value, name):
    """Chuyển về float dương hữu hạn, nếu không hợp lệ thì báo ValueError."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} phải là số dương hữu hạn') from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} phải là số dương hữu hạn')
    return value


class Differential_Robot:
    """Robot hai bánh vi sai: vận tốc bánh → vận tốc thân → pose mới."""

    def __init__(self, L, robot_length, robot_width, wheel_length, wheel_width):
        self.L = _positive(L, 'L')
        self.robot_length = _positive(robot_length, 'robot_length')
        self.robot_width = _positive(robot_width, 'robot_width')
        self.wheel_length = _positive(wheel_length, 'wheel_length')
        self.wheel_width = _positive(wheel_width, 'wheel_width')

        # Pose và vận tốc
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v_l = 0.0
        self.v_r = 0.0
        self.v = 0.0
        self.omega = 0.0

        # Lịch sử quỹ đạo (điểm duy nhất lưu history, GUI đọc lại)
        self.x_data = []
        self.y_data = []
        self.theta_data = []

        # Hình họa: thân xe và hai bánh
        self.robot_body = Rectangle(
            (-self.robot_length / 2, -self.robot_width / 2),
            self.robot_length, self.robot_width,
            fill=False, edgecolor='black', linewidth=2,
        )
        self.left_wheel = Rectangle(
            (-self.wheel_length / 2, -self.wheel_width / 2),
            self.wheel_length, self.wheel_width, color='black',
        )
        self.right_wheel = Rectangle(
            (-self.wheel_length / 2, -self.wheel_width / 2),
            self.wheel_length, self.wheel_width, color='black',
        )

        # Mũi tên hướng xe và vận tốc hai bánh
        self.heading_arrow = FancyArrowPatch(
            (0, 0), (0, 0), arrowstyle='->', color='red', mutation_scale=15,
        )
        self.left_arrow = FancyArrowPatch(
            (0, 0), (0, 0), arrowstyle='->', color='green', mutation_scale=10,
        )
        self.right_arrow = FancyArrowPatch(
            (0, 0), (0, 0), arrowstyle='->', color='orange', mutation_scale=10,
        )

    # ------------------------------------------------------------------
    # Động học
    # ------------------------------------------------------------------
    @property
    def pose(self):
        return self.x, self.y, self.theta

    def set_wheel_velocities(self, v_l, v_r):
        """Gán vận tốc dài của bánh trái/phải (m/s)."""
        self.v_l = float(v_l)
        self.v_r = float(v_r)

    def update_pose(self, dt):
        """Tích phân Euler một bước và ghi lịch sử."""
        dt = _positive(dt, 'dt')
        self.v = (self.v_l + self.v_r) / 2
        self.omega = (self.v_r - self.v_l) / self.L

        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta = self.normalize_angle(self.theta + self.omega * dt)

        self.x_data.append(self.x)
        self.y_data.append(self.y)
        self.theta_data.append(self.theta)

    def predict_pose(self, dt):
        """Trả pose kế tiếp mà không thay đổi trạng thái hiện tại."""
        dt = _positive(dt, 'dt')
        v = (self.v_l + self.v_r) / 2
        omega = (self.v_r - self.v_l) / self.L
        x = self.x + v * np.cos(self.theta) * dt
        y = self.y + v * np.sin(self.theta) * dt
        theta = self.normalize_angle(self.theta + omega * dt)
        return x, y, theta

    def reset(self):
        """Đưa robot về gốc tọa độ, dừng lại và xóa lịch sử."""
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v_l = 0.0
        self.v_r = 0.0
        self.v = 0.0
        self.omega = 0.0
        self.x_data.clear()
        self.y_data.clear()
        self.theta_data.clear()

    @staticmethod
    def normalize_angle(angle):
        """Chuẩn hóa góc về [-pi, pi)."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # ------------------------------------------------------------------
    # Hình họa
    # ------------------------------------------------------------------
    def update_graphics(self, ax):
        """Cập nhật transform của thân xe, bánh xe và các mũi tên."""
        dx = np.cos(self.theta)
        dy = np.sin(self.theta)
        perp_dx = -dy
        perp_dy = dx
        half_L = self.L / 2

        body_transform = (
            transforms.Affine2D().rotate(self.theta).translate(self.x, self.y)
            + ax.transData
        )
        self.robot_body.set_transform(body_transform)

        left_x = self.x + half_L * perp_dx
        left_y = self.y + half_L * perp_dy
        right_x = self.x - half_L * perp_dx
        right_y = self.y - half_L * perp_dy

        self.left_wheel.set_transform(
            transforms.Affine2D().rotate(self.theta).translate(left_x, left_y)
            + ax.transData
        )
        self.right_wheel.set_transform(
            transforms.Affine2D().rotate(self.theta).translate(right_x, right_y)
            + ax.transData
        )

        heading = HEADING_SCALE * self.robot_length
        self.heading_arrow.set_positions(
            (self.x, self.y), (self.x + heading * dx, self.y + heading * dy),
        )
        self.left_arrow.set_positions(
            (left_x, left_y),
            (left_x + ARROW_SCALE * self.v_l * dx,
             left_y + ARROW_SCALE * self.v_l * dy),
        )
        self.right_arrow.set_positions(
            (right_x, right_y),
            (right_x + ARROW_SCALE * self.v_r * dx,
             right_y + ARROW_SCALE * self.v_r * dy),
        )
        return (self.robot_body, self.left_wheel, self.right_wheel,
                self.heading_arrow, self.left_arrow, self.right_arrow)

    def get_robot_corners(self, x=None, y=None, theta=None):
        """Trả bốn góc thân xe, ngược chiều kim đồng hồ, dạng (4, 2).

        Thứ tự: sau-phải, trước-phải, trước-trái, sau-trái. Nếu bỏ trống
        cả ba tham số thì dùng pose hiện tại.
        """
        if x is None and y is None and theta is None:
            x, y, theta = self.x, self.y, self.theta
        elif x is None or y is None or theta is None:
            raise ValueError('cần truyền đủ x, y, theta hoặc bỏ trống cả ba')

        try:
            x, y, theta = float(x), float(y), float(theta)
        except (TypeError, ValueError) as exc:
            raise ValueError('x, y, theta phải là số hữu hạn') from exc
        if not np.all(np.isfinite([x, y, theta])):
            raise ValueError('x, y, theta phải là số hữu hạn')

        half_l = self.robot_length / 2
        half_w = self.robot_width / 2
        corners = np.array([
            [-half_l, -half_w],
            [half_l, -half_w],
            [half_l, half_w],
            [-half_l, half_w],
        ])
        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        return corners @ rotation.T + np.array([x, y])
