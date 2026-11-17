"""Luật điều khiển auto-mode tới goal cho robot vi sai (BTH2.1)."""
import math

GOAL_TOLERANCE = 0.3
K_V = 1.0
K_OMEGA = 2.0
MAX_V = 2.0
MAX_OMEGA = 2.0


def _validate_goal(goal):
    try:
        x, y = float(goal[0]), float(goal[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError('goal phải là cặp (x, y) hữu hạn') from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError('goal phải là cặp (x, y) hữu hạn')
    return x, y


def goal_command(robot, goal, *, tolerance=GOAL_TOLERANCE, k_v=K_V,
                 k_omega=K_OMEGA, max_v=MAX_V, max_omega=MAX_OMEGA):
    """Trả (v_l, v_r, distance, arrived) để lái robot tới goal.

    Quay tại chỗ khi lệch hướng lớn, giảm tốc khi gần tới, clip vận tốc
    theo giới hạn của bài thực hành.
    """
    x_goal, y_goal = _validate_goal(goal)
    dx = x_goal - robot.x
    dy = y_goal - robot.y
    distance = math.hypot(dx, dy)

    if distance < tolerance:
        return 0.0, 0.0, distance, True

    desired_theta = math.atan2(dy, dx)
    angle_error = robot.normalize_angle(desired_theta - robot.theta)

    omega = max(-max_omega, min(max_omega, k_omega * angle_error))
    slow_factor = 1.0 - min(abs(angle_error), math.pi) / (math.pi / 2)
    v = max(0.0, min(max_v, k_v * distance * slow_factor))

    half_track = robot.L / 2
    return v - half_track * omega, v + half_track * omega, distance, False


def point_command(robot, point, *, tolerance=0.1, max_v=1.5,
                  k_omega=2.5, max_omega=2.5):
    """Lái robot tới một điểm bất kỳ (waypoint), trả (v_l, v_r, distance, arrived).

    Khác goal_command: không giảm tốc theo khoảng cách mà quay tại chỗ khi
    lệch hướng lớn, phù hợp để bám biên hoặc đi theo tia tiếp tuyến.
    """
    x_point, y_point = _validate_goal(point)
    dx = x_point - robot.x
    dy = y_point - robot.y
    distance = math.hypot(dx, dy)
    if distance < tolerance:
        return 0.0, 0.0, distance, True

    desired_theta = math.atan2(dy, dx)
    angle_error = robot.normalize_angle(desired_theta - robot.theta)
    omega = max(-max_omega, min(max_omega, k_omega * angle_error))
    v = max_v * max(0.0, math.cos(angle_error))

    half_track = robot.L / 2
    return v - half_track * omega, v + half_track * omega, distance, False
