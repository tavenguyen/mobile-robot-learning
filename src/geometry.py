"""Hình học robot trong mặt phẳng; đơn vị mét và radian."""
import numpy as np


def validate_dimensions(length, width):
    try:
        length, width = float(length), float(width)
    except (TypeError, ValueError) as exc:
        raise ValueError('length và width phải là số dương hữu hạn') from exc
    if not np.all(np.isfinite([length, width])) or min(length, width) <= 0:
        raise ValueError('length và width phải là số dương hữu hạn')
    return length, width


def transform_points(points, state):
    """Quay tọa độ cục bộ theo theta rồi tịnh tiến tới (x, y)."""
    x, y, theta = state
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    return np.asarray(points) @ rotation.T + [x, y]


def rectangle_corners(cx, cy, length, width):
    """Các góc sau-phải, trước-phải, trước-trái, sau-trái."""
    return np.array([[cx-length/2, cy-width/2], [cx+length/2, cy-width/2],
                     [cx+length/2, cy+width/2], [cx-length/2, cy+width/2]])


def robot_polygons(length, width):
    """Trả thân xe và bốn bánh trong hệ tọa độ cục bộ."""
    polygons = [rectangle_corners(0, 0, length, width)]
    for cy in [-.56*width, .56*width]:
        polygons.append(rectangle_corners(0, cy, .22*length, .12*width))
    return polygons


def get_robot_corners(state, length=0.05, width=0.05, *, include_wheels=True):
    """Trả footprint (4, 2) trong hệ bản đồ, ngược chiều kim đồng hồ.

    state=(x,y,theta), đơn vị m và rad. length/width là kích thước thân.
    Mặc định bao cả bánh; không bao các mũi tên chỉ hướng.
    """
    length, width = validate_dimensions(length, width)
    try:
        state = np.asarray(state, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('state phải gồm ba số hữu hạn (x, y, theta)') from exc
    if state.shape != (3,) or not np.all(np.isfinite(state)):
        raise ValueError('state phải gồm ba số hữu hạn (x, y, theta)')
    if not isinstance(include_wheels, (bool, np.bool_)):
        raise ValueError('include_wheels phải là boolean')
    polygons = robot_polygons(length, width)
    points = np.vstack(polygons if include_wheels else polygons[:1])
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    cx, cy = (lower + upper) / 2
    footprint = rectangle_corners(cx, cy, *(upper - lower))
    return transform_points(footprint, state)
