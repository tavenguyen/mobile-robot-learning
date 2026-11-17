"""Hình xe nhìn từ trên xuống; các tọa độ cục bộ tính bằng mét."""
import numpy as np
from matplotlib.patches import Polygon, FancyArrowPatch


if __package__:
    from .geometry import transform_points, rectangle_corners, robot_polygons, validate_dimensions
else:
    from geometry import transform_points, rectangle_corners, robot_polygons, validate_dimensions

_rectangle = rectangle_corners


def create_robot(ax, length=0.05, width=0.05):
    """Tạo hình một lần; trả về các artist và hình học dùng để cập nhật."""
    length, width = validate_dimensions(length, width)
    body_corners, *wheel_corners = robot_polygons(length, width)

    body = Polygon(body_corners, facecolor='red',
                   edgecolor='darkred', alpha=.5, label='Robot', zorder=3)
    ax.add_patch(body)
    polygons = [(body, body_corners)]
    arrows = []
    for corners in wheel_corners:
        cx, cy = corners.mean(axis=0)
        wheel = Polygon(corners, color='black', zorder=4)
        ax.add_patch(wheel)
        polygons.append((wheel, corners))
        endpoints = np.array([[cx, cy], [cx+.35*length, cy]])
        arrow = FancyArrowPatch(endpoints[0], endpoints[1], color='green',
                                arrowstyle='->', mutation_scale=10,
                                shrinkA=0, shrinkB=0, zorder=5)
        ax.add_patch(arrow)
        arrows.append((arrow, endpoints))
    points = np.vstack([p for _, p in polygons] + [p for _, p in arrows])
    extent = np.linalg.norm(points, axis=1).max()
    return dict(polygons=polygons, arrows=arrows, extent=extent)


def update_robot(robot, state):
    """Giữ nguyên artist, chỉ cập nhật hình học theo trạng thái mới."""
    artists = []
    for patch, corners in robot['polygons']:
        patch.set_xy(transform_points(corners, state))
        artists.append(patch)
    for arrow, endpoints in robot['arrows']:
        start, end = transform_points(endpoints, state)
        arrow.set_positions(start, end)
        artists.append(arrow)
    return artists
