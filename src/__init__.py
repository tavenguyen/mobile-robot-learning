"""Các hàm mô phỏng và hiển thị quỹ đạo robot 2D."""
from .simulation import simulate
from .geometry import get_robot_corners
from .visualization import plot_trajectory, animate_trajectory

__all__ = ["simulate", "plot_trajectory", "animate_trajectory", "get_robot_corners"]
