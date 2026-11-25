"""Điểm chạy mô phỏng Bài thực hành 1 & 2.1: python -m src.main [--lidar]."""
import argparse
import tkinter as tk

import matplotlib.patches as patches

if __package__:
    from .GUI import RobotSimulatorGUI
    from .lidar import Lidar
    from .Obstacle import Obstacle
    from .Robot import Differential_Robot
else:
    from GUI import RobotSimulatorGUI
    from lidar import Lidar
    from Obstacle import Obstacle
    from Robot import Differential_Robot

DT = 0.1
L = 1.0


def build_scene():
    """Tạo robot và ba vật cản tĩnh như main script của bài thực hành."""
    robot = Differential_Robot(
        L, robot_length=2.0, robot_width=1.0,
        wheel_length=0.5, wheel_width=0.5,
    )
    obstacles = Obstacle()
    obstacles.add(patches.Rectangle((-5, -5), 3, 3, color='gray'))
    obstacles.add(patches.Circle((10, 10), 2, color='gray'))
    obstacles.add(patches.Rectangle((5, -10), 6, 2, color='dimgray'))
    return robot, obstacles


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Mô phỏng robot vi sai (tùy chọn kèm LiDAR khung cục bộ).')
    parser.add_argument(
        '--lidar', action='store_true',
        help='hiển thị thêm khung LiDAR Scan (Robot Frame) theo tọa độ cục bộ')
    parser.add_argument('--beams', type=int, default=120, help='số tia LiDAR')
    parser.add_argument('--max-range', type=float, default=15.0,
                        help='tầm quét LiDAR tối đa (m)')
    parser.add_argument('--noise', type=float, default=0.02,
                        help='độ lệch chuẩn nhiễu LiDAR (m)')
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    robot, obstacles = build_scene()
    lidar = None
    if args.lidar:
        lidar = Lidar(
            obstacles, num_beams=args.beams,
            max_range=args.max_range, noise_std=args.noise,
        )
    root = tk.Tk()
    root.title('Differential Drive Robot Simulation')
    RobotSimulatorGUI(root, robot, obstacles, dt=DT, lidar=lidar)
    root.mainloop()


if __name__ == '__main__':
    main()
    