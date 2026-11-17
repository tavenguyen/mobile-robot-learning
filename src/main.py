"""Điểm chạy mô phỏng Bài thực hành 1 & 2.1: python -m src.main."""
import tkinter as tk

import matplotlib.patches as patches

if __package__:
    from .GUI import RobotSimulatorGUI
    from .Obstacle import Obstacle
    from .Robot import Differential_Robot
else:
    from GUI import RobotSimulatorGUI
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


def main():
    robot, obstacles = build_scene()
    root = tk.Tk()
    root.title('Differential Drive Robot Simulation')
    RobotSimulatorGUI(root, robot, obstacles, dt=DT)
    root.mainloop()


if __name__ == '__main__':
    main()
    