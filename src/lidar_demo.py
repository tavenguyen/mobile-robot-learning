"""Xem mô phỏng LiDAR riêng, độc lập với mô phỏng né vật cản.

Chạy từ thư mục gốc dự án::

    python -m src.lidar_demo
    python -m src.lidar_demo --beams 360
    python -m src.lidar_demo --animate
    python -m src.lidar_demo --pose 2 -3 0.5 --noise 0.05 --save lidar.png
"""
import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle

if __package__:
    from .lidar import Lidar
    from .Obstacle import Obstacle
else:
    from lidar import Lidar
    from Obstacle import Obstacle


def build_scene():
    """Vài vật cản tĩnh mẫu trong khung toàn cục."""
    obstacles = Obstacle()
    obstacles.add(Rectangle((-6, -4), 3, 2, color='gray'))
    obstacles.add(Circle((5, 3), 1.5, color='gray'))
    obstacles.add(Rectangle(
        (2, -6), 5, 1.2, angle=25, color='dimgray'))
    obstacles.add(Circle((-4, 5), 1.0, color='gray'))
    return obstacles


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Mô phỏng LiDAR 2D và vẽ khung global/local.')
    parser.add_argument(
        '--beams', type=int, default=72,
        help='số tia (mặc định 72; dùng 360 để xem mịn)')
    parser.add_argument(
        '--max-range', type=float, default=12.0,
        help='tầm quét tối đa tính bằng mét (mặc định 12)')
    parser.add_argument(
        '--noise', type=float, default=0.02,
        help='độ lệch chuẩn nhiễu khoảng cách, mét (mặc định 0.02)')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='seed ngẫu nhiên (mặc định 42)')
    parser.add_argument(
        '--pose', nargs=3, type=float, default=(1.0, -1.5, 0.5),
        metavar=('X', 'Y', 'THETA'),
        help='pose tĩnh x y theta (mặc định 1 -1.5 0.5)')
    parser.add_argument(
        '--animate', action='store_true',
        help='cho robot chạy theo quỹ đạo tròn và quét liên tục')
    parser.add_argument(
        '--save', metavar='PATH',
        help='lưu ảnh PNG ở chế độ tĩnh thay vì mở cửa sổ')
    return parser.parse_args(argv)


def _make_lidar(args):
    return Lidar(
        build_scene(),
        num_beams=args.beams,
        max_range=args.max_range,
        noise_std=args.noise,
        seed=args.seed,
    )


def show_static(lidar, pose, save=None):
    scan = lidar.scan(*pose)
    fig, axes = lidar.plot_frames(scan)
    fig.suptitle(
        f'LiDAR: {lidar.num_beams} tia, tầm {lidar.max_range:g} m, '
        f'nhiễu {lidar.noise_std:g} m',
        fontsize=12,
    )
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
        print(f'Đã lưu ảnh: {save}')
        plt.close(fig)
    else:
        plt.show()
    return fig, axes


def show_animation(lidar, frames=240, interval=60):
    fig, (ax_global, ax_local) = plt.subplots(1, 2, figsize=(12, 5.5))

    def update(frame):
        time = frame / 40.0
        x = 2.5 * math.cos(time)
        y = 2.5 * math.sin(time)
        theta = time + math.pi / 2
        scan = lidar.scan(x, y, theta)
        ax_global.clear()
        ax_local.clear()
        lidar.plot_frames(scan, ax_global=ax_global, ax_local=ax_local)
        fig.suptitle(
            f'LiDAR động: khung {frame + 1}/{frames} — '
            f'pose ({x:.2f}, {y:.2f}, {theta:.2f})',
            fontsize=11,
        )
        return ()

    fig.tight_layout()
    animation = FuncAnimation(
        fig, update, frames=frames, interval=interval,
        blit=False, repeat=True,
    )
    plt.show()
    return animation


_ANIMATION = None


def main(argv=None):
    global _ANIMATION
    args = _parse_args(argv)
    if args.animate and args.save:
        raise SystemExit('--save chỉ dùng cho chế độ tĩnh, không dùng với --animate')
    lidar = _make_lidar(args)
    if args.animate:
        _ANIMATION = show_animation(lidar)
    else:
        show_static(lidar, args.pose, args.save)


if __name__ == '__main__':
    main()
