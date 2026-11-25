"""Chạy mô phỏng robot click-to-target kèm LiDAR ở khung cục bộ robot."""
if __package__:
    from .main import main
else:
    from main import main


if __name__ == '__main__':
    main(['--lidar'])
