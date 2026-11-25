"""Hiển thị quỹ đạo và hoạt ảnh bằng Matplotlib."""
import numpy as np

if __package__:
    from .robot_drawing import create_robot, update_robot
else:
    from robot_drawing import create_robot, update_robot


def _axes(states, margin=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set(xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    if margin is None:
        margin = max(.2, np.ptp(states[:, :2], axis=0).max() * .1)
    ax.set_xlim(states[:, 0].min() - margin, states[:, 0].max() + margin)
    ax.set_ylim(states[:, 1].min() - margin, states[:, 1].max() + margin)
    return fig, ax


def plot_trajectory(times, states):
    """Tạo đồ thị; trả về (fig, ax), không tự gọi show()."""
    fig, ax = _axes(states)
    ax.plot(states[:, 0], states[:, 1], label="Trajectory")
    ax.plot(*states[0, :2], "go", label="Start")
    ax.plot(*states[-1, :2], "rx", label="End")
    ax.set_title(f"Trajectory: {times[-1]:g} s")
    ax.legend()
    return fig, ax


def animate_trajectory(times, states, *, length=0.05, width=0.05):
    """Trả về (fig, animation); giữ animation sống đến khi đóng cửa sổ."""
    from matplotlib.animation import FuncAnimation
    fig, ax = _axes(states)
    trail, = ax.plot([], [], "b-", label="Path")
    try:
        robot = create_robot(ax, length, width)
    except ValueError:
        import matplotlib.pyplot as plt
        plt.close(fig)
        raise
    margin = robot['extent'] + max(.2, robot['extent'] * .1)
    ax.set_xlim(states[:, 0].min() - margin, states[:, 0].max() + margin)
    ax.set_ylim(states[:, 1].min() - margin, states[:, 1].max() + margin)
    update_robot(robot, states[0])
    ax.legend(loc="upper right")
    label = ax.text(.02, .95, "", transform=ax.transAxes)
    interval = max(1, round((times[1] - times[0]) * 1000)) if len(times) > 1 else 100
    def update(i):
        trail.set_data(states[:i+1, 0], states[:i+1, 1])
        robot_artists = update_robot(robot, states[i])
        label.set_text(f"t = {times[i]:.2f} s")
        if i < len(times) - 1:
            # TimedAnimation đặt lại timer từ _interval sau mỗi callback.
            # Cập nhật giá trị nguồn để bước cuối ngắn hơn được phát đúng nhịp.
            animation._interval = max(1, round((times[i+1] - times[i]) * 1000))
        return [trail, *robot_artists, label]
    animation = FuncAnimation(fig, update, frames=len(times), interval=interval,
                              repeat=False, blit=False)
    return fig, animation
