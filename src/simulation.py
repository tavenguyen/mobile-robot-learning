"""Mô phỏng chất điểm có hướng trong mặt phẳng (m, s, rad)."""
from __future__ import annotations

import numpy as np
from collections.abc import Callable

if __package__:
    from .controller import motion_step
else:
    from controller import motion_step


def _vector(value, size, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} phải gồm {size} số hữu hạn") from exc
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} phải gồm {size} số hữu hạn")
    return result


def simulate(
    velocity: tuple[float, float] | Callable[[float], tuple[float, float]],
    duration: float = 10.0,
    dt: float = 0.02,
    initial_state: tuple[float, float, float] = (0., 0., 0.),
    *,
    noise_std: tuple[float, float] = (0., 0.),
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mô phỏng chuyển động với nhiễu vận tốc Gaussian ở mỗi bước.

    velocity: cặp (v, w), hoặc hàm velocity(t) trả về cặp đó.
    noise_std: độ lệch chuẩn (sigma_v, sigma_w), đơn vị m/s và rad/s.
    seed: cố định chuỗi ngẫu nhiên để chạy lại được cùng kết quả.
    Trả về times và states, mỗi hàng states là (x, y, theta).
    """
    try:
        duration, dt = float(duration), float(dt)
    except (TypeError, ValueError) as exc:
        raise ValueError("duration và dt phải là số") from exc
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("duration phải hữu hạn và không âm")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt phải hữu hạn và dương")
    initial = _vector(initial_state, 3, "initial_state")
    noise_std = _vector(noise_std, 2, "noise_std")
    if np.any(noise_std < 0):
        raise ValueError("noise_std phải không âm")
    if seed is not None and (isinstance(seed, (bool, np.bool_))
                             or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed phải là số nguyên không âm hoặc None")
    rng = np.random.default_rng(seed)
    if callable(velocity):
        constant_velocity = None
    else:
        constant_velocity = _vector(velocity, 2, "velocity")

    # Giữ bước dt; chỉ bước cuối được rút ngắn.
    times = np.arange(0., duration, dt)
    times = np.append(times, duration)
    states = np.zeros((len(times), 3))
    states[0] = initial
    for i in range(len(times) - 1):
        current_time = times[i]
        step_dt = times[i + 1] - current_time

        # 1. Lấy vận tốc mong muốn tại đầu bước thời gian.
        if constant_velocity is not None:
            v_command, w_command = constant_velocity
        else:
            v_command, w_command = _vector(velocity(current_time), 2, "velocity(t)")

        # Shared update: command velocity -> random noise -> Euler equations.
        states[i + 1] = motion_step(
            states[i], (v_command, w_command), step_dt, noise_std, rng,
        )
    return times, states
