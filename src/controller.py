"""Điều khiển độc lập với GUI."""
import numpy as np


def motion_step(state, velocity, dt, noise_std=(0., 0.), rng=None):
    """Vận tốc lệnh → nhiễu → Euler; trả trạng thái mới."""
    v, w = velocity
    if np.any(np.asarray(noise_std) > 0):
        if rng is None:
            rng = np.random.default_rng()
        noise_v, noise_w = rng.normal(0., noise_std)
        v += noise_v
        w += noise_w
    x, y, theta = state
    return np.array([x + dt*v*np.cos(theta), y + dt*v*np.sin(theta), theta + dt*w])


class RobotController:
    def __init__(self):
        self.state = np.zeros(3)
        self.time = 0.
        self.speed = 0.
        self.turn_remaining = 0.
        self.noise_std = (0., 0.)
        self.rng = np.random.default_rng()
        self.velocity = (0., 0.)

    def drive(self, direction):
        self.speed = .1 * direction
        self.turn_remaining = 0.

    def turn(self, direction):
        self.turn_remaining += direction * np.deg2rad(15)

    def stop(self):
        self.speed = self.turn_remaining = 0.
        self.velocity = (0., 0.)

    def tick(self, dt=.02):
        w = np.clip(self.turn_remaining / dt, -np.pi/4, np.pi/4)
        self.velocity = (self.speed, float(w))
        if self.speed != 0 or w != 0:
            self.state = motion_step(self.state, self.velocity, dt, self.noise_std, self.rng)
        self.turn_remaining -= w * dt
        if abs(self.turn_remaining) < 1e-12:
            self.turn_remaining = 0.
        self.time += dt
        return self.state.copy()
