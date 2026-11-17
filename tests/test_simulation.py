import unittest
import numpy as np
from src import simulate, plot_trajectory, animate_trajectory

class SimulationTests(unittest.TestCase):
    def test_robot_geometry(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from src.robot_drawing import create_robot, update_robot, transform_points
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        robot = create_robot(ax, length=1.2, width=.6)
        self.assertEqual(len(robot['polygons']), 3)
        self.assertEqual(len(robot['arrows']), 2)
        for theta in [0, np.pi/2, .7]:
            state = (2, 3, theta)
            artists = update_robot(robot, state)
            self.assertEqual(len(artists), 5)
            corners = robot['polygons'][0][0].get_xy()[:4]
            np.testing.assert_allclose(corners.mean(axis=0), [2, 3])
            self.assertAlmostEqual(np.linalg.norm(corners[1]-corners[0]), 1.2)
            self.assertAlmostEqual(np.linalg.norm(corners[2]-corners[1]), .6)
            np.testing.assert_allclose(corners[1]-corners[0],
                                       1.2*np.array([np.cos(theta), np.sin(theta)]), atol=1e-12)
            for _, endpoints in robot['arrows']:
                transformed = transform_points(endpoints, state)
                np.testing.assert_allclose(transformed[1]-transformed[0],
                                           .42*np.array([np.cos(theta), np.sin(theta)]), atol=1e-12)
        for length, width in [(0, 1), (-1, 1), (1, np.nan), (np.inf, 1)]:
            with self.assertRaises(ValueError):
                create_robot(ax, length, width)

    def test_noise_reproducibility(self):
        options = dict(duration=.7, dt=.3, noise_std=(.1, .2))
        _, a = simulate((1, .5), seed=42, **options)
        _, b = simulate((1, .5), seed=42, **options)
        _, c = simulate((1, .5), seed=43, **options)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        _, ideal = simulate((1, .5))
        _, zero = simulate((1, .5), noise_std=(0, 0), seed=42)
        np.testing.assert_array_equal(ideal, zero)

    def test_noise_components_and_inputs(self):
        command = np.array([1., 0.])
        initial = np.array([0., 0., 0.])
        _, s = simulate(lambda t: command, initial_state=initial,
                        noise_std=(.1, 0), seed=42)
        np.testing.assert_array_equal(s[:, 1:], 0)
        np.testing.assert_array_equal(command, [1, 0])
        np.testing.assert_array_equal(initial, [0, 0, 0])
        _, s = simulate((0, 0), noise_std=(0, .2), seed=42)
        np.testing.assert_array_equal(s[:, :2], 0)
        self.assertTrue(np.any(s[:, 2] != 0))
        def unused(t):
            raise AssertionError('callback should not run')
        _, s = simulate(unused, duration=0, noise_std=(1, 1), seed=42)
        np.testing.assert_array_equal(s, [[0, 0, 0]])

    def test_invalid_noise(self):
        for noise in [(-1, 0), (1,), (np.nan, 0), (0, np.inf)]:
            with self.assertRaises(ValueError):
                simulate((0, 0), noise_std=noise)
        for seed in [-1, 1.5, True, '42']:
            with self.assertRaises(ValueError):
                simulate((0, 0), seed=seed)

    def test_stationary_and_rotation(self):
        _, s = simulate((0, 0), initial_state=(1, 2, 3))
        np.testing.assert_allclose(s, np.tile([1, 2, 3], (len(s), 1)))
        _, s = simulate((0, 2), duration=2)
        np.testing.assert_allclose(s[-1], [0, 0, 4])

    def test_straight(self):
        _, s = simulate((2, 0), duration=3, initial_state=(0, 0, np.pi/2))
        np.testing.assert_allclose(s[-1, :2], [0, 6], atol=1e-12)

    def test_circle_convergence(self):
        exact = [np.sin(1), 1-np.cos(1)]
        errors = []
        for dt in [.02, .01]:
            _, s = simulate((1, 1), duration=1, dt=dt)
            errors.append(np.linalg.norm(s[-1, :2]-exact))
        self.assertLess(errors[0], .011)
        self.assertLess(errors[1], errors[0])

    def test_callback_and_last_step(self):
        calls = []
        def velocity(t):
            calls.append(t)
            return (1 if t < .3 else 2, 0)
        t, s = simulate(velocity, duration=.7, dt=.3)
        np.testing.assert_allclose(t, [0, .3, .6, .7])
        np.testing.assert_allclose(calls, t[:-1])
        self.assertAlmostEqual(s[-1, 0], 1.1)
        t, s = simulate((1, 1), duration=0)
        self.assertEqual(s.shape, (1, 3))

    def test_invalid(self):
        for kwargs in [dict(dt=0), dict(duration=-1), dict(dt=float('nan')),
                       dict(initial_state=(0, 0)), dict(velocity=(1, float('inf'))),
                       dict(velocity=lambda t: (1,))]:
            with self.assertRaises(ValueError):
                simulate(**({'velocity': (0, 0)} | kwargs))

    def test_plotting(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        t, s = simulate((1, 1), duration=.1)
        fig, ax = plot_trajectory(t, s)
        fig.canvas.draw()
        fig, animation = animate_trajectory(t, s)
        fig.canvas.draw()
        for _ in t:
            animation._step()
        plt.close('all')

    def test_animation_timing_and_artists(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for duration, dt, expected_intervals in [
            (.7, .3, [300, 300, 100, 100]),
            (.6, .3, [300, 300, 300]),
            (0., .3, [100]),
        ]:
            with self.subTest(duration=duration):
                times, states = simulate((1, .5), duration=duration, dt=dt)
                fig, animation = animate_trajectory(times, states)
                self.addCleanup(plt.close, fig)
                ax = fig.axes[0]
                self.assertEqual(
                    [text.get_text() for text in ax.get_legend().get_texts()],
                    ['Path', 'Robot'],
                )
                fig.canvas.draw()
                path, = ax.lines
                body = ax.patches[0]
                patch_count = len(ax.patches)
                for i, expected_interval in enumerate(expected_intervals):
                    self.assertTrue(animation._step())
                    self.assertEqual(animation.event_source.interval, expected_interval)
                    np.testing.assert_allclose(path.get_xdata(), states[:i+1, 0])
                    np.testing.assert_allclose(path.get_ydata(), states[:i+1, 1])
                    np.testing.assert_allclose(body.get_xy()[:4].mean(axis=0), states[i, :2], atol=1e-12)
                    self.assertEqual(len(ax.patches), patch_count)
                    self.assertEqual(ax.texts[0].get_text(), f't = {times[i]:.2f} s')
                self.assertFalse(animation._step())
                self.assertIsNone(animation.event_source)

if __name__ == '__main__':
    unittest.main()
