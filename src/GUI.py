"""Giao diện Tkinter cho mô phỏng robot vi sai (Bài thực hành 1 & 2.1)."""
import math
import time
from dataclasses import replace
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle

if __package__:
    from .avoidance import (
        Bug1Controller, Bug2Controller, TangentBugController,
    )
    from .navigation import goal_command
    from .dwa import DWAConfig, DWAController
    from .global_planner import GlobalPlannerController
    from .hybrid_planner import AStarDWAController
    from .obstacle_editor import (
        MODE_CIRCLE, MODE_RECT, MODE_SELECT, ObstacleEditor,
    )
    from .Robot import Differential_Robot
else:
    from avoidance import Bug1Controller, Bug2Controller, TangentBugController
    from navigation import goal_command
    from dwa import DWAConfig, DWAController
    from global_planner import GlobalPlannerController
    from hybrid_planner import AStarDWAController
    from obstacle_editor import (
        MODE_CIRCLE, MODE_RECT, MODE_SELECT, ObstacleEditor,
    )
    from Robot import Differential_Robot

AVOIDANCE_MODES = ('Tắt né tránh', 'Bug1', 'Bug2', 'Tangent Bug', 'DWA',
                   'Dijkstra', 'A*', 'A* + DWA')
FOLLOW_PHASES = ('FOLLOW', 'RETURN_L', 'LEAVE_Q', 'RETREAT', 'RECOVER')


def _positive(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} phải là số dương hữu hạn') from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} phải là số dương hữu hạn')
    return value


class RobotSimulatorGUI:
    """Cửa sổ mô phỏng: vẽ quỹ đạo, điều khiển tay và báo va chạm."""

    def __init__(self, root, robot, obstacles, dt=0.1, lidar=None):
        self.root = root
        self.robot = robot
        self.obstacles = obstacles
        self.dt = _positive(dt, 'dt')
        self.lidar = lidar
        self.last_collision_state = False
        self.ani = None
        self.ax_lidar = None
        self.local_robot = None
        self.lidar_points_scatter = None
        self.lidar_rays_collection = None
        self.last_lidar_scan = None
        self.last_lidar_local_points = np.empty((0, 2), dtype=float)

        # Auto-mode: click trên bản đồ đặt goal, nút Auto-mode bật/tắt
        self.auto_mode = False
        self.destination = None
        self.destination_marker = None
        self.auto_button = None
        self.toolbar = None

        # Né tránh vật cản (BTH2.1)
        self.avoidance_mode = AVOIDANCE_MODES[0]
        self.avoidance_controller = None
        self.avoidance_combobox = None
        self.avoidance_label = None
        self.follow_x = []
        self.follow_y = []
        self._was_following = False
        self.dwa_config = DWAConfig()
        self.dwa_frame = None
        self.dwa_table = None
        self.dwa_weight_vars = {}
        self.dwa_status_label = None
        self._dwa_last_refresh = float('-inf')

        # Chỉnh sửa vật cản trên bản đồ
        self.obstacle_editor = None
        self.obstacle_mode_var = None
        self.delete_obstacle_button = None
        self.clear_obstacles_button = None
        self.obstacle_status_label = None
        self.obstacle_dimension_vars = {}
        self.obstacle_dimension_entries = {}
        self.apply_obstacle_button = None
        self._dimension_selection = None

        self.setup_gui()
        self.ani = FuncAnimation(
            self.fig, self.update, interval=int(self.dt * 1000),
            blit=False, cache_frame_data=False,
        )

    # ------------------------------------------------------------------
    # Dựng giao diện
    # ------------------------------------------------------------------
    def setup_gui(self):
        if self.lidar is not None:
            self.fig, (self.ax, self.ax_lidar) = plt.subplots(
                1, 2, figsize=(13.5, 6),
            )
            self.fig.subplots_adjust(left=0.06, right=0.97, wspace=0.25)
        else:
            self.fig, self.ax = plt.subplots(figsize=(9, 6))
            # Chừa vùng bên phải cho chú thích, không che bản đồ.
            self.fig.subplots_adjust(right=0.74)
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_aspect('equal')
        self.ax.grid()
        self.ax.set_title('World View' if self.lidar is not None
                          else 'Differential Drive Robot Simulation')
        self.ax.set_xlabel('X position (m)')
        self.ax.set_ylabel('Y position (m)')

        self.path_line, = self.ax.plot([], [], 'b-', label='Path')
        self.follow_line, = self.ax.plot(
            [], [], color='#FF8C00', linewidth=2.0,
            label='Đường né tránh',
        )
        self.robot_dot, = self.ax.plot([], [], 'ro', label='Robot')

        # Lớp hiển thị thuật toán né tránh (BTH2.1)
        self.scan_collection = LineCollection(
            [], colors='#1E90FF', linewidths=0.4, alpha=0.35,
            label='Range sensor',
        )
        self.ax.add_collection(self.scan_collection)
        self.tangent_line, = self.ax.plot(
            [], [], color='#FF4500', linewidth=1.6, alpha=0.9,
            label='Tia tiếp tuyến',
        )
        self.goal_line, = self.ax.plot(
            [], [], color='#008000', linestyle=':', linewidth=1.4,
            label='Tia tới goal',
        )
        self.boundary_line, = self.ax.plot(
            [], [], color='#7F7F7F', linestyle='--', linewidth=1.2,
            label='Biên offset',
        )
        self.m_line, = self.ax.plot(
            [], [], color='#C71585', linestyle='--', linewidth=1.4,
            label='m-line',
        )
        self.leave_line, = self.ax.plot(
            [], [], color='#2E8B57', linestyle='-.', linewidth=1.6,
            label='Đường rời',
        )
        self.hit_marker, = self.ax.plot(
            [], [], marker='X', color='#FF8C00', linestyle='None',
            markersize=12, markeredgewidth=2, label='Hit point',
        )
        self.target_leave_marker, = self.ax.plot(
            [], [], marker='o', markerfacecolor='none', color='#2E8B57',
            linestyle='None', markersize=12, markeredgewidth=2,
            label='Điểm rời dự kiến',
        )
        self.leave_marker, = self.ax.plot(
            [], [], marker='*', color='#00B050', linestyle='None',
            markersize=16, markeredgecolor='black', markeredgewidth=0.6,
            label='Leave point',
        )
        self.dwa_rollout, = self.ax.plot(
            [], [], color='#9400D3', linewidth=2, linestyle='-.',
            label='DWA: quỹ đạo chọn',
        )
        self.global_path_line, = self.ax.plot(
            [], [], color='#228B22', linewidth=2, linestyle='--',
            label='Đường đi toàn cục',
        )
        self.waypoint_marker, = self.ax.plot(
            [], [], marker='D', color='#228B22', linestyle='None',
            markersize=6, label='Waypoint hiện tại',
        )
        if self.ax_lidar is not None:
            self.ax.legend(
                loc='upper right', fontsize=7, framealpha=0.85,
            )
            self._setup_lidar_panel()
        else:
            self.ax.legend(
                loc='upper left', bbox_to_anchor=(1.03, 1.0),
                borderaxespad=0, fontsize=8,
            )

        for obs in self.obstacles.obs:
            self.ax.add_patch(obs)
        self.obstacles.label_obstacles(self.ax)

        self.ax.add_patch(self.robot.robot_body)
        self.ax.add_patch(self.robot.left_wheel)
        self.ax.add_patch(self.robot.right_wheel)
        self.ax.add_patch(self.robot.heading_arrow)
        self.ax.add_patch(self.robot.left_arrow)
        self.ax.add_patch(self.robot.right_arrow)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(
            row=0, column=0, columnspan=5, sticky='nsew', padx=10, pady=10,
        )
        self.canvas.mpl_connect('button_press_event', self.on_canvas_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_canvas_motion)
        self.canvas.mpl_connect('button_release_event', self.on_canvas_release)

        toolbar_frame = tk.Frame(self.root)
        toolbar_frame.grid(row=1, column=0, columnspan=5, pady=5)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        self._create_status_display()
        self._create_controls()
        self._create_dwa_controls()
        self.obstacle_editor = ObstacleEditor(
            self.ax, self.obstacles,
            on_change=self._on_obstacles_changed,
            on_status=self._set_obstacle_status,
        )
        self._bind_keys()
        self._update_obstacle_controls()
        self._set_obstacle_status(
            f'{len(self.obstacles.obs)} vật cản • '
            'Chọn vật cản hoặc chọn công cụ vẽ',
        )
        self.canvas.draw_idle()

        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.robot.update_graphics(self.ax)
        self._update_lidar_view()
        self.update_velocity_labels()

    def _setup_lidar_panel(self):
        limit = max(20.0, float(self.lidar.max_range) + 2.0)
        self.ax_lidar.set_xlim(-limit, limit)
        self.ax_lidar.set_ylim(-limit, limit)
        self.ax_lidar.set_aspect('equal')
        self.ax_lidar.grid()
        self.ax_lidar.set_title('LiDAR Scan (Robot Frame)')
        self.ax_lidar.set_xlabel('Local X / forward (m)')
        self.ax_lidar.set_ylabel('Local Y / left (m)')

        self.lidar_rays_collection = LineCollection(
            [], colors='#1E90FF', linewidths=0.5, alpha=0.3, zorder=1,
        )
        self.ax_lidar.add_collection(self.lidar_rays_collection)
        self.lidar_points_scatter = self.ax_lidar.scatter(
            [], [], s=14, c='#d62728', zorder=3, label='LiDAR hits (local)',
        )
        self.local_robot = Differential_Robot(
            self.robot.L, self.robot.robot_length, self.robot.robot_width,
            self.robot.wheel_length, self.robot.wheel_width,
        )
        for patch in (
            self.local_robot.robot_body,
            self.local_robot.left_wheel,
            self.local_robot.right_wheel,
            self.local_robot.heading_arrow,
            self.local_robot.left_arrow,
            self.local_robot.right_arrow,
        ):
            self.ax_lidar.add_patch(patch)
        self.local_robot.update_graphics(self.ax_lidar)
        self.ax_lidar.legend(loc='upper right', fontsize=8, framealpha=0.85)

    def _update_lidar_view(self):
        if self.lidar is None or self.ax_lidar is None:
            return
        scan = self.lidar.scan(self.robot.x, self.robot.y, self.robot.theta)
        self.last_lidar_scan = scan
        local_hits = scan.local_points[scan.hit]
        self.last_lidar_local_points = local_hits
        if len(local_hits) > 0:
            self.lidar_points_scatter.set_offsets(local_hits)
            origin = np.broadcast_to(self.lidar.mount, local_hits.shape)
            self.lidar_rays_collection.set_segments(
                np.stack((origin, local_hits), axis=1),
            )
        else:
            self.lidar_points_scatter.set_offsets(np.empty((0, 2)))
            self.lidar_rays_collection.set_segments([])
        self.local_robot.set_wheel_velocities(self.robot.v_l, self.robot.v_r)
        self.local_robot.update_graphics(self.ax_lidar)

    def _create_status_display(self):
        self.velocity_frame = ttk.LabelFrame(
            self.root, text='Velocities', padding=(10, 10),
        )
        self.velocity_frame.grid(
            row=0, column=6, rowspan=14, sticky='n', padx=10, pady=10,
        )

        style = ttk.Style()
        style.configure('Linear.TLabel', foreground='#00008B',
                        font=('Arial', 12, 'bold'))
        style.configure('Angular.TLabel', foreground='#8B0000',
                        font=('Arial', 12, 'bold'))
        style.configure('LeftWheel.TLabel', foreground='#006400',
                        font=('Arial', 12, 'bold'))
        style.configure('RightWheel.TLabel', foreground='#B8860B',
                        font=('Arial', 12, 'bold'))

        self.vel_linear_label = ttk.Label(
            self.velocity_frame, text='', style='Linear.TLabel',
        )
        self.vel_linear_label.grid(row=0, column=0, sticky='w', padx=5, pady=5)

        self.vel_angular_label = ttk.Label(
            self.velocity_frame, text='', style='Angular.TLabel',
        )
        self.vel_angular_label.grid(row=1, column=0, sticky='w', padx=5, pady=5)

        self.vel_left_label = ttk.Label(
            self.velocity_frame, text='', style='LeftWheel.TLabel',
        )
        self.vel_left_label.grid(row=2, column=0, sticky='w', padx=5, pady=5)

        self.vel_right_label = ttk.Label(
            self.velocity_frame, text='', style='RightWheel.TLabel',
        )
        self.vel_right_label.grid(row=3, column=0, sticky='w', padx=5, pady=5)

        self.collision_label = ttk.Label(
            self.velocity_frame, text='', foreground='red',
            font=('Arial', 11, 'bold'), wraplength=220,
        )
        self.collision_label.grid(row=4, column=0, sticky='w', padx=5, pady=10)

        self.nav_label = ttk.Label(
            self.velocity_frame, text='', foreground='#00008B',
            font=('Arial', 10), wraplength=220, justify='left',
        )
        self.nav_label.grid(row=5, column=0, sticky='w', padx=5, pady=(0, 5))

        self.avoidance_label = ttk.Label(
            self.velocity_frame,
            text=f'Thuật toán: {self.avoidance_mode}',
            foreground='#8B4513', font=('Arial', 10, 'bold'),
            wraplength=220, justify='left',
        )
        self.avoidance_label.grid(
            row=6, column=0, sticky='w', padx=5, pady=(0, 5))

    def _create_controls(self):
        ttk.Button(self.root, text='↑ Forward',
                   command=self.move_forward).grid(row=2, column=2, pady=10)
        ttk.Button(self.root, text='← Left',
                   command=self.turn_left).grid(row=3, column=1)
        ttk.Button(self.root, text='Stop',
                   command=self.stop_manual).grid(row=3, column=2)
        ttk.Button(self.root, text='Right →',
                   command=self.turn_right).grid(row=3, column=3)
        ttk.Button(self.root, text='↓ Backward',
                   command=self.move_backward).grid(row=4, column=2, pady=10)
        ttk.Button(self.root, text='⟲ Reset',
                   command=self.reset).grid(row=5, column=2, pady=10)
        self.auto_button = ttk.Button(
            self.root, text='▶ Auto-mode: OFF', command=self.toggle_auto_mode,
        )
        self.auto_button.grid(row=5, column=3, pady=10)

        ttk.Label(self.root, text='Thuật toán:').grid(
            row=6, column=1, sticky='e', pady=(0, 10))
        self.avoidance_combobox = ttk.Combobox(
            self.root, values=AVOIDANCE_MODES, state='readonly', width=18,
        )
        self.avoidance_combobox.set(self.avoidance_mode)
        self.avoidance_combobox.grid(
            row=6, column=2, columnspan=2, sticky='w', pady=(0, 10))
        self.avoidance_combobox.bind(
            '<<ComboboxSelected>>', self.on_avoidance_change)

        obstacle_frame = ttk.LabelFrame(
            self.root, text='Vật cản', padding=(8, 8),
        )
        obstacle_frame.grid(
            row=7, column=0, columnspan=5, sticky='ew', padx=10, pady=(0, 10),
        )
        obstacle_frame.columnconfigure(6, weight=1)

        ttk.Label(obstacle_frame, text='Công cụ:').grid(
            row=0, column=0, sticky='w', padx=(0, 6))
        self.obstacle_mode_var = tk.StringVar(value=MODE_SELECT)
        for column, (text, value) in enumerate(
                (('Chọn / Di chuyển', MODE_SELECT),
                 ('Vẽ chữ nhật', MODE_RECT),
                 ('Vẽ hình tròn', MODE_CIRCLE)),
                start=1,
        ):
            ttk.Radiobutton(
                obstacle_frame, text=text, value=value,
                variable=self.obstacle_mode_var,
                command=self._on_obstacle_mode_change,
            ).grid(row=0, column=column, sticky='w', padx=(0, 8))

        self.delete_obstacle_button = ttk.Button(
            obstacle_frame, text='Xoá vật cản',
            command=self._delete_selected_obstacle, state='disabled',
        )
        self.delete_obstacle_button.grid(row=0, column=4, padx=(8, 4))
        self.clear_obstacles_button = ttk.Button(
            obstacle_frame, text='Xoá tất cả',
            command=self._clear_obstacles,
        )
        self.clear_obstacles_button.grid(row=0, column=5, padx=(4, 0))

        dimensions_frame = ttk.Frame(obstacle_frame)
        dimensions_frame.grid(row=1, column=0, columnspan=7, sticky='w',
                              pady=(8, 0))
        for column, (key, label) in enumerate(
                (('radius', 'R (m)'), ('length', 'Length (m)'),
                 ('width', 'Width (m)'))):
            ttk.Label(dimensions_frame, text=label).grid(
                row=0, column=2 * column, padx=(0, 4))
            variable = tk.StringVar(value='')
            entry = ttk.Entry(dimensions_frame, textvariable=variable,
                              width=10, state='disabled')
            entry.grid(row=0, column=2 * column + 1, padx=(0, 12))
            entry.bind('<Return>', self._apply_obstacle_dimensions)
            self.obstacle_dimension_vars[key] = variable
            self.obstacle_dimension_entries[key] = entry
        self.apply_obstacle_button = ttk.Button(
            dimensions_frame, text='Apply', state='disabled',
            command=self._apply_obstacle_dimensions)
        self.apply_obstacle_button.grid(row=0, column=6)

        self.obstacle_status_label = ttk.Label(
            obstacle_frame, text='', foreground='#333333',
            wraplength=760, justify='left',
        )
        self.obstacle_status_label.grid(
            row=2, column=0, columnspan=7, sticky='w', pady=(8, 0))
        ttk.Label(
            obstacle_frame,
            text='Mẹo: chọn công cụ rồi kéo trên bản đồ để vẽ. '
                 'Chọn vật cản để sửa kích thước. Esc huỷ, Delete xoá.',
            foreground='#666666', wraplength=760, justify='left',
        ).grid(row=3, column=0, columnspan=7, sticky='w', pady=(4, 0))


    def _bind_keys(self):
        for key, action in (
                ('<Up>', self.move_forward), ('<Down>', self.move_backward),
                ('<Left>', self.turn_left), ('<Right>', self.turn_right),
                ('<space>', self.stop_manual), ('r', self.reset),
                ('<Escape>', self._cancel_obstacle_action),
                ('<Delete>', self._delete_selected_obstacle),
                ('<BackSpace>', self._delete_selected_obstacle)):
            self.root.bind(key, lambda event, action=action:
                           self._run_shortcut(event, action))

    def _run_shortcut(self, event, action):
        # Không điều khiển robot/xoá vật cản khi người dùng đang nhập số.
        if event.widget.winfo_class() in (
                'Entry', 'TEntry', 'Spinbox', 'TSpinbox', 'Text', 'TCombobox',
                'Treeview'):
            return
        action()

    def _create_dwa_controls(self):
        """Bảng điểm đặt ngoài axes để không che bản đồ."""
        self.dwa_frame = ttk.LabelFrame(self.root, text='DWA', padding=(8, 8))
        self.dwa_frame.grid(row=8, column=0, columnspan=5, sticky='ew',
                            padx=10, pady=(0, 10))
        weights = ttk.Frame(self.dwa_frame)
        weights.grid(row=0, column=0, sticky='w')
        for column, key in enumerate(('alpha', 'beta', 'gamma')):
            ttk.Label(weights, text=key).grid(row=0, column=2 * column)
            variable = tk.StringVar(value=str(getattr(self.dwa_config, key)))
            self.dwa_weight_vars[key] = variable
            entry = ttk.Entry(weights, textvariable=variable, width=8)
            entry.grid(row=0, column=2 * column + 1, padx=(4, 10))
            entry.bind('<Return>', self._apply_dwa_weights)
        ttk.Button(weights, text='Apply', command=self._apply_dwa_weights).grid(
            row=0, column=6)
        self.dwa_status_label = ttk.Label(self.dwa_frame, text='', wraplength=760)
        self.dwa_status_label.grid(row=1, column=0, sticky='w', pady=(4, 4))
        columns = ('selected', 'v', 'w', 'heading', 'dist', 'velocity', 'score')
        self.dwa_table = ttk.Treeview(
            self.dwa_frame, columns=columns, show='headings', height=6)
        for key in columns:
            self.dwa_table.heading(key, text='Chọn' if key == 'selected' else key)
            self.dwa_table.column(key, width=90, minwidth=60, anchor='center')
        self.dwa_table.grid(row=2, column=0, sticky='nsew')
        vertical = ttk.Scrollbar(self.dwa_frame, orient='vertical',
                                 command=self.dwa_table.yview)
        vertical.grid(row=2, column=1, sticky='ns')
        horizontal = ttk.Scrollbar(self.dwa_frame, orient='horizontal',
                                   command=self.dwa_table.xview)
        horizontal.grid(row=3, column=0, sticky='ew')
        self.dwa_table.configure(yscrollcommand=vertical.set,
                                 xscrollcommand=horizontal.set)
        self.dwa_table.tag_configure('selected', background='#E8D8FF')
        self.dwa_frame.columnconfigure(0, weight=1)
        self._sync_dwa_visibility()

    def _sync_dwa_visibility(self):
        if self.dwa_frame is not None:
            if self.avoidance_mode in ('DWA', 'A* + DWA'):
                self.dwa_frame.grid()
            else:
                self.dwa_frame.grid_remove()

    def _apply_dwa_weights(self, event=None):
        try:
            values = {key: float(variable.get())
                      for key, variable in self.dwa_weight_vars.items()}
            if (any(not math.isfinite(value) or value < 0
                    for value in values.values()) or not any(values.values())):
                raise ValueError
            config = replace(self.dwa_config, **values)
        except (ValueError, TypeError):
            self.dwa_status_label.config(
                text='Trọng số phải hữu hạn, không âm và không đồng thời bằng 0.')
        else:
            self.dwa_config = config
            if (self.avoidance_mode in ('DWA', 'A* + DWA')
                    and self.avoidance_controller is not None):
                self.avoidance_controller.config = config
            self.dwa_status_label.config(text='Đã áp dụng trọng số cho bước kế tiếp.')
        return 'break'

    def _clear_dwa_display(self):
        rollout = getattr(self, 'dwa_rollout', None)
        if rollout is not None:
            rollout.set_data([], [])
        table = getattr(self, 'dwa_table', None)
        if table is not None:
            children = table.get_children()
            if children:
                table.delete(*children)
        self._dwa_last_refresh = float('-inf')

    def _update_dwa_display(self):
        controller = self.avoidance_controller
        selected = controller.selected_candidate
        if self.auto_mode and selected is not None:
            trajectory = np.asarray(selected.trajectory)
            self.dwa_rollout.set_data(trajectory[:, 0], trajectory[:, 1])
        else:
            self.dwa_rollout.set_data([], [])
        now = time.monotonic()
        if now - self._dwa_last_refresh < 0.2:
            return
        self._dwa_last_refresh = now
        # Chỉ thay nội dung tối đa 5 lần/giây, giữ vị trí cuộn khi cập nhật.
        position = self.dwa_table.yview()[0]
        children = self.dwa_table.get_children()
        if children:
            self.dwa_table.delete(*children)
        for candidate in controller.candidates:
            chosen = candidate is selected
            values = ('✓' if chosen else '',) + tuple(
                f'{getattr(candidate, key):.4f}'
                for key in ('v', 'w', 'heading', 'dist', 'velocity', 'score'))
            self.dwa_table.insert('', 'end', values=values,
                                  tags=('selected',) if chosen else ())
        self.dwa_table.yview_moveto(position)

    # ------------------------------------------------------------------
    # Vòng lặp mô phỏng
    # ------------------------------------------------------------------
    def update(self, frame=None):
        control = None
        if self.auto_mode and self.destination is not None:
            control = self._auto_control()

        x_next, y_next, theta_next = self.robot.predict_pose(self.dt)

        collided = self.obstacles.collides_pose(
            self.robot, x_next, y_next, theta_next)
        if collided:
            if (self.avoidance_mode in ('Dijkstra', 'A*', 'A* + DWA')
                    and self.avoidance_controller is not None
                    and self.auto_mode):
                # Planner toàn cục dừng khi va chạm, không chạy Bug recovery.
                self.avoidance_controller.on_collision(self.robot)
                self.stop()
                self.auto_mode = False
                self._update_auto_button()
                self.nav_label.config(
                    text=self.avoidance_controller.debug.get('message', 'Va chạm'))
                self.collision_label.config(text='⚠ Collision detected! Robot stopped.')
            elif (self.avoidance_controller is not None and self.auto_mode
                    and self.destination is not None
                    and control is not None
                    and not control.arrived and not control.failed):
                # BTH2.1: contact sensor -> controller tự lùi và bám biên.
                self.avoidance_controller.on_collision(self.robot)
                self.stop()
                if not self.last_collision_state:
                    self.collision_label.config(
                        text='⚠ Collision detected! Đang né tránh…',
                    )
            else:
                self.stop()
                if not self.last_collision_state:
                    self.collision_label.config(
                        text='⚠ Collision detected! Robot stopped.',
                    )
            self.last_collision_state = True
        elif self.robot.v_l != 0 or self.robot.v_r != 0:
            self.robot.update_pose(self.dt)
            if self.last_collision_state:
                self._clear_collision_message()
            self._record_follow_path(control)

        self.path_line.set_data(self.robot.x_data, self.robot.y_data)
        self.follow_line.set_data(self.follow_x, self.follow_y)
        self.robot_dot.set_data([self.robot.x], [self.robot.y])
        self._update_avoidance_overlays()
        self.robot.update_graphics(self.ax)
        self._update_lidar_view()
        self.update_velocity_labels()
        self.canvas.draw_idle()

        artists = (
            self.path_line, self.follow_line, self.robot_dot,
            self.robot.robot_body, self.robot.left_wheel,
            self.robot.right_wheel, self.robot.heading_arrow,
            self.robot.left_arrow, self.robot.right_arrow,
        )
        if self.lidar_points_scatter is not None:
            artists = artists + (self.lidar_points_scatter,)
        return artists

    def _auto_control(self):
        """Chạy một bước điều khiển auto; trả Control hoặc None."""
        if self.avoidance_controller is not None:
            control = self.avoidance_controller.step(
                self.robot, self.destination, self.dt)
            self._last_control = control
            if control.arrived:
                self.stop()
                self.auto_mode = False
                self._update_auto_button()
                self.nav_label.config(
                    text=control.message or 'Destination reached!')
            elif control.failed:
                self.stop()
                self.auto_mode = False
                self._update_auto_button()
                self.nav_label.config(text=control.message)
            else:
                self.robot.set_wheel_velocities(control.v_l, control.v_r)
                self.nav_label.config(text=control.message)
            return control

        v_l, v_r, _, arrived = goal_command(self.robot, self.destination)
        if arrived:
            self.stop()
            self.auto_mode = False
            self._update_auto_button()
            self.nav_label.config(text='Destination reached!')
            return None
        self.robot.set_wheel_velocities(v_l, v_r)
        self.nav_label.config(
            text=f'Goal: ({self.destination[0]:.2f}, '
                 f'{self.destination[1]:.2f})',
        )
        return None

    def _record_follow_path(self, control):
        following = (
            control is not None
            and not control.arrived and not control.failed
            and control.phase in FOLLOW_PHASES
        )
        if not following:
            self._was_following = False
            return
        if not self._was_following:
            # Ngắt nét khi bắt đầu một đoạn bám biên mới.
            self.follow_x.append(float('nan'))
            self.follow_y.append(float('nan'))
            self._was_following = True
        self.follow_x.append(self.robot.x)
        self.follow_y.append(self.robot.y)

    # ------------------------------------------------------------------
    # Hiển thị né tránh
    # ------------------------------------------------------------------
    def _make_controller(self):
        if self.avoidance_mode == 'Bug1':
            return Bug1Controller(self.obstacles)
        if self.avoidance_mode == 'Bug2':
            return Bug2Controller(self.obstacles)
        if self.avoidance_mode == 'Tangent Bug':
            return TangentBugController(self.obstacles)
        if self.avoidance_mode == 'DWA':
            return DWAController(self.obstacles, config=self.dwa_config)
        if self.avoidance_mode == 'A* + DWA':
            return AStarDWAController(self.obstacles, config=self.dwa_config)
        if self.avoidance_mode in ('Dijkstra', 'A*'):
            return GlobalPlannerController(self.obstacles,
                                           algorithm=self.avoidance_mode)
        return None

    def _clear_avoidance_overlays(self):
        self._clear_dwa_display()
        for name in ('global_path_line', 'waypoint_marker'):
            artist = getattr(self, name, None)
            if artist is not None:
                artist.set_data([], [])
        self.scan_collection.set_segments([])
        for line in (self.goal_line, self.tangent_line, self.boundary_line,
                     self.m_line, self.leave_line, self.hit_marker,
                     self.target_leave_marker, self.leave_marker):
            line.set_data([], [])

    def _update_avoidance_overlays(self):
        controller = self.avoidance_controller
        if self.avoidance_mode == 'A* + DWA' and controller is not None:
            self._update_global_display()
            self._update_dwa_display()
            return
        if self.avoidance_mode in ('Dijkstra', 'A*') and controller is not None:
            self._update_global_display()
            return
        if self.avoidance_mode == 'DWA' and controller is not None:
            self._update_dwa_display()
            return
        if controller is None:
            self._clear_avoidance_overlays()
            return
        phase = controller.phase
        if not self.auto_mode and phase not in ('ARRIVED', 'FAILED'):
            self._clear_avoidance_overlays()
            return

        debug = controller.debug
        static_only = phase in ('ARRIVED', 'FAILED')

        # Tia tới goal + range sensor + tia tiếp tuyến chỉ khi đang chạy.
        if not static_only and self.destination is not None:
            self.goal_line.set_data(
                [self.robot.x, self.destination[0]],
                [self.robot.y, self.destination[1]],
            )
        else:
            self.goal_line.set_data([], [])

        scan = debug.get('scan')
        if not static_only and scan is not None and self.auto_mode:
            origin = (self.robot.x, self.robot.y)
            self.scan_collection.set_segments([
                [origin, (float(point[0]), float(point[1]))]
                for point in scan.points
            ])
        else:
            self.scan_collection.set_segments([])

        tangent = debug.get('tangent')
        if not static_only and tangent is not None and self.auto_mode:
            self.tangent_line.set_data(
                [self.robot.x, float(tangent.point[0])],
                [self.robot.y, float(tangent.point[1])],
            )
        else:
            self.tangent_line.set_data([], [])

        # Biên offset của vật cản đang bám.
        boundary = debug.get('boundary')
        if boundary is not None and len(boundary):
            points = np.asarray(boundary, dtype=float)
            closed = np.vstack([points, points[:1]])
            self.boundary_line.set_data(closed[:, 0], closed[:, 1])
        else:
            self.boundary_line.set_data([], [])

        # m-line của Bug2.
        m_line = debug.get('m_line')
        if m_line is not None:
            start, end = m_line
            self.m_line.set_data(
                [float(start[0]), float(end[0])],
                [float(start[1]), float(end[1])],
            )
        else:
            self.m_line.set_data([], [])

        self._set_marker(self.hit_marker, debug.get('hit'))
        self._set_marker(
            self.target_leave_marker, debug.get('target_leave'))
        self._set_marker(self.leave_marker, debug.get('leave'))

        leave = debug.get('leave')
        if leave is not None and self.destination is not None:
            self.leave_line.set_data(
                [float(leave[0]), self.destination[0]],
                [float(leave[1]), self.destination[1]],
            )
        else:
            self.leave_line.set_data([], [])

    def _update_global_display(self):
        controller = self.avoidance_controller
        path = np.asarray(controller.path)
        if path.size:
            self.global_path_line.set_data(path[:, 0], path[:, 1])
        else:
            self.global_path_line.set_data([], [])
        self._set_marker(self.waypoint_marker, controller.debug.get('target'))
        if not self.auto_mode and controller.debug.get('message'):
            self.nav_label.config(text=controller.debug['message'])

    @staticmethod
    def _set_marker(artist, point):
        if point is None:
            artist.set_data([], [])
        else:
            artist.set_data([float(point[0])], [float(point[1])])


    def update_velocity_labels(self):
        v = (self.robot.v_l + self.robot.v_r) / 2
        omega = (self.robot.v_r - self.robot.v_l) / self.robot.L
        self.vel_linear_label.config(text=f'Linear Velocity (v): {v:.2f} m/s')
        self.vel_angular_label.config(
            text=f'Angular Velocity (ω): {omega:.2f} rad/s',
        )
        self.vel_left_label.config(
            text=f'Left Wheel Velocity (v_l): {self.robot.v_l:.2f} m/s',
        )
        self.vel_right_label.config(
            text=f'Right Wheel Velocity (v_r): {self.robot.v_r:.2f} m/s',
        )

    # ------------------------------------------------------------------
    # Điều khiển
    # ------------------------------------------------------------------
    def _clear_collision_message(self):
        self.collision_label.config(text='')
        self.last_collision_state = False

    # ------------------------------------------------------------------
    # Auto-mode
    # ------------------------------------------------------------------
    def _update_auto_button(self):
        if self.auto_button is not None:
            self.auto_button.config(
                text='⏸ Auto-mode: ON' if self.auto_mode
                else '▶ Auto-mode: OFF',
            )

    def _cancel_auto(self):
        if not self.auto_mode:
            self._update_auto_button()
            return
        self.auto_mode = False
        self._update_auto_button()
        self.nav_label.config(text='Auto-mode OFF')

    def on_avoidance_change(self, event=None):
        previous_mode = self.avoidance_mode
        if self.avoidance_combobox is not None:
            self.avoidance_mode = self.avoidance_combobox.get()
        if (self.avoidance_mode in ('DWA', 'Dijkstra', 'A*', 'A* + DWA')
                and previous_mode != self.avoidance_mode):
            self.stop()
        self.avoidance_controller = self._make_controller()
        self._sync_dwa_visibility()
        self.follow_x = []
        self.follow_y = []
        self._was_following = False
        self._clear_avoidance_overlays()
        if (self.avoidance_controller is not None
                and self.destination is not None
                and (self.auto_mode or self.avoidance_mode in (
                    'Dijkstra', 'A*', 'A* + DWA'))):
            self.avoidance_controller.reset(self.robot, self.destination)
        if self.avoidance_label is not None:
            self.avoidance_label.config(
                text=f'Thuật toán: {self.avoidance_mode}')
        if self.avoidance_mode == AVOIDANCE_MODES[0]:
            self.nav_label.config(text='Đi thẳng tới goal.')
        else:
            self.nav_label.config(text=f'Thuật toán: {self.avoidance_mode}')
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Vật cản
    # ------------------------------------------------------------------
    def _on_obstacle_mode_change(self):
        if self.obstacle_editor is None:
            return
        self.obstacle_editor.set_mode(self.obstacle_mode_var.get())
        self._update_obstacle_controls()

    def _delete_selected_obstacle(self):
        if self.obstacle_editor is None:
            return
        self.obstacle_editor.delete_selected()
        self._update_obstacle_controls()

    def _clear_obstacles(self):
        if self.obstacle_editor is None or not self.obstacles.obs:
            return
        if messagebox.askyesno(
                'Xoá tất cả', 'Xoá toàn bộ vật cản trên bản đồ?'):
            self.obstacle_editor.clear_all()
            self._update_obstacle_controls()

    def _cancel_obstacle_action(self):
        if self.obstacle_editor is not None:
            self.obstacle_editor.cancel()
            self._update_obstacle_controls()

    def _update_obstacle_controls(self):
        if self.delete_obstacle_button is not None:
            has_selection = (
                self.obstacle_editor is not None
                and self.obstacle_editor.selected is not None
            )
            self.delete_obstacle_button.config(
                state='normal' if has_selection else 'disabled')
        if self.clear_obstacles_button is not None:
            self.clear_obstacles_button.config(
                state='normal' if self.obstacles.obs else 'disabled')
        self._sync_obstacle_dimensions()

    def _sync_obstacle_dimensions(self, force=False):
        patch = (self.obstacle_editor.selected
                 if self.obstacle_editor is not None else None)
        active = ('radius',) if isinstance(patch, Circle) else (
            ('length', 'width') if isinstance(patch, Rectangle) else ())
        for key, entry in self.obstacle_dimension_entries.items():
            entry.config(state='normal' if key in active else 'disabled')
        if self.apply_obstacle_button is not None:
            self.apply_obstacle_button.config(
                state='normal' if active else 'disabled')
        # Không ghi đè số đang nhập khi chỉ redraw hoặc di chuyển vật cản.
        if force or patch is not self._dimension_selection:
            values = {}
            if isinstance(patch, Circle):
                values['radius'] = patch.radius
            elif isinstance(patch, Rectangle):
                values.update(length=patch.get_width(),
                              width=patch.get_height())
            for key, variable in self.obstacle_dimension_vars.items():
                variable.set(str(float(values[key])) if key in values else '')
            self._dimension_selection = patch

    def _apply_obstacle_dimensions(self, event=None):
        patch = (self.obstacle_editor.selected
                 if self.obstacle_editor is not None else None)
        if patch is None:
            return 'break'
        keys = ('radius',) if isinstance(patch, Circle) else ('length', 'width')
        try:
            values = {key: self.obstacle_dimension_vars[key].get()
                      for key in keys}
            self.obstacle_editor.resize_selected(**values)
        except ValueError as exc:
            self._set_obstacle_status(str(exc))
        else:
            self._sync_obstacle_dimensions(force=True)
        return 'break'

    def _set_obstacle_status(self, text):
        if self.obstacle_status_label is not None:
            self.obstacle_status_label.config(text=text)

    def _on_obstacles_changed(self, reason):
        if reason in ('create', 'move', 'delete', 'clear', 'resize'):
            if getattr(self.avoidance_controller, 'name', None) == 'A* + DWA':
                self.stop()
            self.obstacles.label_obstacles(self.ax)
            self._clear_avoidance_overlays()
            self.follow_x.clear()
            self.follow_y.clear()
            self._was_following = False
            if (self.avoidance_controller is not None
                    and self.destination is not None
                    and (self.auto_mode or self.avoidance_mode in (
                        'Dijkstra', 'A*', 'A* + DWA')
                         or isinstance(
                        self.avoidance_controller, DWAController))):
                self.avoidance_controller.reset(self.robot, self.destination)
        self._update_obstacle_controls()
        if reason == 'resize':
            self._sync_obstacle_dimensions(force=True)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Định tuyến sự kiện chuột: editor vật cản trước, goal sau
    # ------------------------------------------------------------------
    def on_canvas_press(self, event):
        if self.toolbar is not None and self.toolbar.mode:
            return
        if (self.obstacle_editor is not None
                and self.obstacle_editor.on_press(event)):
            self._update_obstacle_controls()
            self.canvas.draw_idle()
            return
        self.on_click(event)
        self._update_obstacle_controls()

    def on_canvas_motion(self, event):
        if self.toolbar is not None and self.toolbar.mode:
            return
        if (self.obstacle_editor is not None
                and self.obstacle_editor.on_motion(event)):
            self.canvas.draw_idle()

    def on_canvas_release(self, event):
        if self.toolbar is not None and self.toolbar.mode:
            return
        if self.obstacle_editor is not None:
            if self.obstacle_editor.on_release(event):
                self.canvas.draw_idle()
            self._update_obstacle_controls()

    def on_click(self, event):
        if event.inaxes is not self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.button != 1:
            return
        if self.toolbar is not None and self.toolbar.mode:
            return

        if self.avoidance_mode == 'A* + DWA':
            self.stop()
        self.destination = (float(event.xdata), float(event.ydata))
        self.auto_mode = True
        self._clear_collision_message()
        self._update_auto_button()
        self.nav_label.config(
            text=f'Goal: ({self.destination[0]:.2f}, '
                 f'{self.destination[1]:.2f}) — Auto-mode ON',
        )
        self.follow_x = []
        self.follow_y = []
        self._was_following = False
        if self.avoidance_controller is not None:
            self.avoidance_controller.reset(self.robot, self.destination)
            self._clear_avoidance_overlays()

        if self.destination_marker is not None:
            self.destination_marker.remove()
        self.destination_marker, = self.ax.plot(
            [self.destination[0]], [self.destination[1]],
            marker='*', color='red', linestyle='None', markersize=15,
            zorder=6,
        )
        self.canvas.draw_idle()

    def toggle_auto_mode(self):
        if self.auto_mode:
            self._cancel_auto()
            self.stop()
        elif self.destination is None:
            self.nav_label.config(
                text='Chưa có goal — click trên bản đồ để đặt goal.',
            )
        else:
            if self.avoidance_mode == 'A* + DWA':
                self.stop()
            self.auto_mode = True
            self._clear_collision_message()
            self.follow_x = []
            self.follow_y = []
            self._was_following = False
            if self.avoidance_controller is not None:
                self.avoidance_controller.reset(self.robot, self.destination)
            self._update_auto_button()
            self.nav_label.config(text='Auto-mode ON')


    def _drive(self, v_l, v_r):
        self._cancel_auto()
        self._clear_collision_message()
        self.robot.set_wheel_velocities(v_l, v_r)
        self.update_velocity_labels()

    def move_forward(self):
        self._drive(1.5, 1.5)

    def move_backward(self):
        self._drive(-1.5, -1.5)

    def turn_left(self):
        self._drive(0.5, 1.5)

    def turn_right(self):
        self._drive(1.5, 0.5)

    def stop(self):
        self.robot.set_wheel_velocities(0.0, 0.0)
        self.update_velocity_labels()

    def stop_manual(self):
        self._cancel_auto()
        self.stop()

    def reset(self):
        if self.obstacle_editor is not None:
            self.obstacle_editor.cancel()
        self.auto_mode = False
        self.destination = None
        if self.destination_marker is not None:
            self.destination_marker.remove()
            self.destination_marker = None
        self._update_auto_button()
        self.nav_label.config(text='')
        self.robot.reset()
        self.avoidance_controller = self._make_controller()
        self.follow_x = []
        self.follow_y = []
        self._was_following = False
        self.path_line.set_data([], [])
        self.follow_line.set_data([], [])
        self.robot_dot.set_data([], [])
        self._clear_avoidance_overlays()
        self._clear_collision_message()
        self.robot.update_graphics(self.ax)
        self._update_lidar_view()
        self.update_velocity_labels()
        self._update_obstacle_controls()
        self.canvas.draw_idle()

    def close(self):
        if self.obstacle_editor is not None:
            self.obstacle_editor.cancel()
        self.robot.set_wheel_velocities(0.0, 0.0)
        if self.ani is not None and self.ani.event_source is not None:
            self.ani.event_source.stop()
        plt.close(self.fig)
        self.root.destroy()
