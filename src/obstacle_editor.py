"""Trình soạn thảo vật cản trên axes Matplotlib.

Module này chỉ phụ trách tương tác chuột và các lớp hiển thị tạm (preview,
highlight). Dữ liệu vật cản thật nằm trong một collection có API tương thích
với ``src.Obstacle.Obstacle`` (``obs``, ``add``, ``pick``, ``index_of``,
``translate``, ``resize``, ``remove_at``, ``clear``).

Không import tkinter: GUI sẽ tự nối các sự kiện chuột của canvas vào
``on_press`` / ``on_motion`` / ``on_release``.
"""
import math

from matplotlib.patches import Circle, Rectangle

# Hằng số chế độ ở cấp module để GUI có thể import trực tiếp.
MODE_SELECT = 'select'
MODE_RECT = 'rect'
MODE_CIRCLE = 'circle'

# Màu/nét dùng cho các lớp phụ trợ.
_PREVIEW_COLOR = '#1E90FF'
_HIGHLIGHT_COLOR = '#FF8C00'
_PREVIEW_ZORDER = 15
_HIGHLIGHT_ZORDER = 20


def _positive_number(value, name):
    """Đổi sang số thực dương hữu hạn, nếu không hợp lệ thì báo lỗi."""
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} phải là số dương hữu hạn') from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} phải là số dương hữu hạn')
    return value


def _remove_artist(artist):
    """Gỡ một artist khỏi axes một cách an toàn (kể cả đã bị gỡ trước đó)."""
    if artist is None:
        return
    try:
        if getattr(artist, 'axes', None) is not None:
            artist.remove()
    except (AttributeError, NotImplementedError, ValueError):
        pass


class ObstacleEditor:
    """Tạo, chọn, di chuyển và xoá vật cản bằng chuột trái."""

    MODE_SELECT = MODE_SELECT
    MODE_RECT = MODE_RECT
    MODE_CIRCLE = MODE_CIRCLE

    _MODES = (MODE_SELECT, MODE_RECT, MODE_CIRCLE)

    def __init__(
        self, ax, obstacles,
        on_change=None, on_status=None,
        min_rect_size=0.2,
        min_circle_radius=0.1,
    ):
        self.ax = ax
        self.obstacles = obstacles
        self.on_change = on_change
        self.on_status = on_status

        self.min_rect_size = _positive_number(min_rect_size, 'min_rect_size')
        self.min_circle_radius = _positive_number(
            min_circle_radius, 'min_circle_radius')

        self._mode = self.MODE_SELECT
        self._selected = None
        self._highlight = None

        # Trạng thái kéo-thả: None | 'move' | 'create_rect' |
        # 'create_circle'.
        self._drag_kind = None
        self._drag_patch = None
        self._drag_original_anchor = None
        self._drag_start = None
        self._drag_last = None
        self._drag_moved = False
        self._pending_select = False

        # Preview hiện tại (Rectangle/Circle nét đứt) và loại của nó.
        self._preview = None
        self._preview_kind = None

    # ------------------------------------------------------------------
    # Thuộc tính công khai
    # ------------------------------------------------------------------
    @property
    def selected(self):
        """Patch đang được chọn, hoặc ``None``."""
        return self._selected

    @property
    def mode(self):
        """Chế độ hiện tại: ``select``, ``rect`` hoặc ``circle``."""
        return self._mode

    # ------------------------------------------------------------------
    # Chế độ và lệnh công khai
    # ------------------------------------------------------------------
    def set_mode(self, mode):
        """Đổi chế độ; mọi thao tác kéo-thả đang dở bị huỷ."""
        if mode not in self._MODES:
            raise ValueError(f'chế độ không hợp lệ: {mode!r}')
        self._abort_drag()
        self._mode = mode
        if mode == self.MODE_SELECT:
            self._status('Chế độ: chọn vật cản.')
        elif mode == self.MODE_RECT:
            self._status('Chế độ: vẽ hình chữ nhật.')
        else:
            self._status('Chế độ: vẽ hình tròn.')

    def delete_selected(self):
        """Xoá vật cản đang chọn; trả ``True`` nếu có xoá."""
        self._abort_drag()
        if self._selected is None:
            self._status('Chưa chọn vật cản để xoá.')
            return False

        index = self.obstacles.index_of(self._selected)
        if index is None:
            self._clear_selection()
            self._status('Vật cản đang chọn không còn tồn tại.')
            return False

        removed = self.obstacles.remove_at(index)
        _remove_artist(removed if removed is not None else self._selected)
        self._clear_selection()
        self._notify_change('delete')
        self._status(f'Đã xoá vật cản {index}.')
        return True

    def resize_selected(self, *, radius=None, length=None, width=None):
        """Đổi kích thước vật đang chọn; chỉ báo thay đổi khi thật sự đổi."""
        self._abort_drag()
        patch = self._selected
        if patch is None:
            return False
        if self.obstacles.index_of(patch) is None:
            self._clear_selection()
            self._status('Vật cản đang chọn không còn tồn tại.')
            return False
        before = ((patch.radius,) if isinstance(patch, Circle) else
                  (patch.get_width(), patch.get_height()))
        self.obstacles.resize(patch, radius=radius, length=length, width=width)
        after = ((patch.radius,) if isinstance(patch, Circle) else
                 (patch.get_width(), patch.get_height()))
        self._sync_highlight()
        if before == after:
            return False
        self._notify_change('resize')
        self._status('Đã đổi kích thước vật cản.')
        return True

    def clear_all(self):
        """Xoá toàn bộ vật cản; trả ``True`` nếu collection có dữ liệu."""
        self._abort_drag()
        count = len(self.obstacles.obs)
        if count == 0:
            self._clear_selection()
            self._status('Không có vật cản để xoá.')
            return False

        old_patches = list(self.obstacles.obs)
        self.obstacles.clear()
        for patch in old_patches:
            _remove_artist(patch)
        self._clear_selection()
        self._notify_change('clear')
        self._status(f'Đã xoá tất cả {count} vật cản.')
        return True

    def cancel(self):
        """Huỷ preview/đang kéo và bỏ chọn."""
        self._abort_drag()
        self._clear_selection()
        self._status('Đã huỷ thao tác.')

    def refresh(self):
        """Đồng bộ selection/highlight/status với collection bên ngoài."""
        if self._drag_kind == 'move' and self._drag_patch is not None:
            if self.obstacles.index_of(self._drag_patch) is None:
                self._abort_drag()

        if self._drag_kind is None and self._preview is not None:
            self._remove_preview()

        if self._selected is not None:
            index = self.obstacles.index_of(self._selected)
            if index is None:
                self._clear_selection()
                self._status(
                    f'Vật cản đã chọn không còn tồn tại. '
                    f'Số vật cản: {len(self.obstacles.obs)}.')
                return
            self._sync_highlight()
            self._status(
                f'Đã chọn vật cản {index}. '
                f'Số vật cản: {len(self.obstacles.obs)}.')
        else:
            self._remove_highlight()
            self._status(f'Số vật cản: {len(self.obstacles.obs)}.')

    # ------------------------------------------------------------------
    # Xử lý sự kiện chuột
    # ------------------------------------------------------------------
    def on_press(self, event):
        """Bắt đầu chọn/di chuyển/vẽ vật cản."""
        if not self._is_left(event):
            return False

        if getattr(event, 'inaxes', None) is not self.ax:
            return False
        point = self._point(event)
        if point is None:
            return False

        self._abort_drag()

        if self._mode == self.MODE_SELECT:
            return self._press_select(point)

        if self._mode in (self.MODE_RECT, self.MODE_CIRCLE):
            return self._press_create(point)

        return False

    def on_motion(self, event):
        """Cập nhật preview hoặc vị trí vật cản đang kéo."""
        if not self._is_left(event):
            return False
        if self._drag_kind is None:
            return False

        point = self._point(event)
        in_main = getattr(event, 'inaxes', None) is self.ax

        if self._drag_kind == 'move':
            if in_main and point is not None:
                self._apply_move(point)
            return True

        if self._drag_kind in ('create_rect', 'create_circle'):
            if in_main and point is not None:
                self._update_preview(self._drag_kind, self._drag_start, point)
                self._drag_last = point
            return True

        return False

    def on_release(self, event):
        """Kết thúc thao tác kéo-thả và tạo/di chuyển vật cản nếu hợp lệ."""
        if not self._is_left(event):
            return False
        kind = self._drag_kind
        if kind is None:
            return False

        point = self._point(event)
        in_main = getattr(event, 'inaxes', None) is self.ax

        if kind == 'move':
            if in_main and point is not None:
                self._apply_move(point)
            moved = self._drag_moved
            pending_select = self._pending_select
            self._reset_drag()
            if moved:
                self._notify_change('move')
                self._status('Đã di chuyển vật cản.')
            elif pending_select:
                self._notify_change('select')
                self._status('Đã chọn vật cản.')
            else:
                # Click lại vật đang chọn: không đổi dữ liệu, chỉ chỉnh status.
                self._status('Đã chọn vật cản.')
            return True

        if kind in ('create_rect', 'create_circle'):
            if not in_main or point is None:
                self._remove_preview()
                self._reset_drag()
                self._status('Đã huỷ tạo vật cản.')
                return True

            self._update_preview(kind, self._drag_start, point)
            self._drag_last = point
            patch = self._finish_create(kind)
            self._remove_preview()
            self._reset_drag()
            if patch is None:
                return True

            self._add_obstacle(patch)
            self._set_selected(patch)
            self._notify_change('create')
            if kind == 'create_rect':
                self._status(
                    f'Đã tạo hình chữ nhật '
                    f'({patch.get_width():.2f} × '
                    f'{patch.get_height():.2f}).')
            else:
                self._status(f'Đã tạo hình tròn (r = {patch.radius:.2f}).')
            return True

        return False

    # ------------------------------------------------------------------
    # Các bước xử lý nội bộ
    # ------------------------------------------------------------------
    def _press_select(self, point):
        """Chọn vật cản trên cùng và chuẩn bị di chuyển."""
        index = self.obstacles.pick(point)
        if index is None:
            had_selection = self._selected is not None
            self._clear_selection()
            if had_selection:
                self._notify_change('select')
                self._status('Đã bỏ chọn vật cản.')
            return False

        patch = self.obstacles.obs[index]
        was_selected = patch is self._selected
        self._set_selected(patch)
        self._drag_kind = 'move'
        self._drag_patch = patch
        self._drag_original_anchor = tuple(
            patch.get_xy() if isinstance(patch, Rectangle) else patch.center)
        self._drag_start = point
        self._drag_last = point
        self._drag_moved = False
        self._pending_select = not was_selected
        self._status(f'Đang di chuyển vật cản {index}...')
        return True

    def _press_create(self, point):
        """Bắt đầu vẽ hình mới từ một góc/tâm."""
        kind = 'create_rect' if self._mode == self.MODE_RECT else (
            'create_circle')
        self._drag_kind = kind
        self._drag_start = point
        self._drag_last = point
        self._drag_moved = False
        self._pending_select = False
        self._update_preview(kind, point, point)
        if kind == 'create_rect':
            self._status('Kéo để vẽ hình chữ nhật.')
        else:
            self._status('Kéo từ tâm để vẽ hình tròn.')
        return True

    def _apply_move(self, point):
        """Dịch vật cản theo hiệu vị trí so với lần cập nhật trước."""
        # Vật cản có thể đã bị gỡ khỏi collection từ bên ngoài khi đang kéo.
        if (self._drag_patch is None
                or self.obstacles.index_of(self._drag_patch) is None):
            self._abort_drag()
            self.refresh()
            return

        last_x, last_y = self._drag_last
        dx = point[0] - last_x
        dy = point[1] - last_y
        if abs(dx) <= 1e-12 and abs(dy) <= 1e-12:
            return
        self.obstacles.translate(self._drag_patch, dx, dy)
        self._drag_last = point
        self._drag_moved = True
        self._sync_highlight()

    def _finish_create(self, kind):
        """Tạo patch từ preview hiện tại, hoặc ``None`` nếu quá nhỏ."""
        start = self._drag_start
        end = self._drag_last
        if start is None or end is None:
            return None

        if kind == 'create_rect':
            x0 = min(start[0], end[0])
            y0 = min(start[1], end[1])
            width = abs(end[0] - start[0])
            height = abs(end[1] - start[1])
            if (width < self.min_rect_size
                    or height < self.min_rect_size):
                self._status('Kích thước quá nhỏ, đã huỷ tạo vật cản.')
                return None
            return Rectangle((x0, y0), width, height, color='gray')

        radius = math.hypot(end[0] - start[0], end[1] - start[1])
        if radius < self.min_circle_radius:
            self._status('Bán kính quá nhỏ, đã huỷ tạo vật cản.')
            return None
        return Circle(start, radius, color='gray')

    def _add_obstacle(self, patch):
        """Thêm patch vào collection và bảo đảm nó đã nằm trên bản đồ."""
        self.obstacles.add(patch)
        if getattr(patch, 'axes', None) is None:
            self.ax.add_patch(patch)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def _ensure_preview(self, kind):
        """Bảo đảm có đúng một patch preview thuộc ``kind``."""
        if (self._preview is not None
                and self._preview_kind == kind
                and getattr(self._preview, 'axes', None) is self.ax):
            return self._preview
        self._remove_preview()
        if kind == 'create_rect':
            patch = Rectangle(
                (0.0, 0.0), 0.0, 0.0,
                fill=False, edgecolor=_PREVIEW_COLOR, linestyle='--',
                linewidth=1.5, zorder=_PREVIEW_ZORDER,
            )
        else:
            patch = Circle(
                (0.0, 0.0), 0.0,
                fill=False, edgecolor=_PREVIEW_COLOR, linestyle='--',
                linewidth=1.5, zorder=_PREVIEW_ZORDER,
            )
        self.ax.add_patch(patch)
        self._preview = patch
        self._preview_kind = kind
        return patch

    def _update_preview(self, kind, start, end):
        """Cập nhật preview khi đang vẽ từ 2 điểm."""
        patch = self._ensure_preview(kind)
        if kind == 'create_rect':
            x0 = min(start[0], end[0])
            y0 = min(start[1], end[1])
            patch.set_xy((x0, y0))
            patch.set_width(abs(end[0] - start[0]))
            patch.set_height(abs(end[1] - start[1]))
        else:
            patch.set_center(start)
            patch.set_radius(
                math.hypot(end[0] - start[0], end[1] - start[1]))

    def _remove_preview(self):
        """Gỡ preview khỏi axes và quên trạng thái của nó."""
        _remove_artist(self._preview)
        self._preview = None
        self._preview_kind = None

    # ------------------------------------------------------------------
    # Selection/highlight
    # ------------------------------------------------------------------
    def _set_selected(self, patch):
        """Đặt patch đang chọn và vẽ highlight màu cam."""
        if patch is None:
            self._clear_selection()
            return
        if patch is not self._selected:
            self._remove_highlight()
            self._selected = patch
        self._sync_highlight()

    def _clear_selection(self):
        """Bỏ chọn và gỡ highlight."""
        self._selected = None
        self._remove_highlight()

    def _sync_highlight(self):
        """Cập nhật highlight theo hình dạng/vị trí hiện tại của patch."""
        patch = self._selected
        if patch is None:
            self._remove_highlight()
            return

        if self.obstacles.index_of(patch) is None:
            self._selected = None
            self._remove_highlight()
            return

        if (self._highlight is not None
                and getattr(self._highlight, 'axes', None) is not self.ax):
            self._highlight = None

        rotation_point = getattr(patch, 'rotation_point', 'xy')
        if self._highlight is not None:
            same_kind = (
                isinstance(patch, Rectangle)
                and isinstance(self._highlight, Rectangle)
            ) or (
                isinstance(patch, Circle)
                and isinstance(self._highlight, Circle)
            )
            same_rotation = (
                not isinstance(patch, Rectangle)
                or getattr(self._highlight, 'rotation_point', 'xy')
                == rotation_point
            )
            if not (same_kind and same_rotation):
                self._remove_highlight()

        if self._highlight is None:
            self._highlight = self._make_highlight(patch, rotation_point)
            self.ax.add_patch(self._highlight)
            return

        if isinstance(patch, Rectangle):
            self._highlight.set_xy(patch.get_xy())
            self._highlight.set_width(patch.get_width())
            self._highlight.set_height(patch.get_height())
            self._highlight.set_angle(patch.get_angle())
        else:
            self._highlight.set_center(patch.center)
            self._highlight.set_radius(patch.radius)
        self._highlight.set_zorder(
            max(_HIGHLIGHT_ZORDER, patch.get_zorder() + 1))

    def _make_highlight(self, patch, rotation_point):
        """Tạo overlay nét đứt màu cam, không sửa style patch gốc."""
        zorder = max(_HIGHLIGHT_ZORDER, patch.get_zorder() + 1)
        if isinstance(patch, Rectangle):
            return Rectangle(
                patch.get_xy(), patch.get_width(), patch.get_height(),
                angle=patch.get_angle(), rotation_point=rotation_point,
                fill=False, edgecolor=_HIGHLIGHT_COLOR, linestyle='--',
                linewidth=2.0, zorder=zorder,
            )
        return Circle(
            patch.center, patch.radius,
            fill=False, edgecolor=_HIGHLIGHT_COLOR, linestyle='--',
            linewidth=2.0, zorder=zorder,
        )

    def _remove_highlight(self):
        """Gỡ overlay highlight khỏi axes."""
        _remove_artist(self._highlight)
        self._highlight = None

    # ------------------------------------------------------------------
    # Trạng thái kéo-thả
    # ------------------------------------------------------------------
    def _reset_drag(self):
        """Xoá toàn bộ trạng thái kéo-thả (không đụng selection/preview)."""
        self._drag_kind = None
        self._drag_patch = None
        self._drag_original_anchor = None
        self._drag_start = None
        self._drag_last = None
        self._drag_moved = False
        self._pending_select = False

    def _abort_drag(self):
        """Huỷ thao tác đang dở và gỡ preview."""
        # Motion dịch patch thật; huỷ phải khôi phục vị trí trước khi kéo,
        # để label và trạng thái tránh vật cản vẫn khớp mà không cần callback.
        patch = self._drag_patch
        if (self._drag_kind == 'move' and self._drag_moved
                and self._drag_original_anchor is not None
                and self.obstacles.index_of(patch) is not None):
            current = (patch.get_xy() if isinstance(patch, Rectangle)
                       else patch.center)
            original = self._drag_original_anchor
            self.obstacles.translate(
                patch, original[0] - current[0], original[1] - current[1])
            self._sync_highlight()
        self._remove_preview()
        self._reset_drag()

    # ------------------------------------------------------------------
    # Tiện ích sự kiện/callback
    # ------------------------------------------------------------------
    @staticmethod
    def _is_left(event):
        """Chỉ nhận chuột trái (motion event thường có button=None)."""
        button = getattr(event, 'button', None)
        return button is None or button == 1 or button == '1'

    @staticmethod
    def _point(event):
        """Lấy ``(x, y)`` dữ liệu hữu hạn từ event, hoặc ``None``."""
        x = getattr(event, 'xdata', None)
        y = getattr(event, 'ydata', None)
        if x is None or y is None:
            return None
        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        return (x, y)

    def _notify_change(self, reason):
        """Báo GUI đúng một lần cho mỗi thao tác đã hoàn tất."""
        if callable(self.on_change):
            self.on_change(reason)

    def _status(self, text):
        """Cập nhật dòng trạng thái tiếng Việt cho GUI."""
        if callable(self.on_status):
            self.on_status(text)
