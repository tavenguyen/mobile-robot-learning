# Two-wheel robot GUI

Mô phỏng robot hai bánh vi sai theo Bài thực hành 1, kèm auto-mode tới goal, ba thuật toán né tránh vật cản của BTH2.1 (Bug1, Bug2, Tangent Bug) và các planner DWA, Dijkstra, A*, A* + DWA.

## Chạy

Từ thư mục gốc dự án, trong môi trường `mobile_robot`:

```powershell
python -m src.main
# Hoặc:
python src/main.py
python -m unittest -v
```

Phụ thuộc: NumPy, Matplotlib (requirements.txt) và Python Tkinter với Tcl/Tk tương thích. Phiên kiểm chứng hiện tại không có Tkinter/display nên chỉ chạy được kiểm tra headless; nếu cửa sổ không mở được trên máy local, kiểm tra cài đặt Tk trong môi trường Conda.

## Controls

- Forward/Backward: đặt `v_l = v_r = ±1.5` m/s; robot chạy liên tục.
- Left: `v_l = 0.5`, `v_r = 1.5` m/s → quay trái với `ω = 1.0` rad/s (với `L = 1.0` m). Right dùng `v_l = 1.5`, `v_r = 0.5`.
- Stop: đặt `v_l = v_r = 0`.
- Reset: đưa robot về `(0, 0, 0)`, xóa quỹ đạo, goal/marker, auto-mode và cảnh báo va chạm.
- Bàn phím: `↑ ↓ ← →`, `Space` (Stop), `R` (Reset).
- Auto-mode: click chuột lên bản đồ để đặt goal (sao đỏ) và bật auto; robot tự quay và tiến tới goal theo P-controller.
- Nút `Auto-mode`: bấm khi đang ON để dừng (giữ goal), bấm lần nữa để chạy tiếp. Chưa có goal thì không bật được.
- Goal được giữ sau khi tới đích hoặc va chạm. Tới trong 0.3 m: dừng, tắt auto và báo "Destination reached!".
- Các nút điều khiển tay (và phím) tự tắt auto nhưng giữ goal.
- Va chạm: GUI dự đoán pose kế tiếp bằng Euler. Chế độ "Tắt né tránh" dừng robot và hiện "⚠ Collision detected!" nhưng auto vẫn ON. Khi bật Bug1/Bug2/Tangent Bug, controller nhận contact/range sensor và tự lùi ra rồi bám biên vật cản; auto vẫn ON trong lúc né tránh.
- Click khi toolbar đang ở chế độ zoom/pan được bỏ qua để không đặt goal nhầm.

## Chỉnh sửa vật cản

Bản đồ dùng toàn bộ vùng biểu đồ; hình tròn luôn hiển thị đúng tỉ lệ:

- Chọn công cụ **Vẽ chữ nhật** hoặc **Vẽ hình tròn** rồi kéo chuột trên bản đồ để vẽ kích thước tuỳ ý. Preview hiện nét đứt trong lúc kéo; `Esc` huỷ.
- Công cụ **Chọn / Di chuyển**: click vật cản để chọn (highlight cam), kéo để di chuyển; nhấn `Esc` hoặc đổi công cụ khi đang kéo để trả về vị trí ban đầu. Click vùng trống để bỏ chọn; click vùng trống vẫn đặt goal/auto như cũ.
- Chọn vật cản rồi nhập **R (m)** hoặc **Length (m)** / **Width (m)** và bấm **Áp dụng** (hoặc Enter). Length theo X, Width theo Y khi chưa xoay; đổi kích thước giữ nguyên tâm và góc xoay. R tối thiểu 0,1 m, mỗi cạnh tối thiểu 0,2 m.
- Nút **Xoá vật cản** hoặc phím `Delete`/`Backspace` xoá vật đang chọn. Nút **Xoá tất cả** xoá toàn bộ vật cản sau khi xác nhận.
- Vật cản vẫn là đối tượng tương tác của bản đồ: có thể chọn, di chuyển, đổi kích thước, xoá; label `Obs i` cập nhật theo vị trí và thứ tự.
- Thêm/xoá/di chuyển/đổi kích thước vật cản khi auto-mode đang chạy sẽ reset controller né tránh tại pose hiện tại và xoá overlay cũ. Nếu vật cản mới chồng lên robot, lần cập nhật kế tiếp sẽ báo va chạm.
- `Reset` chỉ đưa robot về gốc, không xoá vật cản. Dùng **Xoá tất cả** nếu muốn làm sạch bản đồ.

Checklist kiểm tra tay: `tests/manual_obstacle_editor.md`.

Robot mặc định dài 2.0 m, rộng 1.0 m, hai bánh 0.5 × 0.5 m, `L = 1.0` m. Sửa trong `build_scene()` ở `src/main.py`. Cửa sổ hiển thị Path (xanh), đường né tránh (cam), Robot, ba vật cản mẫu và các vector vận tốc bánh.

## Né tránh vật cản (BTH2.1)

Chọn thuật toán ở combobox **Thuật toán**:

- **Tắt né tránh** (mặc định): chỉ dùng `goal_command` như BTH1.
- **Bug1**: contact sensor phát hiện pose kế tiếp sắp chạm thì lùi ra biên offset và bám **đủ một vòng** quanh vật cản. Trong lúc đi, controller ghi nhận dần điểm trên biên gần goal nhất; **hết vòng mới chốt leave point L** (sao xanh). Trước đó GUI chưa hiện “điểm rời dự kiến”. Sau khi chốt L, robot quay lại L rồi đi thẳng tới goal. Hit point hiển thị bằng X cam, biên offset bằng nét đứt xám.
- **Bug2**: giữ **m-line** (nét đứt hồng) từ vị trí bắt đầu auto tới goal. Robot bám biên tới khi cắt lại m-line tại **Q** gần goal hơn hit point **H**, rồi rời tại Q.
- **Tangent Bug**: range sensor 360° (72 tia, hiển thị các tia xanh nhạt) chọn khoảng trống có `h(p) = d(robot, p) + d(p, goal)` nhỏ nhất, vẽ **tia tiếp tuyến** (cam) tới đầu mút và bám biên offset theo tiếp tuyến. Khi đường tới goal không còn bị chặn (line-of-sight), robot ghi **leave point** và đi thẳng tới goal. Đường tới goal được vẽ nét chấm xanh, đường rời nét gạch chấm.

Goal nằm trong vật cản hoặc quá sát vật cản (tâm robot sẽ chạm) được báo **FAILED** ngay thay vì né tránh vô hạn. Đổi thuật toán khi auto đang ON sẽ reset controller tại pose hiện tại và giữ nguyên goal.

## LiDAR mô phỏng

`src/lidar.py` cung cấp class `Lidar` quét vật cản tĩnh và trả đồng thời hai khung toạ độ:

- `world_points`: toạ độ toàn cục của điểm chạm; vật cản vẫn nằm cố định trong `Obstacle`.
- `local_points`: toạ độ trong khung robot, tâm robot `(0, 0)`, `+x` hướng trước, `+y` hướng trái.

```python
from src.lidar import Lidar

lidar = Lidar(obstacles, num_beams=72, max_range=10.0,
              noise_std=0.02, seed=42)
scan = lidar.scan(robot.x, robot.y, robot.theta)

print(scan.world_points.shape)   # (72, 2)
print(scan.local_points.shape)   # (72, 2)
fig, axes = lidar.plot_frames(scan)   # hai khung global/local
```

- Mặc định 72 tia; truyền `num_beams=360` khi cần xem mịn.
- Nhiễu Gaussian cộng vào khoảng cách tia trúng, đơn vị mét; `seed` để tái lập.
- `plot_frames` vẽ khung toàn cục bên trái và khung cục bộ bên phải; trả `(fig, (ax_global, ax_local))`.
- Script xem riêng, không phụ thuộc mô phỏng né vật cản:

```bash
python -m src.lidar_demo
python -m src.lidar_demo --beams 360
python -m src.lidar_demo --animate
```

## Phạm vi

Đã cài đặt BTH1: động học vi sai, mô phỏng thời gian thực, điều khiển bằng nút/bàn phím, phát hiện va chạm. Đã cài đặt auto-mode của BTH2.1: click đặt goal, nút bật/tắt, P-controller và dừng khi tới đích. Đã cài đặt ba thuật toán né tránh chọn từ GUI: Bug1, Bug2 (contact sensor) và Tangent Bug (range sensor 360°), kèm hiển thị hit/leave point, m-line, biên offset, tia tiếp tuyến và đường rời. Đã cài đặt chỉnh sửa vật cản trên bản đồ: vẽ hình chữ nhật/hình tròn, chọn, di chuyển, đổi kích thước và xoá. Đã cài đặt LiDAR 2D mô phỏng riêng với hai khung toạ độ global/local. Chưa cài đặt nhiễu vận tốc bánh, xoay vật cản, undo/redo hoặc lưu bản đồ.

## Source layout

- src/Robot.py: class `Differential_Robot` — động học vi sai, lịch sử quỹ đạo, hình họa và `get_robot_corners`.
- src/Obstacle.py: class `Obstacle` — quản lý Rectangle/Circle, sinh vật cản ngẫu nhiên, kiểm tra va chạm (SAT và circle), `nearest`/`contact_at`/`contains`.
- src/GUI.py: class `RobotSimulatorGUI` — cửa sổ Tkinter + canvas Matplotlib, điều khiển tay, auto-mode, combobox né tránh, hiển thị vận tốc, cảnh báo va chạm và overlay thuật toán.
- src/obstacle_editor.py: class `ObstacleEditor` — vẽ/chọn/di chuyển/đổi kích thước/xoá vật cản, preview và highlight; không phụ thuộc Tkinter.
- src/navigation.py: `goal_command` — P-controller tới goal; `point_command` — lái tới waypoint/tiếp tuyến.
- src/avoidance.py: `Bug1Controller`, `Bug2Controller`, `TangentBugController` — state machine GOAL/RETREAT/FOLLOW/RECOVER/LEAVE và dữ liệu debug cho GUI.
- src/sensors.py: `RangeSensor` 360°, raycast Rectangle/Circle, `clear_path`, `first_blocker`, nhóm khoảng trống và đầu mút tiếp tuyến.
- src/lidar.py: class `Lidar` — quét vật cản tĩnh, trả điểm trong khung toàn cục và khung cục bộ; nhiễu khoảng cách; `plot_frames` vẽ hai khung.
- src/lidar_demo.py: script xem mô phỏng LiDAR riêng (tĩnh hoặc động), không phụ thuộc GUI né vật cản.
- src/boundary.py: `offset_polygon`/`offset_boundary` (biên Minkowski bo góc) và `BoundaryPath` theo độ dài cung.
- src/main.py: điểm chạy, tạo robot và ba vật cản mẫu.
- src/geometry.py, src/simulation.py, src/visualization.py, src/robot_drawing.py: thư viện mô phỏng hàng loạt (xem mục Reuse).
- src/controller.py: bộ điều khiển của bản GUI cũ, hiện chỉ còn dùng trong tests.
- tests/: động học, hình học, vật cản, va chạm, auto-mode, sensor/biên offset, LiDAR, ba thuật toán né tránh và mô phỏng hàng loạt.

## Reuse

```python
from src import simulate, get_robot_corners

times, states = simulate((0.1, 0.2), duration=10, dt=0.02,
                        noise_std=(0.005, 0.05), seed=42)
corners = get_robot_corners(states[-1])
```

States are (x,y,theta), in meters and radians. Euler equations: x+=dt*v*cos(theta), y+=dt*v*sin(theta), theta+=dt*w. Noise is added to v,w before integration. Standard deviations apply per sample, not scaled by sqrt(dt); changing dt changes accumulated noise. Batch noise defaults to zero. A fixed seed reproduces results.

`src.geometry.get_robot_corners` returns four world-frame vertices, counterclockwise: rear-right, front-right, front-left, rear-left. Dimensions default to 0.05 m each. The default footprint includes wheels, excludes arrows; `include_wheels=False` returns body corners only. `Differential_Robot.get_robot_corners` uses the robot's own dimensions and, by default, its current pose. Neither performs collision detection or mapping. Collision checking must consider edges and interior, not only vertices.

No acceleration, wheel slip or ground contact dynamics are modeled. The class-based GUI uses the same Euler step; rendering checks under Agg do not establish that an interactive Tk window works.

## Dynamic Window Approach (DWA)

Chọn **DWA** trong combobox né tránh rồi click vùng trống để đặt goal. DWA lấy mẫu các cặp `(v, w)` trong giới hạn vận tốc/gia tốc, dự đoán quỹ đạo và loại ứng viên va chạm hoặc không thể phanh an toàn. Vận tốc bánh được tính từ cặp tốt nhất.

Gần goal, DWA giới hạn tốc độ theo khoảng cách/horizon và thêm mẫu tốc độ góc hướng về goal để tránh bỏ sót góc quay nhỏ; các mẫu vẫn thuộc cửa sổ động.

Điểm `G = α × heading + β × dist + γ × velocity`, với mặc định α=0.6, β=0.3, γ=0.1. Heading đo độ hướng về goal tại cuối quỹ đạo; dist là khoảng hở nhỏ nhất chia 3 m và giới hạn ở 1; velocity là v/2. Ba thành phần nằm trong [0,1]. Nhập trọng số không âm, hữu hạn và không đồng thời bằng 0, rồi bấm Áp dụng để dùng ở bước tiếp theo.

Bảng cuộn hiển thị các ứng viên hợp lệ theo score giảm dần, đánh dấu cặp được chọn. Map hiển thị quỹ đạo dự đoán tốt nhất; bảng cập nhật tối đa 5 lần/giây. DWA dùng vòng tròn bao robot cộng 0.15 m nên có thể từ chối khe hẹp. Đây là planner cục bộ, không bảo đảm thoát mọi cực trị cục bộ. Khi không còn ứng viên an toàn, robot dừng và tắt auto; sửa map rồi bật lại để tiếp tục.

## Global planner: Dijkstra và A*

Chọn **Dijkstra** hoặc **A*** ở combobox thuật toán rồi click vùng trống đặt goal. Global planner tìm đường trên lưới 0.5 m trong map [-20,20] × [-20,20], mở rộng vật cản theo vòng tròn bao robot cộng biên an toàn. Đường được vẽ riêng trên map; Dijkstra và A* dùng cùng bộ P bám waypoint, không kết hợp DWA/Bug.

Robot lần lượt tiến tới từng waypoint, giảm tốc khi gần điểm và quay tại chỗ ở góc rẽ. Bộ bám đường không tự chọn đường tránh vật cản mới. Sửa vật cản sẽ vô hiệu đường cũ và yêu cầu tính lại; không tìm được đường hoặc gặp va chạm thì robot dừng và báo trạng thái. Lưới và mô hình vòng tròn bao bảo thủ có thể từ chối khe hẹp.

### Heuristic của A*

A* dùng khoảng cách **octile** cho lưới 8 hướng: `max(dx, dy) + (sqrt(2)-1)*min(dx, dy)`, với dx/dy tính bằng mét. Chi phí cạnh ngang/dọc bằng resolution, cạnh chéo bằng resolution × sqrt(2).

Goal có thể nằm giữa các node lưới. Vì vậy heuristic tại node n là giá trị nhỏ nhất của `octile(n, q) + Euclid(q, goal)` trên các node q nối an toàn tới goal. Bảng heuristic được tính trước; không áp dụng octile trực tiếp tới goal ngoài lưới. Cách này giữ cận dưới chi phí và tính nhất quán trên đồ thị tìm kiếm. Dijkstra vẫn dùng h=0. Không thay đổi bộ P bám waypoint.

## Kết hợp A* + DWA

Chọn **A* + DWA**: A* với heuristic Octile tìm đường global, DWA bám mục tiêu cục bộ trên đường để chọn vận tốc. Chế độ này không dùng bộ P hoặc Bug. Đường A* màu xanh nét đứt, mục tiêu cục bộ và quỹ đạo DWA được hiển thị đồng thời; bảng ứng viên và trọng số dùng chung với DWA.

Giữ `G = α·heading + β·dist + γ·velocity`; heading hướng tới mục tiêu cục bộ, còn tới đích chỉ xét goal cuối (0.3 m). Lookahead mặc định1.5m giới hạn ở góc rẽ kế tiếp, không nhảy sang nhánh gần của chữ U. Global margin0.25m, local margin0.15m cộng bán kính bao robot. Khi sửa map/đổi goal, dừng và tính lại đường; đổi trọng số không chạy lại A*.

Không tiến thêm0.1m trong5giây mô phỏng hoặc không có vận tốc an toàn: dừng và thử tính lại đường tối đa2 lần liên tiếp. Không tìm được đường hoặc vẫn kẹt thì tắt auto và thông báo. Chế độ này chưa bảo đảm hội tụ mọi map; vật cản được biết qua map/editor, chưa có vật cản tự chuyển động.

Trong chế độ kết hợp, sau khi lọc an toàn, nếu tồn tại ứng viên có điểm cuối tiến ít nhất0.01m theo hướng đoạn đường đang bám, chỉ chấm G trên nhóm tiến đó; nếu không có, vẫn giữ các ứng viên an toàn để robot quay căn hướng. Đây là chính sách tiến độ riêng của hybrid nhằm tránh đứng yên thắng điểm khoảng hở; DWA độc lập không dùng bộ lọc này, và công thức G không đổi.
