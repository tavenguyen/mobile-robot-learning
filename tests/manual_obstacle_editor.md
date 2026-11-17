# Manual checklist — chỉnh sửa vật cản

Chạy trên máy có Tkinter/display:

```bash
python -m src.main
```

## Tạo vật cản
- [ ] Chọn công cụ “Vẽ chữ nhật”, kéo từ góc này sang góc kia: đúng kích thước, kể cả kéo ngược hướng.
- [ ] Chọn công cụ “Vẽ hình tròn”, kéo từ tâm ra: đúng bán kính.
- [ ] Kéo quá nhỏ: không tạo vật cản, status báo kích thước không hợp lệ.
- [ ] Nhấn Esc khi đang kéo: huỷ preview, không tạo vật cản.

## Chọn và di chuyển
- [ ] Click vật cản: highlight cam, nút “Xoá vật cản” bật.
- [ ] Kéo vật cản: di chuyển theo chuột, giữ nguyên hình dạng.
- [ ] Đang kéo vật cản, nhấn Esc hoặc đổi công cụ: vật cản trở về vị trí ban đầu; label và highlight khớp vị trí.
- [ ] Click vùng trống: bỏ chọn, nút xoá tắt; goal vẫn được đặt như trước.
- [ ] Click vật cản không đặt goal nhầm.
- [ ] Vật cản chồng nhau: click chọn vật trên cùng.

## Kích thước
- [ ] Chọn hình tròn: chỉ R bật; chọn chữ nhật: chỉ Length/Width bật; bỏ chọn: tất cả tắt.
- [ ] Nhập kích thước và Áp dụng/Enter: giữ nguyên tâm, cập nhật highlight và label.
- [ ] Nhập rỗng, chữ, NaN, vô cực hoặc dưới mức tối thiểu: báo lỗi, hình không đổi.
- [ ] Backspace/Delete/mũi tên trong ô nhập hoặc bảng DWA không xoá vật cản hoặc điều khiển robot.

## Xoá
- [ ] Nút “Xoá vật cản” xoá vật đang chọn.
- [ ] Phím Delete/Backspace xoá vật đang chọn.
- [ ] Sau khi xoá: highlight mất, label `Obs i` cập nhật, nút xoá tắt nếu không còn chọn.
- [ ] “Xoá tất cả” có hộp xác nhận và xoá sạch map.
- [ ] Esc bỏ chọn.

## Tích hợp
- [ ] Toolbar zoom/pan không tạo hoặc di chuyển vật cản.
- [ ] Auto-mode vẫn đặt goal khi click vùng trống ở mode Chọn.
- [ ] Đang chạy Bug1/Bug2/Tangent Bug, thêm/xoá/di chuyển/đổi kích thước vật cản: controller reset, overlay cũ biến mất, robot tiếp tục hợp lý.
- [ ] Vẽ vật cản chồng lên robot: lần cập nhật sau báo va chạm và robot dừng.
- [ ] Reset robot: vật cản vẫn còn.
- [ ] Đóng cửa sổ: không lỗi, không còn preview/highlight.

## Layout
- [ ] Không còn palette; map dùng toàn bộ vùng biểu đồ, hình tròn không méo khi co giãn cửa sổ.
- [ ] Status label cập nhật theo thao tác.
- [ ] Cửa sổ co giãn không làm vỡ layout.

## DWA
- [ ] Chọn DWA: robot dừng trước khi chuyển, hiện trọng số và bảng ứng viên ngoài map.
- [ ] Đặt goal trước/sau robot, đường trống: tới goal và dừng; bảng giảm dần theo score, dòng chọn khớp quỹ đạo tím.
- [ ] Thay trọng số khi chạy: hiệu lực lần tiếp theo; số âm, NaN, vô cực hoặc cả ba bằng 0 bị từ chối.
- [ ] Thêm/xoá/move/resize vật cản: bảng và quỹ đạo cũ được xoá, tính lại.
- [ ] Goal bị chắn hoặc robot bị bao vây: dừng, thông báo rõ; sửa map rồi bật auto để thử lại.
- [ ] Đổi sang Bug1/Bug2/Tangent Bug: ẩn phần DWA, các điều khiển cũ hoạt động.
- [ ] Kiểm tra bảng cuộn, legend ngoài map và bố cục trên màn hình thấp; gõ trong ô trọng số không lái robot.

## Global planner
- [ ] Chọn Dijkstra/A*: dừng robot khi chuyển chế độ, ẩn DWA, đặt goal để tìm đường và bám waypoint.
- [ ] Map trống, vật cản trước mặt, bẫy chữ U: đường global hiển thị ngoài các vùng vật cản mở rộng; robot tới goal không cắt góc.
- [ ] Goal bị chặn/ngoài map hoặc không có đường: báo rõ, robot dừng.
- [ ] Sửa map khi chạy/tạm dừng: đường cũ mất, đường được tính lại theo map mới.
- [ ] Reset xoá đường, giữ vật cản; đổi sang DWA/Bug không còn overlay global.

## A* + DWA
- [ ] Chọn hybrid: global path, local target, DWA trajectory/table cùng hiển thị; legend ngoài map.
- [ ] Goal trước/sau, vật cản thẳng hàng, góc90độ và chữ U: tới đích, không nhảy nhánh/cắt qua vật cản.
- [ ] Đổi trọng số không dừng hoặc chạy lại A*; sửa map/goal dừng và tính lại, cả khi tạm dừng.
- [ ] Không đường/kẹt/va chạm: dừng có thông báo, phục hồi giới hạn; reset/đổi mode xoá dữ liệu cũ.
