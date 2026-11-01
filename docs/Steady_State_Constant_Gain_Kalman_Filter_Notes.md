# Steady-state (Constant-gain) Kalman Filter

## Tổng quan

Steady-state Kalman Filter (hay Constant-gain Kalman Filter) là một biến thể của Kalman Filter, trong đó **Kalman Gain (K)** và **Covariance Matrix (P)** sau một thời gian sẽ hội tụ đến giá trị gần như không đổi. Thay vì tính lại chúng ở mỗi frame, ta tính trước (offline) rồi dùng lại.

Ý tưởng:

> Sau khi K và P ổn định, chỉ cần dự đoán trạng thái rồi cập nhật bằng một `K_fixed`.

## Kalman Filter thông thường

Mỗi frame gồm hai bước.

### Prediction

\[
x' = Fx
\]

\[
P' = FPF^T + Q
\]

- `x`: trạng thái
- `F`: ma trận chuyển trạng thái
- `P`: ma trận hiệp phương sai
- `Q`: nhiễu quá trình

### Update

Khi detector (YOLO) trả về measurement `z`:

\[
S = HP'H^T + R
\]

\[
K=P'H^TS^{-1}
\]

\[
x=x'+K(z-Hx')
\]

\[
P=(I-KH)P'
\]

Điều này yêu cầu tính:
- P
- S
- nghịch đảo ma trận
- K

ở **mọi frame, mọi track**.

---

## Steady-state Kalman Filter

Sau nhiều frame:

- P gần như không đổi.
- K cũng gần như không đổi.

Do đó:

```python
x_pred = F_step(x)
x = x_pred + K_fixed @ (z - x_pred[:4])
```

Không cần:

- cập nhật P
- tính S
- tính nghịch đảo
- tính K

### Innovation

```text
innovation = measurement - prediction
```

Ví dụ:

Prediction:

```
(102,198)
```

Measurement:

```
(100,200)
```

Innovation:

```
(-2,+2)
```

Kalman chỉ điều chỉnh prediction theo innovation.

---

## Ưu điểm

- Giảm đáng kể phép nhân ma trận.
- Không cần phép nghịch đảo.
- Giảm CPU trên hệ thống có nhiều track.
- Dễ triển khai trên thiết bị edge.

---

## Nhược điểm

Giả sử đặc tính nhiễu không đổi.

Nếu:

- camera rung
- detector kém ổn định
- điều kiện thay đổi

thì `K_fixed` không còn tối ưu.

---

## Vì sao FastTracker dùng nhiều K?

Mỗi loại đối tượng có chuyển động khác nhau:

- Person
- Car
- Bus

=> nên có thể dùng:

- K_person
- K_car
- K_bus

---

## ByteTrack và chiều cao Bounding Box

ByteTrack điều chỉnh noise theo chiều cao bounding box.

Ví dụ:

- gần camera → bbox lớn
- xa camera → bbox nhỏ

Sai số detector theo tỷ lệ sẽ khác nhau.

Do đó có thể dùng nhiều `K_fixed` theo nhóm chiều cao (height buckets):

| Height | Gain |
|---------|------|
| 0–64 px | K_small |
| 64–128 px | K_medium |
| 128–256 px | K_large |
| >256 px | K_xlarge |

---

# Đánh giá cho bài toán giảng viên

Đặc điểm:

- Camera cố định
- Chỉ detect người
- Người chuyển động chậm
- Khoảng cách khoảng 0–20 m
- Muốn chạy CPU

## Phù hợp?

Có.

Các giả định của Constant-gain Kalman gần như đều được đáp ứng.

### Constant Velocity

Giảng viên thường:

- đi
- dừng
- quay người

không:

- chạy
- tăng tốc mạnh

=> mô hình Constant Velocity hoạt động tốt.

### Camera cố định

Noise của detector ổn định hơn.

P và K hội tụ nhanh.

### Chỉ theo dõi Person

Không cần nhiều bộ tham số theo class.

---

## Lưu ý về khoảng cách 0–20m

Chiều cao bbox thay đổi khá nhiều.

Ví dụ:

- gần: ~600 px
- 5m: ~250 px
- 20m: ~70 px

Nếu chỉ dùng **một** `K_fixed`, hiệu quả có thể giảm ở các khoảng cách rất khác nhau.

Khuyến nghị dùng nhiều K theo height bucket.

Ví dụ:

- <64 px
- 64–128 px
- 128–256 px
- >256 px

---

## Occlusion

Steady-state Kalman vẫn dự đoán tốt trong thời gian mất detection ngắn.

Khả năng giữ ID chủ yếu phụ thuộc:

- Hungarian Matching
- IoU
- ReID
- Target Memory

không phải K.

---

## Lợi ích trên CPU

Kalman thường:

- cập nhật P
- tính S
- inverse
- tính K

Steady-state:

```python
x = x_pred + K_fixed @ innovation
```

Giảm đáng kể chi phí tính toán của tracker.

Lưu ý: trong pipeline YOLO + Tracker, detector vẫn là thành phần chiếm phần lớn thời gian xử lý.

---

# Kết luận

Đối với bài toán theo dõi giảng viên:

- Camera cố định ✔
- Người đi chậm ✔
- Chỉ detect person ✔
- CPU deployment ✔

Steady-state Constant-gain Kalman Filter là lựa chọn phù hợp.

Khuyến nghị thực tế:

**Không dùng một `K_fixed` duy nhất.**

Nên dùng **3–5 bộ `K_fixed` theo chiều cao bounding box** để cân bằng giữa:

- tốc độ
- độ chính xác
- khả năng thích nghi

Đây cũng là hướng tiếp cận gần với ByteTrack nhưng giảm đáng kể chi phí tính toán.
