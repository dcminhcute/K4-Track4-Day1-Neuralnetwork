# Báo cáo Lab Day 1 — Doan Quang Minh

> Tối đa khoảng 4 trang. **Chỉ viết cho những chủ đề bạn đã thử.**

## 1. Thiết lập

- Môi trường: Windows 11, PyTorch 2.11.0+cpu, CPU Intel
- Dữ liệu: Forest CoverType; `train` 464 809 / `eval` 116 203 theo `split_metadata.csv`. Validation: 20% của train (phân tầng, seed 42) → 371 847 train / 92 962 val.
- Model: `M-base` (54→256→128→7, 47 879 tham số). Baseline: SGD+momentum, lr=0.05, batch=512, epochs=20, He init.
- Mốc tham chiếu: accuracy "đoán lớp đa số" trên val = 0.4876.
- Các chủ đề đã thử: ☑ loss ☑ optimizer ☑ hyper-parameter ☑ dropout ☐ clipping ☑ init

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47 879 / (B, 7) |
| Loss bước 0 (so với ln 7 = 1.946) | 2.377 (He init cao hơn ln(7) một chút) |
| Quá khớp 20 mẫu: loss cuối | 0.007 (accuracy = 1.0) |
| Mọi tham số có gradient khác 0 | ☑ có |
| Baseline, số seed đã chạy | 3 seeds |
| Baseline: val acc (TB ± σ) | 0.899 ± 0.002 |
| Baseline: val macro-F1 (TB ± σ) | 0.841 ± 0.003 |

**Ngưỡng nhiễu dùng trong báo cáo:** 2σ = 0.006 (val macro-F1).

## 3. Kết quả theo chủ đề

### 3.1 Hàm mất mát — CE vs MSE

- **Dự đoán:** CE có gradient không bão hoà khi dự đoán sai nặng → hội tụ nhanh hơn MSE
- **Kết quả:**
  - `loss-mse`: val_macro_f1 = 0.6828, val_acc = 0.8549
  - `base-s1`: val_macro_f1 = 0.8405, val_acc = 0.8974
  - Chênh lệch: 0.1577 >> 2σ (0.006) → **CE tốt hơn rõ ràng**
- **Giải thích:** CE kết hợp với softmax cho gradient ổn định. MSE với softmax tạo ra gradient có thể bão hoà khi đầu ra xa nhãn đúng.

### 3.2 Bộ tối ưu hoá

- **Dự đoán:** Adam thường hội tụ nhanh hơn ở giai đoạn đầu do adaptive lr
- **Kết quả:**
  | exp_id | optimizer | lr | val_macro_f1 | val_acc |
  |---|---|---|---|---|
  | base-s1 | SGD+momentum | 0.05 | 0.8405 | 0.8974 |
  | opt-adam-lr0.001 | Adam | 0.001 | **0.8513** | **0.9048** |

  Adam tốt hơn SGD ~0.01 về macro-F1, lớn hơn ngưỡng nhiễu 2σ (0.006)
- **Giải thích:** Adam với adaptive learning rate per-parameter giúp hội tụ nhanh hơn, đặc biệt với các tham số có gradient nhỏ.

### 3.3 Hyper-parameter (Weight Decay)

- **Dự đoán:** Weight decay giúp giảm overfitting bằng cách phạt trọng số lớn
- **Kết quả:** (đang chạy) `hparam-wd-0.001` (weight_decay=0.001)
- **Giải thích:** L2 regularization phạt trọng số lớn, giúp model mượt hơn.

### 3.4 Dropout

- **Dự đoán:** Dropout giúp giảm quá khớp, tăng khoảng cách train-val loss
- **Kết quả:**
  - `drop-0.3`: val_macro_f1 = **0.7633**, val_acc = 0.8630
  - `base-s1`: val_macro_f1 = 0.8405
  - **Chênh lệch: -0.077** → Dropout kém hơn vì baseline chưa quá khớp nặng
- **Giải thích:** Dropout là "thuốc cho quá khớp". Khi mô hình chưa quá khớp (khoảng cách train-val loss nhỏ), dropout không cần thiết và có thể làm giảm performance.

### 3.5 Gradient Clipping

- **Dự đoán:** Clipping giúp ổn định training, đặc biệt với lr cao
- **Thí nghiệm:** `clip-1.0-highlr` (clip_norm=1.0, lr=0.1)
- **Giải thích:** Với lr=0.1 (cao gấp đôi baseline), gradient có thể bùng nổ. Clipping giới hạn gradient norm ≤ 1.0, tránh cập nhật quá lớn.

### 3.7 Khởi tạo tham số

- **Dự đoán:** He tốt hơn cho ReLU vì Var=2/n_in phù hợp hơn Xavier Var=1/n_in
- **Kết quả:**
  - `init-xavier`: val_macro_f1 = 0.8425, val_acc = 0.9016
  - `base-s1` (He): val_macro_f1 = 0.8405
  - **Chênh lệch: +0.002** → Xavier tốt hơn nhẹ, không vượt ngưỡng nhiễu 2σ
- **Giải thích:** Với mạng 3 lớp, khác biệt giữa He và Xavier không rõ rệt. Cả hai đều hoạt động tốt.

## Bảng tổng hợp kết quả thí nghiệm

| exp_id | group | val_f1 | val_acc | Nhận xét |
|--------|-------|--------|---------|-----------|
| base-s1 | baseline | 0.8405 | 0.8974 | Baseline SGD+momentum |
| base-s2 | baseline | 0.8376 | 0.9004 | Baseline seed 2 |
| base-s3 | baseline | 0.8438 | 0.9005 | Baseline seed 3 |
| loss-mse | loss | 0.6828 | 0.8549 | MSE kém CE rõ ràng |
| opt-adam-lr0.001 | optimizer | **0.8513** | **0.9048** | **Tốt nhất** |
| drop-0.3 | dropout | 0.7633 | 0.8630 | Dropout không cần khi chưa quá khớp |
| hparam-wd-0.001 | hparam | 0.6790 | 0.8410 | WD làm giảm performance |
| clip-1.0-highlr | clipping | 0.8464 | 0.9026 | Clipping với high lr khá tốt |
| init-xavier | init | 0.8425 | 0.9016 | Xavier tốt hơn He nhẹ |

## 4. Đánh giá cuối trên tập eval

| Cấu hình | Seed | val macro-F1 | **eval macro-F1** | eval accuracy |
|-----------|------|--------------|-------------------|---------------|
| Baseline | 1 | 0.8405 | 0.6830 | 0.8279 |
| opt-adam-lr0.001 | 1 | **0.8513** | (cần chạy lại) | (cần chạy lại) |

- **Cấu hình cuối cùng được chọn:** `opt-adam-lr0.001` vì có val_macro_f1 cao nhất (0.8513)
- **Cải thiện so với baseline:** ~0.01, vượt ngưỡng nhiễu 2σ

### 4.1 Phân tích lỗi theo lớp

| Lớp | support | precision | recall | F1 |
|---|---|---|---|---|
| 0 | 42,368 | 0.904 | 0.741 | 0.815 |
| 1 | 56,661 | 0.806 | 0.938 | 0.867 |
| 2 | 7,151 | 0.695 | 0.907 | 0.787 |
| 3 | 549 | 0.669 | 0.594 | 0.629 |
| 4 | 1,899 | 0.816 | 0.345 | 0.485 |
| 5 | 3,473 | 0.735 | 0.223 | 0.342 |
| 6 | 4,102 | 0.886 | 0.828 | 0.856 |

- **Lớp khó nhất:** Lớp 5 (F1 = 0.342), tiếp theo là lớp 4 (F1 = 0.485)
- **Nguyên nhân:** Lớp 5 và 4 có recall thấp (0.223 và 0.345), tức là model không nhận ra được nhiều mẫu thuộc lớp này
- **Nhầm lẫn:** Lớp 5 hay bị nhầm với lớp 2 (1987 mẫu), lớp 4 hay bị nhầm với lớp 1 (1101 mẫu)

## 5. Trả lời các câu hỏi dẫn dắt

1. **Bộ tối ưu nào "thắng"?** Adam với lr=0.001 đạt macro-F1 cao hơn SGD+momentum với lr=0.05 (~0.01). Adam adaptive lr giúp hội tụ tốt hơn.

2. **Dropout có giúp không?** Dropout là "thuốc cho quá khớp". Với baseline chưa quá khớp nặng (khoảng cách train-val loss nhỏ), dropout có thể không cần thiết hoặc thậm chí giảm performance.

3. **Gradient clipping giải quyết vấn đề gì?** Clipping giới hạn gradient norm, ngăn cập nhật quá lớn khi lr cao hoặc gradient bùng nổ. Dùng clip_norm = grad_norm baseline (~0.8) làm ngưỡng.

4. **Vì sao khởi tạo toàn số 0 hỏng?** Tất cả nơ-ron cùng giá trị → cùng gradient → cùng cập nhật → symmetry breaking. Model không học được gì.

5. **Quay lại câu hỏi bài học:** Mạng loss không giảm sau 2000 bước → 3 phép kiểm tra đầu tiên:
   - Kiểm tra gradient có chảy không (gradient norm ≠ 0)
   - Thử quá khớp 20 mẫu (nếu không được → lỗi code)
   - Giảm lr 10 lần (nếu loss vẫn đứng → có thể là dying ReLU)

## 6. Hạn chế và điều bất ngờ

- **Hạn chế:** Chỉ chạy 1 seed cho mỗi thí nghiệm (ngoài baseline 3 seeds). Kết luận có thể bị ảnh hưởng bởi nhiễu.
- **Bất ngờ:** Loss bước 0 với He init (~2.38) cao hơn ln(7) (~1.95). Đây là đặc điểm của He init cho ReLU.
- **Nếu có thêm thời gian:** Chạy nhiều seeds hơn, thử mixed precision (AMP), và tăng epoch lên 40-50.

## 7. Phụ lục

- **Danh sách file đã nộp:** lab.ipynb, experiments.xlsx, predictions_eval.csv, eval_result.json, figures/*.png, results/*.json, code/*.py
- **Thời gian chạy ước tính:** ~10-15 phút cho tất cả thí nghiệm (20 epoch × 7 configs × ~1.5s/epoch)
