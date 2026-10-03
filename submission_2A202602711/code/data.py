"""data.py — Phần Part 0: nạp train/eval, tách validation, chuẩn hoá, đưa lên thiết bị.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    train_data = np.load(f"{processed_dir}/train.npz")
    eval_data = np.load(f"{processed_dir}/eval.npz")

    X_train_full = train_data["X"]
    y_train_full = train_data["y"]
    X_eval = eval_data["X"]
    y_eval = eval_data["y"]
    eval_row_id = eval_data["row_id"]

    # Kiểm tra shape và dtype
    assert X_train_full.ndim == 2 and X_train_full.shape[1] == 54, \
        f"X_train_full phải có shape (N, 54), được {X_train_full.shape}"
    assert X_eval.ndim == 2 and X_eval.shape[1] == 54, \
        f"X_eval phải có shape (M, 54), được {X_eval.shape}"
    assert X_train_full.dtype == np.float32, \
        f"X phải là float32, được {X_train_full.dtype}"
    assert X_eval.dtype == np.float32, \
        f"X_eval phải là float32, được {X_eval.dtype}"
    assert y_train_full.dtype == np.int64, \
        f"y phải là int64, được {y_train_full.dtype}"
    assert y_eval.dtype == np.int64, \
        f"y_eval phải là int64, được {y_eval.dtype}"
    assert y_train_full.min() >= 0 and y_train_full.max() <= 6, \
        f"y_train_full phải có giá trị 0..6, được {y_train_full.min()}..{y_train_full.max()}"
    assert y_eval.min() >= 0 and y_eval.max() <= 6, \
        f"y_eval phải có giá trị 0..6, được {y_eval.min()}..{y_eval.max()}"
    assert len(eval_row_id) == len(X_eval), \
        f"eval_row_id và X_eval phải cùng độ dài"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=val_fraction,
        stratify=y,
        random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    X_numeric = X_tr[:, :N_NUMERIC]
    mean = X_numeric.mean(axis=0)
    std = X_numeric.std(axis=0)
    # Tránh std = 0 (thay bằng 1 để tránh chia cho 0)
    std = np.where(std == 0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X_copy = X.copy()
    X_copy[:, :N_NUMERIC] = (X_copy[:, :N_NUMERIC] - mean) / std
    return X_copy


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    # 1. Nạp dữ liệu
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    print(f"Train (full): {X_train_full.shape}, Eval: {X_eval.shape}")

    # 2. Tách validation từ train
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction, seed)
    print(f"Sau khi tách val ({val_fraction*100:.0f}%):")
    print(f"  Train: {X_tr.shape}")
    print(f"  Val:   {X_val.shape}")

    # 3. Tính mean/std CHỈ trên phần train còn lại
    mean, std = fit_standardizer(X_tr)

    # 4. Chuẩn hoá cả ba tập bằng cùng mean/std
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    # 5. Đưa lên device
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.int64, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.int64, device=device)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=device)
    y_eval_t = torch.tensor(y_eval, dtype=torch.int64, device=device)

    # 6. In kiểm tra: accuracy của chiến lược "luôn đoán lớp đa số"
    most_common_class = np.bincount(y_tr).argmax()
    val_majority_acc = (y_val == most_common_class).mean()
    print(f"\nAccuracy 'luôn đoán lớp đa số' (lớp {most_common_class}) trên val: {val_majority_acc:.4f}")
    print(f"(Mốc thấp nhất mà model phải vượt: ≈ 0.4876)")

    # 7. In kiểm tra: mean/std của 10 cột số trên train sau chuẩn hoá
    X_tr_numeric = X_tr[:, :N_NUMERIC]
    tr_mean = X_tr_numeric.mean(axis=0)
    tr_std = X_tr_numeric.std(axis=0)
    print(f"\nTrung bình 10 cột số trên train (sau chuẩn hoá):")
    print(f"  mean ≈ {tr_mean.mean():.6f} (kỳ vọng ≈ 0)")
    print(f"  std  ≈ {tr_std.mean():.6f} (kỳ vọng ≈ 1)")

    # Kiểm tra tỉ lệ lớp
    print(f"\nTỉ lệ lớp trên train/val (xem có phân tầng đúng không):")
    for c in range(7):
        tr_pct = 100 * (y_tr == c).sum() / len(y_tr)
        val_pct = 100 * (y_val == c).sum() / len(y_val)
        print(f"  Lớp {c}: train {tr_pct:6.2f}%, val {val_pct:6.2f}%")

    return {
        "X_tr": X_tr_t,
        "y_tr": y_tr_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_eval": X_eval_t,
        "y_eval": y_eval_t,
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
        "n_train": len(X_tr),
        "n_val": len(X_val),
        "n_eval": len(X_eval),
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; xử lý bằng cách dùng i_start:i_end (tự cắt đúng kích thước còn lại)
    """
    N = len(X)
    if shuffle:
        # Dùng torch.randperm với generator nếu được cung cấp
        perm = torch.randperm(N, generator=generator, device=X.device)
    else:
        perm = torch.arange(N, device=X.device)

    i = 0
    while i < N:
        # Batch cuối có thể nhỏ hơn batch_size (không cần xử lý đặc biệt, cắt tự nhiên)
        end = min(i + batch_size, N)
        idx = perm[i:end]
        yield X[idx], y[idx]
        i = end
