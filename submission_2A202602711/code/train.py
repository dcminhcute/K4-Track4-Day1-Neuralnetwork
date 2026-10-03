"""train.py — Huấn luyện và đánh giá mô hình MLP.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # TODO: chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    f1_scores = []
    for c in range(7):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    return np.mean(f1_scores)


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i+batch_size]
        logits = model(xb)
        preds.append(logits.argmax(dim=1))
    return torch.cat(preds, dim=0)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for i in range(0, len(X), batch_size):
        xb = X[i:i+batch_size]
        yb = y[i:i+batch_size]
        logits = model(xb)
        loss = compute_loss(logits, yb, loss_name)
        total_loss += loss.item() * len(xb)
        all_preds.append(logits.argmax(dim=1))
        all_labels.append(yb)

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)
    N = len(labels)
    avg_loss = total_loss / N
    acc = (preds == labels).float().mean().item()

    # Ma trận nhầm lẫn
    cm = torch.zeros(7, 7, dtype=torch.long, device=labels.device)
    for true_label, pred_label in zip(labels, preds):
        cm[true_label, pred_label] += 1
    cm_np = cm.cpu().numpy()
    macro_f1 = macro_f1_from_confusion(cm_np)

    return {"loss": avg_loss, "acc": acc, "macro_f1": macro_f1}


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ: nn.MSELoss reduction='mean')
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        # One-hot encode y và tính MSE
        y_onehot = F.one_hot(y, num_classes=7).float()
        return F.mse_loss(logits, y_onehot)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt."""
    # 0. Đặt seed và tạo model
    set_seed(cfg["seed"])
    model = MLP(
        hidden=cfg["hidden"],
        dropout=cfg["dropout"],
        init=cfg["init"]
    ).to(data["X_tr"].device)
    assert count_params(model) == EXPECTED_PARAMS[cfg["hidden"]], \
        f"Số tham số {count_params(model)} != {EXPECTED_PARAMS[cfg['hidden']]}"

    optimizer = build_optimizer(
        name=cfg["optimizer"],
        params=model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        momentum=cfg["momentum"]
    )

    # Mixed precision scaler
    scaler = None
    if cfg["precision"] == "fp16":
        scaler = torch.amp.GradScaler("cuda")
    elif cfg["precision"] == "bf16":
        pass  # BF16 thường không cần GradScaler

    # Generator cho shuffle
    gen = torch.Generator(device=data["X_tr"].device)
    set_seed(cfg["seed"])
    gen.manual_seed(cfg["seed"])

    # 1. Step 0 loss
    metrics_0 = evaluate(model, data["X_val"], data["y_val"], cfg["loss"])
    step0_loss = metrics_0["loss"]

    # Lịch sử
    history = {
        "epoch": [], "train_loss": [], "val_loss": [], "val_acc": [],
        "val_macro_f1": [], "grad_norm": [], "epoch_time_s": []
    }
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    diverged = False
    peak_mem_MB = 0.0

    for epoch in range(1, cfg["epochs"] + 1):
        epoch_start = time.time()
        model.train()

        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        n_batches = 0

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], cfg["batch"], gen):
            optimizer.zero_grad(set_to_none=True)

            # Forward với autocast nếu cần
            if cfg["precision"] != "fp32":
                autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = (torch.float16 if cfg["precision"] == "fp16" else torch.bfloat16)
                with torch.amp.autocast(device_type=autocast_device, dtype=dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])

            # Kiểm tra NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Clip gradient
            gn = clip_gradients(model.parameters(), cfg["clip_norm"])

            # Update
            if scaler is not None:
                scaler.unscale_(optimizer)
                gn_actual = clip_gradients(model.parameters(), cfg["clip_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            epoch_loss += loss.item()
            epoch_grad_norm += gn if gn is not None else 0.0
            n_batches += 1

        if diverged:
            break

        # Synchronize và đo thời gian
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start

        # Peak memory
        if torch.cuda.is_available():
            peak_mem_MB = max(peak_mem_MB, torch.cuda.max_memory_allocated() / 1024 / 1024)

        # Đánh giá
        train_loss = epoch_loss / n_batches
        # Đánh giá train ở eval mode (dropout tắt)
        train_metrics = evaluate(model, data["X_tr"], data["y_tr"], cfg["loss"], batch_size=8192)
        val_metrics = evaluate(model, data["X_val"], data["y_val"], cfg["loss"])

        # Lưu vào history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(epoch_grad_norm / n_batches)
        history["epoch_time_s"].append(epoch_time)

        # Cập nhật best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Summary
    summary = {
        "step0_loss": step0_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "val_acc": history["val_acc"][best_epoch - 1] if best_epoch > 0 else None,
        "val_macro_f1": history["val_macro_f1"][best_epoch - 1] if best_epoch > 0 else None,
        "time_per_epoch_s": np.mean(history["epoch_time_s"]) if history["epoch_time_s"] else None,
        "peak_mem_MB": peak_mem_MB,
        "diverged": diverged,
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`."""
    with open(path, "w") as f:
        f.write("row_id,pred\n")
        for rid, pred in zip(row_id, preds):
            f.write(f"{rid},{pred}\n")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions."""
    model = MLP(
        hidden=cfg["hidden"],
        dropout=0.0,  # Không dropout khi đánh giá
        init=cfg["init"]
    ).to(data["X_eval"].device)
    model.load_state_dict({k: v.to(data["X_eval"].device) for k, v in result["best_state"].items()})
    model.eval()

    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
