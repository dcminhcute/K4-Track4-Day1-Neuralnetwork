"""plots.py — Vẽ biểu đồ huấn luyện và so sánh.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    """
    cfg = result["cfg"]
    history = result["history"]
    summary = result["summary"]

    epochs = history["epoch"]
    best_epoch = summary["best_epoch"]

    # Tiêu đề
    title = f"{cfg['exp_id']}: {cfg['description']}\n"
    title += f"{cfg['optimizer']}, lr={cfg['lr']}, batch={cfg['batch']}, "
    title += f"hidden={cfg['hidden']}, dropout={cfg['dropout']}"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=10)

    # (1) Train và Val loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="train_loss", color="blue", alpha=0.7)
    ax.plot(epochs, history["val_loss"], label="val_loss", color="orange", alpha=0.7)
    if best_epoch > 0:
        ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7,
                   label=f"best_epoch={best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Val Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) Val Accuracy và Macro-F1
    ax = axes[1]
    ax.plot(epochs, history["val_acc"], label="val_acc", color="green", marker="o", markersize=3)
    ax.plot(epochs, history["val_macro_f1"], label="val_macro_f1", color="purple", marker="s", markersize=3)
    if best_epoch > 0:
        ax.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.7,
                   label=f"best_epoch={best_epoch}")
        best_idx = best_epoch - 1
        ax.scatter([best_epoch], [history["val_acc"][best_idx]], color="green", s=100, zorder=5)
        ax.scatter([best_epoch], [history["val_macro_f1"][best_idx]], color="purple", s=100, zorder=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_title("Validation Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (3) Grad norm
    ax = axes[2]
    ax.plot(epochs, history["grad_norm"], label="grad_norm", color="red", marker=".", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm (before clipping)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    if not results:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors

    for i, result in enumerate(results):
        cfg = result["cfg"]
        history = result["history"]
        epochs = history["epoch"]
        values = history.get(metric, [])

        if values:
            label = cfg["exp_id"]
            color = colors[i % len(colors)]
            ax.plot(epochs, values, label=label, marker="o" if len(epochs) <= 30 else None,
                    markersize=3, color=color, alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title if title else f"Comparison: {metric}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
