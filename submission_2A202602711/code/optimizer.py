"""optimizer.py — Chọn bộ tối ưu và cắt gradient.

Được dùng torch.optim.* và torch.nn.utils.clip_grad_norm_ (xem README mục 5).
File này gom việc chọn bộ tối ưu và cắt gradient để `train.py` gọn và mọi thí nghiệm công bằng.

Công thức cần hiểu (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import torch
from torch.optim import SGD, Adam, AdamW
from torch.nn.utils import clip_grad_norm_

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer.

    Các bước:
      1. kiểm tra name nằm trong OPTIMIZERS, nếu không raise ValueError
      2. "sgd"          -> torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
         "sgd_momentum" -> torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
         "adam"         -> torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
         "adamw"        -> torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    Chú ý: weight_decay của Adam (L2 trộn vào gradient) khác weight_decay của AdamW (suy giảm tách riêng).
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {name}. Must be one of {OPTIMIZERS}")

    if name == "sgd":
        return SGD(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd_momentum":
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "adam":
        return Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name == "adamw":
        return AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học, ví dụ cosine (slide có ví dụ CosineAnnealingLR).

    Trả về None nếu name là None. Nếu bạn dùng scheduler ở một thí nghiệm, hãy ghi vào bảng (cột notes).
    """
    if name is None:
        return None
    # Tuỳ chọn: có thể thêm các scheduler khác nếu cần
    from torch.optim.lr_scheduler import CosineAnnealingLR
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    raise ValueError(f"Unknown scheduler: {name}")


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    Các bước:
      1. nếu max_norm là None: tính chuẩn toàn cục mà không cắt (dùng inf để không cắt)
      2. ngược lại: total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
      3. return float(total_norm)
    Giá trị trả về chính là `grad_norm` bạn phải ghi lại ở mỗi bước (để thấy "gai" gradient).
    Khi dùng mixed precision FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này.
    """
    if max_norm is None:
        # Không cắt, chỉ tính chuẩn
        total_norm = torch.sqrt(sum(p.grad.norm()**2 for p in params if p.grad is not None))
        return float(total_norm)
    else:
        return float(clip_grad_norm_(params, max_norm))
