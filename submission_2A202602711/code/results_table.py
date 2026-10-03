"""results_table.py — Lưu kết quả ra JSON và điền vào experiments.xlsx.

Nhiệm vụ: lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx (đừng gõ tay hàng chục dòng, rất dễ sai).
"""
from __future__ import annotations

import json
from pathlib import Path

import openpyxl


def save_result(result: dict, results_dir: str = "../results") -> str:
    """Ghi result["cfg"], result["history"], result["summary"] (KHÔNG ghi best_state) ra
    <results_dir>/<exp_id>.json. Trả về đường dẫn file. Tạo thư mục nếu chưa có."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    exp_id = result["cfg"]["exp_id"]

    # Loại bỏ best_state trước khi ghi
    to_save = {
        "cfg": result["cfg"],
        "history": result["history"],
        "summary": result["summary"],
    }

    path = Path(results_dir) / f"{exp_id}.json"
    with open(path, "w") as f:
        json.dump(to_save, f, indent=2)
    return str(path)


def load_results(results_dir: str = "../results") -> list[dict]:
    """Đọc mọi file *.json trong results_dir, trả về danh sách dict (sắp theo exp_id)."""
    results = []
    results_path = Path(results_dir)
    if not results_path.exists():
        return results

    for f in sorted(results_path.glob("*.json")):
        with open(f) as fp:
            results.append(json.load(fp))

    # Sắp xếp theo exp_id
    results.sort(key=lambda r: r["cfg"]["exp_id"])
    return results


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Biến một kết quả thành một dòng của bảng: gộp cfg + summary (+ eval_acc, eval_macro_f1 nếu có)
    + figure_file = f"figures/{exp_id}.png".
    Chỉ truyền eval_scores cho baseline và cấu hình cuối cùng."""
    cfg = result["cfg"]
    summary = result["summary"]

    row = {
        "exp_id": cfg["exp_id"],
        "group": cfg["group"],
        "description": cfg["description"],
        "loss": cfg["loss"],
        "optimizer": cfg["optimizer"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "batch": cfg["batch"],
        "epochs": cfg["epochs"],
        "hidden": str(cfg["hidden"]),
        "dropout": cfg["dropout"],
        "clip_norm": cfg["clip_norm"],
        "precision": cfg["precision"],
        "init": cfg["init"],
        "seed": cfg["seed"],
        "step0_loss": summary.get("step0_loss"),
        "best_val_loss": summary.get("best_val_loss"),
        "best_epoch": summary.get("best_epoch"),
        "final_train_loss": summary.get("final_train_loss"),
        "final_val_loss": summary.get("final_val_loss"),
        "val_acc": summary.get("val_acc"),
        "val_macro_f1": summary.get("val_macro_f1"),
        "time_per_epoch_s": summary.get("time_per_epoch_s"),
        "peak_mem_MB": summary.get("peak_mem_MB"),
        "diverged": summary.get("diverged", False),
        "figure_file": f"figures/{cfg['exp_id']}.png",
        "notes": notes,
    }

    # Thêm eval scores nếu có
    if eval_scores is not None:
        row["eval_acc"] = eval_scores.get("accuracy")
        row["eval_macro_f1"] = eval_scores.get("macro_f1")

    return row


def write_xlsx(rows: list[dict], template_path: str, out_path: str) -> None:
    """Điền các dòng vào sheet "Experiments" của mẫu, từ dòng 2 trở xuống, rồi lưu thành out_path.

    Các bước (openpyxl):
      1. wb = openpyxl.load_workbook(template_path)
      2. ws = wb["Experiments"]; đọc tiêu đề dòng 1 để biết cột nào ứng với khoá nào
      3. với mỗi row: ghi giá trị vào đúng cột; BỎ QUA các cột công thức
      4. wb.save(out_path)
    """
    wb = openpyxl.load_workbook(template_path)
    ws = wb["Experiments"]

    # Đọc tiêu đề dòng 1
    headers = [cell.value for cell in ws[1]]
    header_to_col = {h: i for i, h in enumerate(headers)}

    # Cột công thức cần bỏ qua
    formula_cols = {"step0_gap_vs_lnC", "gap_val_minus_train", "delta_val_f1_vs_base",
                    "beyond_noise", "eval_acc", "eval_macro_f1"}

    # Xóa các dòng cũ (từ dòng 2 trở xuống)
    for row_idx in range(ws.max_row, 1, -1):
        ws.delete_rows(row_idx)

    # Ghi các dòng mới
    for row_data in rows:
        new_row_idx = ws.max_row + 1
        for key, value in row_data.items():
            if key in header_to_col and key not in formula_cols:
                col_idx = header_to_col[key]
                ws.cell(row=new_row_idx, column=col_idx + 1, value=value)

    wb.save(out_path)
