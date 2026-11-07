"""Bác sĩ duyệt dữ liệu SFT trước khi train (human-in-the-loop ở khâu dữ liệu).

Luồng:  artifacts/sft_v2/filtered.jsonl --review_export--> file Excel cho bác sĩ
        --(bác sĩ duyệt trên Excel / Google Sheets)--> --review_import--> reviewed.jsonl --build--> data/sft_v2/

Quy tắc áp dụng kết quả duyệt (doctor_review_v1):
- "Đạt"     : giữ nguyên câu trả lời của AI.
- "Cần sửa" : thay câu trả lời LƯỢT CUỐI (lượt được tính loss khi train) bằng bản bác sĩ viết ở cột
              "Câu trả lời đã sửa". Nếu bác sĩ sửa thẳng vào cột "Câu trả lời của AI" thì dùng bản đó.
- "Loại"    : bỏ mẫu (ghi vào review_rejected.jsonl để phân tích).
- Để trống  : chưa duyệt. Mặc định BỎ (unreviewed="drop") để chỉ train trên dữ liệu đã được bác sĩ xác nhận;
              unreviewed="keep" giữ bản AI và manifest ghi rõ số mẫu chưa duyệt.
Có thể chỉ gửi bác sĩ một MẪU phân tầng theo loại câu hỏi (select_for_review): tỉ lệ "Đạt" trên mẫu là ước lượng chất
lượng của dữ liệu do pipeline tự động tạo; phần không gửi xử lý như "chưa duyệt".
Mọi mâu thuẫn (chọn "Đạt" nhưng có sửa, "Cần sửa" mà không có bản sửa, giá trị lạ, mã trùng/không tồn tại)
đều báo lỗi kèm số dòng — không tự đoán ý bác sĩ.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RULE_VERSION = "doctor_review_v1"

SHEET_GUIDE = "Hướng dẫn"
SHEET_REVIEW = "Duyệt"
SHEET_DOCS = "Tài liệu"
SHEET_META = "_meta"
SHEET_IDS = "_ids"

VERDICT_LABELS = ("Đạt", "Cần sửa", "Loại")
_VERDICT_KEYS = {"dat": "approve", "can sua": "edit", "loai": "reject"}

# Gợi ý trong danh sách thả xuống; bác sĩ vẫn được gõ nội dung khác.
ERROR_TYPES = (
    "Sai kiến thức y khoa",
    "Thiếu cảnh báo đi khám/cấp cứu",
    "Có thể gây hại",
    "Không trả lời đúng câu hỏi",
    "Bỏ qua thông tin người hỏi đã nêu",
    "Diễn đạt chưa phù hợp",
    "Khác",
)

CATEGORY_VI = {
    "red_flag": "Dấu hiệu nguy hiểm",
    "safety_trap": "Bẫy an toàn",
    "clinical_reasoning": "Tình huống lâm sàng",
    "out_of_scope": "Ngoài phạm vi",
    "basic_knowledge": "Kiến thức cơ bản",
}
# Thứ tự trong file duyệt: rủi ro cao lên đầu, để nếu bác sĩ không duyệt hết thì phần quan trọng đã xong.
PRIORITY = {"red_flag": 0, "safety_trap": 1, "clinical_reasoning": 2, "out_of_scope": 3, "basic_knowledge": 4}

# (khóa, tiêu đề cột, độ rộng). Tiêu đề là "khóa" khi đọc lại, nên không đổi tên sau khi đã gửi file cho bác sĩ.
COLUMNS = (
    ("stt", "STT", 6),
    ("id", "Mã", 18),
    ("category", "Loại câu hỏi", 14),
    ("history", "Hội thoại trước", 55),
    ("question", "Câu hỏi", 45),
    ("answer", "Câu trả lời của AI", 75),
    ("verdict", "ĐÁNH GIÁ", 13),
    ("corrected", "Câu trả lời đã sửa", 75),
    ("error_type", "Loại lỗi", 26),
    ("note", "Ghi chú", 35),
)
DOCTOR_KEYS = ("verdict", "corrected", "error_type", "note")

GUIDE_LINES = (
    "HƯỚNG DẪN DUYỆT CÂU TRẢ LỜI",
    "",
    "Mỗi dòng ở sheet \"Duyệt\" là một câu hỏi về bệnh gút (có thể kèm vài lượt hội thoại trước) và câu trả lời do AI viết.",
    "Những câu trả lời được duyệt sẽ dùng để dạy chatbot tư vấn bệnh gút, nên chatbot sẽ học theo đúng cách trả lời này.",
    "",
    "VỚI MỖI DÒNG, CHỌN Ở CỘT \"ĐÁNH GIÁ\":",
    "  • Đạt      : đúng, an toàn, có thể dùng nguyên văn.",
    "  • Cần sửa  : viết lại TOÀN BỘ câu trả lời đúng vào cột \"Câu trả lời đã sửa\".",
    "  • Loại     : sai, nguy hiểm, hoặc không đáng sửa.",
    "  • Để trống : chưa duyệt (dòng chưa duyệt sẽ KHÔNG được dùng để train).",
    "Cột \"Loại lỗi\" và \"Ghi chú\" không bắt buộc, nhưng rất hữu ích khi chọn Cần sửa hoặc Loại.",
    "",
    "TIÊU CHÍ:",
    "  1. Đúng kiến thức y khoa.",
    "  2. An toàn: không đưa liều thuốc (mg, số viên) cho người bệnh tự dùng; không khuyên tự mua, tự ngưng, tự tăng, tự đổi thuốc;",
    "     không chẩn đoán xác định; có cảnh báo khi nào cần đi khám, khi nào cần đi cấp cứu NGAY.",
    "  3. Trả lời đúng điều người hỏi hỏi, và dùng thông tin họ đã nêu ở các lượt trước (tuổi, bệnh thận, thuốc đang dùng...).",
    "  4. Câu hỏi ngoài phạm vi bệnh gút: chatbot nên nói rõ là ngoài phạm vi và khuyên hỏi bác sĩ phù hợp.",
    "",
    "KHI SỬA CÂU TRẢ LỜI:",
    "  • Giữ giọng tư vấn thân thiện, khoảng 80–220 từ; không nêu tên người, tên bệnh viện, địa chỉ, số điện thoại.",
    "  • Khi trả lời, chatbot chỉ được đọc các đoạn tài liệu ở sheet \"Tài liệu\" (cùng STT). Nên sửa dựa trên các đoạn đó;",
    "    nếu cần bổ sung kiến thức không có trong tài liệu, vui lòng ghi rõ ở cột \"Ghi chú\".",
    "  • Không sửa trực tiếp ở cột \"Câu trả lời của AI\"; hãy viết vào cột \"Câu trả lời đã sửa\".",
    "",
    "LƯU Ý:",
    "  • Câu hỏi đã được xếp theo mức độ quan trọng: dấu hiệu nguy hiểm và bẫy an toàn ở trên cùng.",
    "    Nếu không đủ thời gian duyệt hết, xin duyệt từ trên xuống.",
    "  • Không thêm/xóa dòng, không đổi cột \"Mã\" và tên các cột. Có thể lọc/sắp xếp để tiện duyệt.",
)

_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")   # ký tự điều khiển Excel không chấp nhận


def clean_text(value: Any) -> str:
    return _ILLEGAL.sub("", str(value)) if value is not None else ""


def _row_height(texts_widths: list[tuple[str, float]]) -> float:
    """Ước lượng chiều cao dòng (pt) cho chữ xuống dòng; Excel không tự giãn dòng của file tạo bằng code.
    Trần 409 pt là giới hạn của Excel: nội dung dài hơn vẫn đầy đủ, bấm vào ô để đọc hết."""
    lines = 1
    for text, width in texts_widths:
        per_line = max(1.0, width * 1.1)
        n = sum(max(1, -(-len(p) // int(per_line))) for p in str(text).split("\n"))
        lines = max(lines, n)
    return float(min(409, max(15, lines * 15)))


def _landscape(ws) -> None:
    ws.page_setup.orientation = "landscape"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0


def _norm_ws(value: Any) -> str:
    return " ".join(clean_text(value).split())


def _fold(text: str) -> str:
    """Bỏ dấu + chữ thường: 'Cần sửa' -> 'can sua'."""
    text = unicodedata.normalize("NFD", text.strip().lower()).replace("đ", "d")
    return " ".join("".join(ch for ch in text if unicodedata.category(ch) != "Mn").split())


def parse_verdict(value: Any) -> str | None:
    """'Đạt'/'Cần sửa'/'Loại' (không phân biệt hoa thường, có/không dấu) -> approve/edit/reject; trống -> None."""
    text = _fold(clean_text(value))
    if not text:
        return None
    if text not in _VERDICT_KEYS:
        raise ValueError(f"giá trị ĐÁNH GIÁ không hợp lệ: {value!r} (chỉ dùng: {', '.join(VERDICT_LABELS)})")
    return _VERDICT_KEYS[text]


def order_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: (PRIORITY.get(r.get("category"), 9), str(r["id"])))


def select_for_review(rows: list[dict[str, Any]], *, per_category: int | None = None,
                      full_categories: tuple[str, ...] | list[str] = (), seed: int = 2026) -> list[dict[str, Any]]:
    """Chọn mẫu gửi bác sĩ. per_category=None: tất cả. Ngược lại: lấy NGẪU NHIÊN (có seed) tối đa per_category mẫu
    mỗi loại câu hỏi — mẫu phân tầng, nên tỉ lệ Đạt/Cần sửa/Loại ước lượng được chất lượng của cả bộ dữ liệu —
    riêng các loại trong full_categories thì lấy hết (vd. red_flag)."""
    if per_category is None:
        return list(rows)
    import random

    rng = random.Random(seed)
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_cat.setdefault(r.get("category") or "unknown", []).append(r)
    out = []
    for cat in sorted(by_cat):
        group = sorted(by_cat[cat], key=lambda r: str(r["id"]))
        if cat in full_categories:
            out += group
        else:
            rng.shuffle(group)
            out += group[:per_category]
    return out


def format_history(turns: list[str], answers: list[str]) -> str:
    parts = [f"Người hỏi: {t}\nAI: {a}" for t, a in zip(turns[:-1], answers[:-1])]
    return "\n\n".join(parts)


def sheet_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for stt, r in enumerate(order_rows(rows), 1):
        ctx = (r.get("contexts") or [[]])[-1] or []
        out.append({
            "stt": stt,
            "id": str(r["id"]),
            "category": CATEGORY_VI.get(r.get("category"), r.get("category") or ""),
            "history": format_history(r["turns"], r["answers"]),
            "question": r["turns"][-1],
            "answer": r["answers"][-1],
            "docs": [str(c.get("text") or "").strip() for c in ctx],
        })
    return out


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------
def write_review_xlsx(rows: list[dict[str, Any]], path: str | Path, *, source_file: str, source_sha256: str,
                      selection: str = "all") -> int:
    """rows = các mẫu GỬI bác sĩ (đã chọn bằng select_for_review). Danh sách mã được lưu ở sheet ẩn _ids để lúc
    nhập lại phân biệt 'không gửi' với 'đã gửi nhưng bác sĩ chưa duyệt / lỡ xóa dòng'."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    srows = sheet_rows(rows)
    n = len(srows)
    wrap = Alignment(wrap_text=True, vertical="top")
    bold = Font(bold=True)
    grey = PatternFill("solid", fgColor="D9D9D9")
    yellow = PatternFill("solid", fgColor="FFE699")
    light_yellow = PatternFill("solid", fgColor="FFF8DC")

    def put(ws, row, col, value):
        cell = ws.cell(row=row, column=col)
        cell.value = clean_text(value) if isinstance(value, str) else value
        if isinstance(cell.value, str) and cell.value.startswith("="):
            cell.data_type = "s"   # không để Excel hiểu thành công thức
        return cell

    wb = Workbook()
    guide = wb.active
    guide.title = SHEET_GUIDE
    guide.column_dimensions["A"].width = 140
    for i, line in enumerate(GUIDE_LINES, 1):
        c = put(guide, i, 1, line)
        c.alignment = Alignment(wrap_text=True)
        if line and line.upper() == line and not line.startswith(" "):
            c.font = bold
    _landscape(guide)

    ws = wb.create_sheet(SHEET_REVIEW)
    col_of = {key: i for i, (key, _, _) in enumerate(COLUMNS, 1)}
    for key, label, width in COLUMNS:
        c = put(ws, 1, col_of[key], label)
        c.font = bold
        c.fill = yellow if key in DOCTOR_KEYS else grey
        c.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
        ws.column_dimensions[get_column_letter(col_of[key])].width = width
    for i, sr in enumerate(srows, 2):
        for key, _, _ in COLUMNS:
            c = put(ws, i, col_of[key], sr.get(key, ""))
            c.alignment = wrap
            if key in DOCTOR_KEYS:
                c.fill = light_yellow
        link = ws.cell(row=i, column=col_of["id"])
        link.hyperlink = f"#'{SHEET_DOCS}'!A{i}"
        ws.row_dimensions[i].height = _row_height(
            [(clean_text(sr.get(key, "")), width) for key, _, width in COLUMNS if key not in DOCTOR_KEYS])
    ws.row_dimensions[1].height = 30
    ws.freeze_panes = "C2"
    _landscape(ws)
    last = get_column_letter(len(COLUMNS))
    ws.auto_filter.ref = f"A1:{last}{n + 1}"
    if n:
        vcol = get_column_letter(col_of["verdict"])
        dv = DataValidation(type="list", formula1='"' + ",".join(VERDICT_LABELS) + '"', allow_blank=True,
                            showErrorMessage=True, errorTitle="Giá trị không hợp lệ",
                            error="Chọn Đạt, Cần sửa hoặc Loại (hoặc để trống nếu chưa duyệt).")
        ws.add_data_validation(dv)
        dv.add(f"{vcol}2:{vcol}{n + 1}")
        ecol = get_column_letter(col_of["error_type"])
        dv2 = DataValidation(type="list", formula1='"' + ",".join(ERROR_TYPES) + '"', allow_blank=True,
                             showErrorMessage=False)
        ws.add_data_validation(dv2)
        dv2.add(f"{ecol}2:{ecol}{n + 1}")

    docs = wb.create_sheet(SHEET_DOCS)
    n_docs = max((len(sr["docs"]) for sr in srows), default=3) or 3
    headers = ["STT", "Mã", "Câu hỏi"] + [f"Tài liệu {k}" for k in range(1, n_docs + 1)]
    widths = [6, 16, 40] + [70] * n_docs
    for j, (h, w) in enumerate(zip(headers, widths), 1):
        c = put(docs, 1, j, h)
        c.font = bold
        c.fill = grey
        docs.column_dimensions[get_column_letter(j)].width = w
    for i, sr in enumerate(srows, 2):
        values = [sr["stt"], sr["id"], sr["question"], *sr["docs"]]
        for j, v in enumerate(values, 1):
            put(docs, i, j, v).alignment = wrap
        docs.row_dimensions[i].height = _row_height([(clean_text(v), w) for v, w in zip(values, widths)])
    docs.freeze_panes = "D2"
    _landscape(docs)

    meta = wb.create_sheet(SHEET_META)
    for i, (k, v) in enumerate((("rule_version", RULE_VERSION), ("source_file", source_file),
                                ("source_sha256", source_sha256), ("n_rows", n), ("selection", selection),
                                ("exported_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))), 1):
        put(meta, i, 1, k)
        put(meta, i, 2, v)
    meta.sheet_state = "hidden"
    ids = wb.create_sheet(SHEET_IDS)
    for i, sr in enumerate(srows, 1):
        put(ids, i, 1, sr["id"])
    ids.sheet_state = "hidden"

    wb.active = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return n


def read_review_xlsx(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from openpyxl import load_workbook

    wb = load_workbook(path, data_only=True)
    if SHEET_REVIEW not in wb.sheetnames:
        raise ValueError(f"{path}: không có sheet '{SHEET_REVIEW}' (sheet hiện có: {wb.sheetnames})")
    ws = wb[SHEET_REVIEW]
    label_to_key = {label: key for key, label, _ in COLUMNS}
    header = [clean_text(c.value).strip() for c in ws[1]]
    idx = {label_to_key[h]: i for i, h in enumerate(header) if h in label_to_key}
    missing = [label for key, label, _ in COLUMNS if key in ("id", "answer", "verdict", "corrected") and key not in idx]
    if missing:
        raise ValueError(f"{path}: thiếu cột {missing} — đừng đổi tên cột trong file duyệt")
    records = []
    for row_no, row in enumerate(ws.iter_rows(min_row=2, values_only=True), 2):
        rec = {k: (row[i] if i < len(row) else None) for k, i in idx.items()}
        if clean_text(rec.get("id")).strip():
            rec["_row"] = row_no
            records.append(rec)
    meta: dict[str, Any] = {}
    if SHEET_META in wb.sheetnames:
        for k, v, *_ in wb[SHEET_META].iter_rows(values_only=True):
            if k is not None:
                meta[str(k)] = v
    if SHEET_IDS in wb.sheetnames:
        meta["exported_ids"] = [clean_text(v).strip() for (v, *_) in wb[SHEET_IDS].iter_rows(values_only=True)
                                if clean_text(v).strip()]
    return records, meta


# ---------------------------------------------------------------------------
# Áp dụng kết quả duyệt
# ---------------------------------------------------------------------------
def apply_review(rows: list[dict[str, Any]], records: list[dict[str, Any]], *,
                 unreviewed: str = "drop", exported_ids: list[str] | None = None) -> dict[str, Any]:
    """Trả về {"errors", "warnings", "kept", "rejected", "stats"}. Có lỗi thì kept/rejected rỗng.
    exported_ids: các mã đã gửi bác sĩ (None = gửi tất cả). Mẫu không gửi được tính là chưa duyệt."""
    if unreviewed not in ("drop", "keep"):
        raise ValueError("unreviewed phải là 'drop' hoặc 'keep'")
    by_id = {str(r["id"]): r for r in rows}
    errors: list[str] = []
    warnings: list[str] = []
    seen: dict[str, int] = {}
    decisions: dict[str, dict[str, Any]] = {}

    for rec in records:
        rid = clean_text(rec.get("id")).strip()
        where = f"dòng {rec.get('_row', '?')} (mã {rid})"
        if rid not in by_id:
            errors.append(f"{where}: mã không có trong dữ liệu gốc")
            continue
        if rid in seen:
            errors.append(f"{where}: mã bị lặp, đã có ở dòng {seen[rid]}")
            continue
        seen[rid] = rec.get("_row", 0)
        try:
            verdict = parse_verdict(rec.get("verdict"))
        except ValueError as e:
            errors.append(f"{where}: {e}")
            continue

        original = by_id[rid]["answers"][-1]
        corrected = clean_text(rec.get("corrected")).strip()
        in_place = "answer" in rec and _norm_ws(rec.get("answer")) not in ("", _norm_ws(original))
        if in_place and corrected:
            warnings.append(f"{where}: sửa cả cột câu trả lời của AI lẫn cột đã sửa — dùng cột 'Câu trả lời đã sửa'")
        elif in_place:
            corrected = clean_text(rec.get("answer")).strip()
        if _norm_ws(corrected) == _norm_ws(original):
            corrected = ""   # chép lại nguyên văn = không sửa

        if verdict is None:
            if corrected:
                errors.append(f"{where}: có câu trả lời đã sửa nhưng chưa chọn ĐÁNH GIÁ")
            continue
        if verdict == "approve" and corrected:
            errors.append(f"{where}: chọn 'Đạt' nhưng câu trả lời đã bị sửa — đổi thành 'Cần sửa' hoặc xóa phần sửa")
            continue
        if verdict == "edit" and not corrected:
            errors.append(f"{where}: chọn 'Cần sửa' nhưng chưa có câu trả lời đã sửa")
            continue
        decisions[rid] = {"verdict": verdict, "corrected": corrected, "edited_in_place": in_place,
                          "error_type": clean_text(rec.get("error_type")).strip(),
                          "note": clean_text(rec.get("note")).strip(), "row": rec.get("_row")}

    sent = [rid for rid in exported_ids if rid in by_id] if exported_ids is not None else list(by_id)
    missing = [rid for rid in sent if rid not in seen]
    if missing:
        warnings.append(f"{len(missing)} mẫu không có trong file duyệt (bị xóa dòng?) — coi như chưa duyệt: "
                        + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    by_cat: dict[str, Counter] = {}
    error_types: Counter = Counter()
    for r in rows:
        rid = str(r["id"])
        d = decisions.get(rid)
        cat = by_cat.setdefault(r.get("category") or "unknown", Counter())
        if d is None:
            cat["unreviewed"] += 1
            if unreviewed == "keep":
                kept.append({**r, "review": {"verdict": "unreviewed", "rule_version": RULE_VERSION}})
            continue
        cat[d["verdict"]] += 1
        if d["error_type"]:
            error_types[d["error_type"]] += 1
        review = {"verdict": d["verdict"], "error_type": d["error_type"], "note": d["note"],
                  "edited_in_place": d["edited_in_place"], "sheet_row": d["row"], "rule_version": RULE_VERSION}
        if d["verdict"] == "reject":
            rejected.append({**r, "review": review})
        elif d["verdict"] == "edit":
            kept.append({**r, "answers": list(r["answers"][:-1]) + [d["corrected"]],
                         "answer_ai_original": r["answers"][-1], "review": review})
        else:
            kept.append({**r, "review": review})

    total = Counter()
    for c in by_cat.values():
        total.update(c)
    n_reviewed = total["approve"] + total["edit"] + total["reject"]
    stats = {
        "rule_version": RULE_VERSION,
        "unreviewed_policy": unreviewed,
        "n_source": len(rows),
        "n_sent_to_doctor": len(sent),
        "n_reviewed": n_reviewed,
        # Tỉ lệ bác sĩ chấm "Đạt" trên các mẫu đã duyệt = ước lượng chất lượng dữ liệu do pipeline tự động tạo
        "pass_rate": round(total["approve"] / n_reviewed, 4) if n_reviewed else None,
        "pass_rate_by_category": {k: round(v["approve"] / (v["approve"] + v["edit"] + v["reject"]), 4)
                                  for k, v in sorted(by_cat.items()) if v["approve"] + v["edit"] + v["reject"]},
        "n_approve": total["approve"],
        "n_edit": total["edit"],
        "n_reject": total["reject"],
        "n_unreviewed": total["unreviewed"],
        "n_kept": len(kept) if not errors else 0,
        "by_category": {k: dict(v) for k, v in sorted(by_cat.items())},
        "error_types": dict(error_types.most_common()),
    }
    if errors:
        return {"errors": errors, "warnings": warnings, "kept": [], "rejected": [], "stats": stats}
    return {"errors": [], "warnings": warnings, "kept": kept, "rejected": rejected, "stats": stats}
