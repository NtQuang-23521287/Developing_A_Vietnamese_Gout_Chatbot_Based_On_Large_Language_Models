#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chuẩn bị 1 gói ZIP SFT v2 từ dữ liệu RAG + bản Excel bác sĩ nhận xét.

Yêu cầu: chạy tại thư mục gốc repository, Python 3.10–3.12, pip install openpyxl.
Nguồn cố định: artifacts/sft_v2/filtered.jsonl (bản đã dùng để xuất cho bác sĩ).
Không chạy lại notebook 04 bước 1a/1b!

Lần đầu:
  python scripts/prepare_doctor_sft_release.py prepare \
    --doctor-xlsx "/content/sft_v2_reviewĐÃ-CHỈNH(1).xlsx"

Mở artifacts/sft_v2/review/doctor_sft_merge.xlsx, hoàn thiện:
  - "Đáp án cuối": cả câu trả lời sạch, KHÔNG chứa ghi chú màu đỏ;
  - "Trạng thái": Đạt / Cần sửa / Loại;
  - "Bác sĩ xác nhận": CÓ (chỉ sau khi bác sĩ thực sự xác nhận bản cuối);
  - "Lịch sử đã sửa (JSON)": mảng đáp án các lượt TRƯỚC nếu bác sĩ đã sửa lịch sử.

Sau khi bác sĩ xác nhận:
  python scripts/prepare_doctor_sft_release.py build --reviewer "Bác sĩ xác nhận"

Thử nghiệm SFT chỉ dùng AI chưa duyệt (không được gọi "bác sĩ xác minh"):
  python scripts/prepare_doctor_sft_release.py build \
    --reviewer "Bác sĩ xác nhận" --research-include-ai

Đầu ra: artifacts/sft_v2/doctor_sft_package.zip, bên trong:
  data/sft_v2/train.jsonl, val.jsonl, manifest.json, audit JSON và README.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

SOURCE = Path("artifacts/sft_v2/filtered.jsonl")
MERGED = Path("artifacts/sft_v2/review/doctor_sft_merge.xlsx")
PACKAGE = Path("artifacts/sft_v2/doctor_sft_package.zip")
WORK = Path("artifacts/sft_v2")
DATA = Path("data/sft_v2")

HEADERS = [
    "Mã", "Loại câu hỏi", "Câu hỏi", "Hội thoại trước",
    "Đáp án AI gốc", "Bản Excel bác sĩ ghi", "Ghi chú màu đỏ / thay đổi",
    "Đáp án cuối", "Trạng thái", "Bác sĩ xác nhận",
    "Lịch sử đã sửa (JSON)", "Ghi chú kiểm toán", "Nguồn",
]

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def norm(x) -> str:
    return " ".join(str(x or "").split())

def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"Không có {path}. Khôi phục đúng file filtered.jsonl từ Drive; KHÔNG chạy lại filter.")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(x["id"]) for x in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("Dữ liệu nguồn có Mã trùng, không thể tạo dữ liệu SFT.")
    for r in rows:
        turns, answers, contexts = r.get("turns"), r.get("answers"), r.get("contexts")
        if not turns or not answers or not contexts or len(turns) != len(answers) or len(turns) != len(contexts):
            raise SystemExit(f"Thiếu / lệch turns, answers, contexts: {r['id']}")
        if not all(isinstance(c, list) and c for c in contexts):
            raise SystemExit(f"Context RAG rỗng: {r['id']}")
    return rows

def value_text(v) -> str:
    # openpyxl >= 3.1: CellRichText gồm chuỗi và các TextBlock.
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return "".join(getattr(part, "text", str(part)) for part in v)
    return str(v)

def is_red(color) -> bool:
    if color is None:
        return False
    if getattr(color, "type", None) != "rgb":
        return False
    raw = getattr(color, "rgb", "")
    if not isinstance(raw, str) or len(raw) < 6:
        return False
    h = raw[-6:]
    try:
        rr, gg, bb = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return False
    return rr >= 145 and rr >= gg * 1.4 and rr >= bb * 1.3

def extract_red(cell) -> str:
    v = cell.value
    chunks = []
    if isinstance(v, (list, tuple)):
        for part in v:
            f = getattr(part, "font", None)
            if f is not None and is_red(getattr(f, "color", None)):
                chunks.append(getattr(part, "text", str(part)))
    elif is_red(getattr(cell.font, "color", None)):
        chunks.append(value_text(v))
    return "\n".join(s.strip() for s in chunks if s and s.strip())

def format_history(r: dict) -> str:
    return "\n\n".join(
        f"Người hỏi: {q}\nAI: {a}"
        for q, a in zip(r["turns"][:-1], r["answers"][:-1])
    )

def load_doctor_annotations(path: Path) -> tuple[dict, dict]:
    if not path.is_file():
        raise SystemExit(f"Không thấy bản Excel gốc của bác sĩ: {path}")
    wb = load_workbook(path, rich_text=True, data_only=False)
    if "Duyệt" not in wb.sheetnames:
        raise SystemExit(f"{path}: thiếu sheet Duyệt")
    ws = wb["Duyệt"]
    heads = {value_text(c.value).strip(): i for i, c in enumerate(ws[1])}
    needed = ["Mã", "Câu trả lời của AI", "Hội thoại trước", "ĐÁNH GIÁ", "Câu trả lời đã sửa"]
    missing = [h for h in needed if h not in heads]
    if missing:
        raise SystemExit(f"Thiếu cột trong bản bác sĩ: {missing}")
    by_id = {}
    for i in range(2, ws.max_row + 1):
        def cell(h): return ws.cell(i, heads[h] + 1)
        rid = norm(value_text(cell("Mã").value))
        if not rid:
            continue
        if rid in by_id:
            raise SystemExit(f"Mã {rid} lặp trong Excel bác sĩ, dòng {i}")
        ans = cell("Câu trả lời của AI")
        his = cell("Hội thoại trước")
        notes = [extract_red(ans), extract_red(his)]
        if "Ghi chú" in heads:
            n = value_text(cell("Ghi chú").value).strip()
            if n:
                notes.append("GHI CHÚ: " + n)
        by_id[rid] = {
            "answer_annotated": value_text(ans.value),
            "history_annotated": value_text(his.value),
            "red_notes": "\n".join(x for x in notes if x),
            "verdict": value_text(cell("ĐÁNH GIÁ").value).strip(),
            "corrected": value_text(cell("Câu trả lời đã sửa").value).strip(),
            "excel_row": i,
        }
    meta = {}
    if "_meta" in wb.sheetnames:
        for row in wb["_meta"].iter_rows(values_only=True):
            if len(row) >= 2 and row[0] is not None:
                meta[str(row[0])] = row[1]
    return by_id, meta

def prepare(args):
    original = read_jsonl(SOURCE)
    ann, original_meta = load_doctor_annotations(Path(args.doctor_xlsx))
    unknown_ids = set(ann) - {r["id"] for r in original}
    if unknown_ids:
        raise SystemExit(f"Excel có mã không khớp filtered.jsonl: {sorted(unknown_ids)[:10]}")
    if original_meta.get("source_sha256") and original_meta["source_sha256"] != sha(SOURCE):
        raise SystemExit("SHA256 filtered.jsonl KHÁC nguồn gửi bác sĩ. Không sử dụng --force.")
    MERGED.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "Dữ liệu chuẩn hóa"
    ws.append(HEADERS)
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = f"A1:M{len(original)+1}"
    red_n = 0
    modified_n = 0
    for r in sorted(original, key=lambda x: str(x["id"])):
        a = ann.get(r["id"])
        original_ans = r["answers"][-1]
        final = ""
        verdict = ""
        doctor_confirmed = ""
        flags = []
        notes = ""
        annotated_ans = ""
        if a:
            annotated_ans = a["answer_annotated"]
            if a["red_notes"]:
                flags.append("CÓ CHỮ ĐỎ / GHI CHÚ")
                red_n += 1
            if norm(annotated_ans) != norm(original_ans):
                flags.append("ĐÁP ÁN TRONG EXCEL ĐÃ KHÁC NGUỒN")
                modified_n += 1
            if norm(a["history_annotated"]) != norm(format_history(r)):
                flags.append("LỊCH SỬ HỘI THOẠI ĐÃ KHÁC NGUỒN")
            if a["corrected"]:
                flags.append("CỘT ĐÃ SỬA CÓ NỘI DUNG")
            if a["verdict"]:
                flags.append("EXCEL CÓ TRẠNG THÁI: " + a["verdict"])
            notes = a["red_notes"]
            # Không tự lấy đáp án bác sĩ có chèn ghi chú làm đầu ra SFT.
            # Nếu cột "Câu trả lời đã sửa" chứa trọn đáp án thì đưa làm DỰ THẢO,
            # nhưng vẫn yêu cầu bác sĩ xác nhận rõ ràng trong cột J.
            if a["corrected"]:
                final = a["corrected"]
        else:
            flags.append("CHƯA GỬI BÁC SĨ")
        ws.append([
            r["id"], r.get("category", ""), r["turns"][-1], format_history(r),
            original_ans, annotated_ans, notes, final, verdict, doctor_confirmed, "",
            " | ".join(flags), "Bác sĩ" if a else "AI_ONLY",
        ])
    header_fill = PatternFill("solid", fgColor="17365D")
    for c in ws[1]:
        c.fill = header_fill
        c.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    widths = [25, 22, 52, 65, 72, 72, 55, 75, 18, 20, 45, 42, 18]
    for i, width in enumerate(widths, 1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = width
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
        for c in [row[7], row[8], row[9], row[10]]:
            c.fill = PatternFill("solid", fgColor="FFF2CC")
    dv = DataValidation(type="list", formula1='"Đạt,Cần sửa,Loại"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"I2:I{ws.max_row}")
    dv2 = DataValidation(type="list", formula1='"CÓ"', allow_blank=True)
    ws.add_data_validation(dv2)
    dv2.add(f"J2:J{ws.max_row}")
    guide = wb.create_sheet("Hướng dẫn")
    guide.append(["Bước", "Nội dung"])
    instructions = [
        ("1", "Giữ nguyên file Excel gốc bác sĩ gửi. File chuẩn hóa này chỉ là bản làm việc."),
        ("2", "Với các câu có chữ đỏ, xóa thông tin AI cũ mâu thuẫn và viết lại TOÀN BỘ đáp án sạch tại cột H."),
        ("3", "Đạt = đáp án cuối trùng nguyên văn đáp án AI gốc; Cần sửa = đã viết lại toàn bộ đáp án; Loại = bỏ."),
        ("4", "Cột J = CÓ chỉ khi bác sĩ thực sự đã đọc và xác nhận câu trả lời cuối; không tự suy ra từ chữ đỏ."),
        ("5", "Nếu bác sĩ sửa LỊCH SỬ HỘI THOẠI, ghi JSON array các đáp án lượt trước đã sửa vào cột K."),
        ("6", "406 mẫu không được gửi bác sĩ có thể giữ để thử nghiệm --research-include-ai, KHÔNG gọi đã bác sĩ xác minh."),
        ("7", "File filtered.jsonl phải đúng SHA256 nguồn gốc. Không chạy lại notebook 04 bước 1a hoặc 1b."),
        ("8", "Chạy build để xuất ZIP. Với chế độ chính, chỉ nhận dòng có cột J = CÓ và trạng thái hợp lệ."),
    ]
    for ins in instructions:
        guide.append(ins)
    guide.column_dimensions["A"].width = 12
    guide.column_dimensions["B"].width = 115
    meta = wb.create_sheet("_meta")
    meta.sheet_state = "hidden"
    for k, v in [
        ("source_sha256", sha(SOURCE)),
        ("doctor_xlsx", str(Path(args.doctor_xlsx))),
        ("doctor_xlsx_sha256", sha(Path(args.doctor_xlsx))),
        ("source_rows", len(original)),
        ("doctor_rows", len(ann)),
        ("prepared_at", datetime.now(timezone.utc).isoformat()),
    ]:
        meta.append([k, v])
    wb.save(MERGED)
    print(f"Đã xuất: {MERGED} | nguồn {len(original)} | mẫu trong Excel {len(ann)}")
    print(f"Có chữ đỏ/ghi chú tách được: {red_n}; ô đáp án sửa so với nguồn: {modified_n}")
    print("CHƯA THỂ TRAIN: điền Đáp án cuối + Trạng thái + Bác sĩ xác nhận trước khi build.")

def build(args):
    original = read_jsonl(SOURCE)
    if not MERGED.is_file():
        raise SystemExit(f"Chưa có {MERGED}. Chạy 'prepare' trước.")
    wb = load_workbook(MERGED, data_only=True)
    ws = wb["Dữ liệu chuẩn hóa"]
    headers = {norm(c.value): i for i, c in enumerate(ws[1])}
    if any(h not in headers for h in HEADERS):
        raise SystemExit("Thiếu hoặc đổi tên cột trong bản chuẩn hóa.")
    meta = {str(x[0]): x[1] for x in wb["_meta"].iter_rows(values_only=True) if x[0]}
    if meta.get("source_sha256") != sha(SOURCE):
        raise SystemExit("filtered.jsonl thay đổi từ khi prepare; không build để tránh lệch dữ liệu.")
    records = {}
    for nr in range(2, ws.max_row + 1):
        arr = [value_text(ws.cell(nr, i + 1).value).strip() for i in range(len(HEADERS))]
        if not arr[0]:
            continue
        if arr[0] in records:
            raise SystemExit(f"Trùng Mã tại dòng {nr}: {arr[0]}")
        records[arr[0]] = dict(zip(HEADERS, arr))
    if set(records) != {r["id"] for r in original}:
        raise SystemExit("Mã trong Excel không khớp nguồn: có thể đã bị xóa/đổi dòng.")
    kept, rejected, errors = [], [], []
    stats = Counter()
    for r in original:
        x = records[r["id"]]
        verdict = x["Trạng thái"].strip().lower()
        confirmed = x["Bác sĩ xác nhận"].strip().lower() in ("có", "co", "yes")
        final = x["Đáp án cuối"].strip()
        history_fix = x["Lịch sử đã sửa (JSON)"].strip()
        flags = x["Ghi chú kiểm toán"]
        if verdict not in ("", "đạt", "cần sửa", "loại"):
            errors.append(f"{r['id']}: Trạng thái không hợp lệ '{verdict}'")
            continue
        if verdict and not confirmed:
            errors.append(f"{r['id']}: đã chọn Trạng thái nhưng chưa có xác nhận bác sĩ.")
            continue
        if confirmed and not verdict:
            errors.append(f"{r['id']}: có xác nhận nhưng thiếu Trạng thái.")
            continue
        if verdict == "loại":
            stats["rejected"] += 1
            rejected.append(r["id"])
            continue
        if verdict in ("đạt", "cần sửa"):
            if verdict == "đạt" and final and norm(final) != norm(r["answers"][-1]):
                errors.append(f"{r['id']}: chọn Đạt nhưng đáp án đã sửa. Chọn Cần sửa.")
                continue
            if verdict == "cần sửa" and (not final or norm(final) == norm(r["answers"][-1])):
                errors.append(f"{r['id']}: chọn Cần sửa nhưng thiếu đáp án mới hoàn chỉnh.")
                continue
            if "LỊCH SỬ HỘI THOẠI ĐÃ KHÁC NGUỒN" in flags and not history_fix:
                errors.append(f"{r['id']}: bác sĩ sửa lịch sử nhưng chưa điền lịch sử sạch ở cột K.")
                continue
            if ("CÓ CHỮ ĐỎ" in flags or "ĐÁP ÁN TRONG EXCEL ĐÃ KHÁC NGUỒN" in flags) and verdict == "đạt":
                errors.append(f"{r['id']}: có ghi chú/sửa từ bác sĩ; phải duyệt bản sạch, không thể tự coi Đạt.")
                continue
            row = {**r, "answers": list(r["answers"])}
            row["answers"][-1] = final if verdict == "cần sửa" else r["answers"][-1]
            if history_fix:
                try:
                    a = json.loads(history_fix)
                    if not isinstance(a, list) or len(a) != len(r["answers"]) - 1 or not all(isinstance(z, str) and z.strip() for z in a):
                        raise ValueError("sai số lượt hoặc chuỗi rỗng")
                    row["answers"][:-1] = a
                except (ValueError, json.JSONDecodeError) as e:
                    errors.append(f"{r['id']}: lịch sử JSON không hợp lệ: {e}")
                    continue
            stats["doctor_verified"] += 1
            row["review"] = {
                "verdict": "edit" if verdict == "cần sửa" else "approve",
                "doctor_confirmed": True,
                "reviewer": args.reviewer,
                "source": "doctor_final_approved",
            }
            kept.append(row)
        elif args.research_include_ai:
            # Không đưa vào train những mẫu bác sĩ đã sửa/ghi chú nhưng chưa xác nhận.
            if "CHƯA GỬI BÁC SĨ" not in flags and (
                "CÓ CHỮ ĐỎ" in flags or "ĐÁP ÁN TRONG EXCEL ĐÃ KHÁC NGUỒN" in flags
                or "LỊCH SỬ HỘI THOẠI ĐÃ KHÁC NGUỒN" in flags
            ):
                stats["pending_doctor"] += 1
                continue
            row = {**r, "review": {
                "verdict": "unreviewed",
                "doctor_confirmed": False,
                "source": "ai_only_research",
            }}
            stats["ai_unverified"] += 1
            kept.append(row)
        else:
            stats["pending_doctor"] += 1
    if errors:
        print("\n".join(errors[:80]), file=sys.stderr)
        raise SystemExit(f"{len(errors)} lỗi duyệt — chưa tạo train/val.")
    if not kept:
        raise SystemExit("0 mẫu hợp lệ. Chưa có câu nào được bác sĩ xác nhận; không thể train bộ Verified.")
    # SFT RAG-aware phải sử dụng reranker nhất quán với Qwen Base mới.
    lacking = [r["id"] for r in kept if any(
        not all(hit.get("rerank_score") is not None for hit in ctx)
        for ctx in r["contexts"]
    )]
    if lacking:
        raise SystemExit(f"{len(lacking)} mẫu thiếu rerank_score. Kiểm tra protocol, ví dụ {lacking[:5]}.")
    WORK.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    reviewed = WORK / "reviewed.jsonl"
    reviewed.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    audit = {
        "method": "physician_feedback_curated_sft_v2",
        "source_sha256": sha(SOURCE),
        "review_xlsx_sha256": sha(MERGED),
        "source_n": len(original),
        "doctor_verified_n": stats["doctor_verified"],
        "ai_unverified_n": stats["ai_unverified"],
        "rejected_n": stats["rejected"],
        "pending_n": stats["pending_doctor"],
        "research_only": bool(stats["ai_unverified"]),
        "reviewer": args.reviewer,
        "rejected_ids": rejected,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (WORK / "doctor_sft_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    # build_sft_v2.py có sẵn split theo category và build_messages giống inference.
    # review_manifest giữ checksum khớp filtered.jsonl để stage_build không ngộ nhận nguồn.
    manifest = {
        "rule_version": "doctor_sft_release_v1",
        "reviewer": args.reviewer,
        "source_sha256": sha(SOURCE),
        "n_source": len(original),
        "n_reviewed": stats["doctor_verified"] + stats["rejected"],
        "n_approve": sum(r["review"]["verdict"] == "approve" for r in kept),
        "n_edit": sum(r["review"]["verdict"] == "edit" for r in kept),
        "n_reject": stats["rejected"],
        "n_unreviewed": len(original) - stats["doctor_verified"] - stats["rejected"],
        "unreviewed_policy": "keep" if args.research_include_ai else "drop",
        "review_file_sha256": sha(MERGED),
        "imported_at": audit["created_at"],
    }
    (WORK / "review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": "src" + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
    subprocess.run([sys.executable, "scripts/build_sft_v2.py", "build"], check=True, env=env)
    ready = [DATA / "train.jsonl", DATA / "val.jsonl", DATA / "manifest.json", WORK / "doctor_sft_audit.json"]
    for file in ready:
        if not file.exists():
            raise SystemExit(f"Thiếu file sau build: {file}")
    with zipfile.ZipFile(PACKAGE, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for file in ready:
            z.write(file, arcname=str(file))
        z.writestr("README.txt",
            "Gói nghiên cứu SFT. Giải nén ở thư mục gốc repo, sau đó chạy:\n"
            "PYTHONPATH=src python scripts/train_sft.py --config configs/training/sft_qwen3_8b_v3.yaml --preflight-only\n"
            "Kiểm tra preflight trước khi --full-test. Mô hình nghiên cứu KHÔNG dùng tư vấn lâm sàng.\n"
            "Nếu audit.research_only=true: dữ liệu chứa đáp án AI chưa bác sĩ duyệt, KHÔNG gọi Physician-Verified.\n")
    print(f"Đã tạo {PACKAGE} ({PACKAGE.stat().st_size:,} bytes)")
    print(f"Đã xác minh: {stats['doctor_verified']} | AI nghiên cứu chưa duyệt: {stats['ai_unverified']}")
    print(f"Đã loại: {stats['rejected']} | Chưa duyệt chờ: {stats['pending_doctor']}")
    print("Không chạy train nếu chưa đọc audit và kiểm tra thủ công các ví dụ cuối.")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "build"])
    parser.add_argument("--doctor-xlsx", default="artifacts/sft_v2/review/sft_v2_review.xlsx")
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--research-include-ai", action="store_true",
                        help="CHỈ THỬ NGHIỆM: giữ mẫu AI chưa xác minh, loại các mẫu có góp ý bác sĩ chưa xử lý.")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
    else:
        build(args)

if __name__ == "__main__":
    main()
