"""Dựng dữ liệu SFT v2.1 — dữ liệu SFT chính thức — từ bản SFT v2 ngày 16/11 (151 mẫu bác sĩ đã xem ở vòng 1, ghi chú
trực tiếp 19 dòng; nhóm tích hợp ghi chú ngày 16/11). Nhóm dùng kết quả vòng 1, không duyệt lại (quyết định 19/11/2026).

  1. Chuyển về định dạng pipeline, gắn lại chunk KB thật (nhãn nguồn thật) — kiểm tra từng mẫu: ngoài nhãn nguồn,
     không ký tự nào khác bị đổi.
  2. Áp data/sft_v2/doctor_round1_fixes.json (sửa các chỗ bản 16/11 còn sai so với ghi chú bác sĩ).
  3. Kiểm tra mọi mẫu bằng luật bác sĩ (gout_llmops/data/doctor_rules.py). Mẫu bác sĩ đã xem còn vi phạm -> dừng.
  4. (Tùy chọn) --extra-unreviewed: thêm mẫu AI đã lọc nhưng CHƯA gửi bác sĩ (artifacts/sft_v2/filtered.jsonl trên
     Drive) vào tập TRAIN để cân lại phân bố loại câu hỏi; mẫu vi phạm luật bác sĩ bị bỏ.
  5. Ghi data/sft_v2_1/{train,val}.jsonl + manifest.json + fix_log.json và artifacts/sft_v2_1/rows.jsonl
     (định dạng pipeline: dùng làm câu hỏi cho DPO).

Tập val giữ đúng các mẫu val của bản 16/11 (trừ mẫu bị bỏ) để so được với SFT v3.

  PYTHONPATH=src python scripts/fix_sft_v2_data.py
  PYTHONPATH=src python scripts/fix_sft_v2_data.py --extra-unreviewed artifacts/sft_v2/filtered.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from gout_llmops.core.hashing import sha256_file
from gout_llmops.core.io import read_jsonl, write_json, write_jsonl
from gout_llmops.data.doctor_rules import RULESET_VERSION, lint_conversation
from gout_llmops.data.sft_v2_fix import (
    apply_fixes,
    chunk_index,
    row_to_sample,
    same_except_source_labels,
    sample_to_row,
)

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT_PATH = "prompts/system_vi_v1.txt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/sft_v2/train.jsonl")
    ap.add_argument("--val", default="data/sft_v2/val.jsonl")
    ap.add_argument("--fixes", default="data/sft_v2/doctor_round1_fixes.json")
    ap.add_argument("--kb-chunks", default="artifacts/kb/gout_kb_v1/chunks.jsonl")
    ap.add_argument("--extra-unreviewed", default=None,
                    help="filtered.jsonl của build_sft_v2 (mẫu AI đã lọc, chưa gửi bác sĩ) -> thêm vào TRAIN")
    ap.add_argument("--out", default="data/sft_v2_1")
    ap.add_argument("--rows-out", default="artifacts/sft_v2_1/rows.jsonl")
    args = ap.parse_args()
    os.chdir(ROOT)

    system_prompt = Path(SYSTEM_PROMPT_PATH).read_text(encoding="utf-8")
    index = chunk_index(read_jsonl(args.kb_chunks))
    samples = [(s, "train") for s in read_jsonl(args.train)] + [(s, "val") for s in read_jsonl(args.val)]
    split_of = {s["id"]: sp for s, sp in samples}

    rows = [sample_to_row(s, system_prompt, index) for s, _ in samples]
    relabeled = 0
    for (s, _), r in zip(samples, rows, strict=True):   # bước 1: chỉ nhãn nguồn được phép khác
        rebuilt = row_to_sample(r, system_prompt)
        if not same_except_source_labels(s, rebuilt, system_prompt):
            sys.exit(f"{s['id']}: dựng lại khác bản gốc ngoài nhãn nguồn — dừng để kiểm tra")
        relabeled += rebuilt["messages"][0]["content"] != s["messages"][0]["content"]

    fixes_doc = json.loads(Path(args.fixes).read_text(encoding="utf-8"))
    rows, fix_log = apply_fixes(rows, fixes_doc["fixes"])
    for r in rows:
        r["split"] = split_of[r["id"]]
        r["provenance"] = "doctor_round1_sample"

    lint = {r["id"]: v for r in rows if (v := lint_conversation(r["turns"], r["answers"]))}
    if lint:
        for rid, vs in lint.items():
            for v in vs:
                print(f"[VI PHẠM] {rid}: {v['rule']} {v['where']} …{v['excerpt'][:80]}…")
        sys.exit(f"{len(lint)} mẫu bác sĩ đã xem còn vi phạm luật bác sĩ — sửa doctor_round1_fixes.json rồi chạy lại")

    extra, extra_dropped = [], Counter()
    if args.extra_unreviewed:
        seen = {s["id"] for s, _ in samples}
        for r in read_jsonl(args.extra_unreviewed):
            if r["id"] in seen or not r.get("answers") or not r.get("contexts"):
                continue
            v = lint_conversation(r["turns"], r["answers"])
            if v:
                extra_dropped.update(x["rule"] for x in v)
                continue
            extra.append({**r, "split": "train", "provenance": "ai_filtered_unreviewed"})

    out = Path(args.out)
    train = [row_to_sample(r, system_prompt) for r in rows + extra if r["split"] == "train"]
    val = [row_to_sample(r, system_prompt) for r in rows if r["split"] == "val"]
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(Path(args.rows_out), rows)
    if extra:
        write_jsonl(Path(args.rows_out).with_name("rows_extra_unreviewed.jsonl"), extra)
    write_json(out / "fix_log.json", {"fixes": fix_log, "ruleset": RULESET_VERSION})

    def cats(rs):
        return dict(Counter(r["category"] for r in rs))

    manifest = {
        "version": out.name,
        "generated_at": time.strftime("%Y-%m-%d"),
        "n_train": len(train), "n_val": len(val),
        "n_doctor_sample": len(rows), "n_extra_unreviewed": len(extra),
        "by_category": {"train": cats(train), "val": cats(val)},
        "multi_turn": sum(1 for r in rows + extra if len(r["turns"]) > 1),
        "fixes": dict(Counter(f["action"] for f in fix_log)),
        "fixed_ids": [f["id"] for f in fix_log],
        "source_labels_restored": relabeled,
        "lint_ruleset": RULESET_VERSION,
        "extra_unreviewed_dropped_by_rule": dict(extra_dropped),
        "physician_review": ("Bác sĩ xem 151 mẫu (vòng 1, gửi 07/11) và ghi chú trực tiếp vào 19 dòng, không điền cột "
                             "ĐÁNH GIÁ. Nhóm dùng kết quả này, không duyệt lại (quyết định 19/11/2026): dòng không có "
                             "ghi chú được giữ nguyên; dòng có ghi chú được nhóm sửa theo ghi chú."),
        "corrections_rechecked_by_physician": False,
        "provenance": ("151 mẫu gửi bác sĩ -> bác sĩ ghi chú 19 dòng -> nhóm tích hợp ghi chú (16/11, data/sft_v2, "
                       "doctor_feedback_audit.json) -> sửa thêm theo doctor_round1_fixes.json + kiểm tra luật bác sĩ "
                       "(19/11, data/sft_v2_1)."),
        "rag_context": ("chunk KB đầy đủ của lượt cuối, nhãn nguồn thật — giống hệt lúc suy luận "
                        "(bản 16/11 cũng là chunk đầy đủ, chỉ đổi nhãn nguồn thành doctor_review_excerpt_i)"),
        "val_note": "val = các mẫu val của bản 16/11 (trừ mẫu bị bỏ) để so được với SFT v3",
        "format": "build_messages(system_vi_v1 + NGỮ CẢNH RAG lượt cuối, history<=2 cặp) + assistant",
        "inputs": {p: sha256_file(p) for p in (args.train, args.val, args.fixes, args.kb_chunks,
                                               *( [args.extra_unreviewed] if args.extra_unreviewed else []))},
    }
    for f in ("train.jsonl", "val.jsonl"):
        manifest[f"sha256_{f}"] = sha256_file(out / f)
    write_json(out / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("n_train", "n_val", "n_doctor_sample", "n_extra_unreviewed",
                                               "by_category", "fixes", "source_labels_restored",
                                               "extra_unreviewed_dropped_by_rule")}, ensure_ascii=False, indent=1))
    print(f"-> {out}/ và {args.rows_out}")


if __name__ == "__main__":
    main()
