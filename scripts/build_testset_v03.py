"""Gộp file review + case mới -> testset v03 (JSONL), kiểm tra, chia dev/test.

Ví dụ:
  python scripts/build_testset_v03.py \
      --review artifacts/review/testset_v03_review.csv \
      --new data/testset_v03/new_cases.csv

Bản nháp (cho phép dòng pending, dùng để thử pipeline):
  python scripts/build_testset_v03.py --allow-pending --out-dir artifacts/review/draft_v03
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from gout_llmops.core.io import read_jsonl, write_json, write_jsonl

SEP = "||"
ENUMS = {
    "question_type": {"basic_knowledge", "clinical_reasoning", "red_flag", "safety_trap"},
    "risk_level": {"low", "moderate", "high", "critical"},
    "kb_support": {"supported", "partial", "not_supported"},
    "expected_behavior": {"answer", "answer_with_referral", "refuse_and_refer", "urgent_referral"},
    "review_status": {"pending", "approved", "rejected"},
}
RISK_WEIGHT = {"low": 1.0, "moderate": 1.5, "high": 2.0, "critical": 3.0}


def split_list(v) -> list[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    return [x.strip() for x in str(v).split(SEP) if x.strip()]


def clean(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def strip_persona(text: str) -> str:
    return re.sub(r"(bác sĩ|bác sỹ|dr\.?)\s+meddies", "bác sĩ", text, flags=re.I)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_to_case(r: dict, subset: str, persona: bool) -> dict:
    turns_text = [clean(r.get(f"turn_{i}")) for i in (1, 2, 3)]
    turns_text = [t for t in turns_text if t]
    if persona:
        turns_text = [strip_persona(t) for t in turns_text]
    return {
        "case_id": clean(r["case_id"]),
        "scenario": clean(r["scenario"]),
        "subset": subset,
        "target_disease": clean(r.get("target_disease")),
        "source_row_id": int(r["source_row_id"]) if clean(r.get("source_row_id")) else -1,
        "reference_status": clean(r.get("review_status")),
        "turns": [{"turn_id": i + 1, "user": t} for i, t in enumerate(turns_text)],
        "labels": {
            "question_type": clean(r.get("question_type")),
            "risk_level": clean(r.get("risk_level")),
            "risk_weight": RISK_WEIGHT.get(clean(r.get("risk_level")), 1.0),
            "kb_support": clean(r.get("kb_support")),
            "expected_behavior": clean(r.get("expected_behavior")),
        },
        # Chỉ evaluator được đọc khối này — KHÔNG đưa vào prompt/retrieval
        "reference": {
            "required_points": split_list(r.get("required_points")),
            "mandatory_warnings": split_list(r.get("mandatory_warnings")),
            "forbidden_content": split_list(r.get("forbidden_content")),
            "evidence_chunk_ids": split_list(r.get("evidence_chunk_ids")),
            "reference_answer": clean(r.get("reference_answer")),
        },
        "provenance": {
            "label_source": clean(r.get("label_source")),
            "reviewer": clean(r.get("reviewer")),
            "notes": clean(r.get("notes")),
        },
    }


def validate(case: dict, kb_ids: set[str], strict: bool) -> tuple[list[str], list[str]]:
    errors, warns = [], []
    lab, ref = case["labels"], case["reference"]
    for k in ("question_type", "risk_level", "kb_support", "expected_behavior"):
        if lab[k] not in ENUMS[k]:
            (errors if strict else warns).append(f"{k}='{lab[k]}' không hợp lệ")
    n = len(case["turns"])
    if case["scenario"] == "single" and n != 1:
        errors.append(f"single nhưng có {n} lượt")
    if case["scenario"] == "multi" and n != 3:
        errors.append(f"multi phải có đúng 3 lượt, đang có {n}")
    if case["scenario"] not in {"single", "multi"}:
        errors.append(f"scenario='{case['scenario']}'")

    if strict:
        if len(ref["required_points"]) < 2:
            errors.append("required_points < 2")
        if lab["question_type"] == "safety_trap" and lab["expected_behavior"] != "refuse_and_refer":
            warns.append("safety_trap nhưng expected_behavior không phải refuse_and_refer")
        if lab["question_type"] == "red_flag" and lab["expected_behavior"] != "urgent_referral":
            warns.append("red_flag nhưng expected_behavior không phải urgent_referral")
        if lab["risk_level"] in {"high", "critical"} and not ref["mandatory_warnings"]:
            warns.append("risk high/critical nhưng chưa có mandatory_warnings")
        if lab["kb_support"] != "not_supported" and not ref["evidence_chunk_ids"]:
            warns.append("chưa có evidence_chunk_ids")
    bad = [c for c in ref["evidence_chunk_ids"] if c not in kb_ids]
    if bad:
        errors.append(f"chunk_id không có trong KB: {bad}")
    return errors, warns


def stratified_split(cases: list[dict], dev_ratio: float, seed: int) -> tuple[list, list]:
    groups: dict[tuple, list] = defaultdict(list)
    for c in cases:
        groups[(c["labels"]["question_type"], c["scenario"])].append(c)
    rng = random.Random(seed)
    dev, test = [], []
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda c: c["case_id"])
        rng.shuffle(g)
        k = round(len(g) * dev_ratio)
        if len(g) >= 2:
            k = min(max(k, 1), len(g) - 1)  # mỗi nhóm có ít nhất 1 case ở cả hai tập
        dev += g[:k]
        test += g[k:]
    return dev, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default="artifacts/review/testset_v03_review.csv")
    ap.add_argument("--new", default="data/testset_v03/new_cases.csv")
    ap.add_argument("--testset-v02", default="data/data_filtered_final/Group3_Testset_cases_v02.jsonl")
    ap.add_argument("--kb-chunks", default="artifacts/kb/gout_kb_v1/chunks.jsonl")
    ap.add_argument("--out-dir", default="data/testset_v03")
    ap.add_argument("--dev-ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--allow-pending", action="store_true",
                    help="Giữ cả dòng pending (chỉ để thử pipeline, KHÔNG dùng cho kết quả)")
    ap.add_argument("--keep-persona", action="store_true",
                    help="Không thay 'bác sĩ Meddies' bằng 'bác sĩ'")
    args = ap.parse_args()

    kb_ids = {r["chunk_id"] for r in read_jsonl(args.kb_chunks)}
    src_ids = {r["case_id"]: r.get("source_row_id", -1) for r in read_jsonl(args.testset_v02)}

    frames = []
    for path, subset in ((args.review, "group3_v03"), (args.new, "authored_v03")):
        if Path(path).exists():
            df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
            df["_subset"] = subset
            frames.append(df)
            print(f"Đọc {len(df)} dòng từ {path}")
    df = pd.concat(frames, ignore_index=True)
    df["review_status"] = df["review_status"].fillna("pending").str.strip()
    df["source_row_id"] = df["case_id"].map(src_ids)

    keep_status = {"approved", "pending"} if args.allow_pending else {"approved"}
    status_counts = Counter(df["review_status"])
    df = df[df["review_status"].isin(keep_status)]

    cases, all_err, all_warn = [], {}, {}
    for r in df.to_dict("records"):
        c = row_to_case(r, r["_subset"], persona=not args.keep_persona)
        err, warn = validate(c, kb_ids, strict=not args.allow_pending)
        if err:
            all_err[c["case_id"]] = err
        if warn:
            all_warn[c["case_id"]] = warn
        cases.append(c)

    dup = [k for k, v in Counter(c["case_id"] for c in cases).items() if v > 1]
    if dup:
        all_err["_duplicate_case_id"] = dup

    if all_err:
        print("\n❌ LỖI (phải sửa trước khi build):")
        for k, v in list(all_err.items())[:40]:
            print(f"  {k}: {v}")
        raise SystemExit(1)

    dev, test = stratified_split(cases, args.dev_ratio, args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for c in dev:
        c["split"] = "dev"
    for c in test:
        c["split"] = "test"
    write_jsonl(out / "dev.jsonl", dev)
    write_jsonl(out / "test.jsonl", test)
    write_jsonl(out / "cases_all.jsonl", dev + test)

    def dist(rows, key):
        return dict(Counter(c["labels"][key] for c in rows))

    manifest = {
        "version": "testset_v03" + ("_DRAFT" if args.allow_pending else ""),
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "input_status_counts": dict(status_counts),
        "n_cases": len(cases),
        "n_turns": sum(len(c["turns"]) for c in cases),
        "n_dev": len(dev),
        "n_test": len(test),
        "by_scenario": dict(Counter(c["scenario"] for c in cases)),
        "by_question_type": {"all": dist(cases, "question_type"), "dev": dist(dev, "question_type"),
                             "test": dist(test, "question_type")},
        "by_risk_level": {"all": dist(cases, "risk_level"), "dev": dist(dev, "risk_level"),
                          "test": dist(test, "risk_level")},
        "by_kb_support": dist(cases, "kb_support"),
        "kb_chunks_sha256": sha256(Path(args.kb_chunks)),
        "files_sha256": {f: sha256(out / f) for f in ("cases_all.jsonl", "dev.jsonl", "test.jsonl")},
        "warnings": all_warn,
        "rule": "Chỉ dev được dùng để chọn checkpoint/hiệu chỉnh ngưỡng. test chỉ chạy cho kết quả cuối.",
    }
    write_json(out / "manifest.json", manifest)

    print(f"\n✅ {len(cases)} case ({manifest['n_turns']} lượt) -> dev {len(dev)} / test {len(test)}")
    print("question_type:", manifest["by_question_type"]["all"])
    print("risk_level   :", manifest["by_risk_level"]["all"])
    print("scenario     :", manifest["by_scenario"])
    print(f"⚠️  {len(all_warn)} case có cảnh báo (xem manifest.json > warnings)")
    targets = {"safety_trap": 40, "red_flag": 25}
    qt = manifest["by_question_type"]["all"]
    for k, v in targets.items():
        if qt.get(k, 0) < v:
            print(f"   - {k}: mới có {qt.get(k, 0)}, mục tiêu ≥ {v}")
    if manifest["by_scenario"].get("multi", 0) < 30:
        print(f"   - multi-turn: mới có {manifest['by_scenario'].get('multi', 0)}, mục tiêu ≥ 30")


if __name__ == "__main__":
    main()
