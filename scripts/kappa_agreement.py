"""Đo độ đồng thuận giữa 2 người gán nhãn (Bước 1.3).

1) Tạo 2 file mù (đã ẩn nhãn GPT/luật):
   PYTHONPATH=src python3 scripts/kappa_agreement.py make --annotators Q T

2) Mỗi người tự điền question_type và risk_level vào file của mình, KHÔNG bàn với nhau.

3) Tính kết quả:
   PYTHONPATH=src python3 scripts/kappa_agreement.py score --annotators Q T
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

LABELS = {
    "question_type": ["basic_knowledge", "clinical_reasoning", "red_flag", "safety_trap"],
    "risk_level": ["low", "moderate", "high", "critical"],
}
OUT_DIR = Path("artifacts/review/kappa")


def make(args) -> None:
    df = pd.read_csv(args.input, encoding="utf-8-sig", dtype=str).fillna("")
    df = df[df["review_priority"] != "Z"]
    quota = {"A": 10, "B": 5, "C": 15}
    parts = []
    for prio, n in quota.items():
        pool = df[df["review_priority"] == prio]
        parts.append(pool.sample(min(n, len(pool)), random_state=args.seed))
    sample = pd.concat(parts).sample(frac=1, random_state=args.seed)  # xáo trộn, che mức ưu tiên

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # file mù: chỉ có câu hỏi + chunk, KHÔNG có nhãn gợi ý
    blind_cols = ["case_id", "target_disease", "turn_1", "turn_2", "turn_3", "retrieved_chunks"]
    for a in args.annotators:
        b = sample[blind_cols].copy()
        b["question_type"] = ""
        b["risk_level"] = ""
        b["note"] = ""
        b.to_csv(OUT_DIR / f"kappa_{a}.csv", index=False, encoding="utf-8-sig")
    # lưu nhãn GPT riêng để so sánh sau
    sample[["case_id", "review_priority", "question_type", "risk_level"]].to_csv(
        OUT_DIR / "kappa_llm_labels.csv", index=False, encoding="utf-8-sig")

    print(f"Đã tạo {len(sample)} dòng cho mỗi người:")
    for a in args.annotators:
        print(f"  {OUT_DIR / f'kappa_{a}.csv'}")
    print("Giá trị hợp lệ:")
    for k, v in LABELS.items():
        print(f"  {k}: {', '.join(v)}")


def _load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    for k, allowed in LABELS.items():
        d[k] = d[k].str.strip().str.lower()
        bad = d[~d[k].isin(allowed)]
        if len(bad):
            raise SystemExit(f"{path.name}: {k} sai/trống ở {bad['case_id'].tolist()}")
    return d.set_index("case_id")


def score(args) -> None:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    a, b = args.annotators
    da, db = _load(OUT_DIR / f"kappa_{a}.csv"), _load(OUT_DIR / f"kappa_{b}.csv")
    llm = pd.read_csv(OUT_DIR / "kappa_llm_labels.csv", encoding="utf-8-sig",
                      dtype=str).fillna("").set_index("case_id")
    ids = da.index.intersection(db.index)
    result = {"n": int(len(ids))}

    for k, labels in LABELS.items():
        x, y = da.loc[ids, k], db.loc[ids, k]
        agree = float((x == y).mean())
        kappa = cohen_kappa_score(x, y, labels=labels)
        # risk_level có thứ tự -> dùng thêm weighted kappa (lệch 1 bậc phạt nhẹ hơn lệch 3 bậc)
        wk = (cohen_kappa_score(x, y, labels=labels, weights="linear")
              if k == "risk_level" else None)
        vs_llm_a = float((x == llm.loc[ids, k]).mean())
        vs_llm_b = float((y == llm.loc[ids, k]).mean())
        result[k] = {
            "percent_agreement": round(agree, 3),
            "cohen_kappa": round(float(kappa), 3) if kappa == kappa else None,
            "weighted_kappa_linear": round(float(wk), 3) if wk is not None and wk == wk else None,
            f"agreement_{a}_vs_llm": round(vs_llm_a, 3),
            f"agreement_{b}_vs_llm": round(vs_llm_b, 3),
            "label_distribution": {a: x.value_counts().to_dict(), b: y.value_counts().to_dict()},
        }
        print(f"\n=== {k} ===")
        print(f"  % đồng ý {a}-{b}     : {agree:.1%}")
        print(f"  Cohen's kappa        : {result[k]['cohen_kappa']}")
        if wk is not None:
            print(f"  Weighted kappa       : {result[k]['weighted_kappa_linear']}")
        print(f"  {a} khớp GPT / {b} khớp GPT: {vs_llm_a:.1%} / {vs_llm_b:.1%}")
        cm = pd.DataFrame(confusion_matrix(x, y, labels=labels), index=[f"{a}:{lab}" for lab in labels],
                          columns=[f"{b}:{lab}" for lab in labels])
        print(cm.loc[(cm.sum(axis=1) > 0), (cm.sum(axis=0) > 0)].to_string())

    dis = []
    for cid in ids:
        diff = {k: (da.at[cid, k], db.at[cid, k]) for k in LABELS if da.at[cid, k] != db.at[cid, k]}
        if diff:
            dis.append({"case_id": cid, **{f"{k}_{a}": v[0] for k, v in diff.items()},
                        **{f"{k}_{b}": v[1] for k, v in diff.items()},
                        "question": da.at[cid, "turn_1"][:200]})
    pd.DataFrame(dis).to_csv(OUT_DIR / "disagreements.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "kappa_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(f"\n{len(dis)} dòng bất đồng -> {OUT_DIR / 'disagreements.csv'}")
    print(f"Kết quả -> {OUT_DIR / 'kappa_result.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--input", default="artifacts/review/testset_v03_review_triaged.csv")
    m.add_argument("--annotators", nargs=2, default=["Q", "T"])
    m.add_argument("--seed", type=int, default=2026)
    s = sub.add_parser("score")
    s.add_argument("--annotators", nargs=2, default=["Q", "T"])
    args = ap.parse_args()
    make(args) if args.cmd == "make" else score(args)


if __name__ == "__main__":
    main()
