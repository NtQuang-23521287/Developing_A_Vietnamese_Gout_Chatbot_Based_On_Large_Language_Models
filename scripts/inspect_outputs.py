"""Đếm nhanh các dấu hiệu lỗi đã biết trong câu trả lời của một hoặc nhiều run (không tốn tiền).

  PYTHONPATH=src python3 scripts/inspect_outputs.py artifacts/runs/<base> artifacts/runs/<sft> --names Base SFT
"""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

from gout_llmops.core.io import read_jsonl

PATTERNS = {
    "Vinmec / quảng cáo cơ sở": r"vinmec|hệ thống y tế",
    "Răng hàm mặt": r"răng hàm mặt",
    "Hộ lý": r"hộ lý",
    "Nội tiết": r"nội tiết",
    "Cơ xương khớp": r"cơ xương khớp",
    "Trộn tiếng Anh (Rheumatology...)": r"rheumatolog|\bgout flare\b|\buric acid\b",
    "Có số liều (mg/viên)": r"\d+\s*(?:mg|viên)\b",
    "Khuyên đi khám / bác sĩ": r"đi khám|bác sĩ|cơ sở y tế|bệnh viện",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--names", nargs="+")
    args = ap.parse_args()
    names = args.names or [Path(r).name[:30] for r in args.runs]
    stats = {}
    for name, run in zip(names, args.runs):
        preds = [p for p in read_jsonl(Path(run) / "predictions.jsonl") if p.get("status") == "ok"]
        ans = [p["answer"] for p in preds]
        row = {"n_turns": len(ans),
               "median_words": statistics.median(len(a.split()) for a in ans) if ans else 0}
        for label, pat in PATTERNS.items():
            row[label] = sum(1 for a in ans if re.search(pat, a, re.I))
        stats[name] = row
    keys = list(next(iter(stats.values())))
    print("| Chỉ số | " + " | ".join(names) + " |")
    print("| --- |" + " --- |" * len(names))
    for k in keys:
        print(f"| {k} | " + " | ".join(str(stats[n][k]) for n in names) + " |")


if __name__ == "__main__":
    main()
