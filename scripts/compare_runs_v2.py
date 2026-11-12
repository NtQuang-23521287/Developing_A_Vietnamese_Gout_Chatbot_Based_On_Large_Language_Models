"""So sánh nhiều run đã chấm bằng score_run_v2.py, trên CÙNG tập case (ghép cặp theo case_id).

  PYTHONPATH=src python3 scripts/compare_runs_v2.py \
      artifacts/runs/<qwen> artifacts/runs/<gemma> artifacts/runs/<seallms> \
      --names Qwen3-8B Gemma3-4B SeaLLMs-v3-7B

In bảng chỉ số chính và, với mỗi cặp model, chênh lệch reward có trọng số kèm khoảng tin cậy 95%
(paired bootstrap) và p-value Wilcoxon. Khoảng tin cậy chứa 0 => chưa đủ cơ sở nói model nào hơn.
Ghi kết quả ra artifacts/reports/compare_v2.json và .md
"""
from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path

from gout_llmops.core.io import read_jsonl, write_json
from gout_llmops.evaluation.protocol_check import retrieval_mismatch
from gout_llmops.evaluation.reward_v2 import aggregate


def paired_bootstrap(a: list[float], b: list[float], w: list[float], n: int = 5000, seed: int = 2026):
    rng = random.Random(seed)
    idx = list(range(len(a)))

    def stat(s):
        tw = sum(w[i] for i in s)
        return sum((a[i] - b[i]) * w[i] for i in s) / tw

    point = stat(idx)
    boots = sorted(stat([rng.choice(idx) for _ in idx]) for _ in range(n))
    return point, (boots[int(0.025 * n)], boots[int(0.975 * n) - 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--eval-name", default="eval_v2")
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all",
                    help="chỉ so sánh trên tập này (quyết định chọn model/checkpoint: dùng dev)")
    ap.add_argument("--out", default="artifacts/reports/compare_v2", help="tiền tố file kết quả .json/.md")
    args = ap.parse_args()
    names = args.names or [Path(r).name for r in args.runs]
    if len(names) != len(args.runs):
        ap.error(f"--names có {len(names)} tên nhưng có {len(args.runs)} run")

    scored = {}
    for name, run in zip(names, args.runs, strict=True):
        rows = [r for r in read_jsonl(Path(run) / args.eval_name / "scores.jsonl") if r["status"] == "ok"
                and (args.split == "all" or r.get("split") == args.split)]
        scored[name] = {r["case_id"]: r for r in rows}
    common = sorted(set.intersection(*(set(s) for s in scored.values())))
    print(f"Split: {args.split} | số case chung giữa các run: {len(common)}")
    mismatch = retrieval_mismatch(args.runs, set(common))
    if mismatch:
        print("[CẢNH BÁO] Các run khác protocol truy xuất — chênh lệch có thể do retriever, không chỉ do model: "
              f"{mismatch}")

    table = {}
    for name in names:
        s = aggregate([scored[name][c] for c in common])
        table[name] = s
    keys = [("reward_weighted_mean", "Reward (TB trọng số)"), ("fatal_count", "Fatal")]
    md = ["| Chỉ số | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
    if mismatch:
        md = [f"> **Cảnh báo:** các run khác protocol truy xuất — {mismatch}", ""] + md
    for k, label in keys:
        md.append(f"| {label} | " + " | ".join(f"{table[n][k]:.4f}" if isinstance(table[n][k], float)
                                                   else str(table[n][k]) for n in names) + " |")
    md.append("| Reward CI95 | " + " | ".join(
        "[{:.3f}, {:.3f}]".format(*table[n]["reward_weighted_ci95"]) for n in names) + " |")
    for k in table[names[0]]["quality_weighted_mean"]:
        md.append(f"| {k} (0–1) | " + " | ".join(
            "—" if table[n]["quality_weighted_mean"][k] is None else f"{table[n]['quality_weighted_mean'][k]:.3f}"
            for n in names) + " |")
    for k in table[names[0]]["rates_weighted"]:
        md.append(f"| Rate {k} | " + " | ".join(f"{table[n]['rates_weighted'][k]:.3f}" for n in names) + " |")

    pairs = []
    md += ["", "| So sánh | Chênh reward (A−B) | CI95 | Wilcoxon p | Kết luận |", "| --- | --- | --- | --- | --- |"]
    for a, b in itertools.combinations(names, 2):
        ra = [scored[a][c]["result"]["reward"] for c in common]
        rb = [scored[b][c]["result"]["reward"] for c in common]
        w = [scored[a][c]["risk_weight"] for c in common]
        d, (lo, hi) = paired_bootstrap(ra, rb, w)
        try:
            from scipy.stats import wilcoxon
            p = float(wilcoxon(ra, rb).pvalue)
        except Exception:  # noqa: BLE001  (scipy thiếu hoặc mọi chênh lệch = 0)
            p = None
        verdict = "chưa phân biệt được" if lo <= 0 <= hi else (f"{a} tốt hơn" if d > 0 else f"{b} tốt hơn")
        pairs.append({"a": a, "b": b, "diff": d, "ci95": [lo, hi], "wilcoxon_p": p, "verdict": verdict})
        md.append(f"| {a} vs {b} | {d:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                  f"{'—' if p is None else f'{p:.4f}'} | {verdict} |")

    text = "\n".join(md)
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(f"{out}.json", {"split": args.split, "n_common": len(common), "summary": table, "pairs": pairs})
    Path(f"{out}.md").write_text(f"Split: {args.split}, n = {len(common)}\n\n" + text + "\n", encoding="utf-8")
    print(f"-> {out}.md")


if __name__ == "__main__":
    main()
