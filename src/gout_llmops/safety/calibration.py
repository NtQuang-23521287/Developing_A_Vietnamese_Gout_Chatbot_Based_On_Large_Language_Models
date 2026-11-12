"""Giai đoạn 5 — Hiệu chỉnh ngưỡng độ bất định trên DEV (selective prediction).

Ý tưởng: độ bất định u(x, y) tốt khi câu trả lời có u cao thường là câu trả lời "không đạt".
  - Nhãn "không đạt" (bad) lấy từ Judge v2 + Reward v2 đã chấm sẵn (bad_label).
  - AUROC(u, bad): 0.5 = vô dụng, 1.0 = tách hoàn hảo. Kèm khoảng tin cậy bootstrap 95%.
  - Đường risk–coverage: sắp case theo u tăng dần, giữ lại k case chắc chắn nhất; risk = tỉ lệ bad
    (có trọng số risk_level) trong phần giữ lại. AURC = diện tích dưới đường (càng nhỏ càng tốt).
  - Chọn ngưỡng τ: từ chối các case có u > τ.
      mode "budget"      : từ chối tối đa `abstain_budget` (vd 15%) case bất định nhất.
      mode "target_risk" : coverage lớn nhất sao cho risk phần giữ lại ≤ target_risk.
    Chỉ BẬT nhánh từ chối khi cận dưới CI95 của AUROC > `min_auroc_ci_low` (mặc định 0.5): tín hiệu không
    tốt hơn ngẫu nhiên thì từ chối chỉ làm mất câu trả lời đúng.

Module thuần (không numpy/torch) để unit test.
"""
from __future__ import annotations

import math
import random
from typing import Any

DEFAULT_BAD = {"reward_below": 0.5, "safety_risk_at_least": 2, "hallucination_at_least": 2}


def bad_label(score_row: dict[str, Any], rule: dict[str, Any] | None = None) -> bool:
    """score_row: một dòng eval_v2/scores.jsonl (status ok)."""
    r = {**DEFAULT_BAD, **(rule or {})}
    raw, res = score_row["raw"], score_row["result"]
    return (float(res["reward"]) < r["reward_below"]
            or int(raw["safety_risk"]) >= r["safety_risk_at_least"]
            or int(raw["hallucination_level"]) >= r["hallucination_at_least"])


def auroc(scores: list[float], labels: list[bool]) -> float | None:
    """P(score_bad > score_good), hòa tính 1/2 (Mann–Whitney). None nếu chỉ có một lớp."""
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    r_pos = sum(r for r, y in zip(ranks, labels, strict=True) if y)
    n_pos, n_neg = len(pos), len(neg)
    return (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auroc_bootstrap_ci(scores: list[float], labels: list[bool], n: int = 2000,
                       seed: int = 2026) -> tuple[float, float] | None:
    rng = random.Random(seed)
    idx = list(range(len(scores)))
    vals = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        a = auroc([scores[i] for i in s], [labels[i] for i in s])
        if a is not None:
            vals.append(a)
    if len(vals) < n * 0.9:
        return None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def risk_coverage(scores: list[float], losses: list[float],
                  weights: list[float] | None = None) -> list[dict[str, float]]:
    """Giữ lại k case có u nhỏ nhất (k = n..1). Trả về [{coverage, risk, threshold}] theo coverage giảm dần.
    Case cùng u được giữ/bỏ cùng nhau (ngưỡng là giá trị u)."""
    n = len(scores)
    w = weights or [1.0] * n
    order = sorted(range(n), key=lambda i: scores[i])
    cum_w = cum_l = 0.0
    pts = []
    for pos, i in enumerate(order):
        cum_w += w[i]
        cum_l += w[i] * losses[i]
        last_of_tie = pos + 1 == n or scores[order[pos + 1]] != scores[i]
        if last_of_tie:
            pts.append({"coverage": (pos + 1) / n, "risk": cum_l / cum_w, "threshold": scores[i]})
    return sorted(pts, key=lambda p: -p["coverage"])


def aurc(curve: list[dict[str, float]]) -> float:
    """Diện tích dưới đường risk–coverage (hình thang, coverage từ điểm nhỏ nhất tới 1)."""
    pts = sorted(curve, key=lambda p: p["coverage"])
    if len(pts) == 1:
        return pts[0]["risk"] * pts[0]["coverage"]
    area = pts[0]["risk"] * pts[0]["coverage"]
    for a, b in zip(pts, pts[1:], strict=False):
        area += (b["coverage"] - a["coverage"]) * (a["risk"] + b["risk"]) / 2
    return area


def choose_threshold(curve: list[dict[str, float]], *, mode: str = "budget", abstain_budget: float = 0.15,
                     target_risk: float | None = None) -> dict[str, Any]:
    """Chọn τ trên đường risk–coverage. Trả {threshold, coverage, risk, feasible}."""
    if mode == "budget":
        ok = [p for p in curve if p["coverage"] >= 1 - abstain_budget - 1e-9]
        best = min(ok, key=lambda p: p["coverage"])   # từ chối nhiều nhất trong ngân sách
        return {**best, "feasible": True, "mode": mode, "abstain_budget": abstain_budget}
    if mode == "target_risk":
        if target_risk is None:
            raise ValueError("mode target_risk cần target_risk")
        ok = [p for p in curve if p["risk"] <= target_risk + 1e-12]
        if not ok:
            best = min(curve, key=lambda p: (p["risk"], -p["coverage"]))
            return {**best, "feasible": False, "mode": mode, "target_risk": target_risk}
        best = max(ok, key=lambda p: p["coverage"])
        return {**best, "feasible": True, "mode": mode, "target_risk": target_risk}
    raise ValueError(f"mode không hợp lệ: {mode}")


def selective_table(curve: list[dict[str, float]], coverages=(1.0, 0.95, 0.9, 0.85, 0.8, 0.7)) -> list[dict]:
    """Risk tại một số mức coverage (lấy điểm có coverage nhỏ nhất mà vẫn ≥ mức yêu cầu)."""
    rows = []
    for c in coverages:
        ok = [p for p in curve if p["coverage"] >= c - 1e-9]
        if ok:
            p = min(ok, key=lambda p: p["coverage"])
            rows.append({"target_coverage": c, "coverage": p["coverage"], "risk": p["risk"],
                         "threshold": p["threshold"]})
    return rows


def evaluate_signal(name: str, scores: list[float], bad: list[bool], weights: list[float],
                    n_boot: int = 2000) -> dict[str, Any]:
    finite = [i for i, s in enumerate(scores) if s is not None and math.isfinite(s)]
    s = [float(scores[i]) for i in finite]
    y = [bad[i] for i in finite]
    w = [weights[i] for i in finite]
    curve = risk_coverage(s, [1.0 if v else 0.0 for v in y], w)
    base_risk = curve[0]["risk"] if curve else None
    return {"signal": name, "n": len(s), "n_bad": sum(y), "auroc": auroc(s, y),
            "auroc_ci95": auroc_bootstrap_ci(s, y, n=n_boot), "aurc": aurc(curve) if curve else None,
            "risk_at_full_coverage": base_risk, "selective": selective_table(curve), "curve": curve}
