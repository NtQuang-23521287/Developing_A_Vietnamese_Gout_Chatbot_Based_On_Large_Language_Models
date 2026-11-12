"""Kiểm tra hai run có cùng protocol truy xuất không trước khi so sánh.

Đọc trực tiếp context đã lưu trong predictions.jsonl (thứ model thật sự nhìn thấy), không tin vào config:
các run testset v03 ngày 2026-10-25 và SFT v1 có config bật reranker nhưng context không có rerank_score
(reranker nạp lỗi bị nuốt). So một run như vậy với run có reranker là so hai protocol khác nhau.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from gout_llmops.core.io import read_jsonl


def retrieval_profile(run: str | Path, case_ids: set[str] | None = None) -> dict[str, Any]:
    """Tỉ lệ lượt (status ok, có context) mà context mang rerank_score; số context trung bình mỗi lượt."""
    n_turns = n_rerank = n_ctx = 0
    for p in read_jsonl(Path(run) / "predictions.jsonl"):
        if p.get("status") != "ok" or (case_ids is not None and p["case_id"] not in case_ids):
            continue
        ctx = p.get("contexts") or []
        if not ctx:
            continue
        n_turns += 1
        n_ctx += len(ctx)
        n_rerank += any("rerank_score" in c for c in ctx)
    return {"n_turns_with_context": n_turns,
            "reranker_rate": n_rerank / n_turns if n_turns else None,
            "mean_contexts": n_ctx / n_turns if n_turns else None}


def retrieval_mismatch(runs: list[str | Path], case_ids: set[str] | None = None,
                       tol: float = 0.02) -> str | None:
    """None nếu các run cùng protocol truy xuất; ngược lại trả mô tả chênh lệch."""
    profiles = {str(r): retrieval_profile(r, case_ids) for r in runs}
    rates = [p["reranker_rate"] for p in profiles.values() if p["reranker_rate"] is not None]
    sizes = [p["mean_contexts"] for p in profiles.values() if p["mean_contexts"] is not None]
    if (rates and max(rates) - min(rates) > tol) or (sizes and max(sizes) - min(sizes) > 0.05):
        return "; ".join(f"{Path(r).name}: reranker {p['reranker_rate']:.0%}, {p['mean_contexts']:.2f} context/lượt"
                         for r, p in profiles.items() if p["reranker_rate"] is not None)
    return None
