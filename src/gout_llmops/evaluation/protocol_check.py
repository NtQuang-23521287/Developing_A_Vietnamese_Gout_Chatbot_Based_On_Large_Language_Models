"""Kiểm tra hai run có cùng protocol truy xuất không trước khi so sánh.

Reranker và số context: đọc trực tiếp context đã lưu trong predictions.jsonl (thứ model thật sự nhìn thấy), không
tin vào config: các run testset v03 ngày 2026-10-25 và SFT v1 có config bật reranker nhưng context không có
rerank_score (reranker nạp lỗi bị nuốt). So một run như vậy với run có reranker là so hai protocol khác nhau.
Embedding: đọc rag_config.json của run (nạp embedding lỗi thì run dừng hẳn, nên config phản ánh đúng thực tế).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

from gout_llmops.core.io import read_jsonl
from gout_llmops.rag.segment import embedding_identity


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


def run_embedding(run: str | Path) -> dict | None:
    """Embedding truy xuất của run theo rag_config.json; None nếu run không lưu file này."""
    p = Path(run) / "rag_config.json"
    if not p.exists():
        return None
    return embedding_identity(json.loads(p.read_text(encoding="utf-8")).get("embedding") or {})


def _embedding_differs(a: dict, b: dict) -> bool:
    if any(a[k] != b[k] for k in ("model_id", "normalize_embeddings", "word_segmentation")):
        return True
    return bool(a["revision"] and b["revision"] and a["revision"] != b["revision"])  # thiếu revision = không rõ


def retrieval_mismatch(runs: list[str | Path], case_ids: set[str] | None = None,
                       tol: float = 0.02) -> str | None:
    """None nếu các run cùng protocol truy xuất; ngược lại trả mô tả chênh lệch."""
    embs = [(Path(r).name, e) for r in runs if (e := run_embedding(r)) is not None]
    if any(_embedding_differs(a, b) for (_, a), (_, b) in itertools.combinations(embs, 2)):
        return "; ".join(f"{name}: embedding {e['model_id']}, tách từ {e['word_segmentation']}" for name, e in embs)
    profiles = {str(r): retrieval_profile(r, case_ids) for r in runs}
    rates = [p["reranker_rate"] for p in profiles.values() if p["reranker_rate"] is not None]
    sizes = [p["mean_contexts"] for p in profiles.values() if p["mean_contexts"] is not None]
    if (rates and max(rates) - min(rates) > tol) or (sizes and max(sizes) - min(sizes) > 0.05):
        return "; ".join(f"{Path(r).name}: reranker {p['reranker_rate']:.0%}, {p['mean_contexts']:.2f} context/lượt"
                         for r, p in profiles.items() if p["reranker_rate"] is not None)
    return None
