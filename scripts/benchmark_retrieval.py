"""Benchmark truy xuất trên KB thật (gout_kb_v1) với chunk bằng chứng (evidence_chunk_ids) của testset v03.

So các model embedding (mặc định: MiniLM đang dùng, bkai có tách từ, bge-m3), mỗi model có và không có reranker,
theo đúng cách FaissRetriever chạy: vector chuẩn hoá, tìm cosine chính xác, lấy retrieve_k đoạn, reranker giữ top_k.
Câu hỏi = lượt hỏi của các ca một lượt có nhãn bằng chứng (truy xuất chỉ dùng câu hỏi hiện tại, giống protocol).

  PYTHONPATH=src python3 scripts/benchmark_retrieval.py --split dev
  PYTHONPATH=src python3 scripts/benchmark_retrieval.py --split dev --no-reranker      # không cần FlagEmbedding
  PYTHONPATH=src python3 scripts/benchmark_retrieval.py --split dev \\
      --check-run artifacts/runs/20261025T074006Z_baseline_qwen3_8b_7963            # đối chiếu với run thật

Chạy được trên CPU (KB nhỏ). bkai cần tách từ tiếng Việt: pip install pyvi.
Kết quả: artifacts/reports/rag/retrieval_benchmark_<split>.json và .md
Chỉ số là cận dưới: nhãn bằng chứng gán bằng luật/LLM, các tài liệu hướng dẫn trùng nội dung nhau.
"""
from __future__ import annotations

import argparse
import gc
import platform
import statistics as st
import time
from collections import Counter, defaultdict
from pathlib import Path

from gout_llmops.core.config import load_yaml
from gout_llmops.core.io import ensure_dir, read_jsonl, write_json
from gout_llmops.data.testset import load_cases
from gout_llmops.evaluation.retrieval_metrics import (
    hit_at_k,
    mcnemar_exact,
    paired_bootstrap,
    recall_at_k,
    reciprocal_rank,
    wilson_ci,
)
from gout_llmops.rag.segment import get_segmenter

EMBEDDERS = {  # alias -> (model_id, cần tách từ). MiniLM lấy model_id + revision từ rag config.
    "minilm": (None, False),
    "bkai": ("bkai-foundation-models/vietnamese-bi-encoder", True),
    "bge-m3": ("BAAI/bge-m3", False),
}
PRODUCTION = "minilm+rerank"


def load_queries(testset: str, split: str, kb_ids: set[str]) -> tuple[list[dict], dict]:
    cases = load_cases(testset, split=None if split == "all" else split)
    queries, skipped = [], Counter()
    for c in cases:
        gold = set(c.reference.get("evidence_chunk_ids") or [])
        if c.scenario != "single":
            skipped["multi_turn"] += 1
        elif not gold:
            skipped["no_evidence"] += 1
        elif not gold & kb_ids:
            skipped["evidence_not_in_kb"] += 1
        else:
            queries.append({"case_id": c.case_id, "query": c.turns[0].user, "gold": sorted(gold & kb_ids),
                            "question_type": c.question_type})
    return queries, dict(skipped)


def _segment(texts: list[str]) -> list[str]:
    """Tách từ bằng đúng hàm FaissRetriever dùng khi config có word_segmentation: pyvi."""
    try:
        segment = get_segmenter("pyvi")
    except RuntimeError as e:
        raise SystemExit(str(e)) from e
    return [segment(t) for t in texts]


def _resolved_sha(model_id: str, revision: str | None) -> str | None:
    try:
        from huggingface_hub import model_info
        return model_info(model_id, revision=revision).sha
    except Exception:
        return revision


def embed_rank(model_id: str, revision: str | None, segment: bool, chunks: list[str], queries: list[str],
               retrieve_k: int) -> tuple[list[list[int]], dict]:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    t0 = time.perf_counter()
    model = SentenceTransformer(model_id, revision=revision, device="cpu")
    load_s = time.perf_counter() - t0
    docs = _segment(chunks) if segment else chunks
    t0 = time.perf_counter()
    mat = model.encode(docs, normalize_embeddings=True, convert_to_numpy=True, batch_size=16).astype("float32")
    index_s = time.perf_counter() - t0
    ranked, lat = [], []
    for q in (_segment(queries) if segment else queries):
        t0 = time.perf_counter()
        v = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")[0]
        order = np.argsort(-(mat @ v), kind="stable")[:retrieve_k]
        lat.append((time.perf_counter() - t0) * 1000)
        ranked.append([int(i) for i in order])
    info = {"model_id": model_id, "revision": _resolved_sha(model_id, revision), "word_segmentation": segment,
            "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
            "dim": int(mat.shape[1]), "max_seq_length": int(model.max_seq_length),
            "load_s": round(load_s, 1), "index_86_chunks_s": round(index_s, 1),
            "query_ms_median": round(st.median(lat), 1)}
    del model
    gc.collect()
    return ranked, info


class Reranker:
    """FlagReranker như FaissRetriever (normalize=True); cache điểm theo (câu hỏi, chunk) để các pipeline dùng chung."""

    def __init__(self, model_id: str, revision: str | None):
        try:
            from FlagEmbedding import FlagReranker
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise SystemExit("Cần FlagEmbedding cho reranker (hoặc chạy --no-reranker)") from e
        path = snapshot_download(model_id, revision=revision) if revision else model_id
        self.model = FlagReranker(path, use_fp16=False)
        self.info = {"model_id": model_id, "revision": revision}
        self.cache: dict[tuple[int, int], float] = {}
        self.seconds, self.pairs = 0.0, 0

    def rerank(self, qi: int, query: str, cand: list[int], chunks: list[str]) -> list[int]:
        todo = [c for c in cand if (qi, c) not in self.cache]
        if todo:
            t0 = time.perf_counter()
            scores = self.model.compute_score([[query, chunks[c]] for c in todo], normalize=True)
            self.seconds += time.perf_counter() - t0
            self.pairs += len(todo)
            for c, s in zip(todo, scores if isinstance(scores, list) else [scores], strict=True):
                self.cache[(qi, c)] = float(s)
        return sorted(cand, key=lambda c: self.cache[(qi, c)], reverse=True)  # sort ổn định như retriever


def per_query_metrics(ranked_ids: list[str], gold: set[str], top_k: int) -> dict[str, float]:
    return {"hit@3": hit_at_k(ranked_ids, gold, top_k), "recall@3": recall_at_k(ranked_ids, gold, top_k),
            "mrr@12": reciprocal_rank(ranked_ids, gold), "hit@12": hit_at_k(ranked_ids, gold, len(ranked_ids))}


def check_run(run_dir: str, queries: list[dict], pipelines: dict[str, list[list[str]]], top_k: int) -> dict:
    """So top-k của pipeline MiniLM (có/không reranker, theo run) với context đã lưu của một run thật."""
    preds = {(r["case_id"], int(r["turn_id"])): r for r in read_jsonl(Path(run_dir) / "predictions.jsonl")}
    rows = [preds.get((q["case_id"], 1)) for q in queries]
    reranked = any("rerank_score" in c for r in rows if r for c in (r.get("contexts") or []))
    ours = pipelines["minilm+rerank" if reranked else "minilm"]
    same = n = 0
    for r, mine in zip(rows, ours, strict=True):
        if r and r.get("contexts"):
            n += 1
            same += {c["chunk_id"] for c in r["contexts"]} == set(mine[:top_k])
    return {"run": run_dir, "run_has_reranker": reranked, "n": n, "same_top_k_set": same}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rag-config", default="configs/rag/rag_v1.yaml")
    ap.add_argument("--chunks", default="artifacts/kb/gout_kb_v1/chunks.jsonl")
    ap.add_argument("--testset", default="data/testset_v03/cases_all.jsonl")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--models", nargs="+", default=list(EMBEDDERS), choices=list(EMBEDDERS))
    ap.add_argument("--no-reranker", action="store_true")
    ap.add_argument("--check-run", default=None, help="run thật để đối chiếu top-k của MiniLM")
    ap.add_argument("--out", default=None, help="tiền tố file kết quả (mặc định artifacts/reports/rag/...)")
    args = ap.parse_args()
    if args.split in ("test", "all") and not args.final_test:
        raise SystemExit(f"Split {args.split} có tập test: chỉ chạy MỘT lần ở nghiệm thu cuối (thêm --final-test).")

    cfg = load_yaml(args.rag_config)
    retrieve_k, top_k = int(cfg["retrieval"]["retrieve_k"]), int(cfg["retrieval"]["top_k"])
    chunk_rows = read_jsonl(args.chunks)
    chunk_ids, chunk_texts = [c["chunk_id"] for c in chunk_rows], [c["text"] for c in chunk_rows]
    queries, skipped = load_queries(args.testset, args.split, set(chunk_ids))
    print(f"KB: {len(chunk_rows)} chunk | câu hỏi: {len(queries)} | bỏ qua: {skipped}")

    pipelines: dict[str, list[list[str]]] = {}
    models_info, rerank_info = {}, None
    reranker = None if args.no_reranker else Reranker(cfg["reranker"]["model_id"], cfg["reranker"].get("revision"))
    for alias in args.models:
        model_id, seg = EMBEDDERS[alias]
        revision = None
        if alias == "minilm":
            model_id, revision = cfg["embedding"]["model_id"], cfg["embedding"].get("revision")
        print(f"[+] {alias}: {model_id}")
        ranked, info = embed_rank(model_id, revision, seg, chunk_texts, [q["query"] for q in queries], retrieve_k)
        models_info[alias] = info
        pipelines[alias] = [[chunk_ids[i] for i in r] for r in ranked]
        if reranker:
            pipelines[f"{alias}+rerank"] = [
                [chunk_ids[i] for i in reranker.rerank(qi, q["query"], r, chunk_texts)]
                for qi, (q, r) in enumerate(zip(queries, ranked, strict=True))]
    if reranker:
        rerank_info = {**reranker.info, "pairs_scored": reranker.pairs,
                       "ms_per_query_12_pairs": round(reranker.seconds / max(reranker.pairs, 1) * retrieve_k * 1000)}

    per_q = {name: [per_query_metrics(r, set(q["gold"]), top_k) for r, q in zip(rl, queries, strict=True)]
             for name, rl in pipelines.items()}
    n = len(queries)
    summary = {}
    for name, ms in per_q.items():
        hits = sum(m["hit@3"] for m in ms)
        summary[name] = {"hit@3": hits / n, "hit@3_ci95": wilson_ci(hits, n),
                         **{k: sum(m[k] for m in ms) / n for k in ("recall@3", "mrr@12", "hit@12")}}
    ref = PRODUCTION if PRODUCTION in per_q else "minilm"
    comparisons = {}
    for name, ms in per_q.items():
        if name == ref:
            continue
        a, b = [m["hit@3"] for m in ms], [m["hit@3"] for m in per_q[ref]]
        d, lo, hi = paired_bootstrap(a, b)
        dm = paired_bootstrap([m["mrr@12"] for m in ms], [m["mrr@12"] for m in per_q[ref]])
        comparisons[name] = {"vs": ref, "d_hit@3": d, "d_hit@3_ci95": (lo, hi), "mcnemar_p": mcnemar_exact(a, b),
                             "d_mrr@12": dm[0], "d_mrr@12_ci95": (dm[1], dm[2])}
    by_type: dict[str, dict[str, float]] = defaultdict(dict)
    types = Counter(q["question_type"] for q in queries)
    for name, ms in per_q.items():
        for t in types:
            idx = [i for i, q in enumerate(queries) if q["question_type"] == t]
            by_type[t][name] = sum(ms[i]["hit@3"] for i in idx) / len(idx)

    import sentence_transformers
    import torch
    result = {"split": args.split, "rag_config": args.rag_config, "chunks": args.chunks, "n_chunks": len(chunk_ids),
              "n_queries": n, "skipped": skipped, "retrieve_k": retrieve_k, "top_k": top_k,
              "embedders": models_info, "reranker": rerank_info, "summary": summary, "comparisons": comparisons,
              "hit@3_by_question_type": {t: {"n": types[t], **v} for t, v in by_type.items()},
              "check_run": check_run(args.check_run, queries, pipelines, top_k) if args.check_run else None,
              "env": {"device": "cpu", "python": platform.python_version(), "torch": torch.__version__,
                      "sentence_transformers": sentence_transformers.__version__},
              "per_query": [{"case_id": q["case_id"], "gold": q["gold"],
                             **{name: pipelines[name][i] for name in pipelines}} for i, q in enumerate(queries)]}
    out = Path(args.out or f"artifacts/reports/rag/retrieval_benchmark_{args.split}")
    ensure_dir(out.parent)
    write_json(out.with_suffix(".json"), result)
    out.with_suffix(".md").write_text(render_md(result), encoding="utf-8")
    print(render_md(result))
    print(f"-> {out.with_suffix('.json')}")


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_md(r: dict) -> str:
    lines = [f"# Benchmark truy xuất — {r['split']} ({r['n_queries']} câu hỏi một lượt, KB {r['n_chunks']} chunk)", "",
             f"Lấy {r['retrieve_k']} đoạn bằng embedding, giữ {r['top_k']}. Bỏ qua: {r['skipped']}.", "",
             "| Pipeline | Hit@3 [CI95] | Recall@3 | MRR@12 | Hit@12 |", "| --- | --- | --- | --- | --- |"]
    for name, s in r["summary"].items():
        lo, hi = s["hit@3_ci95"]
        lines.append(f"| {name} | {_pct(s['hit@3'])} [{_pct(lo)}; {_pct(hi)}] | {s['recall@3']:.3f} | "
                     f"{s['mrr@12']:.3f} | {_pct(s['hit@12'])} |")
    if r["comparisons"]:
        lines += ["", "| So với | ΔHit@3 [CI95] | McNemar p | ΔMRR@12 [CI95] |", "| --- | --- | --- | --- |"]
        for name, c in r["comparisons"].items():
            lines.append(f"| {name} − {c['vs']} | {100 * c['d_hit@3']:+.1f} điểm % [{100 * c['d_hit@3_ci95'][0]:+.1f}; "
                         f"{100 * c['d_hit@3_ci95'][1]:+.1f}] | {c['mcnemar_p']:.3f} | {c['d_mrr@12']:+.3f} "
                         f"[{c['d_mrr@12_ci95'][0]:+.3f}; {c['d_mrr@12_ci95'][1]:+.3f}] |")
    lines += ["", "| Embedding | Tham số (triệu) | Chiều | Đọc tối đa (token) | Mã hoá 86 chunk (s) "
                  "| Mỗi câu hỏi (ms) |", "| --- | --- | --- | --- | --- | --- |"]
    for alias, m in r["embedders"].items():
        lines.append(f"| {alias} ({m['model_id']}) | {m['params_m']} | {m['dim']} | {m['max_seq_length']} | "
                     f"{m['index_86_chunks_s']} | {m['query_ms_median']} |")
    if r["reranker"]:
        lines += ["", f"Reranker {r['reranker']['model_id']}: khoảng {r['reranker']['ms_per_query_12_pairs']} ms "
                      f"cho 12 cặp mỗi câu hỏi trên CPU."]
    if r["check_run"]:
        c = r["check_run"]
        lines += ["", f"Đối chiếu run thật ({'có' if c['run_has_reranker'] else 'không'} reranker): "
                      f"{c['same_top_k_set']}/{c['n']} câu có cùng tập top-{r['top_k']}."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
