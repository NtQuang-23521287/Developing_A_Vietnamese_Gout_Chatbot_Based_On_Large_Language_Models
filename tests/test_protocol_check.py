import json

from gout_llmops.evaluation.protocol_check import retrieval_mismatch, retrieval_profile


def _run(tmp_path, name, rerank: bool, n_ctx: int = 3, cases=("a", "b")):
    d = tmp_path / name
    d.mkdir()
    rows = []
    for cid in cases:
        ctx = [{"chunk_id": f"c{i}", "retrieval_score": 0.5, **({"rerank_score": 0.9} if rerank else {})}
               for i in range(n_ctx)]
        rows.append({"case_id": cid, "turn_id": 1, "status": "ok", "contexts": ctx})
    rows.append({"case_id": "err", "turn_id": 1, "status": "error"})
    (d / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return d


def test_profile_counts_only_ok_turns_with_context(tmp_path):
    p = retrieval_profile(_run(tmp_path, "r", rerank=True))
    assert p == {"n_turns_with_context": 2, "reranker_rate": 1.0, "mean_contexts": 3.0}
    assert retrieval_profile(_run(tmp_path, "r2", rerank=True), case_ids={"a"})["n_turns_with_context"] == 1


def _rag(run, **emb):
    cfg = {"embedding": {"model_id": "minilm", "revision": "r1", "normalize_embeddings": True, **emb}}
    (run / "rag_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return run


def test_mismatch_detects_embedding(tmp_path):
    a = _rag(_run(tmp_path, "a", rerank=True))
    assert retrieval_mismatch([a, _rag(_run(tmp_path, "b", rerank=True))]) is None
    msg = retrieval_mismatch([a, _rag(_run(tmp_path, "c", rerank=True), model_id="bkai", word_segmentation="pyvi")])
    assert msg and "bkai" in msg and "pyvi" in msg
    assert retrieval_mismatch([a, _rag(_run(tmp_path, "d", rerank=True), revision="r2")]) is not None
    # thiếu revision = không rõ; run không lưu rag_config.json thì chỉ so context
    assert retrieval_mismatch([a, _rag(_run(tmp_path, "e", rerank=True), revision=None)]) is None
    assert retrieval_mismatch([a, _run(tmp_path, "f", rerank=True)]) is None


def test_mismatch_detects_reranker_and_top_k(tmp_path):
    a = _run(tmp_path, "a", rerank=False)
    assert retrieval_mismatch([a, _run(tmp_path, "b", rerank=False)]) is None
    msg = retrieval_mismatch([a, _run(tmp_path, "c", rerank=True)])
    assert msg and "reranker 0%" in msg and "reranker 100%" in msg
    assert retrieval_mismatch([a, _run(tmp_path, "d", rerank=False, n_ctx=5)]) is not None
