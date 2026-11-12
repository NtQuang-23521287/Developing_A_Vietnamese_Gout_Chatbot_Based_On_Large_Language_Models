from __future__ import annotations

from pathlib import Path
from typing import Any

from gout_llmops.core.io import read_jsonl


class FaissRetriever:
    def __init__(self, index_dir: str | Path, rag_config: dict):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("Install RAG dependencies: pip install -e '.[rag]'") from e

        self.faiss = faiss
        self.cfg = rag_config
        self.index_dir = Path(index_dir)
        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))
        self.chunks = read_jsonl(self.index_dir / "chunks.jsonl")
        emb = rag_config["embedding"]
        self.encoder = SentenceTransformer(emb["model_id"], revision=emb.get("revision"))
        self.normalize = bool(emb.get("normalize_embeddings", True))
        self.reranker = None
        rerank = rag_config.get("reranker", {})
        if rerank.get("enabled"):
            # KHÔNG được âm thầm bỏ reranker: trước đây lỗi nạp bị nuốt nên các run testset v03 (2026-10-25,
            # SFT v1) chạy không có reranker trong khi config bật -> protocol lệch mà không ai biết.
            try:
                from FlagEmbedding import FlagReranker
                self.reranker = FlagReranker(
                    rerank["model_id"], use_fp16=True, trust_remote_code=True
                )
            except Exception as e:
                raise RuntimeError(
                    f"Config bật reranker ({rerank.get('model_id')}) nhưng không nạp được: {type(e).__name__}: {e}. "
                    "Sửa môi trường (pip install FlagEmbedding, phiên bản transformers tương thích) "
                    "hoặc tắt reranker trong config một cách tường minh."
                ) from e

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        rcfg = self.cfg["retrieval"]
        retrieve_k = int(rcfg.get("retrieve_k", 12))
        top_k = int(rcfg.get("top_k", 3))
        q = self.encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=self.normalize
        ).astype("float32")
        scores, ids = self.index.search(q, retrieve_k)
        candidates = []
        for score, idx in zip(scores[0].tolist(), ids[0].tolist(), strict=True):
            if idx < 0:
                continue
            row = dict(self.chunks[idx])
            row["retrieval_score"] = float(score)
            candidates.append(row)

        if self.reranker and candidates:
            pairs = [[query, c["text"]] for c in candidates]
            rr = self.reranker.compute_score(pairs, normalize=True)
            if not isinstance(rr, list):
                rr = [rr]
            for c, s in zip(candidates, rr, strict=True):
                c["rerank_score"] = float(s)
            candidates.sort(key=lambda x: x.get("rerank_score", -1e9), reverse=True)
        return candidates[:top_k]
