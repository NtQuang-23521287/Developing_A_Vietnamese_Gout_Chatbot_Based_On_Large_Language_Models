from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gout_llmops.core.io import read_jsonl
from gout_llmops.rag.segment import embedding_identity, get_segmenter


def check_index_embedding(index_dir: str | Path, emb_cfg: dict) -> None:
    """Index phải dựng bằng đúng embedding của config truy xuất.

    Hai model cùng số chiều thì FAISS vẫn chạy, chỉ là kết quả vô nghĩa, nên phải so với manifest của index.
    """
    manifest = Path(index_dir) / "manifest.json"
    if not manifest.exists():
        return
    built = embedding_identity(json.loads(manifest.read_text(encoding="utf-8")).get("embedding") or {})
    wanted = embedding_identity(emb_cfg)
    if built != wanted:
        raise RuntimeError(f"Index {index_dir} dựng bằng {built} nhưng config truy xuất dùng {wanted}. "
                           "Dựng index đúng config: python scripts/build_kb.py --config <rag config>")


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
        check_index_embedding(self.index_dir, emb)
        self.encoder = SentenceTransformer(emb["model_id"], revision=emb.get("revision"))
        self.normalize = bool(emb.get("normalize_embeddings", True))
        self.segment = get_segmenter(emb.get("word_segmentation"))
        self.reranker = None
        rerank = rag_config.get("reranker", {})
        if rerank.get("enabled"):
            # KHÔNG được âm thầm bỏ reranker: trước đây lỗi nạp bị nuốt nên các run testset v03 (2026-10-25,
            # SFT v1) chạy không có reranker trong khi config bật -> protocol lệch mà không ai biết.
            try:
                from FlagEmbedding import FlagReranker
                path = rerank["model_id"]
                if rerank.get("revision"):
                    # Nạp đúng revision ghi trong config. Trước đây revision bị bỏ qua và luôn nạp bản mới nhất
                    # (đến 2026-11-20 bản mới nhất vẫn trùng revision đã ghim nên các run cũ không bị ảnh hưởng).
                    from huggingface_hub import snapshot_download
                    path = snapshot_download(rerank["model_id"], revision=rerank["revision"])
                self.reranker = FlagReranker(path, use_fp16=True, trust_remote_code=True)
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
        q = self.encoder.encode(  # embedding nhận câu hỏi đã tách từ (nếu config yêu cầu); reranker nhận nguyên văn
            [self.segment(query)], convert_to_numpy=True, normalize_embeddings=self.normalize
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
