"""Tách từ tiếng Việt trước khi đưa vào embedding (config: embedding.word_segmentation).

Một số embedding tiếng Việt được huấn luyện trên văn bản đã tách từ, ví dụ
bkai-foundation-models/vietnamese-bi-encoder cần "bệnh_gút" thay vì "bệnh gút". Phải tách từ cả chunk (lúc dựng
index) lẫn câu hỏi (lúc truy xuất) bằng cùng một công cụ. Chỉ embedding nhận văn bản đã tách; chunks.jsonl, prompt
và reranker vẫn dùng văn bản gốc.
"""
from __future__ import annotations

from collections.abc import Callable

SEGMENTERS = ("none", "pyvi")


def get_segmenter(name: str | None) -> Callable[[str], str]:
    """Hàm tách từ theo tên trong config; None hoặc "none" = giữ nguyên văn bản (rag_v1)."""
    if name in (None, "", "none"):
        return lambda text: text
    if name == "pyvi":
        try:
            from pyvi.ViTokenizer import tokenize
        except ImportError as e:
            raise RuntimeError("Config RAG cần tách từ bằng pyvi: pip install -e '.[rag-vi]'") from e
        return tokenize
    raise ValueError(f"embedding.word_segmentation không hỗ trợ {name!r}; chỉ nhận {SEGMENTERS}")


def embedding_identity(emb_cfg: dict) -> dict:
    """Những trường quyết định vector của chunk: index và câu hỏi phải khớp đúng các trường này."""
    return {"model_id": emb_cfg.get("model_id"), "revision": emb_cfg.get("revision"),
            "normalize_embeddings": bool(emb_cfg.get("normalize_embeddings", True)),
            "word_segmentation": emb_cfg.get("word_segmentation") or "none"}
