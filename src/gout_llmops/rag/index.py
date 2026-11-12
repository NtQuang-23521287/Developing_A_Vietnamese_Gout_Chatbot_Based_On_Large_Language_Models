from __future__ import annotations

from pathlib import Path

from gout_llmops.core.hashing import sha256_file
from gout_llmops.core.io import ensure_dir, write_json, write_jsonl
from gout_llmops.rag.chunking import Chunk


def build_faiss_index(chunks: list[Chunk], config: dict) -> dict:
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError("Install RAG dependencies: pip install -e '.[rag]'") from e

    out_dir = ensure_dir(config["output_dir"])
    emb_cfg = config["embedding"]
    model = SentenceTransformer(emb_cfg["model_id"], revision=emb_cfg.get("revision"))
    texts = [c.text for c in chunks]
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=bool(emb_cfg.get("normalize_embeddings", True)),
        show_progress_bar=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    write_jsonl(
        out_dir / "chunks.jsonl",
        ({"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata} for c in chunks),
    )
    source_inputs = []
    for src in config.get("sources", []):
        sp = Path(src["path"])
        source_inputs.append({
            "source_id": src.get("source_id"),
            "path": sp.as_posix(),
            "sha256": sha256_file(sp),
            "tier": src.get("tier"),
            "review_status": src.get("review_status"),
        })
    manifest = {
        "name": config["name"],
        "source_inputs": source_inputs,
        "documents_chunked": len(chunks),
        "embedding": emb_cfg,
        "chunking": config["chunking"],
        "retrieval": config["retrieval"],
        "reranker": config.get("reranker", {}),
        "index_sha256": sha256_file(out_dir / "index.faiss"),
        "chunks_sha256": sha256_file(out_dir / "chunks.jsonl"),
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest
