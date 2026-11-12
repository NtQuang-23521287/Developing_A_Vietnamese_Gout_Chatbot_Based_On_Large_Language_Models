from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gout_llmops.core.hashing import sha256_text
from gout_llmops.rag.documents import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


def char_window_chunks(doc: Document, chunk_chars: int = 700, overlap: int = 120) -> list[Chunk]:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be > 0")
    if overlap < 0 or overlap >= chunk_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_chars")
    text = " ".join(doc.text.split())
    if not text:
        return []
    step = chunk_chars - overlap
    chunks = []
    for idx, start in enumerate(range(0, len(text), step)):
        piece = text[start : start + chunk_chars].strip()
        if not piece:
            continue
        stable = sha256_text(f"{doc.doc_id}|{idx}|{piece}")[:16]
        meta = dict(doc.metadata)
        meta.update({"doc_id": doc.doc_id, "chunk_index": idx, "char_start": start})
        chunks.append(Chunk(chunk_id=f"{doc.doc_id}:chunk:{idx}:{stable}", text=piece, metadata=meta))
        if start + chunk_chars >= len(text):
            break
    return chunks


def chunk_documents(documents: list[Document], chunk_chars: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in documents:
        out.extend(char_window_chunks(doc, chunk_chars=chunk_chars, overlap=overlap))
    return out
