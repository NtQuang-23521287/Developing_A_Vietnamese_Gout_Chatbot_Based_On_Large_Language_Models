from __future__ import annotations

from typing import Any

from gout_llmops.chat.history import tail_history


def format_contexts(contexts: list[dict[str, Any]]) -> str:
    blocks = []
    for i, c in enumerate(contexts, start=1):
        meta = c.get("metadata", {})
        source = meta.get("title") or meta.get("source_id") or c.get("chunk_id", "unknown")
        blocks.append(f"[Nguồn {i}: {source}]\n{c['text']}")
    return "\n\n".join(blocks) if blocks else "(Không có context RAG phù hợp.)"


def build_messages(
    system_prompt: str,
    question: str,
    history: list[dict[str, str]],
    contexts: list[dict[str, Any]],
    max_history_pairs: int,
) -> list[dict[str, str]]:
    system = system_prompt.rstrip() + "\n\nNGỮ CẢNH RAG:\n" + format_contexts(contexts)
    return [
        {"role": "system", "content": system},
        *tail_history(history, max_pairs=max_history_pairs),
        {"role": "user", "content": question},
    ]
