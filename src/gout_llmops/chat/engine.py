from __future__ import annotations

from typing import Any

from gout_llmops.chat.history import append_turn
from gout_llmops.chat.prompt_builder import build_messages


class ChatEngine:
    def __init__(
        self,
        backend,
        retriever,
        system_prompt: str,
        max_history_pairs: int = 2,
        max_input_tokens: int = 4096,
    ):
        self.backend = backend
        self.retriever = retriever
        self.system_prompt = system_prompt
        self.max_history_pairs = max_history_pairs
        self.max_input_tokens = max_input_tokens

    def chat(self, question: str, history: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, str]]]:
        # Baseline v1 retrieval policy is current-question-only. Reference is intentionally absent.
        contexts = self.retriever.retrieve(question) if self.retriever else []
        messages = build_messages(
            system_prompt=self.system_prompt,
            question=question,
            history=history,
            contexts=contexts,
            max_history_pairs=self.max_history_pairs,
        )
        result = self.backend.generate(messages, max_input_tokens=self.max_input_tokens)
        answer = result["answer"]
        new_history = append_turn(history, question, answer)
        record = {
            **result,
            "question": question,
            "contexts": contexts,
            "messages": messages,
        }
        return record, new_history
