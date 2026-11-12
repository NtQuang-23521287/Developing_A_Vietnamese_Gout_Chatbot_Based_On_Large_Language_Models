"""Giai đoạn 5 — SafeChatEngine: ChatEngine + guard + độ bất định + chính sách an toàn (chế độ online).

Cùng hàm `apply_policy` với chế độ offline (scripts/safe_inference.py apply), nên câu trả lời online và offline
giống nhau khi cùng (câu trả lời gốc, guard, độ bất định, ngưỡng).

History lưu câu trả lời CUỐI (đã qua lớp an toàn) — đúng thứ người dùng nhìn thấy.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gout_llmops.chat.engine import ChatEngine
from gout_llmops.chat.history import append_turn
from gout_llmops.safety.guard import OUT_OF_SCOPE, run_guard
from gout_llmops.safety.policy import PolicyConfig, apply_policy


class SafeChatEngine:
    def __init__(self, engine: ChatEngine, policy: PolicyConfig | None = None,
                 uncertainty_fn: Callable[[dict[str, Any]], float] | None = None):
        self.engine = engine
        self.policy = policy or PolicyConfig()
        self.uncertainty_fn = uncertainty_fn
        if self.policy.threshold is not None and uncertainty_fn is None:
            raise ValueError("Có ngưỡng độ bất định nhưng không có uncertainty_fn")

    def chat(self, question: str, history: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, str]]]:
        prior_users = [m["content"] for m in history if m["role"] == "user"]
        guard = run_guard(question, prior_users)

        if self.policy.out_of_scope_refusal and guard.has(OUT_OF_SCOPE):
            decision = apply_policy("", guard, config=self.policy)   # không gọi model, không truy xuất
            record = {"question": question, "contexts": [], "answer": decision.final_answer, "answer_raw": None,
                      "safety": decision.to_dict()}
            return record, append_turn(history, question, decision.final_answer)

        record, _ = self.engine.chat(question, history)
        u = self.uncertainty_fn(record) if self.uncertainty_fn is not None else None
        decision = apply_policy(record["answer"], guard, uncertainty=u, config=self.policy,
                                contexts=record.get("contexts"))
        out = {**record, "answer": decision.final_answer, "answer_raw": record["answer"],
               "safety": decision.to_dict()}
        return out, append_turn(history, question, decision.final_answer)
